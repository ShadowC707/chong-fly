"""Directed rate-network core with a versioned continuous-time contract.

For row-batched states and W[source, target]:
    drive(h,u) = h @ W + routed_input(u) + b
    target(h,u) = A * tanh(drive(h,u))
    dh/dt = (target(h,u) - h) / tau, tau > 0 (seconds)

Exponential Euler freezes target during each step. It is exact for constant
forcing and first-order for a general recurrent system. The class name and
solver alias CfC are retained for API compatibility; this implementation does
not claim equivalence to published CfC/LTC equations.

Structured mode has no dense recurrent backbone; inputs and motor readouts
follow declared routes. An explicitly unconstrained baseline retains the MLP.
These are engineering routes over the supplied graph, not proof of its biology.
"""

from __future__ import annotations

import json
import math
from typing import Optional

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.flight_config import DEFAULT_DT, COORDINATE_VERSION
from core.contracts import DYNAMICS_VERSION, MATRIX_ORIENTATION, TAU_FLOOR_S, canonical_solver
from core.routing import ROUTING_VERSION, RoutedLinear, sensor_mask, motor_mask


# ---------------------------------------------------------------------------
# Utility: LeCun tanh  (Yann LeCun 1998 — better-conditioned than plain tanh)
# ---------------------------------------------------------------------------

class _LeCunTanh(nn.Module):
    """1.7159 · tanh(0.666 · x)  — zero-mean, unit variance at init."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 1.7159 * torch.tanh(0.666 * x)


def _make_activation(name: str) -> nn.Module:
    mapping = {
        "lecun_tanh": _LeCunTanh,
        "tanh":       nn.Tanh,
        "relu":       nn.ReLU,
        "silu":       nn.SiLU,
        "gelu":       nn.GELU,
        "elu":        nn.ELU,
    }
    if name not in mapping:
        raise ValueError(f"Unknown backbone activation '{name}'. "
                         f"Choose from: {list(mapping)}")
    return mapping[name]()


# ---------------------------------------------------------------------------
# BiologicalCfCCell – rate dynamics with a masked directed recurrent operator.
# ---------------------------------------------------------------------------

class BiologicalCfCCell(nn.Module):
    """
    Directed leaky rate cell; historical CfC class name retained for compatibility.

    Parameters
    ----------
    hidden_size     : int   – recurrent dimension k (= macro-cluster count)
    input_size      : int   – sensory input dimension
    mode            : str   – "fixed" | "masked" | "free"
                        fixed  → W_macro is a frozen buffer (non-trainable)
                        masked → W_macro is trainable but constrained to the
                                 sparsity pattern of the pruning mask
                        free   → W_macro is fully trainable (ignores mask)
    backbone_units  : int   – hidden units per backbone MLP layer
    backbone_layers : int   – number of backbone MLP layers (≥ 1)
    backbone_act    : str   – activation for backbone MLP
    backbone_dropout: float – dropout probability inside backbone
    tau_init        : float – initial membrane time-constant τ (seconds)
    dt              : float – default integration timestep Δt (seconds).
                              Can be overridden per forward() call.
    sensor_indices  : dict  – {"lptc_flow": [...], "lc_looming": [...]}
                              input populations enforced by the structured adapter.
    motor_indices   : dict  – {"throttle": [...], "yaw": [...], ...}
                              cluster indices read out as motor commands.
    """

    def __init__(
        self,
        hidden_size: int,
        input_size: int,
        mode: str = "masked",
        backbone_units: int = 64,
        backbone_layers: int = 2,
        backbone_act: str = "lecun_tanh",
        backbone_dropout: float = 0.0,
        tau_init: float = 0.1,
        dt: float = DEFAULT_DT,
        solver_type: str = "exponential_euler",
        sensor_indices: Optional[dict] = None,
        motor_indices: Optional[dict] = None,
        connectivity: str = "structured",
        input_routes: Optional[dict] = None,
    ):
        super().__init__()
        if mode not in ("fixed", "masked", "free"):
            raise ValueError(f"Unknown mode {mode!r}")
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("Default dt must be positive and finite")
        if not math.isfinite(tau_init) or tau_init <= TAU_FLOOR_S:
            raise ValueError(f"tau_init must be finite and greater than {TAU_FLOOR_S} seconds")

        self.hidden_size    = hidden_size
        self.input_size     = input_size
        self.mode           = mode
        self.solver_type    = canonical_solver(solver_type)
        if connectivity not in {"structured", "unconstrained"}:
            raise ValueError("connectivity must be structured or unconstrained")
        if connectivity == "structured" and mode == "free":
            raise ValueError("mode='free' requires explicit connectivity='unconstrained'")
        self.connectivity = connectivity
        self.default_dt = float(dt)
        self._architecture_contract = {
            "hidden_size": hidden_size, "input_size": input_size,
            "backbone_units": backbone_units, "backbone_layers": backbone_layers,
            "backbone_act": backbone_act, "backbone_dropout": backbone_dropout,
            "connectivity": connectivity,
        }
        self.sensor_indices = sensor_indices or {}
        self.motor_indices  = motor_indices or {}

        # ── Membrane time constants  τ  (one per neuron, always trainable) ──
        # Initialise at tau_init; stable inverse softplus includes the positive floor.
        tau_value = tau_init - TAU_FLOOR_S
        tau_raw = (tau_value + math.log(-math.expm1(-tau_value))) * torch.ones(hidden_size)
        self.tau_raw = nn.Parameter(tau_raw)          # softplus(tau_raw) = τ > 0

        # ── Asymptotic amplitude  A  (one per neuron, always trainable) ──────
        self.A = nn.Parameter(torch.ones(hidden_size))

        # ── Backbone MLP  f(x, u; θ)  ────────────────────────────────────────
        self.backbone = nn.Identity()  # no trainable bypass in structured mode
        if connectivity == "unconstrained":
            layers: list[nn.Module] = [nn.Linear(hidden_size + input_size, backbone_units),
                                      _make_activation(backbone_act)]
            for _ in range(backbone_layers - 1):
                layers.extend([nn.Linear(backbone_units, backbone_units), _make_activation(backbone_act)])
                if backbone_dropout > 0.0:
                    layers.append(nn.Dropout(backbone_dropout))
            layers.append(nn.Linear(backbone_units, hidden_size))
            self.backbone = nn.Sequential(*layers)

        # ── W_macro  (recurrent biological weight matrix) ───────────────────
        # Placeholder – populated via from_reduced_model() or set_w_macro()
        k = hidden_size
        W_zeros = torch.zeros(k, k)
        if mode == "fixed":
            self.register_buffer("W_macro", W_zeros)
            self._w_macro_param = None
        else:
            self._w_macro_param = nn.Parameter(W_zeros)
            self.register_buffer("W_macro", None)   # will use param in forward

        # An uninitialized masked graph has no edges. Keep shape stable for checkpoint loading.
        self.register_buffer("_synapse_mask", torch.zeros(k, k, dtype=torch.bool))
        self._hook_handle = None                     # backward hook handle
        if mode == "masked":
            self._hook_handle = self._w_macro_param.register_hook(self._mask_gradient)

        # ── Input projection  W_in  (learnable) ─────────────────────────────
        if connectivity == "structured":
            self.W_in = RoutedLinear(sensor_mask(hidden_size, input_size, self.sensor_indices, input_routes))
            self._architecture_contract["input_routing"] = self.W_in.get_extra_state()
        else:
            self.W_in = nn.Linear(input_size, hidden_size, bias=True)
        nn.init.xavier_uniform_(self.W_in.weight)
        nn.init.zeros_(self.W_in.bias)
        if isinstance(self.W_in, RoutedLinear):
            self.W_in.apply_mask()

        # Bias for backbone output
        self.b = nn.Parameter(torch.zeros(hidden_size))

    # ------------------------------------------------------------------ tau
    @property
    def tau(self) -> torch.Tensor:
        """Positive membrane time constants via softplus."""
        return F.softplus(self.tau_raw) + TAU_FLOOR_S

    def _mask_gradient(self, grad):
        return grad * self._synapse_mask.to(grad.device)

    def get_extra_state(self):
        """Guard against interpreting old trained weights under new dynamics."""
        return {"dynamics_version": DYNAMICS_VERSION, "matrix_orientation": MATRIX_ORIENTATION,
                "coordinate_version": COORDINATE_VERSION,
                "routing_version": ROUTING_VERSION,
                "solver": self.solver_type, "default_dt": self.default_dt, "mode": self.mode,
                "architecture": dict(self._architecture_contract)}

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise RuntimeError("Incompatible neural contract in checkpoint; rebuild/retrain the policy")

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        if state_dict.get(prefix + "_extra_state") != self.get_extra_state():
            error_msgs.append(prefix + "incompatible or missing neural contract (legacy checkpoint); retrain required")
            return
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    # ------------------------------------------------------------------ W_macro accessors

    def _effective_W(self) -> torch.Tensor:
        """Return the W_macro tensor used in forward (buffer or param)."""
        if self.mode == "fixed":
            return self.W_macro                    # frozen buffer
        return self._w_macro_param                 # masked / free param

    # ------------------------------------------------------------------ topology loading

    def set_w_macro(
        self,
        W: np.ndarray | sp.csr_matrix | torch.Tensor,
        mask: Optional[np.ndarray | sp.csr_matrix | torch.Tensor] = None,
    ):
        """
        Initialise W_macro from a numpy array, scipy CSR, or torch tensor.

        Parameters
        ----------
        W    : source_target weight matrix (k × k); W[source, target]
        mask : binary sparsity mask (k × k).  If None and mode == "masked",
               the mask is inferred from the non-zero entries of W.
        """
        parameter = self._effective_W()
        W_dense = torch.as_tensor(W.toarray() if sp.issparse(W) else W,
                                  dtype=parameter.dtype, device=parameter.device)
        shape = (self.hidden_size, self.hidden_size)
        if W_dense.shape != shape or not torch.isfinite(W_dense).all():
            raise ValueError(f"W must be a finite {shape} source_target matrix")
        if mask is None:
            mask_tensor = W_dense != 0
        else:
            raw_mask = torch.as_tensor(mask.toarray() if sp.issparse(mask) else mask,
                                       device=parameter.device)
            if raw_mask.shape != shape or not torch.isfinite(raw_mask).all():
                raise ValueError(f"mask must be finite with shape {shape}")
            mask_tensor = (raw_mask != 0) & (W_dense != 0)
        with torch.no_grad():
            parameter.copy_(W_dense if self.mode == "free" else W_dense * mask_tensor)
            self._synapse_mask.copy_(mask_tensor)

    # ------------------------------------------------------------------ forward

    def forward(self, input: torch.Tensor, hx: torch.Tensor,
                dt: Optional[float] = None) -> torch.Tensor:
        """Integrate dh/dt = (A*tanh(drive(h,u))-h)/tau.

        Both solvers evaluate drive at the old state. Exponential Euler freezes
        that target for this step; it is not an exact nonlinear ODE solution.
        """
        dt = self.default_dt if dt is None else dt
        if not math.isfinite(dt) or dt < 0:
            raise ValueError("dt must be finite and nonnegative")
        if dt == 0:
            return hx
        W = self._effective_W()
        if self.mode == "masked":
            W = W * self._synapse_mask
        f = hx @ W + self.W_in(input) + self.b
        if self.connectivity == "unconstrained":
            f = f + self.backbone(torch.cat([hx, input], dim=-1))
        target = self.A * torch.tanh(f)
        if self.solver_type == "euler":
            return hx + (dt / self.tau) * (target - hx)
        # expm1 avoids cancellation when dt is very small.
        fraction = -torch.expm1(-dt / self.tau)
        return hx + fraction * (target - hx)

    # ------------------------------------------------------------------ constructor helpers

    @classmethod
    def from_reduced_model(
        cls,
        model,                         # ReducedModel instance
        input_size: int,
        mode: str = "masked",
        prune_mask_model=None,         # optional separate MagnitudePruner model for mask
        pruning_sparsity: Optional[float] = None,
        solver_type: str = "exponential_euler",
        **kwargs,
    ) -> "BiologicalCfCCell":
        """
        Construct a BiologicalCfCCell pre-loaded with a ReducedModel's W_macro.

        Parameters
        ----------
        model           : ReducedModel (spectral or centrality)
        input_size      : sensory input dimension
        mode            : "fixed" | "masked" | "free"
        prune_mask_model: optional ReducedModel from MagnitudePruner — its
                          sparsity pattern is used as the synapse mask.
        pruning_sparsity: optional float in [0.0, 1.0) for magnitude pruning.
        solver_type     : 'exponential_euler' | 'euler'; old spellings are aliases
        **kwargs        : forwarded to BiologicalCfCCell.__init__
        """
        k = model.k
        cell = cls(
            hidden_size=k,
            input_size=input_size,
            mode=mode,
            solver_type=solver_type,
            sensor_indices=model.sensor_index_map,
            motor_indices=model.motor_index_map,
            **kwargs,
        )

        W = model.W
        # Choose mask / pruning source
        if pruning_sparsity is not None:
            if not math.isfinite(pruning_sparsity) or not 0 <= pruning_sparsity < 1:
                raise ValueError("pruning_sparsity must be in [0, 1)")
            W_dense = W.toarray() if sp.issparse(W) else np.array(W, copy=True)
            k_pct = float(pruning_sparsity) * 100.0
            threshold = float(np.percentile(np.abs(W_dense), k_pct))
            mask = ((np.abs(W_dense) >= threshold) & (W_dense != 0)).astype(np.float32)
            W_pruned = W_dense * mask
            cell.set_w_macro(W_pruned, mask=mask)
        elif prune_mask_model is not None:
            if prune_mask_model.k != model.k or not np.array_equal(prune_mask_model.cluster_map, model.cluster_map):
                raise ValueError("Pruning mask must use the same cluster mapping as the model")
            mask = prune_mask_model.W   # CSR matrix → boolean mask
            cell.set_w_macro(W, mask=mask)
        else:
            mask = None                 # inferred from W zeros
            cell.set_w_macro(W, mask=mask)

        return cell

    # ------------------------------------------------------------------ apply mask (post-step)

    @torch.no_grad()
    def apply_topology_mask(self):
        """
        Hard-zero W_macro entries outside the sparsity mask.
        Call after each optimizer.step() when mode == "masked".
        """
        if self.mode == "masked" and self._synapse_mask is not None:
            mask = self._synapse_mask.to(self._w_macro_param.device).float()
            self._w_macro_param.mul_(mask)
        if isinstance(self.W_in, RoutedLinear):
            self.W_in.apply_mask()

    # ------------------------------------------------------------------ repr

    def extra_repr(self) -> str:
        nnz = int(self._synapse_mask.sum()) if self._synapse_mask is not None else "?"
        return (f"hidden={self.hidden_size}, input={self.input_size}, "
                f"mode={self.mode}, solver={self.solver_type}, synapse_nnz={nnz}, dt={self.default_dt}s")


# ---------------------------------------------------------------------------
# BiologicalCfCNetwork – wraps the cell into a sequence model with motor head
# ---------------------------------------------------------------------------

class BiologicalCfCNetwork(nn.Module):
    """
    Full recurrent network: BiologicalCfCCell + linear motor output head.

    In structured mode, both the cell input and this motor head use declared
    routes. ChongFlyMSPPolicy replaces the generic head with its single DN head.
    Dense input/backbone/readout is available only in the unconstrained baseline.

    Parameters
    ----------
    cell        : BiologicalCfCCell
    output_dim  : int  — total number of motor output scalars
                  (e.g. 4: throttle, yaw_L, yaw_R, pitch_roll)
    return_sequences : bool — if True, return all hidden states; else last only
    """

    def __init__(
        self,
        cell: BiologicalCfCCell,
        output_dim: int = 4,
        return_sequences: bool = False,
        include_motor_head: bool = True,
    ):
        super().__init__()
        self.cell             = cell
        self.output_dim       = output_dim
        self.return_sequences = return_sequences

        # The optional generic head follows the same channel mapping as the policy.
        self.motor_head = None
        if include_motor_head:
            if cell.connectivity == "structured":
                if output_dim != 4:
                    raise ValueError("Structured motor mapping requires four output channels")
                self.motor_head = RoutedLinear(motor_mask(cell.hidden_size, cell.motor_indices))
            else:
                self.motor_head = nn.Linear(cell.hidden_size, output_dim, bias=True)
            nn.init.xavier_uniform_(self.motor_head.weight)
            nn.init.zeros_(self.motor_head.bias)
            if isinstance(self.motor_head, RoutedLinear):
                self.motor_head.apply_mask()

    def forward(
        self,
        inputs: torch.Tensor,                   # (batch, T, input_size)
        hx: Optional[torch.Tensor] = None,      # (batch, hidden_size) or None
        dt: Optional[float] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        inputs : (batch, T, input_size)
        hx     : initial hidden state; zeros if None
        dt     : integration timestep

        Returns
        -------
        outputs : (batch, T, output_dim)  if return_sequences
                  (batch, output_dim)     if not return_sequences
        h_last  : (batch, hidden_size)    final hidden state
        """
        if self.motor_head is None:
            raise RuntimeError("This network is a policy encoder; use ChongFlyMSPPolicy for motor outputs")
        B, T, _ = inputs.shape
        if hx is None:
            hx = torch.zeros(B, self.cell.hidden_size,
                             device=inputs.device, dtype=inputs.dtype)

        out_seq = []
        h = hx
        for t in range(T):
            h = self.cell(inputs[:, t, :], h, dt=dt)
            out_seq.append(h)

        # Stack: (batch, T, hidden_size)
        all_h = torch.stack(out_seq, dim=1)

        # Motor projection
        motor_out = self.motor_head(all_h)          # (batch, T, output_dim)

        if self.return_sequences:
            return motor_out, h
        else:
            return motor_out[:, -1, :], h           # last timestep only

    @torch.no_grad()
    def post_step(self):
        """
        Call after optimizer.step() to enforce sparsity mask on W_macro.
        """
        self.cell.apply_topology_mask()
        if isinstance(self.motor_head, RoutedLinear):
            self.motor_head.apply_mask()


# ---------------------------------------------------------------------------
# Factory: load from saved ReducedModel meta file
# ---------------------------------------------------------------------------

def build_network_from_meta(
    meta_path: str,
    input_size: int,
    output_dim: int,
    mode: str = "masked",
    prune_meta_path: Optional[str] = None,
    pruning_sparsity: Optional[float] = None,
    solver_type: str = "exponential_euler",
    ablate_cx: Optional[bool] = None,
    backbone_units: int = 64,
    backbone_layers: int = 2,
    backbone_act: str = "lecun_tanh",
    backbone_dropout: float = 0.0,
    tau_init: float = 0.1,
    dt: float = DEFAULT_DT,
    return_sequences: bool = False,
    connectivity: str = "structured",
    input_routes: Optional[dict] = None,
    include_motor_head: bool = True,
) -> BiologicalCfCNetwork:
    """
    One-call factory: load a ReducedModel from disk and build the full network.

    Parameters
    ----------
    meta_path       : path to meta_spectral_k64.json (or centrality)
    input_size      : sensory input channels (e.g. 2 for FlowX/Y)
    output_dim      : motor output channels (e.g. 4)
    mode            : "fixed" | "masked" | "free"
    prune_meta_path : optional meta_magnitude_p90_k12942.json — its sparsity
                      pattern is used as synapse mask (overrides W zeros)
    pruning_sparsity: optional float in [0.0, 1.0) for magnitude pruning
    solver_type     : 'exponential_euler' | 'euler'; old spellings are aliases
    ablate_cx       : if True, load _nocx version of model if available
    **kwargs        : forwarded to BiologicalCfCCell

    Returns
    -------
    BiologicalCfCNetwork ready for training
    """
    # Import here to avoid circular dependency at module level
    import os
    try:
        from generator.graph_reducer import ReducedModel
    except ImportError:
        from bio_pipeline.graph_reducer import ReducedModel

    if ablate_cx and "_nocx" not in meta_path:
        nocx_meta = meta_path.replace(".json", "_nocx.json")
        if os.path.exists(nocx_meta):
            meta_path = nocx_meta

    model = ReducedModel.load(meta_path)

    prune_model = None
    if prune_meta_path is not None:
        prune_model = ReducedModel.load(prune_meta_path)

    cell = BiologicalCfCCell.from_reduced_model(
        model,
        input_size=input_size,
        mode=mode,
        prune_mask_model=prune_model,
        pruning_sparsity=pruning_sparsity,
        solver_type=solver_type,
        backbone_units=backbone_units,
        backbone_layers=backbone_layers,
        backbone_act=backbone_act,
        backbone_dropout=backbone_dropout,
        tau_init=tau_init,
        dt=dt,
        connectivity=connectivity,
        input_routes=input_routes,
    )

    return BiologicalCfCNetwork(cell, output_dim=output_dim,
        return_sequences=return_sequences, include_motor_head=include_motor_head)
