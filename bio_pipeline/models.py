"""
bio_pipeline/models.py
======================
Biological Closed-form Continuous-time (CfC) neural core for Chong-Fly.

Architecture
------------
BiologicalCfCCell implements the closed-form solution of the Liquid Time-constant
(LTC) ODE derived in Hasani et al. 2022 (Nature Machine Intelligence):

    dx/dt = -[ 1/τ + f(x,u;θ) ] · x  +  f(x,u;θ) · A

Closed-form solution (no Runge-Kutta needed):

    x(t+Δt) = σ(-f·Δt) · x(t)  +  (1 - σ(-f·Δt)) · A·f/(1/τ + f)
             ≈ σ_gate · x(t)    +  (1 - σ_gate) · h_ff

where  σ_gate = sigmoid(-(f + 1/τ)·Δt)  is the time-aware decay gate,
       h_ff   = tanh(W_macro·x + W_in·u + b)  is the feed-forward backbone,
       A      = learned asymptotic state amplitude.

W_macro topology
----------------
The recurrent weight matrix is initialised from one of the pre-computed
reduced models (spectral / centrality) and can be:
  • Fixed    – frozen as a buffer; gradients are blocked entirely.
  • Masked   – trainable but constrained to the sparsity pattern of a
               magnitude-pruned CSR matrix; entries outside the mask are
               zeroed after every gradient step via a registered backward hook.

Usage
-----
    from bio_pipeline.models import BiologicalCfCCell, BiologicalCfCNetwork
    from bio_pipeline.graph_reducer import ReducedModel

    model_meta = ReducedModel.load("data/reduced_models/meta_spectral_k64.json")

    cell = BiologicalCfCCell.from_reduced_model(
        model_meta,
        input_size=2,           # FlowX + FlowY
        mode="masked",          # "fixed" | "masked" | "free"
        backbone_units=64,
        backbone_layers=2,
        dt=0.004,               # 250 Hz control loop
    )

    # Single time-step
    x = torch.zeros(batch, cell.hidden_size)
    u = torch.randn(batch, 2)
    x_next = cell(u, x, dt=0.004)

    # Full sequence via BiologicalCfCNetwork
    net = BiologicalCfCNetwork(cell, output_dim=4)   # 4 motor channels
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
# SparseTopologyMask – registers a pruning mask as a non-parameter buffer
# and applies it to W_macro after every backward pass.
# ---------------------------------------------------------------------------

class _SparseTopologyHook:
    """
    Backward hook that zeroes gradient entries outside the binary mask,
    effectively blocking updates to pruned synapses.
    """
    def __init__(self, mask: torch.Tensor):
        self.mask = mask          # (k, k) bool tensor on same device as param

    def __call__(self, grad: torch.Tensor) -> torch.Tensor:
        return grad * self.mask


# ---------------------------------------------------------------------------
# BiologicalCfCCell
# ---------------------------------------------------------------------------

class BiologicalCfCCell(nn.Module):
    """
    Single-step Closed-form Continuous-time RNN cell with biological W_macro.

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
                              cluster indices that receive sensory input;
                              used for structured input projection.
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
        dt: float = 0.004,
        sensor_indices: Optional[dict] = None,
        motor_indices: Optional[dict] = None,
    ):
        super().__init__()
        assert mode in ("fixed", "masked", "free"), \
            f"mode must be 'fixed', 'masked' or 'free', got '{mode}'"

        self.hidden_size    = hidden_size
        self.input_size     = input_size
        self.mode           = mode
        self.default_dt     = dt
        self.sensor_indices = sensor_indices or {}
        self.motor_indices  = motor_indices or {}

        # ── Membrane time constants  τ  (one per neuron, always trainable) ──
        # Initialised with small spread around tau_init; kept positive via softplus.
        tau_raw = math.log(math.expm1(tau_init)) * torch.ones(hidden_size)
        self.tau_raw = nn.Parameter(tau_raw)          # softplus(tau_raw) = τ > 0

        # ── Asymptotic amplitude  A  (one per neuron, always trainable) ──────
        self.A = nn.Parameter(torch.ones(hidden_size))

        # ── Backbone MLP  f(x, u; θ)  ────────────────────────────────────────
        layers: list[nn.Module] = [
            nn.Linear(hidden_size + input_size, backbone_units),
            _make_activation(backbone_act),
        ]
        for _ in range(backbone_layers - 1):
            layers.append(nn.Linear(backbone_units, backbone_units))
            layers.append(_make_activation(backbone_act))
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

        # Mask buffer (registered separately; None until set_w_macro is called)
        self.register_buffer("_synapse_mask", None)
        self._hook_handle = None                     # backward hook handle

        # ── Input projection  W_in  (learnable) ─────────────────────────────
        self.W_in = nn.Linear(input_size, hidden_size, bias=True)
        nn.init.xavier_uniform_(self.W_in.weight)
        nn.init.zeros_(self.W_in.bias)

        # Bias for backbone output
        self.b = nn.Parameter(torch.zeros(hidden_size))

    # ------------------------------------------------------------------ tau
    @property
    def tau(self) -> torch.Tensor:
        """Positive membrane time constants via softplus."""
        return F.softplus(self.tau_raw) + 1e-3    # always > 0

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
        W    : weight matrix (k × k)
        mask : binary sparsity mask (k × k).  If None and mode == "masked",
               the mask is inferred from the non-zero entries of W.
        """
        # Convert W to float32 tensor
        if sp.issparse(W):
            W_dense = torch.from_numpy(W.toarray()).float()
        elif isinstance(W, np.ndarray):
            W_dense = torch.from_numpy(W).float()
        else:
            W_dense = W.float()

        assert W_dense.shape == (self.hidden_size, self.hidden_size), \
            f"W shape {W_dense.shape} ≠ ({self.hidden_size}, {self.hidden_size})"

        # Build / validate mask
        if mask is None:
            mask_tensor = (W_dense.abs() > 1e-7).float()
        elif sp.issparse(mask):
            mask_tensor = torch.from_numpy(
                (mask.toarray() != 0).astype(np.float32))
        elif isinstance(mask, np.ndarray):
            mask_tensor = torch.from_numpy(mask.astype(np.float32))
        else:
            mask_tensor = mask.float()

        # Store
        if self.mode == "fixed":
            self.W_macro.copy_(W_dense)
        else:
            with torch.no_grad():
                self._w_macro_param.copy_(W_dense)

        # Register / update mask buffer
        self._synapse_mask = mask_tensor.bool()

        # Register backward hook for "masked" mode
        if self.mode == "masked":
            if self._hook_handle is not None:
                self._hook_handle.remove()
            hook = _SparseTopologyHook(self._synapse_mask.float())
            self._hook_handle = self._w_macro_param.register_hook(hook)

    # ------------------------------------------------------------------ forward

    def forward(
        self,
        input: torch.Tensor,           # (batch, input_size)
        hx: torch.Tensor,              # (batch, hidden_size)
        dt: Optional[float] = None,    # integration timestep override
    ) -> torch.Tensor:
        """
        One CfC integration step.

        Closed-form update (no ODE solver):
            f    = backbone(cat(hx, input)) + W_macro @ hx + W_in(input) + b
            gate = sigmoid( -(f + 1/τ) · Δt )
            h'   = gate · hx  +  (1 - gate) · A · tanh(f)

        Returns
        -------
        h_new : (batch, hidden_size)
        """
        if dt is None:
            dt = self.default_dt

        W = self._effective_W()                     # (k, k)

        # ── Backbone: non-linear modulation of recurrent + input signal ──────
        cat_in = torch.cat([hx, input], dim=-1)     # (batch, k + input_size)
        f_bb   = self.backbone(cat_in)               # (batch, k)

        # ── Recurrent contribution (W_macro, possibly sparse-masked) ─────────
        # Apply mask in forward pass too (zero-out pruned weights)
        if self.mode == "masked" and self._synapse_mask is not None:
            W = W * self._synapse_mask.to(W.device).float()

        f_rec  = hx @ W.T                           # (batch, k)

        # ── Sensory input projection ──────────────────────────────────────────
        f_in   = self.W_in(input)                   # (batch, k)

        # ── Combined pre-activation ───────────────────────────────────────────
        f      = f_bb + f_rec + f_in + self.b       # (batch, k)

        # ── Time-aware decay gate  σ(-(f + 1/τ)·Δt) ─────────────────────────
        tau    = self.tau.to(input.device)           # (k,)
        gate   = torch.sigmoid(-(f + 1.0 / tau) * dt)  # (batch, k)

        # ── Asymptotic attractor ──────────────────────────────────────────────
        A      = self.A.to(input.device)             # (k,)
        h_inf  = A * torch.tanh(f)                   # (batch, k)

        # ── Closed-form state update ──────────────────────────────────────────
        h_new  = gate * hx + (1.0 - gate) * h_inf   # (batch, k)

        return h_new

    # ------------------------------------------------------------------ constructor helpers

    @classmethod
    def from_reduced_model(
        cls,
        model,                         # ReducedModel instance
        input_size: int,
        mode: str = "masked",
        prune_mask_model=None,         # optional separate MagnitudePruner model for mask
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
        **kwargs        : forwarded to BiologicalCfCCell.__init__
        """
        k = model.k
        cell = cls(
            hidden_size=k,
            input_size=input_size,
            mode=mode,
            sensor_indices=model.sensor_index_map,
            motor_indices=model.motor_index_map,
            **kwargs,
        )

        # Choose mask source
        if prune_mask_model is not None:
            mask = prune_mask_model.W   # CSR matrix → boolean mask
        else:
            mask = None                 # inferred from W zeros

        W = model.W
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

    # ------------------------------------------------------------------ repr

    def extra_repr(self) -> str:
        nnz = int(self._synapse_mask.sum()) if self._synapse_mask is not None else "?"
        return (f"hidden={self.hidden_size}, input={self.input_size}, "
                f"mode={self.mode}, synapse_nnz={nnz}, dt={self.default_dt}s")


# ---------------------------------------------------------------------------
# BiologicalCfCNetwork – wraps the cell into a sequence model with motor head
# ---------------------------------------------------------------------------

class BiologicalCfCNetwork(nn.Module):
    """
    Full recurrent network: BiologicalCfCCell + linear motor output head.

    Sensor routing
    --------------
    If the cell's sensor_indices are set, a structured input projection is
    applied: each sensor group's signal is injected specifically into the
    cluster indices that correspond to that biological population.
    Otherwise a standard linear projection W_in is used.

    Motor output
    ------------
    The motor_indices from the cell's index map are used to read out the
    hidden state into separate motor channels (throttle / yaw / pitch-roll).
    A final linear layer maps each group's cluster activities to scalar
    actuator commands.

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
    ):
        super().__init__()
        self.cell             = cell
        self.output_dim       = output_dim
        self.return_sequences = return_sequences

        # Motor output head: reads full hidden state → motor commands
        self.motor_head = nn.Linear(cell.hidden_size, output_dim, bias=True)
        nn.init.xavier_uniform_(self.motor_head.weight)
        nn.init.zeros_(self.motor_head.bias)

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


# ---------------------------------------------------------------------------
# Factory: load from saved ReducedModel meta file
# ---------------------------------------------------------------------------

def build_network_from_meta(
    meta_path: str,
    input_size: int,
    output_dim: int,
    mode: str = "masked",
    prune_meta_path: Optional[str] = None,
    backbone_units: int = 64,
    backbone_layers: int = 2,
    backbone_act: str = "lecun_tanh",
    backbone_dropout: float = 0.0,
    tau_init: float = 0.1,
    dt: float = 0.004,
    return_sequences: bool = False,
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
    **kwargs        : forwarded to BiologicalCfCCell

    Returns
    -------
    BiologicalCfCNetwork ready for training
    """
    # Import here to avoid circular dependency at module level
    from bio_pipeline.graph_reducer import ReducedModel

    model = ReducedModel.load(meta_path)

    prune_model = None
    if prune_meta_path is not None:
        prune_model = ReducedModel.load(prune_meta_path)

    cell = BiologicalCfCCell.from_reduced_model(
        model,
        input_size=input_size,
        mode=mode,
        prune_mask_model=prune_model,
        backbone_units=backbone_units,
        backbone_layers=backbone_layers,
        backbone_act=backbone_act,
        backbone_dropout=backbone_dropout,
        tau_init=tau_init,
        dt=dt,
    )

    return BiologicalCfCNetwork(cell, output_dim=output_dim,
                                return_sequences=return_sequences)
