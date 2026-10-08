"""
simulation/policy.py
====================
ChongFlyMSPPolicy — Biological CfC → MSP/RC PWM translation layer.

Sensor layout (66 values total)
--------------------------------
    [0:2]   FlowX, FlowY     — optical flow (m/s, normalised ±1)
    [2:66]  ToF_0 … ToF_63   — 8×8 ToF depth grid (m, normalised 0…1)

Output channels (4 RC PWM values, µs)
--------------------------------------
    ch[0]  Throttle   1000 + sigmoid(NNout) × 1000   ∈ [1000, 2000]
    ch[1]  Roll       1500 +    tanh(NNout) × 500    ∈ [1000, 2000]
    ch[2]  Pitch      1500 +    tanh(NNout) × 500    ∈ [1000, 2000]
    ch[3]  Yaw        1500 +    tanh(NNout) × 500    ∈ [1000, 2000]

Architecture
------------

    SensorInputLayer          — structured input normalisation
           ↓  (batch, 66)
    BiologicalCfCNetwork      — recurrent CfC with W_macro topology
           ↓  (batch, k)   hidden state
    DNProjectionHead          — sparse read-out from DN motor clusters
           ↓  (batch, 4)   raw motor logits
    PWMOutputLayer            — affine transform → RC PWM µs
           ↓  (batch, 4)   [throttle, roll, pitch, yaw] in µs

Usage
-----
    from simulation.policy import ChongFlyMSPPolicy

    policy = ChongFlyMSPPolicy.from_meta(
        meta_path  = "data/reduced_models/meta_spectral_k64.json",
        mode       = "masked",
        backbone_units  = 64,
        backbone_layers = 2,
        dt         = 0.004,       # 250 Hz
    )

    # Numpy path (flight-loop friendly, no grad)
    pwm = policy.step_np(flow_xy=np.array([0.1, -0.05]),
                         tof_8x8=np.zeros(64))
    # → array([1500, 1502, 1498, 1501], dtype=float32)  [µs]

    # Torch path (training)
    obs = torch.randn(batch, 66)
    pwm_t, h = policy(obs, hx)      # (batch, 4), (batch, k)
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Make project root importable when run directly
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    from core.models import BiologicalCfCCell, BiologicalCfCNetwork
except (ImportError, AttributeError):
    BiologicalCfCCell = Any  # type: ignore
    BiologicalCfCNetwork = Any  # type: ignore


# ─────────────────────────────────────────────────────────────────────────────
# Constants (imported from centralized config)
# ─────────────────────────────────────────────────────────────────────────────

from configs.flight_config import (
    SENSOR_DIM_BASE,
    MEMORY_DIM,
    SENSOR_DIM,
    FLOW_DIM,
    TOF_DIM,
    N_CONTROLS,
    PWM_MIN,
    PWM_MID,
    PWM_MAX,
    PWM_HALF,
    CH_THROTTLE,
    CH_ROLL,
    CH_PITCH,
    CH_YAW,
    CHANNEL_KEYS,
    DEFAULT_DT,
    MEMORY_DEFAULT_DISTANCE, PWM_HOVER,
)



# ─────────────────────────────────────────────────────────────────────────────
# SensorInputLayer
# ─────────────────────────────────────────────────────────────────────────────

class SensorInputLayer(nn.Module):
    """
    Normalise and optionally scale the 74-D (or 66-D legacy) sensor vector.

    FlowX/Y       : expected range ±1 (already normalised by caller)
    ToF 8×8       : expected range [0, 1] (distance / max_range)
    Memory Ring 8 : expected range [0, 1] (egocentric obstacle distance)

    An optional learnable affine rescaling (per-channel gain + bias) is
    applied after normalisation so the network can adapt to sensor offsets.
    """

    def __init__(self, sensor_dim: int = SENSOR_DIM, learnable_scale: bool = True,
                 encoding: str = 'distance-v1'):
        super().__init__()
        if encoding not in {'distance-v1', 'threat-v1', 'proximity-v1', 'proximity-mean-v1'}:
            raise ValueError('Unknown sensor encoding')
        self.sensor_dim = sensor_dim
        self.encoding = encoding

        if learnable_scale:
            # Per-sensor gain (initialised to 1) and bias (initialised to 0)
            self.gain = nn.Parameter(torch.ones(sensor_dim))
            self.bias = nn.Parameter(torch.zeros(sensor_dim))
        else:
            self.register_buffer("gain", torch.ones(sensor_dim))
            self.register_buffer("bias", torch.zeros(sensor_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, 74) or (batch, 66) — raw sensor vector
            x[:, 0:2]   = [FlowX, FlowY]    ∈ [-1, 1]
            x[:, 2:66]  = ToF pixels        ∈ [0, 1]
            x[:, 66:74] = Egocentric Memory ∈ [0, 1]
        Returns normalised (batch, sensor_dim).
        """
        if x.shape[-1] == SENSOR_DIM_BASE and self.sensor_dim == SENSOR_DIM:
            # Legacy 66-D passed to 74-D layer -> pad with safe default memory (1.0)
            pad = torch.ones(*x.shape[:-1], MEMORY_DIM, dtype=x.dtype, device=x.device)
            x = torch.cat([x, pad], dim=-1)
        elif x.shape[-1] == SENSOR_DIM and self.sensor_dim == SENSOR_DIM_BASE:
            # 74-D passed to 66-D layer -> drop memory suffix
            x = x[..., :SENSOR_DIM_BASE]

        if self.encoding == 'threat-v1':
            # An engineering encoding of distance, not a claim about LC responses.
            # Flow keeps its sign; ToF/memory have zero drive in a clear scene.
            x = torch.cat([x[..., :FLOW_DIM], 1-x[..., FLOW_DIM:]], dim=-1)
        elif self.encoding in {'proximity-v1', 'proximity-mean-v1'}:
            from configs.flight_config import APF_DISTANCE_THRESHOLD_M, TOF_RAYCASTER_MAX_RANGE_M
            # A per-pixel distance encoding only. No action, laterality, or
            # sensor->motor shortcut: all commands still traverse the graph.
            depth = torch.clamp(1-x[..., FLOW_DIM:SENSOR_DIM_BASE] /
                                (APF_DISTANCE_THRESHOLD_M/TOF_RAYCASTER_MAX_RANGE_M), min=0)
            if self.encoding == 'proximity-mean-v1':
                # Explicit experimental divisive normalization: preserve the
                # spatial pattern, but decouple total drive from occupied area.
                # Zero-support stays zero; one pixel has a bounded factor 32.
                support = (depth > 0).sum(dim=-1, keepdim=True).clamp_min(1)
                depth = depth * (32./support)
            x = torch.cat([x[..., :FLOW_DIM], depth, 1-x[..., SENSOR_DIM_BASE:]], dim=-1)
        return x * self.gain + self.bias

    def from_numpy(
        self,
        flow_xy: np.ndarray,
        tof_8x8: np.ndarray,
        memory_ring: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        """
        Convenience: pack numpy arrays → (1, sensor_dim) tensor.

        Parameters
        ----------
        flow_xy     : (2,) float32  [FlowX, FlowY] normalised ±1
        tof_8x8     : (64,) float32  ToF pixels normalised [0,1]
        memory_ring : optional (8,) float32 egocentric spatial memory
        """
        flow_part = np.asarray(flow_xy, dtype=np.float32).ravel()[:FLOW_DIM]
        tof_part = np.asarray(tof_8x8, dtype=np.float32).ravel()[:TOF_DIM]
        parts = [flow_part, tof_part]
        if memory_ring is not None:
            parts.append(np.asarray(memory_ring, dtype=np.float32).ravel()[:MEMORY_DIM])

        raw = np.concatenate(parts)
        return torch.from_numpy(raw).unsqueeze(0).to(self.gain.device)


# ─────────────────────────────────────────────────────────────────────────────
# DNProjectionHead
# ─────────────────────────────────────────────────────────────────────────────

class DNProjectionHead(nn.Module):
    """Weighted readout from explicitly mapped DN clusters, without implicit fallbacks.

    pitch_roll is a supported legacy alias for both channels. Shared clusters
    remain shared; masking cannot restore motor degrees of freedom lost in reduction.
    """
    def __init__(self, hidden_size, motor_index_map=None, *, allow_dense_fallback=False):
        super().__init__()
        import warnings
        from core.routing import RoutedLinear, motor_mask, structural_rank
        self.hidden_size = hidden_size
        self.motor_index_map = motor_index_map or {}
        self.proj = RoutedLinear(motor_mask(hidden_size, self.motor_index_map, allow_dense_fallback))
        self.use_sparse = True
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        import math
        target_th = max(.001, min(.999, (PWM_HOVER-PWM_MIN)/(PWM_MAX-PWM_MIN)))
        with torch.no_grad():
            self.proj.bias[CH_THROTTLE] = math.log(target_th/(1-target_th))
        self.proj.apply_mask()
        self.structural_rank = structural_rank(self._proj_mask)
        if self.structural_rank < N_CONTROLS:
            warnings.warn(f"Motor mapping supports at most {self.structural_rank}/4 independent "
                          "readout directions; reduction has merged motor roles",
                          UserWarning, stacklevel=2)

    @property
    def _proj_mask(self):
        return self.proj.route_mask

    def forward(self, h):
        return self.proj(h)


class PWMOutputLayer(nn.Module):
    """
    Affine transform: raw NN logits → RC PWM microseconds.

    Throttle  ch[0]: sigmoid(x) ∈ [0,1]  → 1000 + out × 1000  ∈ [1000, 2000]
    Roll      ch[1]: tanh(x)   ∈ [-1,1]  → 1500 + out × 500   ∈ [1000, 2000]
    Pitch     ch[2]: tanh(x)   ∈ [-1,1]  → 1500 + out × 500   ∈ [1000, 2000]
    Yaw       ch[3]: tanh(x)   ∈ [-1,1]  → 1500 + out × 500   ∈ [1000, 2000]
    """

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        logits : (batch, 4)  raw motor signals
        Returns pwm : (batch, 4)  in microseconds [1000, 2000]
        """
        throttle = PWM_MIN  + torch.sigmoid(logits[:, CH_THROTTLE:CH_THROTTLE+1]) * (PWM_MAX - PWM_MIN)
        roll     = PWM_MID  + torch.tanh(logits[:, CH_ROLL:CH_ROLL+1])       * PWM_HALF
        pitch    = PWM_MID  + torch.tanh(logits[:, CH_PITCH:CH_PITCH+1])     * PWM_HALF
        yaw      = PWM_MID  + torch.tanh(logits[:, CH_YAW:CH_YAW+1])        * PWM_HALF
        return torch.cat([throttle, roll, pitch, yaw], dim=-1)   # (batch, 4)

    @staticmethod
    def decode_np(pwm: np.ndarray) -> dict[str, float]:
        """Human-readable dict from a (4,) numpy PWM array."""
        return {
            "throttle_us": float(pwm[CH_THROTTLE]),
            "roll_us":     float(pwm[CH_ROLL]),
            "pitch_us":    float(pwm[CH_PITCH]),
            "yaw_us":      float(pwm[CH_YAW]),
            "throttle_pct": round((float(pwm[CH_THROTTLE]) - PWM_MIN) / (PWM_MAX - PWM_MIN) * 100, 1),
            "roll_norm":    round((float(pwm[CH_ROLL])  - PWM_MID) / PWM_HALF, 4),
            "pitch_norm":   round((float(pwm[CH_PITCH]) - PWM_MID) / PWM_HALF, 4),
            "yaw_norm":     round((float(pwm[CH_YAW])   - PWM_MID) / PWM_HALF, 4),
        }


# ─────────────────────────────────────────────────────────────────────────────
# ChongFlyMSPPolicy — top-level module
# ─────────────────────────────────────────────────────────────────────────────

class ChongFlyMSPPolicy(nn.Module):
    """
    Full sensor→PWM policy pipeline:

        SensorInputLayer (66)
            ↓
        BiologicalCfCNetwork (k)
            ↓
        DNProjectionHead (4 logits)
            ↓
        PWMOutputLayer → [1000…2000] µs

    Parameters
    ----------
    cfc_network     : BiologicalCfCNetwork  (pre-built or from factory)
    sensor_dim      : total sensory input size (default 66)
    learnable_scale : whether SensorInputLayer has trainable gain/bias
    """

    def __init__(
        self,
        cfc_network: BiologicalCfCNetwork,
        sensor_dim: int = SENSOR_DIM,
        learnable_scale: bool = True,
        allow_dense_motor_fallback: bool = False,
        sensor_encoding: str = 'distance-v1',
        neutral_origin: bool = False,
    ):
        super().__init__()
        cell = cfc_network.cell
        if type(neutral_origin) is not bool:
            raise ValueError('neutral_origin must be an explicit boolean')
        if neutral_origin and (sensor_encoding not in {'threat-v1', 'proximity-v1', 'proximity-mean-v1'} or cell.connectivity != 'structured'):
            raise ValueError('Neutral origin requires threat/proximity encoding and structured connectivity')
        self.neutral_origin = neutral_origin
        self.training_admission = getattr(cfc_network, 'graph_provenance', {}).get('training_ready')
        if self.training_admission is not None and type(self.training_admission) is not bool:
            raise ValueError('Invalid graph training admission flag')

        self.sensor_layer  = SensorInputLayer(sensor_dim, learnable_scale, sensor_encoding)
        if sensor_encoding != 'distance-v1':
            # Preserve the historical distance contract, but reject loading its
            # weights into a differently encoded model (even with strict=False).
            cell._architecture_contract['sensor_encoding'] = sensor_encoding
            if sensor_encoding in {'proximity-v1', 'proximity-mean-v1'}:
                from configs.flight_config import APF_DISTANCE_THRESHOLD_M, TOF_RAYCASTER_MAX_RANGE_M
                cell._architecture_contract['sensor_encoding_parameters'] = {
                    'threshold_m':APF_DISTANCE_THRESHOLD_M, 'max_range_m':TOF_RAYCASTER_MAX_RANGE_M}
                if sensor_encoding == 'proximity-mean-v1':
                    cell._architecture_contract['sensor_encoding_parameters']['reference_active_pixels'] = 32
        self.cfc_network   = cfc_network
        if allow_dense_motor_fallback and cell.connectivity != "unconstrained":
            raise ValueError("Dense motor fallback requires an explicit unconstrained baseline")
        self.dn_head = DNProjectionHead(cell.hidden_size, cell.motor_indices,
                                        allow_dense_fallback=allow_dense_motor_fallback)
        if neutral_origin:
            # Rate deviations around a zero engineering reference. Cruise pitch
            # retains its readout bias; constant neural/input drive and yaw bias
            # cannot manufacture a turn with zero sensors and zero hidden state.
            cell._architecture_contract['neutral_origin'] = 'zero-drive-yaw-v1'
            with torch.no_grad():
                for bias in (self.sensor_layer.bias, cell.W_in.bias, cell.b):
                    bias.zero_()
                    bias.requires_grad_(False)
                cell.W_in.bias_mask.zero_()
                self.dn_head.proj.bias_mask[CH_YAW] = False
                self.dn_head.proj.apply_mask()
        self.cfc_network.motor_head = None  # the policy owns the single active DN readout
        self.sensor_dim = sensor_dim
        if sensor_dim != cell.input_size:
            raise ValueError("Policy sensor_dim must match the cell input_size")
        self.routing_diagnostics = {"connectivity": cell.connectivity,
                                    "motor_structural_rank": self.dn_head.structural_rank}
        if cell.connectivity == "structured":
            from core.routing import sensor_motor_paths
            self.routing_diagnostics["unmapped_input_channels"] = (~cell.W_in.route_mask.any(dim=0)).nonzero().flatten().tolist()
            overlap = cell.W_in.route_mask.any(dim=1) & self.dn_head._proj_mask.any(dim=0)
            self.routing_diagnostics["input_motor_overlap_clusters"] = overlap.nonzero().flatten().tolist()
            self.routing_diagnostics["sensor_motor_paths"] = sensor_motor_paths(
                cell._effective_W().detach() != 0, cell.sensor_indices, self.dn_head._proj_mask)
        self.pwm_layer     = PWMOutputLayer()
        self.default_dt    = getattr(cell, "default_dt", DEFAULT_DT)

        # Cache for stateful inference
        self._hx: Optional[torch.Tensor] = None

    # ─────────────────────────────────── forward (training / batched)

    def forward(
        self,
        obs: torch.Tensor,                     # (batch, 66) or (batch, T, 66)
        hx: Optional[torch.Tensor] = None,
        dt: Optional[float] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        obs  : (batch, 66)     — single-step observation
               (batch, T, 66)  — sequence observation
        hx   : (batch, k) initial hidden state; zeros if None
        dt   : integration timestep override

        Returns
        -------
        pwm    : (batch, 4) or (batch, T, 4)   RC PWM µs
        h_last : (batch, k)                     final hidden state
        """
        # ── Normalise sensor input ─────────────────────────────────────────
        sequence_mode = obs.dim() == 3
        if sequence_mode:
            B, T, D = obs.shape
            obs_flat = obs.reshape(B * T, D)
            obs_norm = self.sensor_layer(obs_flat).reshape(B, T, self.sensor_dim)
        else:
            obs_norm = self.sensor_layer(obs)          # (batch, 66)
            obs_norm = obs_norm.unsqueeze(1)           # (batch, 1, 66)
            T = 1

        # ── CfC recurrent forward — raw hidden states ─────────────────────
        # Run the CfC cell directly to obtain (batch, T, k) hidden states.
        # This policy owns one DN readout; the unused generic motor head is removed.
        cell = self.cfc_network.cell
        B = obs_norm.shape[0]
        if hx is None:
            hx = torch.zeros(B, cell.hidden_size,
                             device=obs_norm.device, dtype=obs_norm.dtype)

        h_states = []
        h = hx
        for t in range(obs_norm.shape[1]):
            h = cell(obs_norm[:, t, :], h, dt=dt)
            h_states.append(h)

        h_seq  = torch.stack(h_states, dim=1)   # (batch, T, k)
        h_last = h                               # (batch, k)

        # ── DN projection ──────────────────────────────────────────────────
        if sequence_mode:
            B2, T2, K = h_seq.shape
            logits = self.dn_head(h_seq.reshape(B2 * T2, K)).reshape(B2, T2, N_CONTROLS)
            pwm    = self.pwm_layer(logits.reshape(B2 * T2, N_CONTROLS)).reshape(B2, T2, N_CONTROLS)
        else:
            logits = self.dn_head(h_seq[:, 0, :])     # (batch, k) → (batch, 4)
            pwm    = self.pwm_layer(logits)            # (batch, 4)

        return pwm, h_last

    # ─────────────────────────────────── stateful single-step inference

    @torch.no_grad()
    def step(
        self,
        obs: torch.Tensor,           # (1, 66) or (66,)
        dt: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Stateful single-step inference (no grad).
        Maintains internal hidden state across calls.

        Returns pwm : (4,) tensor in µs.
        """
        self.eval()
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)          # (1, 66)

        pwm, self._hx = self.forward(obs, hx=self._hx, dt=dt)
        return pwm.squeeze(0)               # (4,)

    @torch.no_grad()
    def step_np(
        self,
        flow_xy: np.ndarray,                         # (2,)   FlowX, FlowY  ∈ [-1, 1]
        tof_8x8: np.ndarray,                         # (64,)  ToF pixels     ∈ [0, 1]
        memory_ring: Optional[np.ndarray] = None,   # (8,)   Egocentric memory ∈ [0, 1]
        dt: Optional[float] = None,
    ) -> np.ndarray:
        """
        Numpy convenience wrapper for the flight-control loop.
        Returns pwm : (4,) float32 numpy array [throttle, roll, pitch, yaw] µs.
        """
        obs_t = self.sensor_layer.from_numpy(flow_xy, tof_8x8, memory_ring=memory_ring)
        pwm_t = self.step(obs_t, dt=dt)
        return pwm_t.cpu().numpy()

    def reset_state(self):
        """Clear internal recurrent state (call at episode start)."""
        self._hx = None

    # ─────────────────────────────────── post-optimiser mask enforcement

    def post_step(self):
        """Call after optimizer.step() to enforce W_macro sparsity."""
        self.cfc_network.post_step()
        self.dn_head.proj.apply_mask()

    # ─────────────────────────────────── factory

    @classmethod
    def from_meta(
        cls,
        meta_path: str,
        mode: str = "masked",
        prune_meta_path: Optional[str] = None,
        sensor_dim: int = SENSOR_DIM,
        backbone_units: int = 64,
        backbone_layers: int = 2,
        backbone_act: str = "lecun_tanh",
        backbone_dropout: float = 0.0,
        tau_init: float = 0.1,
        dt: float = DEFAULT_DT,
        solver_type: str = "exponential_euler",
        pruning_sparsity: Optional[float] = None,
        ablate_cx: Optional[bool] = None,
        learnable_scale: bool = True,
        connectivity: str = "structured",
        input_routes: Optional[dict] = None,
        allow_dense_motor_fallback: bool = False,
        preserve_signs: bool = False,
        sensor_encoding: str = 'distance-v1',
        neutral_origin: bool = False,
    ) -> "ChongFlyMSPPolicy":
        """
        Build ChongFlyMSPPolicy directly from a ReducedModel meta file.

        Parameters
        ----------
        meta_path       : e.g. "data/reduced_models/meta_spectral_k64.json"
        mode            : "fixed" | "masked" | "free"
        prune_meta_path : optional magnitude-pruner meta for synapse mask
        sensor_dim      : sensory input size (default 66 = 2 flow + 64 ToF)
        solver_type     : 'exponential_euler' | 'euler'; old spellings remain aliases
        pruning_sparsity: optional float in [0.0, 1.0)
        ablate_cx       : whether Central Complex is ablated (loads _nocx meta if available)
        """
        from core.models import build_network_from_meta

        net = build_network_from_meta(
            meta_path=meta_path,
            input_size=sensor_dim,
            output_dim=N_CONTROLS,
            mode=mode,
            prune_meta_path=prune_meta_path,
            pruning_sparsity=pruning_sparsity,
            solver_type=solver_type,
            ablate_cx=ablate_cx,
            backbone_units=backbone_units,
            backbone_layers=backbone_layers,
            backbone_act=backbone_act,
            backbone_dropout=backbone_dropout,
            tau_init=tau_init,
            dt=dt,
            return_sequences=True,
            connectivity=connectivity,
            input_routes=input_routes,
            include_motor_head=False,
            preserve_signs=preserve_signs,
        )
        return cls(net, sensor_dim=sensor_dim, learnable_scale=learnable_scale,
                   allow_dense_motor_fallback=allow_dense_motor_fallback,
                   sensor_encoding=sensor_encoding, neutral_origin=neutral_origin)

    # ─────────────────────────────────── repr

    def extra_repr(self) -> str:
        k = self.cfc_network.cell.hidden_size
        return (f"sensor_dim={self.sensor_layer.sensor_dim}, hidden={k}, "
                f"output={N_CONTROLS}×PWM[{int(PWM_MIN)}–{int(PWM_MAX)}µs], "
                f"mode={self.cfc_network.cell.mode}")
