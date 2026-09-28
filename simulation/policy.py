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

from bio_pipeline.models import BiologicalCfCCell, BiologicalCfCNetwork


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

SENSOR_DIM    = 66          # 2 flow + 64 ToF
FLOW_DIM      = 2           # FlowX, FlowY
TOF_DIM       = 64          # 8×8 ToF grid
N_CONTROLS    = 4           # throttle, roll, pitch, yaw

# RC PWM limits (µs)
PWM_MIN       = 1000.0
PWM_MID       = 1500.0
PWM_MAX       = 2000.0
PWM_HALF      = 500.0       # half-swing for attitude channels

# Channel indices
CH_THROTTLE   = 0
CH_ROLL       = 1
CH_PITCH      = 2
CH_YAW        = 3


# ─────────────────────────────────────────────────────────────────────────────
# SensorInputLayer
# ─────────────────────────────────────────────────────────────────────────────

class SensorInputLayer(nn.Module):
    """
    Normalise and optionally scale the 66-D sensor vector.

    FlowX/Y  : expected range ±1 (already normalised by caller)
    ToF 8×8  : expected range [0, 1] (distance / max_range)

    An optional learnable affine rescaling (per-channel gain + bias) is
    applied after normalisation so the network can adapt to sensor offsets.
    """

    def __init__(self, sensor_dim: int = SENSOR_DIM, learnable_scale: bool = True):
        super().__init__()
        self.sensor_dim = sensor_dim

        if learnable_scale:
            # Per-sensor gain (initialised to 1) and bias (initialised to 0)
            self.gain = nn.Parameter(torch.ones(sensor_dim))
            self.bias = nn.Parameter(torch.zeros(sensor_dim))
        else:
            self.register_buffer("gain", torch.ones(sensor_dim))
            self.register_buffer("bias", torch.zeros(sensor_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, 66) — raw sensor vector
            x[:, 0:2]   = [FlowX, FlowY]   ∈ [-1, 1]
            x[:, 2:66]  = ToF pixels        ∈ [0, 1]
        Returns normalised (batch, 66).
        """
        return x * self.gain + self.bias

    def from_numpy(self, flow_xy: np.ndarray, tof_8x8: np.ndarray) -> torch.Tensor:
        """
        Convenience: pack numpy arrays → (1, 66) tensor.

        Parameters
        ----------
        flow_xy : (2,) float32  [FlowX, FlowY] normalised ±1
        tof_8x8 : (64,) float32  ToF pixels normalised [0,1]
        """
        raw = np.concatenate([
            np.asarray(flow_xy,  dtype=np.float32).ravel()[:FLOW_DIM],
            np.asarray(tof_8x8,  dtype=np.float32).ravel()[:TOF_DIM],
        ])
        return torch.from_numpy(raw).unsqueeze(0)   # (1, 66)


# ─────────────────────────────────────────────────────────────────────────────
# DNProjectionHead
# ─────────────────────────────────────────────────────────────────────────────

class DNProjectionHead(nn.Module):
    """
    Sparse read-out from DN motor cluster indices → 4 raw motor logits.

    For each motor channel (throttle, roll, pitch, yaw), the corresponding
    DN cluster activations are averaged and mapped through a small linear
    projection.  This mirrors the biological premotor→motor pathway:
    descending neurons (DNs) → thoracic motor circuits.

    If motor_index_map is not provided (or empty), falls back to a full
    linear projection from hidden_size → N_CONTROLS.

    Parameters
    ----------
    hidden_size      : k  (CfC hidden dimension)
    motor_index_map  : {"throttle": [i, j, ...], "roll": [...], ...}
                       Cluster indices from ReducedModel.motor_index_map.
    """

    # Canonical channel ordering
    _CHANNEL_KEYS = ("throttle", "roll", "pitch", "yaw")

    def __init__(
        self,
        hidden_size: int,
        motor_index_map: Optional[dict[str, list[int]]] = None,
    ):
        super().__init__()
        self.hidden_size     = hidden_size
        self.motor_index_map = motor_index_map or {}

        # Build per-channel index buffers
        self._ch_indices: list[Optional[torch.Tensor]] = []
        for key in self._CHANNEL_KEYS:
            # Try direct key, then fall back to partial match
            idx = self._resolve_key(key)
            if idx:
                buf = torch.tensor(idx, dtype=torch.long)
                self.register_buffer(f"_idx_{key}", buf)
                self._ch_indices.append(buf)
            else:
                self.register_buffer(f"_idx_{key}", None)
                self._ch_indices.append(None)

        # Per-channel linear projector: mean-pool → scalar
        # Input dim = hidden_size (fallback) or 1 (mean of DN activations)
        self.use_sparse = any(i is not None for i in self._ch_indices)

        if self.use_sparse:
            # 1-D per-channel: mean(DN activations) → 1 scalar per channel
            self.proj = nn.Linear(hidden_size, N_CONTROLS, bias=True)
            # Mask: only the DN indices for each channel are active
            self._build_sparse_proj_mask()
        else:
            # Full projection (no structural prior)
            self.proj = nn.Linear(hidden_size, N_CONTROLS, bias=True)

        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def _resolve_key(self, key: str) -> list[int]:
        """
        Match motor_index_map key to channel name.
        Supports exact match and substring match (e.g. "pitch_roll" → pitch & roll).
        """
        if key in self.motor_index_map:
            return self.motor_index_map[key]
        # substring search
        for mkey, indices in self.motor_index_map.items():
            if key in mkey or mkey in key:
                return indices
        return []

    def _build_sparse_proj_mask(self):
        """
        Create a (N_CONTROLS, hidden_size) boolean mask where row i is True
        only for the DN cluster indices of channel i.
        """
        mask = torch.zeros(N_CONTROLS, self.hidden_size, dtype=torch.bool)
        for ch_idx, key in enumerate(self._CHANNEL_KEYS):
            idx_buf = getattr(self, f"_idx_{key}", None)
            if idx_buf is not None and len(idx_buf) > 0:
                valid = idx_buf[idx_buf < self.hidden_size]
                mask[ch_idx, valid] = True
        # If a channel has no mapped indices, activate all (fallback)
        empty_rows = ~mask.any(dim=1)
        mask[empty_rows, :] = True
        self.register_buffer("_proj_mask", mask)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        h : (batch, hidden_size)
        Returns raw motor logits : (batch, N_CONTROLS)
        """
        if self.use_sparse:
            # Apply sparse mask to projection weights in forward
            W_masked = self.proj.weight * self._proj_mask.float()
            return F.linear(h, W_masked, self.proj.bias)
        return self.proj(h)


# ─────────────────────────────────────────────────────────────────────────────
# PWMOutputLayer
# ─────────────────────────────────────────────────────────────────────────────

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
    ):
        super().__init__()
        cell = cfc_network.cell

        self.sensor_layer  = SensorInputLayer(sensor_dim, learnable_scale)
        self.cfc_network   = cfc_network
        self.dn_head       = DNProjectionHead(cell.hidden_size, cell.motor_indices)
        self.pwm_layer     = PWMOutputLayer()

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
            obs_norm = self.sensor_layer(obs_flat).reshape(B, T, D)
        else:
            obs_norm = self.sensor_layer(obs)          # (batch, 66)
            obs_norm = obs_norm.unsqueeze(1)           # (batch, 1, 66)
            T = 1

        # ── CfC recurrent forward — raw hidden states ─────────────────────
        # Run the CfC cell directly to obtain (batch, T, k) hidden states.
        # We bypass BiologicalCfCNetwork.motor_head (used in training/env)
        # and feed h_seq straight into our DNProjectionHead.
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
        flow_xy: np.ndarray,               # (2,)   FlowX, FlowY  ∈ [-1, 1]
        tof_8x8: np.ndarray,               # (64,)  ToF pixels     ∈ [0, 1]
        dt: Optional[float] = None,
    ) -> np.ndarray:
        """
        Numpy convenience wrapper for the flight-control loop.
        Returns pwm : (4,) float32 numpy array [throttle, roll, pitch, yaw] µs.
        """
        obs_t = self.sensor_layer.from_numpy(flow_xy, tof_8x8)
        pwm_t = self.step(obs_t, dt=dt)
        return pwm_t.cpu().numpy()

    def reset_state(self):
        """Clear internal recurrent state (call at episode start)."""
        self._hx = None

    # ─────────────────────────────────── post-optimiser mask enforcement

    def post_step(self):
        """Call after optimizer.step() to enforce W_macro sparsity."""
        self.cfc_network.post_step()

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
        dt: float = 0.004,
        solver_type: str = "CfC",
        pruning_sparsity: Optional[float] = None,
        ablate_cx: Optional[bool] = None,
        learnable_scale: bool = True,
    ) -> "ChongFlyMSPPolicy":
        """
        Build ChongFlyMSPPolicy directly from a ReducedModel meta file.

        Parameters
        ----------
        meta_path       : e.g. "data/reduced_models/meta_spectral_k64.json"
        mode            : "fixed" | "masked" | "free"
        prune_meta_path : optional magnitude-pruner meta for synapse mask
        sensor_dim      : sensory input size (default 66 = 2 flow + 64 ToF)
        solver_type     : 'CfC' | 'Euler_dt_0.02'
        pruning_sparsity: optional float in [0.0, 1.0)
        ablate_cx       : whether Central Complex is ablated (loads _nocx meta if available)
        """
        from bio_pipeline.models import build_network_from_meta

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
        )
        return cls(net, sensor_dim=sensor_dim, learnable_scale=learnable_scale)

    # ─────────────────────────────────── repr

    def extra_repr(self) -> str:
        k = self.cfc_network.cell.hidden_size
        return (f"sensor_dim={SENSOR_DIM}, hidden={k}, "
                f"output=4×PWM[1000–2000µs], "
                f"mode={self.cfc_network.cell.mode}")
