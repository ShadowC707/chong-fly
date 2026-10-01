"""
simulation/pmw3901_emulator.py
==============================
Emulation of the PMW3901 optical flow sensor (PixArt Imaging).

Computes translational optical flow rates across the ground surface:
    FlowX = clip(v_body_x / h_ground, -max_flow, max_flow) / max_flow
    FlowY = clip(v_body_y / h_ground, -max_flow, max_flow) / max_flow

Includes:
- Gyro compensation (derotation) to isolate translation from rotational body rates
- Surface height clamping (valid operating altitude [0.08m, 3.5m])
- Optional measurement noise and surface texture dropouts
- Structured (2,) output matching ChongFlyMSPPolicy sensor layout [FlowX, FlowY]
"""

from __future__ import annotations

import math
from typing import Optional, Tuple
import numpy as np

from configs.flight_config import (
    SENSORS,
)


class PMW3901FlowSensor:
    """
    PMW3901 Optical Flow Sensor emulator for micro-drones.
    """

    def __init__(
        self,
        min_altitude: float = SENSORS.flow_min_altitude_m,     # minimum sensing distance (meters)
        max_altitude: float = SENSORS.flow_max_altitude_m,     # maximum tracking altitude (meters)
        max_flow_rate: float = SENSORS.flow_max_rate_rads,     # maximum angular flow rate (rad/s) for +/- 1.0 normalization
        derotate_with_gyro: bool = True,                       # subtract angular velocity (p, q)
        noise_std: float = SENSORS.flow_noise_std,             # Gaussian measurement noise std
    ):
        self.min_altitude = min_altitude
        self.max_altitude = max_altitude
        self.max_flow_rate = max_flow_rate
        self.derotate_with_gyro = derotate_with_gyro
        self.noise_std = noise_std

    def compute_flow(
        self,
        v_world: np.ndarray,             # [vx, vy, vz] in world frame (m/s)
        rot_matrix: np.ndarray,          # (3, 3) body-to-world rotation matrix R
        altitude_above_surface: float,   # distance to surface directly below (meters)
        omega_body: Optional[np.ndarray] = None, # [p, q, r] body angular velocity (rad/s)
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Calculates normalized optical flow vector [FlowX, FlowY] in [-1.0, 1.0].
        
        Args:
            v_world: linear velocity in world coordinates (m/s)
            rot_matrix: rotation matrix R such that v_world = R @ v_body
            altitude_above_surface: true perpendicular distance to ground / obstacle (m)
            omega_body: body angular velocity [p, q, r] in rad/s
            rng: optional random generator for sensor noise

        Returns:
            flow_xy: np.ndarray of shape (2,) with [FlowX, FlowY] in [-1.0, 1.0]
        """
        # 1. Transform velocity from world frame into drone body frame: v_body = R.T @ v_world
        v_body = rot_matrix.T @ v_world
        vx_body = float(v_body[0])
        vy_body = float(v_body[1])

        # 2. Altitude protection against division by zero
        h = float(np.clip(altitude_above_surface, self.min_altitude, self.max_altitude))

        # 3. Pure translational optical flow rates (rad/s)
        # Looking down: forward velocity (+vx) produces negative backward motion of ground (-rad/s)
        # Here we follow the convention: flow tracks forward/lateral body speed normalized
        flow_rate_x = vx_body / h
        flow_rate_y = vy_body / h

        # 4. Optional gyro derotation (if sensor lacks onboard derotation or is compensated)
        # Note: if raw sensor includes rotation: flow_raw_x = flow_rate_x - q, flow_raw_y = flow_rate_y + p
        # With derotation = True, we output the pure velocity-dependent flow (as done by flight stacks).
        if not self.derotate_with_gyro and omega_body is not None:
            p = float(omega_body[0])
            q = float(omega_body[1])
            flow_rate_x = flow_rate_x - q
            flow_rate_y = flow_rate_y + p

        # 5. Normalization to [-1.0, 1.0]
        norm_flow_x = float(np.clip(flow_rate_x / self.max_flow_rate, -1.0, 1.0))
        norm_flow_y = float(np.clip(flow_rate_y / self.max_flow_rate, -1.0, 1.0))

        # 6. Add sensor noise if requested
        if self.noise_std > 0.0 and rng is not None:
            norm_flow_x += float(rng.normal(0.0, self.noise_std))
            norm_flow_y += float(rng.normal(0.0, self.noise_std))
            norm_flow_x = float(np.clip(norm_flow_x, -1.0, 1.0))
            norm_flow_y = float(np.clip(norm_flow_y, -1.0, 1.0))

        return np.array([norm_flow_x, norm_flow_y], dtype=np.float32)
