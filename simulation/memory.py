"""
simulation/memory.py
====================
Egocentric Ring Buffer (Spatial Memory) for Drone Simulation.

Maintains an O(1) 1D Ring Buffer of 8 directional sectors (45° slices)
around the drone:
    Sector 0: Front        (0°)
    Sector 1: Front-Right  (45°)
    Sector 2: Right        (90°)
    Sector 3: Back-Right   (135°)
    Sector 4: Back         (180°)
    Sector 5: Back-Left    (225° / -135°)
    Sector 6: Left         (270° / -90°)
    Sector 7: Front-Left   (315° / -45°)

Rotation physics:
    When the drone turns Right (+Yaw), an obstacle in Front (Sector 0) shifts
    to the Left (Sector 7 for 45°, Sector 6 for 90°).
    Implemented via np.roll.

Decay (forgetting):
    Unobserved sectors slowly decay towards 1.0 (safe max distance) by decay_rate
    at each step to accommodate odometry drift.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence, Union
import numpy as np

from configs.flight_config import (
    MEMORY_NUM_SECTORS,
    MEMORY_DECAY_RATE,
    MEMORY_DEFAULT_DISTANCE,
    TOF_DIM,
    TOF_ROWS,
    TOF_COLS,
    SENSORS,
)

# Directional Sector Indices
SECTOR_FRONT = 0
SECTOR_FRONT_RIGHT = 1
SECTOR_RIGHT = 2
SECTOR_BACK_RIGHT = 3
SECTOR_BACK = 4
SECTOR_BACK_LEFT = 5
SECTOR_LEFT = 6
SECTOR_FRONT_LEFT = 7


class EgocentricMemoryWrapper:
    """
    Egocentric 8-sector Ring Buffer for spatial obstacle tracking.
    """

    NUM_SECTORS: int = MEMORY_NUM_SECTORS
    SECTOR_ANGLE_RAD: float = math.pi / 4.0  # 45 degrees

    def __init__(
        self,
        num_sectors: int = MEMORY_NUM_SECTORS,
        decay_rate: float = MEMORY_DECAY_RATE,
        default_distance: float = MEMORY_DEFAULT_DISTANCE,
    ):
        self.num_sectors = int(num_sectors)
        self.sector_angle = (2.0 * math.pi) / float(self.num_sectors)
        self.decay_rate = float(decay_rate)
        self.default_distance = float(default_distance)

        self.memory = np.full(self.num_sectors, self.default_distance, dtype=np.float32)
        self.accumulated_yaw = 0.0
        self.last_yaw: Optional[float] = None

    def reset(self) -> None:
        """Resets spatial memory to default safe distance and clears yaw state."""
        self.memory.fill(self.default_distance)
        self.accumulated_yaw = 0.0
        self.last_yaw = None

    @staticmethod
    def extract_center_tof(obs_tof: Any) -> float:
        """
        Extracts minimum distance from central ToF pixels.
        - If obs_tof is a scalar, clamps to [0, 1].
        - If obs_tof is 8x8 or 64-element array, extracts central 4x4 (rows 2..5, cols 2..5).
        - Otherwise takes minimum of all finite elements.
        """
        if isinstance(obs_tof, (int, float, np.floating, np.integer)):
            return float(np.clip(obs_tof, 0.0, 1.0))

        if hasattr(obs_tof, "detach"):
            obs_tof = obs_tof.detach().cpu().numpy()

        arr = np.asarray(obs_tof, dtype=np.float32)
        if arr.size == 0:
            return 1.0

        if arr.size == TOF_DIM:
            grid = arr.reshape((TOF_ROWS, TOF_COLS))
            center = grid[SENSORS.tof_center_row_start:SENSORS.tof_center_row_end, SENSORS.tof_center_col_start:SENSORS.tof_center_col_end]
            valid = center[np.isfinite(center)]
            if valid.size > 0:
                return float(np.clip(np.min(valid), 0.0, 1.0))
            return 1.0

        valid = arr[np.isfinite(arr)]
        if valid.size > 0:
            return float(np.clip(np.min(valid), 0.0, 1.0))
        return 1.0

    def update(
        self,
        tof_center_dist: Union[float, np.ndarray, Sequence[float]],
        delta_yaw_rad: Optional[float] = None,
        current_yaw_rad: Optional[float] = None,
    ) -> np.ndarray:
        """
        Advances spatial memory by one step:
        1. Decay: unobserved sectors (1..7) decay towards default_distance (1.0).
        2. Rotation: accumulates yaw and shifts sectors (np.roll) when |accumulated_yaw| >= 45°.
        3. Vision: writes the central ToF distance into Sector 0 (Front).

        Parameters
        ----------
        tof_center_dist : float or 8x8/64 ToF array.
        delta_yaw_rad : delta yaw angle in radians (turn Right > 0, turn Left < 0).
        current_yaw_rad : optional absolute yaw angle in radians.

        Returns
        -------
        1D numpy array of shape (8,) with sector distances.
        """
        # 1. Resolve delta yaw
        if delta_yaw_rad is None and current_yaw_rad is not None:
            if self.last_yaw is not None:
                dy = current_yaw_rad - self.last_yaw
                dy = (dy + math.pi) % (2.0 * math.pi) - math.pi
                delta_yaw_rad = dy
            else:
                delta_yaw_rad = 0.0
            self.last_yaw = float(current_yaw_rad)
        elif delta_yaw_rad is None:
            delta_yaw_rad = 0.0

        # 2. Decay unobserved sectors (1..num_sectors-1) before new observation rotation
        if self.decay_rate > 0.0:
            self.memory[1:] = np.clip(
                self.memory[1:] + self.decay_rate, 0.0, self.default_distance
            )

        # 3. Accumulated rotation & circular shift
        self.accumulated_yaw += float(delta_yaw_rad)
        shifts = int(self.accumulated_yaw / self.sector_angle)
        if shifts != 0:
            # Turn Right (+Yaw, shifts > 0) -> Sector 0 moves to Sector 7 / 6 (to the Left)
            # np.roll(arr, -shifts) moves index 0 to (0 - shifts) % 8
            self.memory = np.roll(self.memory, -shifts)
            self.accumulated_yaw -= float(shifts) * self.sector_angle

        # 4. Vision: update Sector 0 (Front)
        front_dist = self.extract_center_tof(tof_center_dist)
        self.memory[0] = front_dist

        return self.memory.copy()

    def get_memory(self) -> np.ndarray:
        """Returns a copy of the current 8-sector memory array."""
        return self.memory.copy()