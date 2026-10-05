# Interface for NVIDIA Isaac Gym and MSP protocol integration
from __future__ import annotations

import abc
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from configs.flight_config import (
    DEFAULT_DT,
    TARGET_ALTITUDE_M,
    MAX_TILT_ANGLE_RAD,
    LASER_MIN_RANGE_M,
    LASER_MAX_RANGE_M,
    LASER_NOISE_STD,
    PWM_HOVER,
    PWM_LEVEL_ROLL,
    PWM_MID,
    PWM_NEUTRAL_YAW,
    SENSORS,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Laser Sensor & Raycasting Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LaserHitResult:
    """Detailed hit result from downward laser rangefinder."""
    distance: float             # Measured distance along laser beam in meters
    distance_norm: float        # Normalized distance in [0.0, 1.0] (1.0 = clear path / max range)
    hit_point: np.ndarray       # 3D world coordinates [x, y, z] of hit point
    hit_normal: np.ndarray      # Surface normal vector at contact point [nx, ny, nz]
    is_valid: bool              # True if distance < max_range
    surface_type: str           # "floor", "box", "cylinder", or "out_of_range"


class DownwardLaserSensor:
    """
    Downward-facing laser distance sensor (ToF Altimeter).
    
    Mounted underneath the quadcopter body facing along -Z body axis:
        d_body = [0.0, 0.0, -1.0]
    When the drone rolls, pitches, or translates, the beam is rotated by the
    drone's attitude matrix:
        d_world = R_body_to_world @ d_body
    
    Computes exact analytical ray intersections with:
      - Ground floor (z = z_min)
      - Box obstacles (slab intersection)
      - Cylindrical pillars
    """

    def __init__(
        self,
        min_range: float = LASER_MIN_RANGE_M,        # 2 cm minimum sensing distance
        max_range: float = LASER_MAX_RANGE_M,        # 6 meters maximum laser range
        noise_std: float = LASER_NOISE_STD,          # 3 mm standard measurement noise
        mount_offset: Optional[np.ndarray] = None, # position relative to drone COM [x, y, z]
    ):
        self.min_range = min_range
        self.max_range = max_range
        self.noise_std = noise_std
        self.mount_offset = mount_offset if mount_offset is not None else np.array([0.0, 0.0, -0.018], dtype=np.float32)

    def measure(
        self,
        drone_pos: np.ndarray,
        rot_matrix: np.ndarray,
        floor_z: float = 0.0,
        boxes: Optional[List[Any]] = None,
        cylinders: Optional[List[Any]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> LaserHitResult:
        """
        Casts a laser ray downward from the sensor emitter.
        """
        # Sensor origin in world coordinates
        origin = drone_pos + rot_matrix @ self.mount_offset
        # Direction along body -Z transformed into world coordinates
        direction = rot_matrix @ np.array([0.0, 0.0, -1.0], dtype=np.float32)
        direction /= np.linalg.norm(direction)

        px, py, pz = float(origin[0]), float(origin[1]), float(origin[2])
        dx, dy, dz = float(direction[0]), float(direction[1]), float(direction[2])

        min_t = self.max_range
        hit_surf = "out_of_range"
        hit_norm = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        eps = 1e-6

        # 1. Ground floor intersection (z = floor_z)
        if dz < -eps:
            t_floor = (floor_z - pz) / dz
            if 0.0 < t_floor < min_t:
                min_t = t_floor
                hit_surf = "floor"
                hit_norm = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        # 2. Box obstacles intersection (slab method)
        if boxes:
            for box in boxes:
                inv_dx = 1.0 / dx if abs(dx) > eps else (1.0 / (eps * (1.0 if dx >= 0 else -1.0)))
                inv_dy = 1.0 / dy if abs(dy) > eps else (1.0 / (eps * (1.0 if dy >= 0 else -1.0)))
                inv_dz = 1.0 / dz if abs(dz) > eps else (1.0 / (eps * (1.0 if dz >= 0 else -1.0)))

                t_b1 = (box.x_min - px) * inv_dx
                t_b2 = (box.x_max - px) * inv_dx
                t_near_x, t_far_x = min(t_b1, t_b2), max(t_b1, t_b2)

                t_b3 = (box.y_min - py) * inv_dy
                t_b4 = (box.y_max - py) * inv_dy
                t_near_y, t_far_y = min(t_b3, t_b4), max(t_b3, t_b4)

                t_b5 = (box.z_min - pz) * inv_dz
                t_b6 = (box.z_max - pz) * inv_dz
                t_near_z, t_far_z = min(t_b5, t_b6), max(t_b5, t_b6)

                t_enter = max(max(t_near_x, t_near_y), t_near_z)
                t_exit = min(min(t_far_x, t_far_y), t_far_z)

                if t_enter <= t_exit and t_enter > 0.0 and t_enter < min_t:
                    min_t = t_enter
                    hit_surf = "box"
                    hit_norm = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        # 3. Cylindrical obstacle intersection
        if cylinders:
            for cyl in cylinders:
                cx, cy, r = cyl.center_x, cyl.center_y, cyl.radius
                ox = px - cx
                oy = py - cy
                A = dx * dx + dy * dy
                B = ox * dx + oy * dy
                C = ox * ox + oy * oy - r * r
                disc = B * B - A * C
                if disc >= 0.0 and A > eps:
                    s_disc = math.sqrt(disc)
                    for t_cand in [(-B - s_disc) / A, (-B + s_disc) / A]:
                        if 0.0 < t_cand < min_t:
                            hit_z = pz + t_cand * dz
                            if floor_z <= hit_z <= cyl.height:
                                min_t = t_cand
                                hit_surf = "cylinder"
                                hit_norm = np.array([(px + t_cand * dx - cx) / r, (py + t_cand * dy - cy) / r, 0.0], dtype=np.float32)

        # Add Gaussian measurement noise if valid reading
        dist = min_t
        if self.noise_std > 0.0 and rng is not None and dist < self.max_range:
            dist += float(rng.normal(0.0, self.noise_std))
        dist = float(np.clip(dist, self.min_range, self.max_range))

        hit_point = origin + dist * direction
        is_valid = (dist < self.max_range)
        norm_dist = dist / self.max_range

        return LaserHitResult(
            distance=dist,
            distance_norm=norm_dist,
            hit_point=hit_point.astype(np.float32),
            hit_normal=hit_norm.astype(np.float32),
            is_valid=is_valid,
            surface_type=hit_surf,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Universal Isaac Observation and Action Container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class IsaacObservation:
    """
    Rich sensory observation container for Chong-Fly simulation models.
    
    Contains all physical sensor readouts:
    - Downward Laser Rangefinder (distance to ground/obstacle underneath)
    - 8x8 Depth Matrix (VL53L5CX, 45° FOV)
    - Optical Sensor (PMW3901 flow + incremental/cumulative displacement)
    - IMU (gyro rates and accelerations)
    - Flight state (attitude euler, velocity, altitude, payload)
    """
    # ── Downward Laser Sensor ────────────────────────────────────────────────
    laser_distance: float               # Laser distance to surface directly below (meters)
    laser_distance_norm: float          # Normalized distance in [0, 1] (1.0 = max range)
    laser_hit_point: np.ndarray         # (3,) world coordinates of laser ground contact point
    laser_valid: bool                   # True if surface is within laser sensing range

    # ── 8x8 Depth Matrix (45° FOV) ───────────────────────────────────────────
    depth_8x8: np.ndarray               # (8, 8) normalized depth values in [0, 1] (1.0 = clear)
    depth_flat: np.ndarray              # (64,) flattened 1D array of depth values

    # ── Optical Sensor (Displacement & Surface Flow) ─────────────────────────
    optical_flow: np.ndarray            # (2,) [FlowX, FlowY] normalized in [-1.0, 1.0]
    displacement_step: np.ndarray       # (2,) [dx, dy] translational displacement this step (m)
    displacement_total: np.ndarray      # (2,) [x_disp, y_disp] cumulative integrated displacement (m)

    # ── Inertial Measurement Unit (IMU) ──────────────────────────────────────
    imu_gyro: np.ndarray                # (3,) [p, q, r] body angular velocities (rad/s)
    imu_accel: np.ndarray               # (3,) [ax, ay, az] estimated body acceleration (m/s^2)
    attitude_euler: np.ndarray          # (3,) [roll, pitch, yaw] Euler angles in radians

    # ── Kinematic State ──────────────────────────────────────────────────────
    position: np.ndarray                # (3,) [x, y, z] world position (m)
    velocity: np.ndarray                # (3,) [vx, vy, vz] world velocity (m/s)

    # ── Drone Configuration ──────────────────────────────────────────────────
    payload_mass: float                 # Current attached cargo/payload mass (kg)
    total_mass: float                   # Base mass (0.130 kg) + payload_mass (kg)
    hover_throttle: float               # Calculated hover throttle for current mass [0.0, 1.0]
    sim_time: float                     # Simulation elapsed time (seconds)
    step_count: int                     # Simulation step index

    def to_chong_fly_66d(self) -> np.ndarray:
        """
        Returns standard 66-D vector [FlowX, FlowY, ToF_00 ... ToF_63]
        for backward compatibility with Drosophila connectome policies.
        """
        return np.concatenate([self.optical_flow, self.depth_flat], axis=0).astype(np.float32)

    def to_extended_67d(self) -> np.ndarray:
        """
        Returns extended 67-D vector [FlowX, FlowY, LaserAltNorm, ToF_00 ... ToF_63].
        Injects the downward laser altimeter into sensory representations.
        """
        laser_val = np.array([self.laser_distance_norm], dtype=np.float32)
        return np.concatenate([self.optical_flow, laser_val, self.depth_flat], axis=0).astype(np.float32)

    def to_full_vector(self) -> np.ndarray:
        """
        Returns full 82-dimensional vector containing all sensor readings,
        IMU, displacement, attitude, and payload state for advanced RL training.
        """
        return np.concatenate([
            self.optical_flow,                      # (2,)
            self.displacement_step,                 # (2,)
            self.displacement_total,                # (2,)
            np.array([self.laser_distance_norm, self.laser_distance], dtype=np.float32), # (2,)
            self.imu_gyro,                          # (3,)
            self.imu_accel,                         # (3,)
            self.attitude_euler,                    # (3,)
            np.array([self.payload_mass, self.hover_throttle], dtype=np.float32), # (2,)
            self.depth_flat,                        # (64,)
        ], axis=0).astype(np.float32)

    def to_dict(self) -> Dict[str, Any]:
        """Returns structured dictionary of observations."""
        return {
            "laser": {
                "distance_m": self.laser_distance,
                "distance_norm": self.laser_distance_norm,
                "hit_point": self.laser_hit_point,
                "valid": self.laser_valid,
            },
            "flow": {
                "optical_flow": self.optical_flow,
                "displacement_step": self.displacement_step,
                "displacement_total": self.displacement_total,
            },
            "imu": {
                "gyro_rad_s": self.imu_gyro,
                "accel_m_s2": self.imu_accel,
                "attitude_rad": self.attitude_euler,
            },
            "kinematics": {
                "position_m": self.position,
                "velocity_m_s": self.velocity,
            },
            "status": {
                "payload_kg": self.payload_mass,
                "total_mass_kg": self.total_mass,
                "hover_throttle": self.hover_throttle,
                "time_s": self.sim_time,
                "step": self.step_count,
            },
        }


@dataclass
class IsaacAction:
    """
    Flight control command passed from a neural model or algorithm to the drone simulation.
    
    Supports two primary control modes:
    1. Setpoint Mode (Recommended for high-level neural policies):
       - thrust: collective thrust in [0.0, 1.0]
       - roll: target roll angle in radians [-max_angle, +max_angle]
       - pitch: target pitch angle in radians [-max_angle, +max_angle]
       - yaw_rate: target yaw rate in rad/s [-max_yaw_rate, +max_yaw_rate]
    2. RC PWM Mode (Betaflight standard):
       - throttle_pwm: [1000, 2000] µs
       - roll_pwm: [1000, 2000] µs (1500 = center)
       - pitch_pwm: [1000, 2000] µs (1500 = center)
       - yaw_pwm: [1000, 2000] µs (1500 = center)
    """
    action_type: str = "setpoints"   # "setpoints" or "pwm"
    thrust: float = 0.5
    roll: float = 0.0
    pitch: float = 0.0
    yaw_rate: float = 0.0
    throttle_pwm: float = PWM_HOVER
    roll_pwm: float = PWM_LEVEL_ROLL
    pitch_pwm: float = PWM_MID
    yaw_pwm: float = PWM_NEUTRAL_YAW

    @classmethod
    def from_setpoints(
        cls,
        thrust: float,
        roll: float = 0.0,
        pitch: float = 0.0,
        yaw_rate: float = 0.0,
    ) -> IsaacAction:
        """Creates an action using physical flight setpoints."""
        return cls(
            action_type="setpoints",
            thrust=float(thrust),
            roll=float(roll),
            pitch=float(pitch),
            yaw_rate=float(yaw_rate),
        )

    @classmethod
    def from_pwm(
        cls,
        throttle: float = PWM_HOVER,
        roll: float = PWM_LEVEL_ROLL,
        pitch: float = PWM_MID,
        yaw: float = PWM_NEUTRAL_YAW,
    ) -> IsaacAction:
        """Creates an action using standard RC PWM microseconds [1000, 2000]."""
        return cls(
            action_type="pwm",
            throttle_pwm=float(throttle),
            roll_pwm=float(roll),
            pitch_pwm=float(pitch),
            yaw_pwm=float(yaw),
        )

    def to_array(self) -> np.ndarray:
        """Returns 4-element array for direct drone env stepping."""
        if self.action_type == "pwm":
            return np.array([self.throttle_pwm, self.roll_pwm, self.pitch_pwm, self.yaw_pwm], dtype=np.float32)
        return np.array([self.thrust, self.roll, self.pitch, self.yaw_rate], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Base Isaac Model Interface & Implementations
# ─────────────────────────────────────────────────────────────────────────────

class BaseIsaacModel(abc.ABC):
    """Universal Abstract Base Class for any model plugged into Chong-Fly simulation."""

    @abc.abstractmethod
    def step(self, obs: IsaacObservation) -> Union[IsaacAction, np.ndarray]:
        """Evaluates one model inference step given the current Isaac observation."""
        pass

    def reset(self) -> None:
        """Called when the simulation restarts or drone respawns."""
        pass


class ConnectomeModelAdapter(BaseIsaacModel):
    """Adapts any ChongFlyMSPPolicy biological connectome model into the Isaac interface."""

    def __init__(self, policy: Any):
        self.policy = policy

    @classmethod
    def from_meta(
        cls,
        meta_path: str,
        solver_type: str = "exponential_euler",
        pruning_sparsity: Optional[float] = None,
        ablate_cx: Optional[bool] = None,
    ) -> ConnectomeModelAdapter:
        from simulation.policy import ChongFlyMSPPolicy
        pol = ChongFlyMSPPolicy.from_meta(
            meta_path=meta_path,
            mode="fixed",
            dt=DEFAULT_DT,
            solver_type=solver_type,
            pruning_sparsity=pruning_sparsity,
            ablate_cx=ablate_cx,
        )
        return cls(pol)

    def reset(self) -> None:
        self.policy.reset_state()

    def step(self, obs: IsaacObservation) -> IsaacAction:
        pwm = self.policy.step_np(obs.optical_flow, obs.depth_flat)
        return IsaacAction.from_pwm(
            throttle=pwm[0],
            roll=pwm[1],
            pitch=pwm[2],
            yaw=pwm[3],
        )


class AutonomousLaserNavigatorModel(BaseIsaacModel):
    """
    Intelligent reference model integrating all 3 sensor modalities:
    1. Downward Laser Sensor: Maintains precise hover altitude.
    2. 8x8 Depth Matrix (45° FOV): Detects obstacles in 4 visual quadrants.
    3. Optical Flow & Displacement: Opposes translational ground drift.
    """

    def __init__(
        self,
        target_altitude: float = TARGET_ALTITUDE_M,
        kp_alt: float = 0.40,
        kd_alt: float = 0.15,
        avoidance_gain: float = 0.35,
        flow_damping_gain: float = 0.20,
    ):
        self.target_altitude = target_altitude
        self.kp_alt = kp_alt
        self.kd_alt = kd_alt
        self.avoidance_gain = avoidance_gain
        self.flow_damping_gain = flow_damping_gain
        self.last_laser_alt = target_altitude

    def reset(self) -> None:
        self.last_laser_alt = self.target_altitude

    def step(self, obs: IsaacObservation) -> IsaacAction:
        laser_alt = obs.laser_distance
        alt_err = self.target_altitude - laser_alt
        alt_rate = (laser_alt - self.last_laser_alt) / DEFAULT_DT
        self.last_laser_alt = laser_alt

        base_th = obs.hover_throttle
        thrust_cmd = base_th + self.kp_alt * alt_err - self.kd_alt * alt_rate
        thrust_cmd = float(np.clip(thrust_cmd, 0.05, 0.95))

        depth_mat = obs.depth_8x8
        mid_col = depth_mat.shape[1] // 2 if depth_mat.ndim == 2 else TOF_COLS // 2
        left_threat = float(1.0 - np.mean(depth_mat[:, :mid_col]))
        right_threat = float(1.0 - np.mean(depth_mat[:, mid_col:]))
        center_threat = float(1.0 - np.mean(depth_mat[SENSORS.tof_center_row_start:SENSORS.tof_center_row_end, SENSORS.tof_center_col_start:SENSORS.tof_center_col_end]))

        target_roll = (left_threat - right_threat) * self.avoidance_gain
        target_pitch = -center_threat * self.avoidance_gain

        # +FlowY is leftward velocity; +roll accelerates right in FLU.
        target_roll += obs.optical_flow[1] * self.flow_damping_gain
        target_pitch -= obs.optical_flow[0] * self.flow_damping_gain

        max_angle = MAX_TILT_ANGLE_RAD
        target_roll = float(np.clip(target_roll, -max_angle, max_angle))
        target_pitch = float(np.clip(target_pitch, -max_angle, max_angle))

        return IsaacAction.from_setpoints(
            thrust=thrust_cmd,
            roll=target_roll,
            pitch=target_pitch,
            yaw_rate=0.0,
        )
