"""
configs/flight_config.py
========================
Centralized configuration repository for Chong-Fly.

All flight control logic parameters, physical constants, sensor dimensions,
actuator bounds, APF coefficients, and environmental constraints across the
pipeline are defined exclusively here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


# ─────────────────────────────────────────────────────────────────────────────
# 1. Sensors & Dimensions Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SensorsConfig:
    flow_dim: int = 2                       # Optical flow (FlowX, FlowY)
    tof_dim: int = 64                       # 8x8 ToF distance grid
    tof_rows: int = 8                       # Grid height
    tof_cols: int = 8                       # Grid width
    memory_dim: int = 8                     # Egocentric ring buffer sectors
    sensor_dim_base: int = 66               # Legacy 66-D (2 flow + 64 ToF)
    sensor_dim: int = 74                    # Current 74-D (2 flow + 64 ToF + 8 memory)
    n_controls: int = 4                     # 4 channels: Throttle, Roll, Pitch, Yaw

    tof_max_range_m: float = 3.0            # Max measurement range for ToF
    tof_raycaster_max_range_m: float = 3.5  # Max measurement range for Isaac ToF raycaster
    tof_fov_h_deg: float = 45.0             # Horizontal FOV (degrees)
    tof_fov_v_deg: float = 45.0             # Vertical FOV (degrees)
    tof_min_clamp_dist: float = 0.1         # Protection against division by zero in depth
    tof_center_row_start: int = 2           # Central region row start
    tof_center_row_end: int = 6             # Central region row end
    tof_center_col_start: int = 2           # Central region col start
    tof_center_col_end: int = 6             # Central region col end

    laser_min_range_m: float = 0.02         # 2 cm minimum downward laser range
    laser_max_range_m: float = 6.0          # 6 m maximum downward laser range
    laser_noise_std: float = 0.003          # 3 mm measurement noise std

    flow_min_altitude_m: float = 0.08       # Minimum operating altitude for flow sensor
    flow_max_altitude_m: float = 3.50       # Maximum operating altitude for flow sensor
    flow_max_rate_rads: float = 4.0         # Max angular rate for flow normalization
    flow_noise_std: float = 0.01            # Measurement noise std
    flow_z_safe_m: float = 0.2              # Ground clearance divisor clamp for flow


# ─────────────────────────────────────────────────────────────────────────────
# 2. Actuation & PWM Channel Limits Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ActuatorConfig:
    pwm_min: float = 1000.0                 # Minimum RC PWM command (µs)
    pwm_mid: float = 1500.0                 # Neutral / hover midpoint RC PWM (µs)
    pwm_max: float = 2000.0                 # Maximum RC PWM command (µs)
    pwm_half: float = 500.0                 # Half-swing deflection (µs) for attitude

    pwm_hover: float = 1500.0               # Hover baseline PWM (µs)
    pwm_level_roll: float = 1500.0          # Neutral roll PWM (µs)
    pwm_neutral_yaw: float = 1500.0         # Neutral yaw PWM (µs)
    pwm_cruise_pitch: float = 1600.0        # Forward cruising pitch PWM (µs)
    pwm_brake_pitch: float = 1300.0         # Hard braking pitch PWM (µs)
    pwm_turn_left_yaw: float = 1100.0       # Sharp left evasion yaw PWM (µs)
    pwm_turn_right_yaw: float = 1900.0      # Sharp right evasion yaw PWM (µs)

    # Physical saturation boundary thresholds
    pwm_saturation_low: float = 1100.0      # Motor choke low threshold (µs)
    pwm_saturation_high: float = 1900.0     # Motor choke high threshold (µs)
    max_saturation_ratio: float = 0.15      # Max allowable saturated steps ratio before kill

    # Channel indexing
    ch_throttle: int = 0
    ch_roll: int = 1
    ch_pitch: int = 2
    ch_yaw: int = 3
    channel_keys: Tuple[str, ...] = ("throttle", "roll", "pitch", "yaw")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Physics & Drone Dynamics Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PhysicsConfig:
    default_dt: float = 0.004               # 250 Hz physics integration step (s)
    gravity: float = 9.81                   # Gravitational acceleration (m/s^2)
    drone_mass_kg: float = 0.035            # 35g Crazyflie-class micro-quadcopter
    drone_mass_isaac_kg: float = 0.130      # 130g base micro-quad in Isaac
    drag_coeff: float = 0.25                # Translational drag coefficient
    angular_damping: float = 4.0            # Angular rate damping
    thrust_gain: float = 2.0                # Maximum thrust-to-weight ratio (2.0 gives exact gravity balance at 0.5 hover throttle)
    target_altitude_m: float = 1.0          # Nominal flight altitude (m)
    altitude_hold_gain: float = 8.0         # P-gain for altitude stabilization
    max_tilt_angle_rad: float = 0.44        # ~25 degrees maximum roll / pitch angle
    attitude_tau: float = 0.05              # 50 ms attitude tracking time constant
    yaw_rate_gain: float = 3.0              # Body yaw angular rate multiplier (rad/s per norm cmd)

    # Termination / Crash conditions
    tumble_angle_threshold_rad: float = 1.2 # ~70 degrees tumble limit
    ground_crash_alt_m: float = 0.02        # Altitude threshold for ground collision (m)
    obstacle_crash_dist_m: float = 0.05     # Distance threshold for obstacle impact (m)
    back_wall_dist_m: float = 3.5           # Max backward drift limit (m)

    # Reset random noise ranges
    reset_alt_noise_m: float = 0.05
    reset_vel_noise_mps: float = 0.05
    reset_att_noise_rad: float = 0.02
    reset_obstacle_dist_min_m: float = 0.50
    reset_obstacle_dist_max_m: float = 0.90


# ─────────────────────────────────────────────────────────────────────────────
# 4. Arena & Environment Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ArenaConfig:
    room_x_min: float = -4.0
    room_x_max: float = 4.0
    room_y_min: float = -4.0
    room_y_max: float = 4.0
    room_z_min: float = 0.0
    room_z_max: float = 5.0

    obstacle_default_dist_m: float = 2.0
    obstacle_detection_threshold_m: float = 0.8
    obstacle_evasion_yaw_threshold_rad: float = 0.20
    obstacle_cleared_yaw_threshold_rad: float = 0.40
    obstacle_respawn_dist_min_m: float = 1.0
    obstacle_respawn_dist_max_m: float = 2.0
    obstacle_initial_dist_min_m: float = 0.6
    obstacle_initial_dist_max_m: float = 1.4
    turn_clearance_dist_m: float = 0.15
    turn_max_steps: int = 35
    obstacle_clearance_min_m: float = 0.20


# ─────────────────────────────────────────────────────────────────────────────
# 5. Artificial Potential Fields (APF) Expert Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class APFConfig:
    distance_threshold_m: float = 0.8       # Clearance threshold to trigger avoidance (m)
    max_sensor_range_m: float = 3.0         # Sensor normalization range (m)
    k_repulsive: float = 20.0               # Repulsive potential field gain
    noise_std_pwm: float = 15.0             # Gaussian actuator perturbation std (µs)
    
    # Smooth braking & turning gains
    max_pitch_brake_pwm: float = 280.0      # Maximum braking deflection from cruising pitch (µs)
    repulsion_scale: float = 600.0          # Characteristic scaling factor for rational saturation
    yaw_gain: float = 0.20                  # Yaw turning deflection gain per unit force
    min_dist_clamp: float = 0.1             # Epsilon distance to avoid division by zero (normalized)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Spatial Memory (Egocentric Ring Buffer) Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SpatialMemoryConfig:
    num_sectors: int = 8                    # 8 sectors (45 degrees each)
    decay_rate: float = 0.02                # Forgetting rate towards safe distance per step
    default_distance: float = 1.0           # Default cleared distance (normalized)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Metrics, Telemetry & Energy Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MetricsConfig:
    voxel_size_m: float = 0.5               # Voxel grid discretization cube size (m)
    saccade_yaw_threshold_pwm: float = 150.0# Min PWM change on Yaw to classify as a saccade
    min_speed_hover_mps: float = 0.10       # Minimum horizontal velocity for forward ratio tracking
    threshold_crab: float = 0.50            # Threshold below which flight is classified as crab flight
    
    # Cost function weights in DroneSimulationEnv
    cost_weight_alt: float = 1.0
    cost_weight_tilt: float = 2.0
    cost_weight_vel: float = 0.5
    cost_weight_jitter: float = 0.001

    # Actuator power model constants
    power_throttle_coeff: float = 15.0      # Base throttle power constant (Watts)
    power_attitude_coeff: float = 5.0       # Base attitude deflection power constant (Watts)
    hover_throttle_normalized: float = 0.5  # Normalized throttle corresponding to steady hover


# ─────────────────────────────────────────────────────────────────────────────
# Singletons for structured access
# ─────────────────────────────────────────────────────────────────────────────

SENSORS = SensorsConfig()
ACTUATORS = ActuatorConfig()
PHYSICS = PhysicsConfig()
ARENA = ArenaConfig()
APF = APFConfig()
MEMORY = SpatialMemoryConfig()
METRICS = MetricsConfig()


# ─────────────────────────────────────────────────────────────────────────────
# Direct module-level constants (for lightweight, zero-overhead access)
# ─────────────────────────────────────────────────────────────────────────────

# Sensor dimensions
SENSOR_DIM_BASE = SENSORS.sensor_dim_base
MEMORY_DIM = SENSORS.memory_dim
SENSOR_DIM = SENSORS.sensor_dim
FLOW_DIM = SENSORS.flow_dim
TOF_DIM = SENSORS.tof_dim
N_CONTROLS = SENSORS.n_controls
TOF_ROWS = SENSORS.tof_rows
TOF_COLS = SENSORS.tof_cols
TOF_MAX_RANGE_M = SENSORS.tof_max_range_m
TOF_RAYCASTER_MAX_RANGE_M = SENSORS.tof_raycaster_max_range_m
TOF_MIN_CLAMP_DIST = SENSORS.tof_min_clamp_dist
FLOW_Z_SAFE_M = SENSORS.flow_z_safe_m
LASER_MIN_RANGE_M = SENSORS.laser_min_range_m
LASER_MAX_RANGE_M = SENSORS.laser_max_range_m
LASER_NOISE_STD = SENSORS.laser_noise_std
FLOW_MIN_ALTITUDE_M = SENSORS.flow_min_altitude_m
FLOW_MAX_ALTITUDE_M = SENSORS.flow_max_altitude_m
FLOW_MAX_RATE_RADS = SENSORS.flow_max_rate_rads
FLOW_NOISE_STD = SENSORS.flow_noise_std

# RC PWM limits & channels (µs)
PWM_MIN = ACTUATORS.pwm_min
PWM_MID = ACTUATORS.pwm_mid
PWM_MAX = ACTUATORS.pwm_max
PWM_HALF = ACTUATORS.pwm_half
PWM_HOVER = ACTUATORS.pwm_hover
PWM_LEVEL_ROLL = ACTUATORS.pwm_level_roll
PWM_NEUTRAL_YAW = ACTUATORS.pwm_neutral_yaw
PWM_CRUISE_PITCH = ACTUATORS.pwm_cruise_pitch
PWM_BRAKE_PITCH = ACTUATORS.pwm_brake_pitch
PWM_TURN_LEFT_YAW = ACTUATORS.pwm_turn_left_yaw
PWM_TURN_RIGHT_YAW = ACTUATORS.pwm_turn_right_yaw
PWM_SATURATION_LOW = ACTUATORS.pwm_saturation_low
PWM_SATURATION_HIGH = ACTUATORS.pwm_saturation_high
MAX_SATURATION_RATIO = ACTUATORS.max_saturation_ratio

CH_THROTTLE = ACTUATORS.ch_throttle
CH_ROLL = ACTUATORS.ch_roll
CH_PITCH = ACTUATORS.ch_pitch
CH_YAW = ACTUATORS.ch_yaw
CHANNEL_KEYS = ACTUATORS.channel_keys

# Physics
DEFAULT_DT = PHYSICS.default_dt
GRAVITY = PHYSICS.gravity
DRONE_MASS_KG = PHYSICS.drone_mass_kg
DRONE_MASS_ISAAC_KG = PHYSICS.drone_mass_isaac_kg
DRAG_COEFF = PHYSICS.drag_coeff
ANGULAR_DAMPING = PHYSICS.angular_damping
THRUST_GAIN = PHYSICS.thrust_gain
TARGET_ALTITUDE_M = PHYSICS.target_altitude_m
ALTITUDE_HOLD_GAIN = PHYSICS.altitude_hold_gain
MAX_TILT_ANGLE_RAD = PHYSICS.max_tilt_angle_rad
ATTITUDE_TAU = PHYSICS.attitude_tau
YAW_RATE_GAIN = PHYSICS.yaw_rate_gain
TUMBLE_ANGLE_THRESHOLD_RAD = PHYSICS.tumble_angle_threshold_rad
GROUND_CRASH_ALT_M = PHYSICS.ground_crash_alt_m
OBSTACLE_CRASH_DIST_M = PHYSICS.obstacle_crash_dist_m
BACK_WALL_DIST_M = PHYSICS.back_wall_dist_m

# Arena
ROOM_X_MIN = ARENA.room_x_min
ROOM_X_MAX = ARENA.room_x_max
ROOM_Y_MIN = ARENA.room_y_min
ROOM_Y_MAX = ARENA.room_y_max
ROOM_Z_MIN = ARENA.room_z_min
ROOM_Z_MAX = ARENA.room_z_max
OBSTACLE_DEFAULT_DIST_M = ARENA.obstacle_default_dist_m
OBSTACLE_DETECTION_THRESHOLD_M = ARENA.obstacle_detection_threshold_m
OBSTACLE_EVASION_YAW_THRESHOLD_RAD = ARENA.obstacle_evasion_yaw_threshold_rad
OBSTACLE_CLEARED_YAW_THRESHOLD_RAD = ARENA.obstacle_cleared_yaw_threshold_rad
OBSTACLE_RESPAWN_DIST_MIN_M = ARENA.obstacle_respawn_dist_min_m
OBSTACLE_RESPAWN_DIST_MAX_M = ARENA.obstacle_respawn_dist_max_m
OBSTACLE_INITIAL_DIST_MIN_M = ARENA.obstacle_initial_dist_min_m
OBSTACLE_INITIAL_DIST_MAX_M = ARENA.obstacle_initial_dist_max_m
TURN_CLEARANCE_DIST_M = ARENA.turn_clearance_dist_m
TURN_MAX_STEPS = ARENA.turn_max_steps
OBSTACLE_CLEARANCE_MIN_M = ARENA.obstacle_clearance_min_m

# APF Expert
APF_DISTANCE_THRESHOLD_M = APF.distance_threshold_m
APF_MAX_SENSOR_RANGE_M = APF.max_sensor_range_m
APF_K_REPULSIVE = APF.k_repulsive
APF_NOISE_STD_PWM = APF.noise_std_pwm
APF_MAX_PITCH_BRAKE_PWM = APF.max_pitch_brake_pwm
APF_REPULSION_SCALE = APF.repulsion_scale
APF_YAW_GAIN = APF.yaw_gain
APF_MIN_DIST_CLAMP = APF.min_dist_clamp
APF_D0 = APF.distance_threshold_m / APF.max_sensor_range_m

# Memory
MEMORY_NUM_SECTORS = MEMORY.num_sectors
MEMORY_DECAY_RATE = MEMORY.decay_rate
MEMORY_DEFAULT_DISTANCE = MEMORY.default_distance

# Metrics & Energy
VOXEL_SIZE_M = METRICS.voxel_size_m
SACCADE_YAW_THRESHOLD_PWM = METRICS.saccade_yaw_threshold_pwm
MIN_SPEED_HOVER_MPS = METRICS.min_speed_hover_mps
THRESHOLD_CRAB = METRICS.threshold_crab
COST_WEIGHT_ALT = METRICS.cost_weight_alt
COST_WEIGHT_TILT = METRICS.cost_weight_tilt
COST_WEIGHT_VEL = METRICS.cost_weight_vel
COST_WEIGHT_JITTER = METRICS.cost_weight_jitter
POWER_THROTTLE_COEFF = METRICS.power_throttle_coeff
POWER_ATTITUDE_COEFF = METRICS.power_attitude_coeff
HOVER_THROTTLE_NORMALIZED = METRICS.hover_throttle_normalized
