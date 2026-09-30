"""
training/env.py
===============
Closed-Loop Drone Flight Simulation Environment for Chong-Fly.

Simulates the 6-DOF dynamics of a micro-drone controlled by ChongFlyMSPPolicy:
  - Sensor layer (66-D):
      [0:2]   FlowX, FlowY optical flow (normalized ±1)
      [2:66]  ToF 8x8 distance grid (normalized 0..1)
  - Actuation (4 PWM channels in [1000, 2000] µs):
      ch0: Throttle (1000..2000 µs, hover ~1500)
      ch1: Roll     (1000..2000 µs, neutral 1500)
      ch2: Pitch    (1000..2000 µs, neutral 1500)
      ch3: Yaw      (1000..2000 µs, neutral 1500)
  - Disturbance injections:
      - Optomotor drift stimulus (horizontal velocity perturbation)
      - Looming threat approach (obstacle moving into field of view)

Complexity:
  Each step performs O(1) physics updates, while the policy forward step
  is O(N) where N = k_clusters macro-nodes.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple

import numpy as np

from configs.flight_config import (
    SENSOR_DIM,
    SENSOR_DIM_BASE,
    FLOW_DIM,
    TOF_DIM,
    TOF_ROWS,
    TOF_COLS,
    TOF_MAX_RANGE_M,
    MEMORY_DIM,
    N_CONTROLS,
    PWM_MIN,
    PWM_MID,
    PWM_MAX,
    PWM_HALF,
    PWM_HOVER,
    PWM_SATURATION_LOW,
    PWM_SATURATION_HIGH,
    MAX_SATURATION_RATIO,
    CH_THROTTLE,
    CH_ROLL,
    CH_PITCH,
    CH_YAW,
    DEFAULT_DT,
    GRAVITY,
    DRONE_MASS_KG,
    DRONE_MASS_ISAAC_KG,
    DRAG_COEFF,
    ANGULAR_DAMPING,
    THRUST_GAIN,
    TARGET_ALTITUDE_M,
    ALTITUDE_HOLD_GAIN,
    MAX_TILT_ANGLE_RAD,
    ATTITUDE_TAU,
    YAW_RATE_GAIN,
    TUMBLE_ANGLE_THRESHOLD_RAD,
    GROUND_CRASH_ALT_M,
    OBSTACLE_CRASH_DIST_M,
    BACK_WALL_DIST_M,
    OBSTACLE_DEFAULT_DIST_M,
    OBSTACLE_DETECTION_THRESHOLD_M,
    OBSTACLE_EVASION_YAW_THRESHOLD_RAD,
    OBSTACLE_CLEARED_YAW_THRESHOLD_RAD,
    OBSTACLE_RESPAWN_DIST_MIN_M,
    OBSTACLE_RESPAWN_DIST_MAX_M,
    MEMORY_DECAY_RATE,
    MEMORY_DEFAULT_DISTANCE,
    VOXEL_SIZE_M,
    MIN_SPEED_HOVER_MPS,
    THRESHOLD_CRAB,
    COST_WEIGHT_ALT,
    COST_WEIGHT_TILT,
    COST_WEIGHT_VEL,
    COST_WEIGHT_JITTER,
    POWER_THROTTLE_COEFF,
    POWER_ATTITUDE_COEFF,
    HOVER_THROTTLE_NORMALIZED,
    FLOW_Z_SAFE_M,
    SENSORS,
    PHYSICS,

    MAX_SIM_TIME_S,
    CONTROL_DT,
    STAGNATION_TIME_LIMIT_S,
    MIN_SPEED_HOVER_MPS,
)


#class DroneSimulationEnv:
#    """
#    Fast closed-loop drone flight simulator for Level 2 behavioral evaluation.
#    """
#
#    def __init__(
#        self,
#        dt: float = DEFAULT_DT,
#        target_altitude: float = TARGET_ALTITUDE_M,
#        gravity: float = GRAVITY,
#        mass: float = DRONE_MASS_KG,
#        drag_coeff: float = DRAG_COEFF,
#        angular_damping: float = ANGULAR_DAMPING,
#        thrust_gain: float = THRUST_GAIN,
#    ):
#        self.dt = dt
#        self.target_altitude = target_altitude
#        self.g = gravity
#        self.mass = mass
#        self.drag = drag_coeff
#        self.damping = angular_damping
#        self.max_thrust_acc = gravity * thrust_gain
#
#        # State vectors:
#        # pos = [x, y, z]
#        # vel = [vx, vy, vz]
#        # att = [phi (roll), theta (pitch), psi (yaw)] in radians
#        # omega = [p, q, r] angular velocities in rad/s
#        self.pos = np.zeros(3, dtype=np.float32)
#        self.vel = np.zeros(3, dtype=np.float32)
#        self.att = np.zeros(3, dtype=np.float32)
#        self.omega = np.zeros(3, dtype=np.float32)
#
#        # Obstacle state
#        self.obstacle_dist = OBSTACLE_DEFAULT_DIST_M
#        self.obstacle_active = True
#
#        # Last PWM for jitter measurement
#        self.last_pwm = np.full(N_CONTROLS, PWM_HOVER, dtype=np.float32)
#
#    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
#        """
#        Reset simulation state to hover baseline with small random noise.
#        """
#        rng = np.random.default_rng(seed)
#        self.pos = np.array([0.0, 0.0, self.target_altitude + rng.uniform(-PHYSICS.reset_alt_noise_m, PHYSICS.reset_alt_noise_m)], dtype=np.float32)
#        self.vel = rng.uniform(-PHYSICS.reset_vel_noise_mps, PHYSICS.reset_vel_noise_mps, size=3).astype(np.float32)
#        self.att = rng.uniform(-PHYSICS.reset_att_noise_rad, PHYSICS.reset_att_noise_rad, size=3).astype(np.float32)
#        self.omega = np.zeros(3, dtype=np.float32)
#
#        self.obstacle_dist = float(rng.uniform(PHYSICS.reset_obstacle_dist_min_m, PHYSICS.reset_obstacle_dist_max_m))
#        self.last_pwm = np.full(N_CONTROLS, PWM_HOVER, dtype=np.float32)
#
#        return self._get_observations()
#
#    def _get_observations(self) -> Tuple[np.ndarray, np.ndarray]:
#        """
#        Synthesize FlowX/Y and 8x8 ToF grid from physical state.
#        """
#        # Optical Flow: horizontal velocity relative to ground altitude in body frame
#        z_safe = max(FLOW_Z_SAFE_M, float(self.pos[2]))
#        psi = float(self.att[2])
#        v_fwd = float(self.vel[0] * math.cos(psi) + self.vel[1] * math.sin(psi))
#        v_lat = float(-self.vel[0] * math.sin(psi) + self.vel[1] * math.cos(psi))
#        flow_x = np.clip(v_fwd / z_safe, -1.0, 1.0)
#        flow_y = np.clip(v_lat / z_safe, -1.0, 1.0)
#        flow_xy = np.array([flow_x, flow_y], dtype=np.float32)
#
#        # ToF 8x8 Grid: distance normalized to [0, 1]
#        # Looming obstacle projects onto central pixels
#        max_range = TOF_MAX_RANGE_M
#        grid = np.ones((TOF_ROWS, TOF_COLS), dtype=np.float32)
#        norm_dist = np.clip(self.obstacle_dist / max_range, 0.0, 1.0)
#
#        # Central 4x4 pixels represent forward looming zone
#        grid[SENSORS.tof_center_row_start:SENSORS.tof_center_row_end, SENSORS.tof_center_col_start:SENSORS.tof_center_col_end] = norm_dist
#        tof_8x8 = grid.ravel().astype(np.float32)
#
#        return flow_xy, tof_8x8
#
#    def step(
#        self,
#        pwm: np.ndarray,
#        dt: Optional[float] = None,
#    ) -> Tuple[Tuple[np.ndarray, np.ndarray], float, bool, dict[str, Any]]:
#        """
#        Advance physics by one time step with given PWM actuators.
#
#        Args:
#            pwm: (4,) array [throttle, roll, pitch, yaw] in [1000, 2000] µs.
#            dt: optional timestep override.
#
#        Returns:
#            ((flow_xy, tof_8x8), step_cost, done, info)
#        """
#        step_dt = self.dt if dt is None else dt
#        pwm = np.asarray(pwm, dtype=np.float32)
#
#        # 1. Normalize PWM to control inputs [-1, 1] or [0, 1]
#        throttle_norm = np.clip((pwm[CH_THROTTLE] - PWM_MIN) / (PWM_MAX - PWM_MIN), 0.0, 1.0)
#        roll_cmd      = np.clip((pwm[CH_ROLL] - PWM_MID) / PWM_HALF, -1.0, 1.0)
#        pitch_cmd     = np.clip((pwm[CH_PITCH] - PWM_MID) / PWM_HALF, -1.0, 1.0)
#        yaw_cmd       = np.clip((pwm[CH_YAW] - PWM_MID) / PWM_HALF, -1.0, 1.0)
#
#        # 2. Translational Dynamics
#        # Vertical thrust: 0.5 throttle gives exactly gravity compensation
#        # Vertical acceleration with altitude hold stabilization
#        vertical_thrust_acc = throttle_norm * self.max_thrust_acc
#        a_z = (
#            vertical_thrust_acc
#            - self.g
#            - (self.drag * self.vel[2])
#            + ALTITUDE_HOLD_GAIN * (self.target_altitude - self.pos[2])
#        )
#
#        # Attitude dynamics: Angle mode tracking (target angle proportional to stick)
#        target_phi = roll_cmd * MAX_TILT_ANGLE_RAD
#        target_theta = pitch_cmd * MAX_TILT_ANGLE_RAD
#        att_tau = ATTITUDE_TAU
#
#        self.att[0] += (target_phi - self.att[0]) * min(1.0, step_dt / att_tau)
#        self.att[1] += (target_theta - self.att[1]) * min(1.0, step_dt / att_tau)
#        self.att[2] += (yaw_cmd * YAW_RATE_GAIN) * step_dt
#
#        self.omega[0] = (target_phi - self.att[0]) / att_tau
#        self.omega[1] = (target_theta - self.att[1]) / att_tau
#        self.omega[2] = yaw_cmd * YAW_RATE_GAIN
#
#        phi, theta, psi = self.att
#
#        # Horizontal acceleration in world frame (rotated by yaw psi)
#        ax_body = self.g * math.sin(theta)
#        ay_body = -self.g * math.sin(phi)
#        a_x = ax_body * math.cos(psi) - ay_body * math.sin(psi) - (self.drag * self.vel[0])
#        a_y = ax_body * math.sin(psi) + ay_body * math.cos(psi) - (self.drag * self.vel[1])
#
#        # Integrate translation
#        self.vel[0] += a_x * step_dt
#        self.vel[1] += a_y * step_dt
#        self.vel[2] += a_z * step_dt
#
#        self.pos[0] += self.vel[0] * step_dt
#        self.pos[1] += self.vel[1] * step_dt
#        self.pos[2] = max(0.0, self.pos[2] + self.vel[2] * step_dt)
#
#        # 4. Obstacle relative movement
#        # Obstacle approaches along body forward axis
#        v_forward = self.vel[0] * math.cos(psi) + self.vel[1] * math.sin(psi)
#        self.obstacle_dist = max(0.0, self.obstacle_dist - v_forward * step_dt)
#
#        # 5. Cost / Performance evaluation
#        alt_err = abs(float(self.pos[2]) - self.target_altitude)
#        tilt_err = float(self.att[0]**2 + self.att[1]**2)
#        vel_err = float(self.vel[0]**2 + self.vel[1]**2)
#        jitter = float(np.mean(np.abs(pwm - self.last_pwm)))
#        self.last_pwm = pwm.copy()
#
#        step_cost = COST_WEIGHT_ALT * alt_err + COST_WEIGHT_TILT * tilt_err + COST_WEIGHT_VEL * vel_err + COST_WEIGHT_JITTER * jitter
#
#        # 6. Termination condition
#        tumbled = abs(self.att[0]) > TUMBLE_ANGLE_THRESHOLD_RAD or abs(self.att[1]) > TUMBLE_ANGLE_THRESHOLD_RAD
#        ground_crash = self.pos[2] <= GROUND_CRASH_ALT_M
#        obstacle_crash = self.obstacle_dist <= OBSTACLE_CRASH_DIST_M
#        hit_back_wall = self.obstacle_dist >= BACK_WALL_DIST_M
#        done = tumbled or ground_crash or obstacle_crash or hit_back_wall
#
#        info = {
#            "alt_err": alt_err,
#            "tilt_err": tilt_err,
#            "vel_err": vel_err,
#            "jitter": jitter,
#            "obstacle_dist": self.obstacle_dist,
#            "tumbled": tumbled,
#            "ground_crash": ground_crash,
#            "obstacle_crash": obstacle_crash,
#        }
#
#        obs = self._get_observations()
#        return obs, step_cost, done, info


# Встав це в optimizer/evaluate.py (замість старої функції simulate_policy_rollout)
import os
import torch
from simulation.metrics import (
    calculate_jitter_pr,
    calculate_saccades_yaw,
    VoxelTracker,
    PhysicsTelemetryTracker,
)
from simulation.memory import EgocentricMemoryWrapper
from optimizer.pretrain import pretrain_policy
from typing import Any, Optional, Tuple, Dict, Union
import numpy as np


def simulate_policy_rollout(
        policy: Any,
        env: Optional[Any] = None,  # Використовуй DroneSimulationEnv, але залишив Any для універсальності
        eval_steps: int = 100,
        dt: Optional[float] = None,
        seed: int = 42,
) -> Tuple[Tuple[float, float, float], dict]:
    """
    Runs closed-loop simulation with Math Signal Metrics and Physical Constraints.
    """
    # Якщо env не передано, створюємо локальний швидкий симулятор для Optuna
    if env is None:
        policy_dt = getattr(policy, "default_dt", None)
        if policy_dt is None and hasattr(policy, "cfc_network"):
            policy_dt = getattr(getattr(policy.cfc_network, "cell", None), "default_dt", DEFAULT_DT)
        step_dt = float(policy_dt) if policy_dt is not None else DEFAULT_DT
        
        from simulation.drone_env import OpticalFlowOUWrapper
        from simulation.drone_env import DroneSimulationEnv
        raw_env = DroneSimulationEnv(dt=step_dt)
        env = OpticalFlowOUWrapper(raw_env, seed=seed)

    if hasattr(policy, "reset_state"):
        policy.reset_state()

    reset_res = env.reset(seed=seed)
    if isinstance(reset_res, tuple) and len(reset_res) == 2:
        obs_flow, obs_tof = reset_res
    elif hasattr(env, "get_chong_fly_obs"):
        obs_flow, obs_tof = env.get_chong_fly_obs()
    elif isinstance(reset_res, np.ndarray) and reset_res.shape[0] >= SENSOR_DIM_BASE:
        obs_flow = reset_res[:FLOW_DIM]
        obs_tof = reset_res[FLOW_DIM:(FLOW_DIM + TOF_DIM)]
    else:
        obs_flow = np.zeros(FLOW_DIM, dtype=np.float32)
        obs_tof = np.ones(TOF_DIM, dtype=np.float32)

    steps_survived = 0
    total_energy_j = 0.0

    # Масив для нашої математики
    pwm_history = []

    # Трекер просторового покриття (дискретизація R^3 -> Z^3)
    voxel_tracker = VoxelTracker(voxel_size=VOXEL_SIZE_M)

    # Фізичний фільтр-вбивця
    saturation_violations = 0

    rng = np.random.default_rng(seed)
    walls_avoided = 0
    in_evasion = False
    fwd_speeds = []

    # Трекер фізичної телеметрії (Mean Clearance, Kinetic Energy, Crab Flight)
    telemetry_tracker = PhysicsTelemetryTracker(min_speed_hover=MIN_SPEED_HOVER_MPS, threshold_crab=THRESHOLD_CRAB)

    # Йогоцентричний буфер просторової пам'яті (8 секторів, 74-D простір сенсорів)
    memory_wrapper = EgocentricMemoryWrapper(decay_rate=MEMORY_DECAY_RATE)
    if hasattr(env, "att"):
        last_yaw = float(env.att[2])
    elif hasattr(env, "physics") and hasattr(env.physics, "quaternion_to_euler"):
        last_yaw = float(env.physics.quaternion_to_euler(env.physics.quat)[2])
    else:
        last_yaw = 0.0
    evasion_yaw_initial = last_yaw
    memory_8 = memory_wrapper.get_memory()

    for step in range(eval_steps):
        # Відстежуємо позицію дрона
        pos = getattr(env, "pos", None)
        if pos is None and hasattr(env, "get_position"):
            pos = env.get_position()
        elif pos is None and hasattr(env, "physics"):
            pos = getattr(env.physics, "pos", None)
        if pos is not None:
            voxel_tracker.update(pos)

        # Отримуємо поточний кут курсу (Yaw)
        if hasattr(env, "att"):
            current_yaw = float(env.att[2])
        elif hasattr(env, "physics") and hasattr(env.physics, "quaternion_to_euler"):
            current_yaw = float(env.physics.quaternion_to_euler(env.physics.quat)[2])
        else:
            current_yaw = 0.0

        # Отримуємо швидкість і трансформуємо її у зв'язану систему координат (Body Frame)
        vel = getattr(env, "vel", None)
        if vel is None and hasattr(env, "get_velocity"):
            vel = env.get_velocity()
        elif vel is None and hasattr(env, "physics"):
            vel = getattr(env.physics, "vel", None)

        vx_w = float(vel[0]) if vel is not None else 0.0
        vy_w = float(vel[1]) if vel is not None else 0.0
        vz_w = float(vel[2]) if vel is not None else 0.0
        v_fwd = vx_w * math.cos(current_yaw) + vy_w * math.sin(current_yaw)
        v_lat = -vx_w * math.sin(current_yaw) + vy_w * math.cos(current_yaw)
        vel_body = np.array([v_fwd, v_lat, vz_w], dtype=np.float32)
        fwd_speeds.append(v_fwd)

        # ─── ПРАВИЛО АКУЛИ (Kill-Switch за пасивність) ───
        # Скільки кроків становить наш ліміт стагнації (напр. 1.5 сек / 0.02 = 75 кроків)
        stagnation_steps = int(STAGNATION_TIME_LIMIT_S / step_dt)
        if steps_survived > stagnation_steps:
            # Дивимось середню швидкість ТІЛЬКИ за останні 1.5 секунди
            recent_speed = sum(fwd_speeds[-stagnation_steps:]) / stagnation_steps
            if recent_speed < MIN_SPEED_HOVER_MPS:
                print(f"KILLED AT {steps_survived * step_dt:.2f}s: Stagnation (Zombie detected)")
                break  # Вбиваємо симуляцію достроково!

        # Оновлюємо фізичну телеметрію ТІЛЬКИ в Body Frame (запобігає хибному crab flight на поворотах)
        telemetry_tracker.update_step(obs_tof, vel_body)

        # Оновлюємо егоцентричну пам'ять (8 секторів) за кутом курсу (Yaw)
        delta_yaw = current_yaw - last_yaw
        delta_yaw = (delta_yaw + math.pi) % (2.0 * math.pi) - math.pi
        last_yaw = current_yaw
        memory_8 = memory_wrapper.update(obs_tof, delta_yaw_rad=delta_yaw)

        # Формуємо розширене 74-D спостереження [FlowX, FlowY, ToF_64, Mem_8]
        obs_74 = np.concatenate([
            np.asarray(obs_flow, dtype=np.float32).ravel(),
            np.asarray(obs_tof, dtype=np.float32).ravel(),
            np.asarray(memory_8, dtype=np.float32).ravel(),
        ])

        if hasattr(policy, "step_np"):
            try:
                pwm = policy.step_np(obs_flow, obs_tof, memory_ring=memory_8, dt=dt)
            except TypeError:
                pwm = policy.step_np(obs_flow, obs_tof, dt=dt)
        elif hasattr(policy, "step"):
            pwm = policy.step(obs_74, dt=dt)
            if hasattr(pwm, "cpu"):
                pwm = pwm.cpu().numpy()
        elif callable(policy):
            try:
                pwm = policy(obs_74)
            except TypeError:
                pwm = policy(obs_flow, obs_tof)
            if hasattr(pwm, "cpu"):
                pwm = pwm.cpu().numpy()
        else:
            pwm = np.full(N_CONTROLS, PWM_HOVER, dtype=np.float32)

        # 1. Захист від математичного вибуху (Spectral Radius > 1)
        if np.isnan(pwm).any() or np.isinf(pwm).any():
            print(f"FAILED AT {steps_survived}: NaN generated")
            failure_metrics = {
                "fatal_failure": True,
                "failure_reason": "NaN generated",
                "crashed": True,
                "walls_avoided": walls_avoided,
                "mean_fwd_speed": 0.0,
                "survival_time_s": float(steps_survived * env.dt),
                "survival_ratio": float(steps_survived / eval_steps),
                "saturation_ratio": 1.0,
                "jitter_pr_l2": 9999.0,
                "saccades_yaw_count": 0,
                "roughness_score": 9999.0,
                "coverage_count": 0,
                "coverage_volume": 0.0,
                "mean_clearance": 0.0,
                "impact_energy_j": 9999.0,
                "forward_ratio_median": 0.0,
                "is_crab_flight": False,
                "memory_sectors": memory_8.copy() if hasattr(memory_8, "copy") else memory_8,
            }
            return (float("inf"), float("inf"), 0.0), failure_metrics

        # 2. ФІЗИЧНИЙ ФІЛЬТР: Якщо мотори виходять за межі сатурації, вони задихаються
        if np.any(pwm < PWM_SATURATION_LOW) or np.any(pwm > PWM_SATURATION_HIGH):
            saturation_violations += 1

        pwm_history.append(pwm.copy())

        # Енергетична модель
        u_t = np.clip((pwm[CH_THROTTLE] - PWM_MIN) / (PWM_MAX - PWM_MIN), 0.0, 1.0)
        u_r = np.clip((pwm[CH_ROLL] - PWM_MID) / PWM_HALF, -1.0, 1.0)
        u_p = np.clip((pwm[CH_PITCH] - PWM_MID) / PWM_HALF, -1.0, 1.0)
        u_y = np.clip((pwm[CH_YAW] - PWM_MID) / PWM_HALF, -1.0, 1.0)
        p_actuators = POWER_THROTTLE_COEFF * ((u_t / HOVER_THROTTLE_NORMALIZED) ** 2) + POWER_ATTITUDE_COEFF * (u_r ** 2 + u_p ** 2 + u_y ** 2)
        total_energy_j += p_actuators * env.dt

        try:
            step_res = env.step(pwm, dt=dt)
        except TypeError:
            step_res = env.step(pwm)

        # Безперервний респавн стіни під час rollout
        if hasattr(env, "obstacle_dist"):
            if env.obstacle_dist < OBSTACLE_DETECTION_THRESHOLD_M and abs(delta_yaw) > OBSTACLE_CLEARED_YAW_THRESHOLD_RAD:
                walls_avoided += 1
                env.obstacle_dist = float(np.random.uniform(OBSTACLE_RESPAWN_DIST_MIN_M, OBSTACLE_RESPAWN_DIST_MAX_M))
            elif env.obstacle_dist <= SENSORS.tof_min_clamp_dist:  # Якщо не встиг і майже врізався
                pass  # Краш відпрацює нижче

        next_obs, step_cost, done, info = step_res
        if isinstance(next_obs, tuple) and len(next_obs) == 2:
            obs_flow, obs_tof = next_obs
        elif hasattr(env, "get_chong_fly_obs"):
            obs_flow, obs_tof = env.get_chong_fly_obs()
        elif isinstance(next_obs, np.ndarray) and next_obs.shape[0] >= SENSOR_DIM_BASE:
            obs_flow = next_obs[:FLOW_DIM]
            obs_tof = next_obs[FLOW_DIM:(FLOW_DIM + TOF_DIM)]

        # ── Неперервне середовище: ухилення від перешкод та їх респавн ──
        dist = float(getattr(env, "obstacle_dist", 99.0))
        if dist < OBSTACLE_DETECTION_THRESHOLD_M and not in_evasion:
            in_evasion = True
            evasion_yaw_initial = current_yaw

        if in_evasion:
            yaw_deflection = abs((current_yaw - evasion_yaw_initial + math.pi) % (2.0 * math.pi) - math.pi)
            if (yaw_deflection > OBSTACLE_EVASION_YAW_THRESHOLD_RAD) and dist >= SENSORS.tof_min_clamp_dist:
                walls_avoided += 1
                in_evasion = False
                # Респавн наступної перешкоди попереду за новим курсом
                new_dist = float(rng.uniform(OBSTACLE_RESPAWN_DIST_MIN_M, OBSTACLE_RESPAWN_DIST_MAX_M))
                env.obstacle_dist = new_dist
                grid = np.ones((TOF_ROWS, TOF_COLS), dtype=np.float32)
                grid[SENSORS.tof_center_row_start:SENSORS.tof_center_row_end, SENSORS.tof_center_col_start:SENSORS.tof_center_col_end] = min(1.0, new_dist / TOF_MAX_RANGE_M)
                obs_tof = grid.ravel().astype(np.float32)

        steps_survived += 1

        if done:
            # Якщо краш настав до завершення часу — фіксуємо кінетичну енергію удару
            final_vel = getattr(env, "vel", None)
            if final_vel is None and hasattr(env, "get_velocity"):
                final_vel = env.get_velocity()
            elif final_vel is None and hasattr(env, "physics"):
                final_vel = getattr(env.physics, "vel", None)
            mass_obj = getattr(getattr(env, "physics", None), "total_mass", getattr(env, "mass", DRONE_MASS_ISAAC_KG))
            drone_mass = float(mass_obj() if callable(mass_obj) else (mass_obj if mass_obj is not None else DRONE_MASS_ISAAC_KG))
            telemetry_tracker.register_crash(final_vel, mass=drone_mass)
            print(f"CRASH: {info}"); break

    # ── ПІСЛЯ ПОЛЬОТУ: Оцінка Метрик ──
    survival_ratio = steps_survived / eval_steps
    survival_time_s = steps_survived * env.dt
    mean_power_w = total_energy_j / max(1e-4, survival_time_s)
    mean_fwd_speed = float(np.mean(fwd_speeds)) if fwd_speeds else 0.0

    # Просторове покриття
    coverage_count = voxel_tracker.get_coverage_count()
    coverage_volume = voxel_tracker.get_coverage_volume()

    # Фізична телеметрія
    telemetry_summary = telemetry_tracker.compute_summary()

    # Викликаємо метрики сигналу
    jitter_pr = calculate_jitter_pr(pwm_history)
    saccades_yaw = calculate_saccades_yaw(pwm_history)

    # Фізичний фільтр / Kill-Switches:
    saturation_ratio = saturation_violations / max(1, steps_survived)
    is_saturated = saturation_ratio > MAX_SATURATION_RATIO
    is_crab = bool(telemetry_summary["is_crab_flight"])
    is_crash = bool(telemetry_summary["is_crash"]) or (steps_survived < eval_steps)

    if is_saturated or is_crab:
        failure_reasons = []
        if is_saturated:
            failure_reasons.append("PWM Saturated")
        if is_crab:
            failure_reasons.append(
                f"Crab Flight (median {telemetry_summary['forward_ratio_median']:.2f} < 0.50)"
            )

        #return 9999.0, {
        #    "fatal_failure": True,
        #    "failure_reason": ", ".join(failure_reasons),
        #    "crashed": is_crash,
        #    "walls_avoided": walls_avoided,
        #    "mean_fwd_speed": mean_fwd_speed,
        #    "coverage_count": coverage_count,
        #    "coverage_volume": float(coverage_volume),
        #    "coverage_bonus": 0.0,
        #    "mean_clearance": float(telemetry_summary["mean_clearance"]),
        #    "impact_energy_j": float(telemetry_summary["impact_energy_j"]),
        #    "forward_ratio_median": float(telemetry_summary["forward_ratio_median"]),
        #    "is_crab_flight": is_crab,
        #    "memory_sectors": memory_8.copy(),
        #}

    # Multi-Objective Vector (Pareto Front)
    # 1. Hardware Optimization (Minimize model size / non-zero weights)
    hardware_obj = 0.0
    if hasattr(policy, "parameters"):
        hardware_obj = float(sum((p != 0).sum().item() for p in policy.parameters()))
        
    # 2. Smoothness (Minimize Jitter on Pitch/Roll ONLY)
    jitter_obj = float(jitter_pr)
    
    # 3. Exploration / Survival (Maximize unique voxels visited)
    if survival_time_s < 2.0: # HYPERPARAMETER??????
        exploration_obj = 0.0
    else:
        exploration_obj = float(coverage_count)

    pareto_vector = (hardware_obj, jitter_obj, exploration_obj)

    metrics = {
        "fatal_failure": False,
        "crashed": is_crash,
        "walls_avoided": walls_avoided,
        "mean_fwd_speed": mean_fwd_speed,
        "survival_time_s": float(survival_time_s),
        "survival_ratio": float(survival_ratio),
        "saturation_ratio": float(saturation_ratio),
        "jitter_pr_l2": jitter_pr,
        "saccades_yaw_count": saccades_yaw,
        "roughness_score": float(jitter_pr + saccades_yaw),
        "coverage_count": coverage_count,
        "coverage_volume": coverage_volume,
        "mean_clearance": float(telemetry_summary["mean_clearance"]),
        "impact_energy_j": float(telemetry_summary["impact_energy_j"]),
        "forward_ratio_median": float(telemetry_summary["forward_ratio_median"]),
        "is_crab_flight": False,
        "memory_sectors": memory_8.copy(),
    }

    return pareto_vector, metrics


# ─────────────────────────────────────────────────────────────────────────────
# Default Trainable Policy (Fallback / Agile Recurrent Controller)
# ─────────────────────────────────────────────────────────────────────────────

class DefaultFlightPolicy(torch.nn.Module):
    """
    Agile 74-D recurrent flight policy using GRU temporal integration.
    Used for Optuna trials or when specific connectome ReducedModel artifacts
    are being synthesized.
    """

    def __init__(self, sensor_dim: int = SENSOR_DIM, hidden_dim: int = 32):
        super().__init__()
        self.sensor_dim = sensor_dim
        self.hidden_dim = hidden_dim
        self.fc_in = torch.nn.Linear(sensor_dim, hidden_dim)
        self.gru = torch.nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc_out = torch.nn.Linear(hidden_dim, N_CONTROLS)
        torch.nn.init.constant_(self.fc_out.bias, PWM_HOVER)
        self._hx: Optional[torch.Tensor] = None

    def reset_state(self) -> None:
        """Resets recurrent hidden state."""
        self._hx = None

    def forward(
        self,
        x: torch.Tensor,
        hx: Optional[torch.Tensor] = None,
        dt: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        squeeze_batch = False
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, 1, 74]
            squeeze_batch = True

        h_feat = torch.tanh(self.fc_in(x))
        out, h_last = self.gru(h_feat, hx)
        pwm = self.fc_out(out)
        pwm = torch.clamp(pwm, PWM_MIN, PWM_MAX)

        if squeeze_batch:
            return pwm.squeeze(1), h_last
        return pwm, h_last

    @torch.no_grad()
    def step_np(
        self,
        flow_xy: np.ndarray,
        tof_8x8: np.ndarray,
        memory_ring: Optional[np.ndarray] = None,
        dt: Optional[float] = None,
    ) -> np.ndarray:
        parts = [np.asarray(flow_xy).ravel()[:FLOW_DIM], np.asarray(tof_8x8).ravel()[:TOF_DIM]]
        if memory_ring is not None:
            parts.append(np.asarray(memory_ring).ravel()[:MEMORY_DIM])
        elif self.sensor_dim == SENSOR_DIM:
            parts.append(np.full(MEMORY_DIM, MEMORY_DEFAULT_DISTANCE, dtype=np.float32))

        raw = np.concatenate(parts).astype(np.float32)
        t_in = torch.from_numpy(raw).unsqueeze(0).unsqueeze(0).to(next(self.parameters()).device)  # [1, 1, 74]
        pwm_t, self._hx = self.forward(t_in, self._hx, dt=dt)
        return pwm_t.squeeze().cpu().numpy()

    def post_step(self) -> None:
        pass


def create_model(
    trial_or_params: Any = None,
    sensor_dim: int = SENSOR_DIM,
    base_dir: str = "data/reduced_models",
    allow_fallback: bool = False,
) -> Any:
    """
    Constructs a flight policy from an Optuna Trial or dictionary parameters.
    Attempts ChongFlyMSPPolicy.from_meta if metadata and weight matrices are available.
    If allow_fallback is False, raises FileNotFoundError if metadata or weights are missing.
    """
    params: Dict[str, Any] = {}
    if trial_or_params is not None:
        if hasattr(trial_or_params, "suggest_categorical"):
            params["k_clusters"] = trial_or_params.suggest_categorical("k_clusters", [32, 64])
            params["pruning_sparsity"] = trial_or_params.suggest_float("pruning_sparsity", 0.50, 0.90)
            params["solver_type"] = trial_or_params.suggest_categorical("solver_type", ["CfC"]) # cut off , "Euler_dt_0.02"
            params["ablate_cx"] = trial_or_params.suggest_categorical("ablate_cx", [False])
        elif isinstance(trial_or_params, dict):
            params = dict(trial_or_params)

    k = params.get("k_clusters", 64)
    sparsity = params.get("pruning_sparsity", 0.60)
    solver = params.get("solver_type", "CfC")
    ablate_cx = params.get("ablate_cx", False)

    _ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    suffix = "_nocx" if ablate_cx else ""
    meta_name = f"meta_spectral_k{k}{suffix}.json"
    meta_path = os.path.join(_ROOT_DIR, base_dir, meta_name)

    if not os.path.exists(meta_path):
        if not allow_fallback:
            raise FileNotFoundError(f"Connectome metadata file '{meta_path}' not found for k={k}, ablate_cx={ablate_cx}")
        return DefaultFlightPolicy(sensor_dim=sensor_dim)

    # Валідація наявності бінарної матриці ваг на диску (без мовчазного проковтування)
    import json
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_dict = json.load(f)
    w_file = meta_dict.get("w_file")
    w_path = os.path.join(os.path.dirname(meta_path), w_file) if w_file else ""
    if not os.path.exists(w_path):
        if not allow_fallback:
            raise FileNotFoundError(f"Weight matrix file '{w_path}' referenced in '{meta_name}' not found on disk")
        return DefaultFlightPolicy(sensor_dim=sensor_dim)

    try:
        from simulation.policy import ChongFlyMSPPolicy
        dt = 0.02 if solver == "Euler_dt_0.02" else DEFAULT_DT
        return ChongFlyMSPPolicy.from_meta(
            meta_path=meta_path,
            sensor_dim=sensor_dim,
            solver_type=solver,
            pruning_sparsity=sparsity,
            ablate_cx=ablate_cx,
            dt=dt,
        )
    except Exception as e:
        if not allow_fallback:
            raise
        pass

    return DefaultFlightPolicy(sensor_dim=sensor_dim)


def objective(
    trial: Any = None,
    dataset_path: str = "data/reflex_dataset.pt",
    pretrain: bool = True,
    pretrain_epochs: int = 15,
    subset_ratio: float = 0.7,
    #eval_steps: int = 100,
    seed: int = 42,
    device: str = "auto",
) -> Tuple[float, float, float]:
    """
    Optuna Multi-Objective Function:
    1. Proposes hyperparameters & builds policy (create_model).
    2. Rapid Behavioral Cloning pretraining (pretrain_policy).
    3. Closed-loop arena simulation rollout (simulate_policy_rollout).
    4. Records trial telemetry and returns Pareto vector (Hardware, Jitter, Exploration).
    """
    policy = create_model(trial, sensor_dim=SENSOR_DIM, allow_fallback=False)


    import torch
    
    if device == "auto":
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        target_device = torch.device(device)
        if target_device.type == "cuda" and not torch.cuda.is_available():
            print(f"WARNING: CUDA not available, falling back to CPU.")
            target_device = torch.device("cpu")
            
    # Move policy to device
    try:
        policy = policy.to(target_device)
    except Exception as e:
        print(f"WARNING: Could not move policy to {target_device}: {e}")
        
    if pretrain:
        trial_seed = getattr(trial, "number", seed) if trial is not None else seed
        policy = pretrain_policy(
            policy=policy,
            dataset_path=dataset_path,
            epochs=pretrain_epochs,
            subset_ratio=subset_ratio,
            seed=trial_seed,
            device=target_device,
        )

    total_eval_steps = int(MAX_SIM_TIME_S / CONTROL_DT)

    pareto_vector, metrics = simulate_policy_rollout(
        policy=policy,
        eval_steps=total_eval_steps,
        dt=CONTROL_DT,
        seed=seed,
    )

    if trial is not None and hasattr(trial, "set_user_attr"):
        # Автоматично зберігаємо ВСІ метрики, які зібрав симулятор
        for key, value in metrics.items():
            # Optuna не вміє зберігати масиви в БД, тому переводимо пам'ять у рядок
            if key == "memory_sectors":
                value = str(value)
            trial.set_user_attr(key, value)

    return pareto_vector
