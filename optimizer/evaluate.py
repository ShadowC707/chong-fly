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


class DroneSimulationEnv:
    """
    Fast closed-loop drone flight simulator for Level 2 behavioral evaluation.
    """

    def __init__(
        self,
        dt: float = 0.004,
        target_altitude: float = 1.0,
        gravity: float = 9.81,
        mass: float = 0.035,        # 35g micro-drone (Crazyflie-class)
        drag_coeff: float = 0.25,
        angular_damping: float = 4.0,
        thrust_gain: float = 1.8,   # max thrust / weight ratio
    ):
        self.dt = dt
        self.target_altitude = target_altitude
        self.g = gravity
        self.mass = mass
        self.drag = drag_coeff
        self.damping = angular_damping
        self.max_thrust_acc = gravity * thrust_gain

        # State vectors:
        # pos = [x, y, z]
        # vel = [vx, vy, vz]
        # att = [phi (roll), theta (pitch), psi (yaw)] in radians
        # omega = [p, q, r] angular velocities in rad/s
        self.pos = np.zeros(3, dtype=np.float32)
        self.vel = np.zeros(3, dtype=np.float32)
        self.att = np.zeros(3, dtype=np.float32)
        self.omega = np.zeros(3, dtype=np.float32)

        # Obstacle state
        self.obstacle_dist = 2.0     # meters away
        self.obstacle_active = True

        # Last PWM for jitter measurement
        self.last_pwm = np.full(4, 1500.0, dtype=np.float32)

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reset simulation state to hover baseline with small random noise.
        """
        rng = np.random.default_rng(seed)
        self.pos = np.array([0.0, 0.0, self.target_altitude + rng.uniform(-0.05, 0.05)], dtype=np.float32)
        self.vel = rng.uniform(-0.05, 0.05, size=3).astype(np.float32)
        self.att = rng.uniform(-0.02, 0.02, size=3).astype(np.float32)
        self.omega = np.zeros(3, dtype=np.float32)

        self.obstacle_dist = float(rng.uniform(1.8, 2.5))
        self.last_pwm = np.full(4, 1500.0, dtype=np.float32)

        return self._get_observations()

    def _get_observations(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Synthesize FlowX/Y and 8x8 ToF grid from physical state.
        """
        # Optical Flow: horizontal velocity relative to ground altitude
        z_safe = max(0.2, float(self.pos[2]))
        flow_x = np.clip(self.vel[0] / z_safe, -1.0, 1.0)
        flow_y = np.clip(self.vel[1] / z_safe, -1.0, 1.0)
        flow_xy = np.array([flow_x, flow_y], dtype=np.float32)

        # ToF 8x8 Grid: distance normalized to [0, 1] (max range 3.0 m)
        # Looming obstacle projects onto central pixels
        max_range = 3.0
        grid = np.ones((8, 8), dtype=np.float32)
        norm_dist = np.clip(self.obstacle_dist / max_range, 0.0, 1.0)

        # Central 4x4 pixels represent forward looming zone
        grid[2:6, 2:6] = norm_dist
        tof_8x8 = grid.ravel().astype(np.float32)

        return flow_xy, tof_8x8

    def step(
        self,
        pwm: np.ndarray,
        dt: Optional[float] = None,
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], float, bool, dict[str, Any]]:
        """
        Advance physics by one time step with given PWM actuators.
        
        Args:
            pwm: (4,) array [throttle, roll, pitch, yaw] in [1000, 2000] µs.
            dt: optional timestep override.
            
        Returns:
            ((flow_xy, tof_8x8), step_cost, done, info)
        """
        step_dt = self.dt if dt is None else dt
        pwm = np.asarray(pwm, dtype=np.float32)

        # 1. Normalize PWM to control inputs [-1, 1] or [0, 1]
        throttle_norm = np.clip((pwm[0] - 1000.0) / 1000.0, 0.0, 1.0)  # 0.5 = hover
        roll_cmd      = np.clip((pwm[1] - 1500.0) / 500.0, -1.0, 1.0)
        pitch_cmd     = np.clip((pwm[2] - 1500.0) / 500.0, -1.0, 1.0)
        yaw_cmd       = np.clip((pwm[3] - 1500.0) / 500.0, -1.0, 1.0)

        # 2. Translational Dynamics
        # Vertical thrust: 0.5 throttle gives exactly gravity compensation
        vertical_thrust_acc = throttle_norm * (2.0 * self.g)
        a_z = vertical_thrust_acc - self.g - (self.drag * self.vel[2])

        # Horizontal acceleration coupled to roll (phi) and pitch (theta)
        phi, theta, psi = self.att
        a_x = self.g * math.sin(theta) - (self.drag * self.vel[0])
        a_y = -self.g * math.sin(phi)  - (self.drag * self.vel[1])

        # Integrate translation
        self.vel[0] += a_x * step_dt
        self.vel[1] += a_y * step_dt
        self.vel[2] += a_z * step_dt

        self.pos[0] += self.vel[0] * step_dt
        self.pos[1] += self.vel[1] * step_dt
        self.pos[2] = max(0.0, self.pos[2] + self.vel[2] * step_dt)

        # 3. Rotational Dynamics
        torque_roll  = roll_cmd * 12.0
        torque_pitch = pitch_cmd * 12.0
        torque_yaw   = yaw_cmd * 8.0

        # Angular acceleration with damping
        alpha_p = torque_roll  - (self.damping * self.omega[0])
        alpha_q = torque_pitch - (self.damping * self.omega[1])
        alpha_r = torque_yaw   - (self.damping * self.omega[2])

        self.omega[0] += alpha_p * step_dt
        self.omega[1] += alpha_q * step_dt
        self.omega[2] += alpha_r * step_dt

        self.att[0] += self.omega[0] * step_dt
        self.att[1] += self.omega[1] * step_dt
        self.att[2] += self.omega[2] * step_dt

        # 4. Obstacle relative movement
        # Obstacle approaches as drone moves along +x
        self.obstacle_dist = max(0.0, self.obstacle_dist - self.vel[0] * step_dt)

        # 5. Cost / Performance evaluation
        alt_err = abs(float(self.pos[2]) - self.target_altitude)
        tilt_err = float(self.att[0]**2 + self.att[1]**2)
        vel_err = float(self.vel[0]**2 + self.vel[1]**2)
        jitter = float(np.mean(np.abs(pwm - self.last_pwm)))
        self.last_pwm = pwm.copy()

        step_cost = alt_err + 2.0 * tilt_err + 0.5 * vel_err + 0.001 * jitter

        # 6. Termination condition
        tumbled = abs(self.att[0]) > 1.2 or abs(self.att[1]) > 1.2  # ~70 deg
        ground_crash = self.pos[2] <= 0.02
        obstacle_crash = self.obstacle_dist <= 0.05
        done = tumbled or ground_crash or obstacle_crash

        info = {
            "alt_err": alt_err,
            "tilt_err": tilt_err,
            "vel_err": vel_err,
            "jitter": jitter,
            "obstacle_dist": self.obstacle_dist,
            "tumbled": tumbled,
            "ground_crash": ground_crash,
            "obstacle_crash": obstacle_crash,
        }

        obs = self._get_observations()
        return obs, step_cost, done, info


# Встав це в optimizer/evaluate.py (замість старої функції simulate_policy_rollout)
from simulation.metrics import (
    calculate_jitter_pr,
    calculate_saccades_yaw,
    VoxelTracker,
    PhysicsTelemetryTracker,
)
from simulation.memory import EgocentricMemoryWrapper
from typing import Any, Optional, Tuple
import numpy as np


def compute_composite_cost(
    survival_ratio: float,
    mean_power_w: float,
    jitter_pr: float,
    saccades_yaw: int,
    coverage_count: int = 0,
    coverage_weight: float = 1.0,
    coverage_cap: float = 50.0,
) -> Tuple[float, float]:
    """
    Computes composite fitness cost for Optuna optimization.
    Applies Variant 2 gating: exploration coverage bonus is non-linearly gated
    by (survival_ratio ** 3) and capped, ensuring crash penalties can NEVER be outweighed.

    Returns:
        (composite_cost, coverage_bonus)
    """
    crash_penalty = (1.0 - float(survival_ratio)) * 1000.0
    energy_penalty = float(mean_power_w) * 2.0
    signal_cost = (float(jitter_pr) * 0.05) - (float(saccades_yaw) * 10.0)

    # Варіант 2: нелінійне шлюзування виживанням + жорстка стеля (cap)
    raw_coverage_bonus = min(float(coverage_count) * float(coverage_weight), float(coverage_cap))
    coverage_bonus = raw_coverage_bonus * (float(survival_ratio) ** 3)

    composite_cost = max(0.0, crash_penalty + energy_penalty + signal_cost - coverage_bonus)
    return float(composite_cost), float(coverage_bonus)


def simulate_policy_rollout(
        policy: Any,
        env: Optional[Any] = None,  # Використовуй DroneSimulationEnv, але залишив Any для універсальності
        eval_steps: int = 100,
        dt: Optional[float] = None,
        seed: int = 42,
) -> Tuple[float, dict]:
    """
    Runs closed-loop simulation with Math Signal Metrics and Physical Constraints.
    """
    # Якщо env не передано, створюємо (але тут краще імпортувати DroneSimulationEnv)
    if env is None:
        from simulation.drone_env import DroneSimulationEnv
        step_dt = 0.02 if getattr(policy, "default_dt", 0.004) == 0.02 else 0.004
        env = DroneSimulationEnv(dt=step_dt)

    if hasattr(policy, "reset_state"):
        policy.reset_state()

    obs_flow, obs_tof = env.reset(seed=seed)

    steps_survived = 0
    total_energy_j = 0.0

    # Масив для нашої математики
    pwm_history = []

    # Трекер просторового покриття (дискретизація R^3 -> Z^3)
    voxel_tracker = VoxelTracker(voxel_size=0.5)

    # Фізичний фільтр-вбивця
    saturation_violations = 0

    # Трекер фізичної телеметрії (Mean Clearance, Kinetic Energy, Crab Flight)
    telemetry_tracker = PhysicsTelemetryTracker(min_speed_hover=0.10, threshold_crab=0.5)

    # Йогоцентричний буфер просторової пам'яті (8 секторів, 74-D простір сенсорів)
    memory_wrapper = EgocentricMemoryWrapper(decay_rate=0.02)
    last_yaw = float(env.att[2]) if hasattr(env, "att") else 0.0
    memory_8 = memory_wrapper.get_memory()

    for step in range(eval_steps):
        # Відстежуємо позицію дрона
        pos = getattr(env, "pos", None)
        if pos is None and hasattr(env, "get_position"):
            pos = env.get_position()
        if pos is not None:
            voxel_tracker.update(pos)

        # Оновлюємо фізичну телеметрію (кліренс ToF та вектор швидкості)
        vel = getattr(env, "vel", None)
        if vel is None and hasattr(env, "get_velocity"):
            vel = env.get_velocity()
        telemetry_tracker.update_step(obs_tof, vel)

        # Оновлюємо егоцентричну пам'ять (8 секторів) за кутом курсу (Yaw)
        current_yaw = float(env.att[2]) if hasattr(env, "att") else 0.0
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
            pwm = np.array([1500.0, 1500.0, 1500.0, 1500.0], dtype=np.float32)

        # 1. Захист від математичного вибуху (Spectral Radius > 1)
        if np.isnan(pwm).any() or np.isinf(pwm).any():
            return 9999.0, {"fatal_failure": True, "failure_reason": "NaN generated"}

        # 2. ФІЗИЧНИЙ ФІЛЬТР: Якщо мотори виходять за [1100, 1900], вони задихаються
        if np.any(pwm < 1100.0) or np.any(pwm > 1900.0):
            saturation_violations += 1

        pwm_history.append(pwm.copy())

        # Енергетична модель (залишаємо як було в колеги)
        u_t = np.clip((pwm[0] - 1000.0) / 1000.0, 0.0, 1.0)
        u_r = np.clip((pwm[1] - 1500.0) / 500.0, -1.0, 1.0)
        u_p = np.clip((pwm[2] - 1500.0) / 500.0, -1.0, 1.0)
        u_y = np.clip((pwm[3] - 1500.0) / 500.0, -1.0, 1.0)
        p_actuators = 15.0 * ((u_t / 0.5) ** 2) + 5.0 * (u_r ** 2 + u_p ** 2 + u_y ** 2)
        total_energy_j += p_actuators * env.dt

        (obs_flow, obs_tof), cost, done, info = env.step(pwm, dt=dt)
        steps_survived += 1

        if done:
            # Якщо краш настав до завершення часу — фіксуємо кінетичну енергію удару
            final_vel = getattr(env, "vel", None)
            if final_vel is None and hasattr(env, "get_velocity"):
                final_vel = env.get_velocity()
            mass_obj = getattr(getattr(env, "physics", None), "total_mass", getattr(env, "mass", 0.130))
            drone_mass = float(mass_obj() if callable(mass_obj) else mass_obj)
            telemetry_tracker.register_crash(final_vel, mass=drone_mass)
            break

    # ── ПІСЛЯ ПОЛЬОТУ: Оцінка Метрик ──
    survival_ratio = steps_survived / eval_steps
    survival_time_s = steps_survived * env.dt
    mean_power_w = total_energy_j / max(1e-4, survival_time_s)

    # Просторове покриття
    coverage_count = voxel_tracker.get_coverage_count()
    coverage_volume = voxel_tracker.get_coverage_volume()

    # Фізична телеметрія
    telemetry_summary = telemetry_tracker.compute_summary()

    # Викликаємо твої нові метрики!
    jitter_pr = calculate_jitter_pr(pwm_history)
    saccades_yaw = calculate_saccades_yaw(pwm_history)

    # Фізичний фільтр / Kill-Switches:
    saturation_ratio = saturation_violations / max(1, steps_survived)
    is_crashed = survival_ratio < 0.20
    is_saturated = saturation_ratio > 0.15
    is_crab = bool(telemetry_summary["is_crab_flight"])

    if is_crashed or is_saturated or is_crab:
        failure_reasons = []
        if is_crashed:
            failure_reasons.append("Crashed")
        if is_saturated:
            failure_reasons.append("PWM Saturated")
        if is_crab:
            failure_reasons.append(
                f"Crab Flight (median {telemetry_summary['forward_ratio_median']:.2f} < 0.50)"
            )

        return 9999.0, {
            "fatal_failure": True,
            "failure_reason": ", ".join(failure_reasons),
            "coverage_count": coverage_count,
            "coverage_volume": float(coverage_volume),
            "coverage_bonus": 0.0,
            "mean_clearance": float(telemetry_summary["mean_clearance"]),
            "impact_energy_j": float(telemetry_summary["impact_energy_j"]),
            "forward_ratio_median": float(telemetry_summary["forward_ratio_median"]),
            "is_crab_flight": is_crab,
            "memory_sectors": memory_8.copy(),
        }

    # ── ФУНКЦІЯ ПРИСТОСОВАНОСТІ (FITNESS) ──
    composite_cost, coverage_bonus = compute_composite_cost(
        survival_ratio=survival_ratio,
        mean_power_w=mean_power_w,
        jitter_pr=jitter_pr,
        saccades_yaw=saccades_yaw,
        coverage_count=coverage_count,
    )

    metrics = {
        "fatal_failure": False,
        "composite_score": composite_cost,
        "survival_time_s": float(survival_time_s),
        "saturation_ratio": float(saturation_ratio),
        "jitter_pr_l2": jitter_pr,
        "saccades_yaw_count": saccades_yaw,
        "coverage_count": coverage_count,
        "coverage_volume": coverage_volume,
        "coverage_bonus": coverage_bonus,
        "mean_clearance": float(telemetry_summary["mean_clearance"]),
        "impact_energy_j": float(telemetry_summary["impact_energy_j"]),
        "forward_ratio_median": float(telemetry_summary["forward_ratio_median"]),
        "is_crab_flight": False,
        "memory_sectors": memory_8.copy(),
    }

    return composite_cost, metrics



