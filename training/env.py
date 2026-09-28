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


def simulate_policy_rollout(
    policy: Any,
    env: Optional[DroneSimulationEnv] = None,
    eval_steps: int = 100,
    dt: Optional[float] = None,
    seed: int = 42,
) -> Tuple[float, dict[str, Any]]:
    """
    Runs closed-loop simulation of the policy in DroneSimulationEnv for O(N) Level 2 evaluation.
    
    Returns:
        (composite_behavior_score: float, detailed_metrics: dict)
    """
    if env is None:
        step_dt = 0.02 if getattr(policy, "default_dt", 0.004) == 0.02 else 0.004
        env = DroneSimulationEnv(dt=step_dt)

    if hasattr(policy, "reset_state"):
        policy.reset_state()

    obs_flow, obs_tof = env.reset(seed=seed)
    total_cost = 0.0
    total_energy_j = 0.0
    steps_survived = 0
    jitters = []
    alt_errors = []
    tilt_errors = []

    for step in range(eval_steps):
        # Policy step (O(N))
        if hasattr(policy, "step_np"):
            pwm = policy.step_np(obs_flow, obs_tof, dt=dt)
        else:
            pwm = np.array([1500.0, 1500.0, 1500.0, 1500.0], dtype=np.float32)

        # Check for numerical NaN/Inf
        if np.isnan(pwm).any() or np.isinf(pwm).any():
            return 999.0, {
                "fatal_failure": True,
                "failure_reason": "Policy generated NaN or Inf PWM actuators",
                "survival_ratio": steps_survived / eval_steps,
                "survival_time_s": steps_survived * env.dt,
                "energy_cost_j": 999.0,
            }

        # Energy consumption calculation:
        # Hover baseline is ~15W for a micro-quadrotor (at normalized throttle u_t=0.5)
        u_t = np.clip((pwm[0] - 1000.0) / 1000.0, 0.0, 1.0)
        u_r = np.clip((pwm[1] - 1500.0) / 500.0, -1.0, 1.0)
        u_p = np.clip((pwm[2] - 1500.0) / 500.0, -1.0, 1.0)
        u_y = np.clip((pwm[3] - 1500.0) / 500.0, -1.0, 1.0)
        p_actuators = 15.0 * ((u_t / 0.5) ** 2) + 5.0 * (u_r**2 + u_p**2 + u_y**2)
        total_energy_j += p_actuators * env.dt

        (obs_flow, obs_tof), cost, done, info = env.step(pwm, dt=dt)
        total_cost += cost
        steps_survived += 1
        jitters.append(info["jitter"])
        alt_errors.append(info["alt_err"])
        tilt_errors.append(info["tilt_err"])

        if done:
            break

    survival_ratio = steps_survived / eval_steps
    survival_time_s = steps_survived * env.dt
    mean_cost = total_cost / max(1, steps_survived)
    mean_power_w = total_energy_j / max(1e-4, survival_time_s)
    
    # Crash penalty if drone did not survive full evaluation
    crash_penalty = (1.0 - survival_ratio) * 10.0
    composite_score = float(mean_cost + crash_penalty)

    metrics = {
        "fatal_failure": survival_ratio < 0.20,  # immediate fatal crash
        "failure_reason": "Drone crashed or tumbled almost immediately" if survival_ratio < 0.20 else "",
        "composite_score": composite_score,
        "survival_ratio": float(survival_ratio),
        "survival_time_s": float(survival_time_s),
        "energy_cost_j": float(total_energy_j),
        "mean_power_w": float(mean_power_w),
        "steps_survived": int(steps_survived),
        "mean_altitude_error": float(np.mean(alt_errors)) if alt_errors else 0.0,
        "mean_tilt_error": float(np.mean(tilt_errors)) if tilt_errors else 0.0,
        "mean_pwm_jitter": float(np.mean(jitters)) if jitters else 0.0,
        "final_obstacle_dist": float(env.obstacle_dist),
    }

    return composite_score, metrics
