"""
generator/generate_reflex_dataset.py
====================================
Offline Reflex Dataset Generator for Behavioral Cloning (Chong-Fly).

Generates expert flight demonstrations teaching collision avoidance reflexes:
- When front clearance < 0.8 m: brake (Pitch = 1300) and sharp turn (Yaw = 1900 or 1100).
- When front clearance >= 0.8 m: advance (Pitch = 1600, Yaw = 1500).
- State Latching (Hysteresis): prevents high-frequency chattering mid-turn.
- Stochasticity: 50% left / 50% right turn probability, plus Gaussian actuator noise (+-20 PWM).
- Sequential Format: outputs (X: [N, T, 74], Y: [N, T, 4]) sequences for recurrent CfC BPTT.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from simulation.memory import EgocentricMemoryWrapper, SECTOR_FRONT
#from optimizer.evaluate import DroneSimulationEnv
from simulation.drone_env import DroneSimulationEnv
from configs.flight_config import (
    SENSOR_DIM,
    FLOW_DIM,
    TOF_DIM,
    TOF_ROWS,
    TOF_COLS,
    MEMORY_DIM,
    N_CONTROLS,
    PWM_MIN,
    PWM_MAX,
    PWM_HOVER,
    PWM_LEVEL_ROLL,
    PWM_CRUISE_PITCH,
    PWM_NEUTRAL_YAW,
    DEFAULT_DT,
    APF_DISTANCE_THRESHOLD_M,
    APF_MAX_SENSOR_RANGE_M,
    APF_K_REPULSIVE,
    APF_NOISE_STD_PWM,
    APF_MAX_PITCH_BRAKE_PWM,
    APF_REPULSION_SCALE,
    APF_YAW_GAIN,
    APF_MIN_DIST_CLAMP,
    MEMORY_DECAY_RATE,
    OBSTACLE_INITIAL_DIST_MIN_M,
    OBSTACLE_INITIAL_DIST_MAX_M,
    OBSTACLE_RESPAWN_DIST_MIN_M,
    OBSTACLE_RESPAWN_DIST_MAX_M,
    TURN_CLEARANCE_DIST_M,
    TURN_MAX_STEPS,
    OBSTACLE_CLEARANCE_MIN_M,
    SENSORS,
    TOF_MAX_RANGE_M,
)


class ExpertReflexPolicy:
    """
    Artificial Potential Fields (APF) Expert flight controller.
    Provides a continuous, differentiable target for Behavioral Cloning using Khatib's formula.
    """

    def __init__(
        self,
        distance_threshold_m: float = APF_DISTANCE_THRESHOLD_M,
        max_sensor_range_m: float = APF_MAX_SENSOR_RANGE_M,
        noise_std_pwm: float = APF_NOISE_STD_PWM,
        seed: Optional[int] = None,
        k_repulsive: float = APF_K_REPULSIVE,
        max_pitch_brake_pwm: float = APF_MAX_PITCH_BRAKE_PWM,
        repulsion_scale: float = APF_REPULSION_SCALE,
        yaw_gain: float = APF_YAW_GAIN,
        min_dist_clamp: float = APF_MIN_DIST_CLAMP,
        pwm_hover: float = PWM_HOVER,
        pwm_level_roll: float = PWM_LEVEL_ROLL,
        pwm_cruise_pitch: float = PWM_CRUISE_PITCH,
        pwm_neutral_yaw: float = PWM_NEUTRAL_YAW,
        pwm_min: float = PWM_MIN,
        pwm_max: float = PWM_MAX,
    ):
        self.distance_threshold_m = float(distance_threshold_m)
        self.max_sensor_range_m = float(max_sensor_range_m)
        # Normalized distance threshold (d_0) in [0, 1]
        self.d0 = self.distance_threshold_m / self.max_sensor_range_m
        self.noise_std_pwm = float(noise_std_pwm)
        self.rng = np.random.default_rng(seed)
        self.k_repulsive = float(k_repulsive)
        self.max_pitch_brake_pwm = float(max_pitch_brake_pwm)
        self.repulsion_scale = float(repulsion_scale)
        self.yaw_gain = float(yaw_gain)
        self.min_dist_clamp = float(min_dist_clamp)
        self.pwm_hover = float(pwm_hover)
        self.pwm_level_roll = float(pwm_level_roll)
        self.pwm_cruise_pitch = float(pwm_cruise_pitch)
        self.pwm_neutral_yaw = float(pwm_neutral_yaw)
        self.pwm_min = float(pwm_min)
        self.pwm_max = float(pwm_max)

        self.latched_turn = None  # Retained for compatibility with generator loop

        # Precompute normals for the 8x8 ToF grid
        # Columns 0..7 map to horizontal normal nx from -1.0 (left) to 1.0 (right)
        cols = np.linspace(-1.0, 1.0, TOF_COLS)
        self.nx = np.tile(cols, (TOF_ROWS, 1)).flatten()  # 64-D array of x-normals

    def reset(self) -> None:
        self.latched_turn = None

    def step(self, obs_74: np.ndarray) -> np.ndarray:
        """
        Produces expert 4-D PWM command [throttle, roll, pitch, yaw] based on APF.
        """
        obs = np.asarray(obs_74, dtype=np.float32).ravel()

        throttle = self.pwm_hover
        roll = self.pwm_level_roll
        pitch_base = self.pwm_cruise_pitch
        yaw_base = self.pwm_neutral_yaw

        # ToF Grid (first 64 elements of the depth part)
        tof = obs[FLOW_DIM:(FLOW_DIM + TOF_DIM)] if obs.size >= (FLOW_DIM + TOF_DIM) else np.ones(TOF_DIM, dtype=np.float32)

        # Protection against division by zero
        d = np.maximum(tof, self.min_dist_clamp)

        active_mask = d < self.d0

        if np.any(active_mask):
            d_active = d[active_mask]
            
            # Khatib's Formula: F_i = k * (1/d_i - 1/d0) / (d_i^2)
            force_magnitudes = self.k_repulsive * (1.0 / d_active - 1.0 / self.d0) * (1.0 / (d_active ** 2))
            
            # Use mean instead of sum to prevent 64x blowup
            mean_repulsion = float(np.mean(force_magnitudes))
            
            # Directional force for Yaw.
            # If obstacle is at nx (e.g., -1 left), it pushes us right (positive force).
            force_x = float(np.mean(force_magnitudes * (-self.nx[active_mask])))
            
            # Continuous smooth braking using rational saturation to prevent violent jumps near d -> 0
            braking = self.max_pitch_brake_pwm * (mean_repulsion / (self.repulsion_scale + mean_repulsion))
            pitch = pitch_base - braking
            
            # Yaw turn
            yaw = yaw_base + force_x * self.yaw_gain
        else:
            pitch = pitch_base
            yaw = yaw_base

        pwm = np.array([throttle, roll, pitch, yaw], dtype=np.float32)

        if self.noise_std_pwm > 0.0:
            noise = self.rng.normal(0.0, self.noise_std_pwm, size=N_CONTROLS).astype(np.float32)
            pwm += noise

        return np.clip(pwm, self.pwm_min, self.pwm_max)


def generate_reflex_dataset(
    num_episodes: int = 50,
    seq_len: int = 200,
    dt: float = DEFAULT_DT,
    distance_threshold_m: float = APF_DISTANCE_THRESHOLD_M,
    noise_std_pwm: float = APF_NOISE_STD_PWM,
    seed: int = 42,
    output_path: Optional[str] = None,
    pwm_hover: float = PWM_HOVER,
    pwm_level_roll: float = PWM_LEVEL_ROLL,
    pwm_cruise_pitch: float = PWM_CRUISE_PITCH,
    pwm_neutral_yaw: float = PWM_NEUTRAL_YAW,
    pwm_min: float = PWM_MIN,
    pwm_max: float = PWM_MAX,
) -> Dict[str, Any]:
    """
    Generates demonstration sequences of (X: 74-D, Y: 4-D PWM) using ExpertReflexPolicy.

    Returns:
        dict with keys:
          'X': torch.Tensor of shape (num_episodes, seq_len, 74)
          'Y': torch.Tensor of shape (num_episodes, seq_len, 4)
          'metadata': dict of training parameters
    """
    rng = np.random.default_rng(seed)
    expert = ExpertReflexPolicy(
        distance_threshold_m=distance_threshold_m,
        noise_std_pwm=noise_std_pwm,
        seed=seed,
        pwm_hover=pwm_hover,
        pwm_level_roll=pwm_level_roll,
        pwm_cruise_pitch=pwm_cruise_pitch,
        pwm_neutral_yaw=pwm_neutral_yaw,
        pwm_min=pwm_min,
        pwm_max=pwm_max,
    )

    all_x: List[np.ndarray] = []
    all_y: List[np.ndarray] = []

    for ep in range(num_episodes):
        ep_seed = int(rng.integers(0, 1_000_000))
        env = DroneSimulationEnv(dt=dt, engine="standalone", headless=True)

        obs = env.reset(seed=ep_seed)
        obs_flow = obs[:FLOW_DIM]
        
        obstacle_dist = float(rng.uniform(OBSTACLE_INITIAL_DIST_MIN_M, OBSTACLE_INITIAL_DIST_MAX_M))
        
        def get_yaw() -> float:
            q = env.physics.quat
            return float(math.atan2(2.0 * (q[0] * q[3] + q[1] * q[2]), 1.0 - 2.0 * (q[2]**2 + q[3]**2)))
            
        def mock_tof(dist: float) -> np.ndarray:
            grid = np.ones((TOF_ROWS, TOF_COLS), dtype=np.float32)
            norm_dist = np.clip(dist / TOF_MAX_RANGE_M, 0.0, 1.0)
            grid[SENSORS.tof_center_row_start:SENSORS.tof_center_row_end, SENSORS.tof_center_col_start:SENSORS.tof_center_col_end] = norm_dist
            return grid.ravel()

        obs_tof = mock_tof(obstacle_dist)

        memory_wrapper = EgocentricMemoryWrapper(decay_rate=MEMORY_DECAY_RATE)
        expert.reset()

        last_yaw = get_yaw()
        turn_steps = 0
        ep_x: List[np.ndarray] = []
        ep_y: List[np.ndarray] = []

        for step in range(seq_len):
            # 1. Update Egocentric Ring Buffer
            current_yaw = get_yaw()
            delta_yaw = current_yaw - last_yaw
            delta_yaw = (delta_yaw + math.pi) % (2.0 * math.pi) - math.pi
            last_yaw = current_yaw

            memory_8 = memory_wrapper.update(obs_tof, delta_yaw_rad=delta_yaw)

            # 2. Build 74-D observation vector
            obs_74 = np.concatenate([
                np.asarray(obs_flow, dtype=np.float32).ravel()[:FLOW_DIM],
                np.asarray(obs_tof, dtype=np.float32).ravel()[:TOF_DIM],
                np.asarray(memory_8, dtype=np.float32).ravel()[:MEMORY_DIM],
            ])

            # 3. Query expert action
            pwm_expert = expert.step(obs_74)

            ep_x.append(obs_74)
            ep_y.append(pwm_expert)

            # 4. Advance physics simulation
            obs_66, cost, done, info = env.step(pwm_expert)
            
            vel = env.physics.vel
            v_forward = float(vel[0] * math.cos(current_yaw) + vel[1] * math.sin(current_yaw))
            obstacle_dist = max(0.0, obstacle_dist - v_forward * dt)
            
            obs_flow = obs_66[:FLOW_DIM]
            obs_tof = mock_tof(obstacle_dist)

            # Continuous reflex environment: complete turn maneuver and respawn next obstacle
            if expert.latched_turn is not None:
                turn_steps += 1
                if turn_steps >= TURN_MAX_STEPS or obstacle_dist <= TURN_CLEARANCE_DIST_M:
                    expert.latched_turn = None
                    turn_steps = 0
                    obstacle_dist = float(rng.uniform(OBSTACLE_RESPAWN_DIST_MIN_M, OBSTACLE_RESPAWN_DIST_MAX_M))
                    obs_tof = np.ones(TOF_DIM, dtype=np.float32)
            else:
                turn_steps = 0
                if obstacle_dist <= OBSTACLE_CLEARANCE_MIN_M:
                    obstacle_dist = float(rng.uniform(OBSTACLE_RESPAWN_DIST_MIN_M, OBSTACLE_RESPAWN_DIST_MAX_M))
                    obs_tof = np.ones(TOF_DIM, dtype=np.float32)

            if done:
                # If crashed or tumbled, reset state to maintain full sequence length
                obs = env.reset(seed=ep_seed + step + 1)
                obs_flow = obs[:FLOW_DIM]
                obstacle_dist = float(rng.uniform(OBSTACLE_INITIAL_DIST_MIN_M, OBSTACLE_INITIAL_DIST_MAX_M))
                obs_tof = mock_tof(obstacle_dist)
                last_yaw = get_yaw()
                expert.reset()
                turn_steps = 0

        all_x.append(np.array(ep_x, dtype=np.float32))
        all_y.append(np.array(ep_y, dtype=np.float32))

    X_tensor = torch.from_numpy(np.array(all_x, dtype=np.float32))  # [N, T, 74]
    Y_tensor = torch.from_numpy(np.array(all_y, dtype=np.float32))  # [N, T, 4]

    dataset = {
        "X": X_tensor,
        "Y": Y_tensor,
        "metadata": {
            "num_episodes": num_episodes,
            "seq_len": seq_len,
            "dt": dt,
            "sensor_dim": SENSOR_DIM,
            "action_dim": N_CONTROLS,
            "distance_threshold_m": distance_threshold_m,
            "seed": seed,
        },
    }

    if output_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        torch.save(dataset, output_path)

    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Reflex Dataset for Chong-Fly")
    parser.add_argument("--episodes", type=int, default=60, help="Number of flight episodes")
    parser.add_argument("--seq_len", type=int, default=250, help="Steps per episode")
    parser.add_argument("--output", type=str, default="data/reflex_dataset.pt", help="Output path")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    args = parser.parse_args()

    out_file = os.path.join(_ROOT, args.output) if not os.path.isabs(args.output) else args.output
    print(f"Generating reflex dataset: {args.episodes} episodes x {args.seq_len} steps -> {out_file}")
    data = generate_reflex_dataset(
        num_episodes=args.episodes,
        seq_len=args.seq_len,
        seed=args.seed,
        output_path=out_file,
    )
    print(f"Saved dataset: X shape = {data['X'].shape}, Y shape = {data['Y'].shape}")
