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
from optimizer.evaluate import DroneSimulationEnv


class ExpertReflexPolicy:
    """
    Rule-based expert flight controller with state latching and stochastic evasion.
    """

    def __init__(
        self,
        distance_threshold_m: float = 0.8,
        max_sensor_range_m: float = 3.0,
        noise_std_pwm: float = 15.0,
        seed: Optional[int] = None,
    ):
        self.distance_threshold_m = float(distance_threshold_m)
        self.max_sensor_range_m = float(max_sensor_range_m)
        # Normalized distance threshold in [0, 1]
        self.threshold_norm = self.distance_threshold_m / self.max_sensor_range_m
        self.noise_std_pwm = float(noise_std_pwm)
        self.rng = np.random.default_rng(seed)

        # State Latching: -1 = turning left (Yaw 1100), +1 = turning right (Yaw 1900), None = straight
        self.latched_turn: Optional[int] = None

    def reset(self) -> None:
        """Reset internal latching state at episode start."""
        self.latched_turn = None

    def get_front_clearance(self, obs_74: np.ndarray) -> float:
        """
        Extracts minimum frontal distance from Sector 0 of Egocentric Memory
        and central ToF pixels.
        obs_74 layout:
          [0:2]   FlowX, FlowY
          [2:66]  ToF 8x8 pixels
          [66:74] Egocentric Memory 8 sectors (idx 66 is Sector 0: Front)
        """
        obs = np.asarray(obs_74, dtype=np.float32).ravel()
        if obs.size >= 74:
            front_mem = float(obs[66])
            # Central 4x4 ToF grid (indices in 64-D array)
            tof_grid = obs[2:66].reshape((8, 8))
            center_tof = float(np.min(tof_grid[2:6, 2:6]))
            return min(front_mem, center_tof)
        elif obs.size >= 66:
            # Fallback to ToF central pixels if memory not appended
            tof_grid = obs[2:66].reshape((8, 8))
            return float(np.min(tof_grid[2:6, 2:6]))
        return 1.0

    def step(self, obs_74: np.ndarray) -> np.ndarray:
        """
        Produces expert 4-D PWM command [throttle, roll, pitch, yaw] based on obs_74.
        """
        front_dist = self.get_front_clearance(obs_74)

        throttle = 1500.0  # Hover baseline
        roll = 1500.0      # Level roll

        if front_dist < self.threshold_norm:
            # 1. OBSTACLE DETECTED: Active braking
            pitch = 1300.0

            # State Latching: pick and lock turn direction if not already latching
            if self.latched_turn is None:
                self.latched_turn = 1 if self.rng.random() < 0.5 else -1

            yaw = 1900.0 if self.latched_turn == 1 else 1100.0
        else:
            # 2. FRONT CLEAR: Release latch and cruise forward
            self.latched_turn = None
            pitch = 1600.0
            yaw = 1500.0

        pwm = np.array([throttle, roll, pitch, yaw], dtype=np.float32)

        # Inject light Gaussian noise to prevent degenerate cloning
        if self.noise_std_pwm > 0.0:
            noise = self.rng.normal(0.0, self.noise_std_pwm, size=4).astype(np.float32)
            pwm += noise

        return np.clip(pwm, 1000.0, 2000.0)


def generate_reflex_dataset(
    num_episodes: int = 50,
    seq_len: int = 200,
    dt: float = 0.004,
    distance_threshold_m: float = 0.8,
    noise_std_pwm: float = 15.0,
    seed: int = 42,
    output_path: Optional[str] = None,
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
    )

    all_x: List[np.ndarray] = []
    all_y: List[np.ndarray] = []

    for ep in range(num_episodes):
        ep_seed = int(rng.integers(0, 1_000_000))
        env = DroneSimulationEnv(dt=dt)
        obs_flow, obs_tof = env.reset(seed=ep_seed)

        memory_wrapper = EgocentricMemoryWrapper(decay_rate=0.02)
        expert.reset()

        last_yaw = float(env.att[2])
        ep_x: List[np.ndarray] = []
        ep_y: List[np.ndarray] = []

        for step in range(seq_len):
            # 1. Update Egocentric Ring Buffer
            current_yaw = float(env.att[2])
            delta_yaw = current_yaw - last_yaw
            delta_yaw = (delta_yaw + math.pi) % (2.0 * math.pi) - math.pi
            last_yaw = current_yaw

            memory_8 = memory_wrapper.update(obs_tof, delta_yaw_rad=delta_yaw)

            # 2. Build 74-D observation vector
            obs_74 = np.concatenate([
                np.asarray(obs_flow, dtype=np.float32).ravel()[:2],
                np.asarray(obs_tof, dtype=np.float32).ravel()[:64],
                np.asarray(memory_8, dtype=np.float32).ravel()[:8],
            ])

            # 3. Query expert action
            pwm_expert = expert.step(obs_74)

            ep_x.append(obs_74)
            ep_y.append(pwm_expert)

            # 4. Advance physics simulation
            (obs_flow, obs_tof), cost, done, info = env.step(pwm_expert, dt=dt)

            # Continuous reflex environment: respawn obstacle ahead if avoided or cleared
            if env.obstacle_dist <= 0.20 or (expert.latched_turn is None and env.obstacle_dist < 0.8):
                # Reposition obstacle ahead for another encounter
                env.obstacle_dist = float(rng.uniform(1.8, 2.8))

            if done:
                # If crashed or tumbled, reset state to maintain full sequence length
                obs_flow, obs_tof = env.reset(seed=ep_seed + step + 1)
                last_yaw = float(env.att[2])
                expert.reset()

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
            "sensor_dim": 74,
            "action_dim": 4,
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
