#!/usr/bin/env python3
"""
examples/custom_model_template.py
==================================
Template and Reference Guide for Plugging Custom Models into Chong-Fly SITL.

Platform Specifications:
------------------------
- Base Drone Mass: 130 g (0.130 kg)
- Dimensions: 15 cm x 15 cm Frame
- Payload Capacity: 1.2 kg - 1.5 kg (peak thrust up to 28 N, TWR >= 1.7 at 1.63 kg AUW)
- Sensors:
  1. Downward Laser Rangefinder: Single-beam ToF altimeter along body -Z
  2. 8x8 Depth Matrix: 64 distance zones covering 45° FOV (VL53L5CX)
  3. Optical Sensor: Measures surface flow rates and translational displacement (PMW3901)

How to Plug In Your Own Model:
------------------------------
1. Subclass `BaseSITLModel` from `simulation.sitl_interface`.
2. Implement the `step(self, obs: SITLObservation) -> SITLAction` method.
3. Access sensor readings directly from `obs`:
     - `obs.laser_distance`: Float (meters) from downward laser
     - `obs.depth_8x8`: NumPy array (8, 8) normalized in [0, 1] (45° FOV)
     - `obs.optical_flow`: NumPy array (2,) [FlowX, FlowY] in [-1, 1]
     - `obs.displacement_total`: Cumulative translational displacement [X, Y] (meters)
     - `obs.hover_throttle`: Theoretical collective thrust needed to hover with active payload
4. Return a `SITLAction`:
     - Physical setpoints: `SITLAction.from_setpoints(thrust=0.68, roll=0.0, pitch=0.0, yaw_rate=0.0)`
     - Betaflight RC PWM: `SITLAction.from_pwm(throttle=1680, roll=1500, pitch=1500, yaw=1500)`
"""

from __future__ import annotations

import math
import os
import sys
from typing import Union

import numpy as np

# Ensure project root is in python path
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from simulation.drone_env import DroneDynamicsParams, DroneSimulationEnv
from simulation.sitl_interface import (
    BaseSITLModel,
    SITLAction,
    SITLObservation,
)


class MyCustomDronePolicy(BaseSITLModel):
    """
    Example Custom Neural / Algorithmic Flight Model.
    
    Demonstrates multi-modal sensor fusion:
    - Downward laser for terrain-following altitude control
    - 8x8 depth matrix (45° FOV) for obstacle avoidance
    - Optical displacement tracking to cancel ground drift
    """

    def __init__(self, target_altitude: float = 1.0):
        self.target_altitude = target_altitude
        self.last_laser_alt = target_altitude

    def reset(self) -> None:
        """Called whenever the episode resets or drone respawns."""
        self.last_laser_alt = self.target_altitude

    def step(self, obs: SITLObservation) -> SITLAction:
        """
        Calculates flight action from observation.
        
        Args:
            obs: SITLObservation containing:
                 - obs.laser_distance (m)
                 - obs.depth_8x8 (8, 8) normalized in [0, 1]
                 - obs.optical_flow (2,) normalized in [-1, 1]
                 - obs.displacement_step & obs.displacement_total (m)
                 - obs.hover_throttle (adapted for current 0.0 - 1.5kg payload)
        """
        # 1. Downward Laser Altitude Control (PD loop with physical hover feedforward)
        laser_alt = obs.laser_distance
        alt_error = self.target_altitude - laser_alt
        alt_rate = (laser_alt - self.last_laser_alt) / 0.004
        self.last_laser_alt = laser_alt

        # Base throttle automatically scales for base mass (130g) + payload (e.g. 1.2kg)
        thrust_cmd = obs.hover_throttle + 0.40 * alt_error - 0.15 * alt_rate
        thrust_cmd = float(np.clip(thrust_cmd, 0.05, 0.95))

        # 2. 8x8 Depth Matrix (45° FOV) Lateral Obstacle Evasion
        # Depth values: 1.0 = clear, 0.0 = obstacle in close proximity
        left_clearance = float(np.mean(obs.depth_8x8[:, :4]))
        right_clearance = float(np.mean(obs.depth_8x8[:, 4:]))
        center_clearance = float(np.mean(obs.depth_8x8[2:6, 2:6]))

        # Steer roll towards the clearer sector
        target_roll = (right_clearance - left_clearance) * 0.30

        # Pitch back if frontal path is obstructed
        target_pitch = 0.0
        if center_clearance < 0.70:
            target_pitch = -0.25 * (0.70 - center_clearance)

        # 3. Optical Displacement Damping (oppose lateral and forward drift)
        target_roll -= obs.optical_flow[1] * 0.15
        target_pitch -= obs.optical_flow[0] * 0.15

        # Clamp attitude setpoints to +/- 25 degrees
        max_angle = math.radians(25.0)
        target_roll = float(np.clip(target_roll, -max_angle, max_angle))
        target_pitch = float(np.clip(target_pitch, -max_angle, max_angle))

        return SITLAction.from_setpoints(
            thrust=thrust_cmd,
            roll=target_roll,
            pitch=target_pitch,
            yaw_rate=0.0,
        )


def run_evaluation(
    model: BaseSITLModel,
    duration_s: float = 2.0,
    payload_kg: float = 1.2,
    headless: bool = True,
):
    """Runs a standalone evaluation loop for a custom model."""
    print("=" * 80)
    print("   🛸 RUNNING CHONG-FLY SITL MODEL EVALUATION")
    print("=" * 80)
    print(f"  • Base Drone:      130 g | 15 cm x 15 cm Frame")
    print(f"  • Cargo Payload:   {payload_kg:.2f} kg (Total AUW: {(0.130 + payload_kg)*1000:.0f} g)")
    print(f"  • Flight Duration: {duration_s:.1f} s ({int(duration_s / 0.004)} steps @ 250 Hz)")
    print(f"  • Sensors Active:  Downward Laser, 8x8 Depth (45° FOV), PMW3901 Flow/Disp")
    print("=" * 80 + "\n")

    # Initialize SITL Environment
    params = DroneDynamicsParams(mass=0.130, payload_mass=payload_kg)
    env = DroneSimulationEnv(
        dt=0.004,
        target_altitude=1.0,
        dynamics_params=params,
        headless=headless,
    )
    env.reset(seed=42)
    model.reset()

    total_steps = int(duration_s / env.dt)
    alt_errors = []

    for step in range(total_steps):
        # 1. Read rich SITL sensory observation
        sitl_obs = env.get_sitl_obs()

        # 2. Model inference
        action = model.step(sitl_obs)

        # 3. Step physics and Betaflight avionics
        obs, reward, done, info = env.step(action)
        laser_dist = info["laser_alt"]
        alt_errors.append(abs(laser_dist - 1.0))

        if step % 50 == 0:
            print(f"Step {step:3d} | Down Laser: {laser_dist:4.2f}m | Target: 1.00m | Hover Th: {info['hover_throttle']*100:.1f}% | Power: {info['power_w']:.1f}W")

        if done:
            print("⚠️ Episode terminated prematurely.")
            break

    mean_err = np.mean(alt_errors)
    print("\n" + "=" * 80)
    print(f"✅ Evaluation Complete! Mean Laser Altitude Error: {mean_err * 1000:.1f} mm")
    print("=" * 80 + "\n")
    env.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Chong-Fly Custom Model SITL Runner")
    parser.add_argument("--duration", type=float, default=2.0, help="Flight duration (s)")
    parser.add_argument("--payload", type=float, default=1.2, help="Payload mass in kg [0.0 - 1.5]")
    parser.add_argument("--gui", action="store_true", help="Launch interactive 3D visualizer")
    args = parser.parse_args()

    my_policy = MyCustomDronePolicy(target_altitude=1.0)
    run_evaluation(my_policy, duration_s=args.duration, payload_kg=args.payload, headless=not args.gui)
