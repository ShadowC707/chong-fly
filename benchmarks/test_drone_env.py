"""
simulation/test_drone_env.py
============================
Validation suite for DroneSimulationEnv, BetaflightCascadedPID, ToFRaycaster,
and PMW3901FlowSensor.
"""

import os
import sys
import math
import unittest
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from simulation.avionics_filter import BetaflightCascadedPID, PIDConstants, PT1Filter

from simulation.pmw3901_emulator import PMW3901FlowSensor
from simulation.drone_env import (
    DroneSimulationEnv,
    QuadcopterDynamics,
    DroneDynamicsParams,
    ToFRaycaster,
    RoomBoundaries,
    CylinderObstacle,
    BoxObstacle,
)
from simulation.policy import ChongFlyMSPPolicy


class TestBetaflightPID(unittest.TestCase):
    """Test Betaflight cascaded PID controller and quad-X motor mixer."""

    def setUp(self):
        self.pid = BetaflightCascadedPID(dt=0.004, angle_p_gain=5.0)

    def test_pt1_filter(self):
        """Test PT1 1st-order low-pass filter frequency response."""
        lpf = PT1Filter(cutoff_hz=50.0, dt=0.004)
        # Step response
        val = 0.0
        for _ in range(50):
            val = lpf.update(1.0)
        self.assertAlmostEqual(val, 1.0, delta=0.05)

    def test_outer_loop_angle_mode(self):
        """Test that angle error creates proportional target rate."""
        target_pitch = math.radians(10.0)
        target_roll = 0.0
        measured_euler = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        measured_omega = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        motors, debug = self.pid.compute_motor_commands(
            target_thrust=0.5,
            target_pitch=target_pitch,
            target_roll=target_roll,
            target_yaw_rate=0.0,
            measured_euler=measured_euler,
            measured_omega=measured_omega,
        )
        # Target pitch rate should be positive (P_angle * 10 deg)
        rate_target = debug["rate_target"]
        self.assertGreater(rate_target[1], 0.0) # q_target > 0
        self.assertAlmostEqual(rate_target[0], 0.0, places=4) # p_target == 0

    def test_rate_pid_anti_windup(self):
        """Test anti-windup clamping under persistent rate error."""
        # Force constant rate error
        for _ in range(500):
            self.pid.compute_motor_commands(
                target_thrust=0.5,
                target_pitch=0.0,
                target_roll=math.radians(20.0),
                target_yaw_rate=0.0,
                measured_euler=np.zeros(3, dtype=np.float32),
                measured_omega=np.zeros(3, dtype=np.float32),
            )
        self.assertLessEqual(abs(self.pid.pid_roll.iterm), self.pid.pid_roll.constants.iterm_limit)

    def test_quad_x_motor_mixer_bounds(self):
        """Test that motor commands remain clipped within [0.0, 1.0]."""
        motors, _ = self.pid.compute_motor_commands(
            target_thrust=1.0,
            target_pitch=math.radians(35.0),
            target_roll=math.radians(35.0),
            target_yaw_rate=math.radians(200.0),
            measured_euler=np.zeros(3, dtype=np.float32),
            measured_omega=np.zeros(3, dtype=np.float32),
        )
        self.assertEqual(motors.shape, (4,))
        self.assertTrue(np.all(motors >= 0.0))
        self.assertTrue(np.all(motors <= 1.0))


class TestQuadcopterDynamics(unittest.TestCase):
    """Test 6-DOF rigid body equations of motion."""

    def setUp(self):
        self.params = DroneDynamicsParams()
        self.dyn = QuadcopterDynamics(self.params, dt=0.004)
        self.dyn.reset(np.array([0.0, 0.0, 1.0]))

    def test_hover_equilibrium(self):
        """Hover throttle should maintain roughly constant altitude."""
        hover_throttle = np.full(4, self.dyn.hover_throttle, dtype=np.float32)
        initial_z = self.dyn.pos[2]
        for _ in range(100): # 0.4 seconds
            self.dyn.step(hover_throttle)
        # Altitude should remain within 5 cm of initial
        self.assertAlmostEqual(self.dyn.pos[2], initial_z, delta=0.08)

    def test_roll_torque_generation(self):
        """Differential left/right throttle should produce roll acceleration."""
        # Motor layout: m1, m2 (right), m3, m4 (left)
        # Increasing left motors (m3, m4) rolls drone to the right (positive roll rate)
        cmd = np.array([0.4, 0.4, 0.7, 0.7], dtype=np.float32)
        for _ in range(20):
            self.dyn.step(cmd)
        self.assertGreater(self.dyn.omega[0], 0.0) # p > 0

    def test_quaternion_euler_roundtrip(self):
        """Test conversion between Euler angles and quaternions."""
        roll, pitch, yaw = 0.25, -0.15, 0.80
        q = QuadcopterDynamics.euler_to_quaternion(roll, pitch, yaw)
        e = QuadcopterDynamics.quaternion_to_euler(q)
        self.assertAlmostEqual(roll, float(e[0]), delta=1e-4)
        self.assertAlmostEqual(pitch, float(e[1]), delta=1e-4)
        self.assertAlmostEqual(yaw, float(e[2]), delta=1e-4)


class TestToFRaycaster(unittest.TestCase):
    """Test 8x8 ToF distance sensor raycaster."""

    def setUp(self):
        self.tof = ToFRaycaster(rows=8, cols=8, fov_h_deg=45.0, fov_v_deg=45.0, max_range=3.5)
        self.room = RoomBoundaries(x_min=-3.0, x_max=3.0, y_min=-3.0, y_max=3.0, z_min=0.0, z_max=3.0)

    def test_ray_directions_count(self):
        """Check 64 directional rays generated."""
        self.assertEqual(self.tof.body_ray_dirs.shape, (64, 3))
        # Unit norm
        norms = np.linalg.norm(self.tof.body_ray_dirs, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_wall_detection_straight_ahead(self):
        """Drone at (0, 0, 1.0) facing +X wall at x=3.0 should measure distance 3.0 m."""
        pos = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        rot = np.eye(3, dtype=np.float32) # facing +X
        distances = self.tof.cast_rays(pos, rot, self.room)
        self.assertEqual(distances.shape, (64,))
        # All normalized distances must be in [0, 1]
        self.assertTrue(np.all(distances >= 0.0))
        self.assertTrue(np.all(distances <= 1.0))
        # Central forward rays should be ~ 3.0 / 3.5 ≈ 0.857
        central_ray = distances[27] # near center
        self.assertAlmostEqual(central_ray, 3.0 / 3.5, delta=0.1)

    def test_cylinder_obstacle_detection(self):
        """Cylinder placed directly ahead at x=1.0 should shorten central rays."""
        pos = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        rot = np.eye(3, dtype=np.float32)
        cyl = [CylinderObstacle(center_x=1.0, center_y=0.0, radius=0.2, height=3.0)]

        dist_clear = self.tof.cast_rays(pos, rot, self.room)
        dist_cyl = self.tof.cast_rays(pos, rot, self.room, cylinders=cyl)

        # Central rays should be shorter due to obstacle
        self.assertLess(np.min(dist_cyl), np.min(dist_clear))
        # Distance to cylinder front is 1.0 - 0.2 = 0.8 m -> normalized 0.8 / 3.5 ≈ 0.228
        self.assertAlmostEqual(np.min(dist_cyl), 0.8 / 3.5, delta=0.05)


class TestPMW3901FlowSensor(unittest.TestCase):
    """Test optical flow sensor emulation."""

    def setUp(self):
        self.sensor = PMW3901FlowSensor(min_altitude=0.08, max_altitude=3.5, derotate_with_gyro=True, noise_std=0.0)

    def test_stationary_flow(self):
        """Zero velocity should yield zero optical flow."""
        v = np.zeros(3, dtype=np.float32)
        R = np.eye(3, dtype=np.float32)
        flow = self.sensor.compute_flow(v, R, altitude_above_surface=1.0)
        np.testing.assert_allclose(flow, [0.0, 0.0], atol=1e-5)

    def test_forward_velocity_flow(self):
        """Forward velocity (+vx) at 1m altitude produces positive FlowX."""
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        R = np.eye(3, dtype=np.float32)
        flow = self.sensor.compute_flow(v, R, altitude_above_surface=1.0)
        self.assertGreater(flow[0], 0.0)
        self.assertAlmostEqual(flow[1], 0.0, delta=1e-4)

    def test_altitude_scaling(self):
        """Higher altitude should reduce optical flow magnitude for same linear velocity."""
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        R = np.eye(3, dtype=np.float32)
        flow_low = self.sensor.compute_flow(v, R, altitude_above_surface=0.5)
        flow_high = self.sensor.compute_flow(v, R, altitude_above_surface=2.0)
        self.assertGreater(abs(flow_low[0]), abs(flow_high[0]))


class TestDroneSimulationEnv(unittest.TestCase):
    """Test complete DroneSimulationEnv integration."""

    def setUp(self):
        self.env = DroneSimulationEnv(dt=0.004, target_altitude=1.0)

    def test_reset_shape(self):
        """Reset should return 66-dimensional observation."""
        obs = self.env.reset()
        self.assertEqual(obs.shape, (66,))
        # Flow in [-1, 1], ToF in [0, 1]
        self.assertTrue(np.all(obs[0:2] >= -1.0) and np.all(obs[0:2] <= 1.0))
        self.assertTrue(np.all(obs[2:66] >= 0.0) and np.all(obs[2:66] <= 1.0))

    def test_chong_fly_obs_helper(self):
        """Check get_chong_fly_obs() returns (2,) and (64,) arrays."""
        self.env.reset()
        flow_xy, tof_8x8 = self.env.get_chong_fly_obs()
        self.assertEqual(flow_xy.shape, (2,))
        self.assertEqual(tof_8x8.shape, (64,))

    def test_step_with_setpoints(self):
        """Test environment step with direct flight setpoints."""
        self.env.reset()
        # [thrust, roll_cmd, pitch_cmd, yaw_rate_cmd]
        action = np.array([self.env.physics.hover_throttle, 0.0, 0.0, 0.0], dtype=np.float32)
        obs, reward, done, info = self.env.step(action, action_type="setpoints")

        self.assertEqual(obs.shape, (66,))
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIn("position", info)
        self.assertIn("motor_commands", info)

    def test_step_with_pwm(self):
        """Test environment step with Chong-Fly [1000, 2000] µs RC PWM commands."""
        self.env.reset()
        pwm_hover = np.array([1500.0, 1500.0, 1500.0, 1500.0], dtype=np.float32)
        obs, reward, done, info = self.env.step(pwm_hover, action_type="pwm")

        self.assertEqual(obs.shape, (66,))
        self.assertFalse(done)

    def test_closed_loop_policy_integration(self):
        """Test closed-loop flight with ChongFlyMSPPolicy."""
        policy = ChongFlyMSPPolicy.from_meta(
            "data/reduced_models/meta_spectral_k32.json",
            mode="fixed",
            dt=0.004,
        )
        policy.reset_state()
        self.env.reset(seed=42)

        for _ in range(50): # 50 steps = 0.20 seconds of 250 Hz flight
            flow_xy, tof_8x8 = self.env.get_chong_fly_obs()
            pwm = policy.step_np(flow_xy, tof_8x8)
            obs, reward, done, info = self.env.step(pwm)
            if done:
                break

        # Drone should have survived the initial 50 steps
        self.assertFalse(done)
        self.assertGreater(info["position"][2], 0.2)


if __name__ == "__main__":
    unittest.main()
