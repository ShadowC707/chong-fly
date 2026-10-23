"""Observable signs in the FLU world and right-positive RC interface."""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from simulation.drone_env import DroneSimulationEnv, ToFRaycaster, RoomBoundaries, BoxObstacle
from simulation.drone_interface import AutonomousLaserNavigatorModel
from simulation.pmw3901_emulator import PMW3901FlowSensor
from simulation.memory import EgocentricMemoryWrapper, SECTOR_LEFT


def level_env():
    env = DroneSimulationEnv(engine='standalone', headless=True)
    env.boxes, env.cylinders = [], []
    env.reset(initial_pos=np.array([0., 0., 1.]), seed=3)
    env.physics.quat[:] = [1, 0, 0, 0]
    env.physics.vel[:] = 0
    return env


@pytest.mark.parametrize('channel,axis,expected_sign', [(1, 1, -1), (2, 0, 1), (3, 2, -1)])
def test_positive_rc_roll_pitch_yaw_moves_right_forward_turns_right(channel, axis, expected_sign):
    env = level_env()
    pwm = np.array([1000 + 1000*env.physics.hover_throttle, 1500, 1500, 1500])
    pwm[channel] += 100
    for _ in range(80):
        env.step(pwm, action_type='pwm')
    motion = env.physics.omega if channel == 3 else env.physics.vel
    assert motion[axis] * expected_sign > .01


def test_native_positive_yaw_setpoint_remains_left_positive():
    env = level_env()
    for _ in range(80):
        env.step([env.physics.hover_throttle, 0, 0, .4], action_type='setpoints')
    assert env.physics.omega[2] > .1


def test_tof_columns_are_left_to_right_and_rows_top_to_bottom():
    rays = ToFRaycaster().body_ray_dirs.reshape(8, 8, 3)
    assert np.all(rays[:, 0, 1] > 0) and np.all(rays[:, -1, 1] < 0)
    assert np.all(rays[0, :, 2] > 0) and np.all(rays[-1, :, 2] < 0)


def test_left_obstacle_hits_left_image_half_and_mirror_hits_right():
    caster = ToFRaycaster()
    room = RoomBoundaries(-10, 10, -10, 10, -10, 10)
    left = caster.cast_rays(np.array([0., 0., 1.]), np.eye(3), room,
                           boxes=[BoxObstacle(.5, .6, .03, .4, .3, 1.7)]).reshape(8, 8)
    right = caster.cast_rays(np.array([0., 0., 1.]), np.eye(3), room,
                            boxes=[BoxObstacle(.5, .6, -.4, -.03, .3, 1.7)]).reshape(8, 8)
    assert left[:, :4].min() < .3 and np.all(left[:, 4:] == 1)
    np.testing.assert_allclose(left[:, ::-1], right)


def test_flow_tracks_body_forward_and_left_at_rotated_heading():
    sensor = PMW3901FlowSensor(noise_std=0)
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.]])
    flow = sensor.compute_flow(R @ np.array([1., .5, 0]), R, 1.)
    np.testing.assert_allclose(flow, [.25, .125])


def test_memory_adapter_moves_front_obstacle_left_after_physical_right_turn():
    mem = EgocentricMemoryWrapper(decay_rate=0)
    mem.update(.2, current_yaw_rad=0)
    native_yaw = -math.pi/2
    ring = mem.update(1., current_yaw_rad=-native_yaw)
    assert ring[SECTOR_LEFT] == pytest.approx(.2)


def test_navigation_flow_feedback_brakes_leftward_drift():
    obs = SimpleNamespace(laser_distance=1., hover_throttle=.213,
                          depth_8x8=np.ones((8, 8)), optical_flow=np.array([.2, .2]))
    action = AutonomousLaserNavigatorModel().step(obs)
    assert action.roll > 0  # positive roll accelerates toward body right
    assert action.pitch < 0


def test_isaac_state_adapter_converts_body_angular_velocity_to_world_frame():
    from simulation.drone_env import IsaacGymDroneSim, QuadcopterDynamics
    # Test the adapter without requiring the external Isaac renderer/runtime.
    adapter = IsaacGymDroneSim.__new__(IsaacGymDroneSim)
    adapter.root_states = np.zeros((1, 13))
    adapter.root_tensor, adapter.sim = None, None
    adapter.gym = SimpleNamespace(set_actor_root_state_tensor=lambda *args: None)
    q = QuadcopterDynamics.euler_to_quaternion(0, 0, math.pi/2)
    adapter.sync_state(np.zeros(3), q, np.zeros(3), np.array([1., 0., 0.]))
    np.testing.assert_allclose(adapter.root_states[0, 10:13], [0, 1, 0], atol=1e-6)
