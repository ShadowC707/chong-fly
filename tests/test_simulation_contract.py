"""Physical/sensor regressions: these assertions describe observable behavior."""
import numpy as np
import pytest

from simulation.drone_env import DroneSimulationEnv, RoomBoundaries, BoxObstacle, CylinderObstacle


def make_env(**kwargs):
    env = DroneSimulationEnv(engine="standalone", headless=True, **kwargs)
    env.reset(initial_pos=np.array([0., 0., 1.]), seed=7)
    return env


@pytest.mark.parametrize("position", [[1.5, 0., 1.], [0., 2.4, 1.]])
def test_inside_obstacle_is_a_crash(position):
    env = make_env()
    env.reset(initial_pos=np.array(position), seed=7)
    _, _, done, info = env.step([env.physics.hover_throttle, 0, 0, 0])
    assert done and info["crashed"]
    assert info["collision_kind"] in {"box", "cylinder"}


@pytest.mark.parametrize("kind", ["box", "cylinder"])
def test_swept_collision_catches_crossing_with_both_endpoints_clear(kind):
    env = make_env()
    env.boxes = [BoxObstacle(.9, 1.1, -.1, .1, 0., 2.)] if kind == "box" else []
    env.cylinders = [CylinderObstacle(1., 0., .1, 2.)] if kind == "cylinder" else []
    # Prescribe a physics segment to isolate contact detection from flight PID.
    env.physics.step = lambda motors: setattr(env.physics, "pos", np.array([2., 0., 1.]))
    _, _, done, info = env.step([env.physics.hover_throttle, 0, 0, 0])
    assert done and info["collision_kind"] == kind


def test_propeller_envelope_hits_without_center_penetration():
    env = make_env(collision_radius_m=.12)
    env.boxes = [BoxObstacle(.1, .5, -.1, .1, .5, 1.5)]
    env.cylinders = []
    _, _, done, info = env.step([env.physics.hover_throttle, 0, 0, 0])
    assert done and info["collision_kind"] == "box"


def test_safe_flight_above_short_obstacle_does_not_collide():
    env = make_env()
    env.boxes = []
    env.cylinders = [CylinderObstacle(0., 0., .2, .4)]
    assert not env.step([env.physics.hover_throttle, 0, 0, 0])[2]


def test_room_lower_bounds_are_not_assumed_symmetric():
    env = make_env(room=RoomBoundaries(x_min=-1., x_max=4.))
    env.reset(initial_pos=np.array([-1.1, 0., 1.]), seed=1)
    _, _, done, info = env.step([env.physics.hover_throttle, 0, 0, 0])
    assert done and info["collision_kind"] == "room"


def test_observation_reads_do_not_advance_sensors_or_odometry():
    env = make_env()
    assert np.array_equal(env.displacement_total, np.zeros(2))
    env.physics.vel[:] = [1., 0., 0.]
    obs, _, _, _ = env.step([env.physics.hover_throttle, 0, 0, 0])
    displacement = env.displacement_total.copy()
    for _ in range(4):
        flow, tof = env.get_chong_fly_obs()
        np.testing.assert_array_equal(np.r_[flow, tof], obs)
        rich = env.get_isaac_obs()
        np.testing.assert_array_equal(rich.optical_flow, obs[:2])
        np.testing.assert_array_equal(rich.depth_flat, obs[2:])
    np.testing.assert_array_equal(env.displacement_total, displacement)
    assert env.step_count == 1
    flow[:] = 42  # callers cannot corrupt the stored snapshot
    assert not np.any(env.get_chong_fly_obs()[0] == 42)


@pytest.mark.parametrize("action", [[np.nan, 0, 0, 0], [np.inf, 0, 0, 0], [1, 2, 3]])
def test_bad_action_is_rejected_before_state_mutation(action):
    env = make_env()
    pos = env.physics.pos.copy()
    with pytest.raises(ValueError):
        env.step(action)
    assert env.step_count == 0
    np.testing.assert_array_equal(env.physics.pos, pos)


def test_reset_repeats_sensor_stream_even_after_another_episode():
    env = make_env()
    def trace():
        result = [env.reset(seed=17)]
        for _ in range(10):
            result.append(env.step([env.physics.hover_throttle, 0, 0, 0])[0])
        return np.array(result)
    first = trace()
    env.reset(seed=99)
    np.testing.assert_array_equal(first, trace())
