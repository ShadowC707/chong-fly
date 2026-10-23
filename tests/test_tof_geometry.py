"""Analytic visibility cases: geometry must agree before teacher evaluation."""
import numpy as np
import pytest

from simulation.drone_env import ToFRaycaster, RoomBoundaries, BoxObstacle, CylinderObstacle


def ray(direction, position, *, boxes=None, cylinders=None, floor=0.):
    sensor = ToFRaycaster(rows=1, cols=1, fov_h_deg=0, fov_v_deg=0)
    direction = np.asarray(direction, dtype=np.float32)
    sensor.body_ray_dirs[:] = direction/np.linalg.norm(direction)
    # Runtime warnings are failures: parallel slab intersections must not do 0/0.
    with np.errstate(divide='raise', invalid='raise'):
        return float(sensor.cast_rays(np.array(position), np.eye(3),
            RoomBoundaries(-10, 10, -10, 10, floor, 10),
            boxes=boxes, cylinders=cylinders)[0])*sensor.max_range


def test_vertical_ray_hits_cylinder_top_before_floor():
    assert ray([0, 0, -1], [0, 0, 2], cylinders=[CylinderObstacle(0, 0, .3, 1)]) == pytest.approx(1.)


def test_oblique_ray_hits_cylinder_top_before_far_side():
    assert ray([1, 0, -.5], [0, 0, 1.5],
               cylinders=[CylinderObstacle(1, 0, .3, 1)]) == pytest.approx(np.sqrt(1.25))


@pytest.mark.parametrize('y', [0., .5, 1.])
def test_parallel_box_ray_inside_or_on_slab_has_a_finite_hit(y):
    assert ray([1, 0, 0], [0, y, 1], boxes=[BoxObstacle(1, 2, 0, 1, 0, 2)]) == pytest.approx(1.)


def test_parallel_box_ray_outside_slab_does_not_create_a_phantom_hit():
    assert ray([1, 0, 0], [0, 1.01, 1], boxes=[BoxObstacle(1, 2, 0, 1, 0, 2)]) == pytest.approx(3.5)


def test_nearer_box_occludes_cylinder_cap():
    assert ray([0, 0, -1], [0, 0, 2], boxes=[BoxObstacle(-.2, .2, -.2, .2, 1.3, 1.5)],
               cylinders=[CylinderObstacle(0, 0, .3, 1)]) == pytest.approx(.5)


def test_point_rays_can_miss_a_thin_centered_object():
    sensor = ToFRaycaster()
    room = RoomBoundaries(-10, 10, -10, 10, 0, 4)
    position = np.array([0., 0., 1.])
    empty = sensor.cast_rays(position, np.eye(3), room)
    thin = sensor.cast_rays(position, np.eye(3), room,
        boxes=[BoxObstacle(.6, .62, -.005, .005, .2, 1.8)])
    np.testing.assert_array_equal(thin, empty)
    # Move the same object onto an existing central ray: now it is visible.
    y = .6*np.tan(np.deg2rad(45/14))
    visible = sensor.cast_rays(position, np.eye(3), room,
        boxes=[BoxObstacle(.6, .62, y-.005, y+.005, .2, 1.8)])
    assert visible.min()*sensor.max_range < .65


def test_challenge_scenes_have_reproducible_mirrored_visibility():
    from simulation.tof_challenge_scenes import ToFChallengeScene
    from scripts.probe_tof_geometry import visibility
    grids = []
    for name in ('near_left_far_right', 'near_right_far_left'):
        env = ToFChallengeScene(name)
        try:
            first = env.reset(seed=42)
            env.reset(seed=99)
            np.testing.assert_array_equal(env.reset(seed=42), first)
            grids.append(np.array(visibility(env)['scene_depth']).reshape(8, 8))
            assert env.specification()['initial_position'] == env.physics.pos.tolist()
        finally:
            env.close()
    np.testing.assert_allclose(grids[0][:, ::-1], grids[1])


def test_rollout_records_new_geometry_and_excludes_old_benchmark_version():
    from optimizer.rollout import simulate_policy_rollout, BENCHMARK_VERSION
    from scripts.audit_tof_threats import TeacherAdapter
    from generator.generate_reflex_dataset import ExpertReflexPolicy
    from simulation.altitude_control import AltitudeHold
    from simulation.tof_challenge_scenes import ToFChallengeScene
    env = ToFChallengeScene('thin_on_ray')
    try:
        _, report = simulate_policy_rollout(TeacherAdapter(ExpertReflexPolicy), env=env,
                                            eval_steps=2, altitude_hold=AltitudeHold())
        assert report['sensor_geometry_version'] == ToFRaycaster.geometry_version
        assert report['benchmark_version'] == BENCHMARK_VERSION
        assert report['benchmark_version'] != 'flight-benchmark-v8'
    finally:
        env.close()


def test_demonstrations_record_sensor_geometry_version():
    from generator.generate_reflex_dataset import generate_reflex_dataset
    data = generate_reflex_dataset(num_episodes=1, seq_len=2, seed=42)
    assert data['metadata']['sensor_geometry_version'] == ToFRaycaster.geometry_version
