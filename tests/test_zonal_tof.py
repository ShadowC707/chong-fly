import numpy as np
import pytest


def test_mixed_zone_mean_minimum_and_supported_group_are_distinct():
    from simulation.zonal_tof import ZonalConfig, aggregate_ranges
    depths = np.array([.5]*2+[3.]*14)
    assert aggregate_ranges(depths, ZonalConfig(rule='mean'))[0] == pytest.approx(2.6875)
    assert aggregate_ranges(depths, ZonalConfig(rule='minimum'))[0] == .5
    assert aggregate_ranges(depths, ZonalConfig(rule='supported'))[0] == .5
    isolated = np.array([.5]+[3.]*15)
    result, support = aggregate_ranges(isolated, ZonalConfig(rule='supported'))
    assert result == 3. and support == 15/16


def test_group_diameter_is_anchored_not_chained():
    from simulation.zonal_tof import ZonalConfig, aggregate_ranges
    # Adjacent differences fit .05 but the three samples do not form one group.
    depths = np.array([.50, .54, .58]+[3.]*13)
    assert aggregate_ranges(depths, ZonalConfig(rule='supported', min_coverage=.18))[0] == 3.


@pytest.mark.parametrize('rule', ['mean', 'minimum', 'supported'])
def test_no_hit_stays_max_range_without_invented_support(rule):
    from simulation.zonal_tof import ZonalConfig, aggregate_ranges
    distance, support = aggregate_ranges(np.full(16, 3.5), ZonalConfig(rule=rule))
    assert distance == 3.5 and support == 0


def test_no_return_counts_against_coverage_and_mean_includes_its_range_cap():
    from simulation.zonal_tof import ZonalConfig, aggregate_ranges
    values = np.array([.5]+[3.5]*15)
    assert aggregate_ranges(values, ZonalConfig(rule='supported')) == (3.5, 0.)
    assert aggregate_ranges(values, ZonalConfig(rule='mean'))[0] == pytest.approx(3.3125)


@pytest.mark.parametrize('values', [[np.nan]*16, [-1.]*16, [4.]*16, []])
def test_invalid_samples_are_not_converted_to_free_space(values):
    from simulation.zonal_tof import ZonalConfig, aggregate_ranges
    with pytest.raises(ValueError):
        aggregate_ranges(np.array(values), ZonalConfig())


@pytest.mark.parametrize('kwargs', [dict(samples_per_axis=0), dict(samples_per_axis=2.5),
    dict(rule='unknown'), dict(min_coverage=0), dict(group_span_m=float('nan'))])
def test_invalid_configuration_fails(kwargs):
    from simulation.zonal_tof import ZonalConfig
    with pytest.raises(ValueError):
        ZonalConfig(**kwargs)


def test_sampling_shape_and_mirror_symmetry_and_legacy_is_unchanged():
    from simulation.zonal_tof import ZonalConfig, ZonalToF
    from simulation.drone_env import ToFRaycaster, RoomBoundaries, BoxObstacle
    old = ToFRaycaster(); before = old.body_ray_dirs.copy()
    sensor = ZonalToF(ZonalConfig(samples_per_axis=4, rule='minimum'))
    dirs = sensor.subray_directions
    assert dirs.shape == (8, 8, 16, 3)
    np.testing.assert_allclose(np.linalg.norm(dirs, axis=-1), 1., atol=1e-6)
    room = RoomBoundaries(-10, 10, -10, 10, 0, 4)
    args = (np.array([0., 0., 1.]), np.eye(3), room)
    left = sensor.cast_rays(*args, boxes=[BoxObstacle(.6,.62,.03,.08,.2,1.8)]).reshape(8,8)
    right = sensor.cast_rays(*args, boxes=[BoxObstacle(.6,.62,-.08,-.03,.2,1.8)]).reshape(8,8)
    np.testing.assert_allclose(left[:, ::-1], right)
    assert left.min()*3.5 < .65
    np.testing.assert_array_equal(old.body_ray_dirs, before)


def test_minimum_sees_thin_center_with_sufficient_sampling_but_mean_dilutes_it():
    from simulation.zonal_tof import ZonalConfig, ZonalToF
    from simulation.drone_env import RoomBoundaries, BoxObstacle
    args = (np.array([0.,0.,1.]), np.eye(3), RoomBoundaries(-10,10,-10,10,0,4))
    objects = [BoxObstacle(.6,.62,-.005,.005,.2,1.8)]
    near = ZonalToF(ZonalConfig(samples_per_axis=8,rule='minimum')).cast_rays(*args, boxes=objects)
    mean = ZonalToF(ZonalConfig(samples_per_axis=8,rule='mean')).cast_rays(*args, boxes=objects)
    assert near.min()*3.5 < .8
    assert mean.min()*3.5 > .8


def test_supported_rule_has_an_explicit_coverage_boundary():
    from simulation.zonal_tof import ZonalConfig, aggregate_ranges
    cfg = ZonalConfig(samples_per_axis=8, rule='supported')
    below = aggregate_ranges(np.array([.6]*7+[3.5]*57), cfg)
    at = aggregate_ranges(np.array([.6]*8+[3.5]*56), cfg)
    assert below == (3.5, 0.)
    assert at == pytest.approx((.6, .125))


def test_zonal_rotation_uses_the_shared_world_transform():
    from simulation.zonal_tof import ZonalConfig, ZonalToF
    from simulation.drone_env import RoomBoundaries, BoxObstacle
    sensor = ZonalToF(ZonalConfig(rule='minimum'))
    room = RoomBoundaries(-10, 10, -10, 10, 0, 4)
    pos = np.array([0., 0., 1.])
    straight = sensor.cast_rays(pos, np.eye(3), room,
                               boxes=[BoxObstacle(.6,.62,-.1,.1,.2,1.8)])
    rotated = sensor.cast_rays(pos, np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]]), room,
                              boxes=[BoxObstacle(-.1,.1,.6,.62,.2,1.8)])
    np.testing.assert_allclose(straight, rotated)


def test_experiment_pairs_physics_but_labels_custom_sensor_results_separately():
    from scripts.probe_zonal_tof import physical_case, VARIANTS, EXPERIMENT_VERSION
    from optimizer.rollout import BENCHMARK_VERSION
    row = physical_case('thin_center_gap', steps=2, seed=42, coverage=.125, span=.05)
    assert set(row['results']) == set(VARIANTS)
    observation_hashes = set()
    for result in row['results'].values():
        assert result['benchmark_version'] == EXPERIMENT_VERSION != BENCHMARK_VERSION
        assert result['rollout_engine_version'] == BENCHMARK_VERSION
        assert result['sensor_contract']['geometry_version'] == result['sensor_geometry_version']
        assert result['horizon_completed'] and not result['fatal_failure']
        assert len(result['navigation_trace']) == 2
        observation_hashes.add(result['initial_observation_sha256'])
    # The manipulation must actually change what the same teacher observes.
    assert len(observation_hashes) > 1
