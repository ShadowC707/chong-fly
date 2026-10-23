"""Analytic checks before using curved/tilted geometry in sensor experiments."""
import numpy as np
import pytest


def test_sphere_near_root_tangent_miss_and_inside_exit():
    from simulation.tof_surfaces import Sphere, cast_surfaces
    sphere = Sphere((1., 0., 0.), .2)
    assert cast_surfaces([0,0,0], [[1,0,0]], [sphere])[0] == pytest.approx(.8)
    assert cast_surfaces([0,.2,0], [[1,0,0]], [sphere])[0] == pytest.approx(1.)
    assert cast_surfaces([0,.21,0], [[1,0,0]], [sphere])[0] == 3.5
    assert cast_surfaces([1,0,0], [[1,0,0]], [sphere])[0] == pytest.approx(.2)


def test_patch_is_finite_two_sided_and_parallel_safe():
    from simulation.tof_surfaces import PlanePatch, cast_surfaces
    panel = PlanePatch((.6,0,0), (0,1,0), (0,0,1), .1, .2)
    assert cast_surfaces([0,0,0], [[1,0,0]], [panel])[0] == pytest.approx(.6)
    assert cast_surfaces([0,.11,0], [[1,0,0]], [panel])[0] == 3.5
    assert cast_surfaces([0,0,0], [[0,1,0]], [panel])[0] == 3.5
    assert cast_surfaces([1,0,0], [[-1,0,0]], [panel])[0] == pytest.approx(.4)


def test_tilted_plane_matches_plane_equation():
    from simulation.tof_surfaces import PlanePatch, cast_surfaces
    s = np.sqrt(.5)
    panel = PlanePatch((.6,0,0), (-s,s,0), (0,0,1), 1., 1.)
    # Plane x+y=.6; origin y=.1, direction +x -> t=.5.
    assert cast_surfaces([0,.1,0], [[1,0,0]], [panel])[0] == pytest.approx(.5)


def test_occlusion_order_and_range_cap():
    from simulation.tof_surfaces import Sphere, cast_surfaces
    near, far = Sphere((1.,0,0),.2), Sphere((2.,0,0),.3)
    a = cast_surfaces([0,0,0], [[1,0,0]], [far, near])
    b = cast_surfaces([0,0,0], [[1,0,0]], [near, far])
    np.testing.assert_array_equal(a, b)
    assert a[0] == pytest.approx(.8)
    assert cast_surfaces([0,0,0], [[1,0,0]], [near], max_range=.5)[0] == .5


@pytest.mark.parametrize('bad', [0., -1., float('nan')])
def test_rejects_invalid_sphere_radius(bad):
    from simulation.tof_surfaces import Sphere
    with pytest.raises(ValueError):
        Sphere((1.,0,0), bad)


def test_rejects_invalid_patch_basis():
    from simulation.tof_surfaces import PlanePatch
    with pytest.raises(ValueError):
        PlanePatch((1,0,0), (0,1,0), (0,1,0), 1, 1)
    with pytest.raises(ValueError):
        PlanePatch((1,0,0), (0,2,0), (0,0,1), 1, 1)


@pytest.mark.parametrize('direction', [[[2,0,0]], [[np.nan,0,0]], [[0,0,0]]])
def test_rejects_nonunit_or_nonfinite_rays(direction):
    from simulation.tof_surfaces import cast_surfaces
    with pytest.raises(ValueError):
        cast_surfaces([0,0,0], direction, [])


def test_front_patch_matches_existing_box_front_and_rotation_is_invariant():
    from simulation.tof_surfaces import PlanePatch, cast_surfaces
    from simulation.drone_env import ToFRaycaster, RoomBoundaries, BoxObstacle
    caster = ToFRaycaster()
    panel = PlanePatch((.6,0,0), (0,1,0), (0,0,1), 1., 1.)
    ranges = cast_surfaces([0,0,0], caster.body_ray_dirs, [panel])
    legacy = caster.cast_rays(np.zeros(3), np.eye(3), RoomBoundaries(-10,10,-10,10,-10,10),
                             boxes=[BoxObstacle(.6,.7,-1,1,-1,1)])*3.5
    np.testing.assert_allclose(ranges, legacy, atol=1e-6)
    rotation = np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    turned = PlanePatch((0,.6,0), (-1,0,0), (0,0,1), 1., 1.)
    np.testing.assert_allclose(cast_surfaces([0,0,0], caster.body_ray_dirs@rotation.T, [turned]), ranges)


def test_small_patch_can_be_sampled_then_rejected_by_support():
    from simulation.tof_surfaces import PlanePatch, cast_surfaces
    from simulation.zonal_tof import ZonalConfig, ZonalToF, aggregate_ranges
    sensor = ZonalToF(ZonalConfig(samples_per_axis=8))
    az = np.deg2rad(2.8125)
    center = (.6, .6*np.tan(az), .6*np.tan(az)/np.cos(az))
    panel = PlanePatch(center, (0,1,0), (0,0,1), .006, .006)
    ranges = cast_surfaces([0,0,0], sensor.subray_directions.reshape(-1,3), [panel]).reshape(64,-1)
    assert ranges.min() < .8
    grouped = [aggregate_ranges(x, sensor.config)[0] for x in ranges]
    assert min(grouped) == 3.5


def test_probe_distinguishes_missing_samples_from_discarded_near_ranges():
    from scripts.probe_tof_surfaces import observation_metrics
    empty = np.full((64,16),3.5)
    outputs = np.full(64,3.5)
    assert observation_metrics(empty,outputs)['state'] == 'sampling_miss'
    empty[7,0] = .6
    missed = observation_metrics(empty,outputs)
    assert missed['state'] == 'aggregation_miss' and missed['lost_near_zones'] == 1
    outputs[7] = .6
    retained = observation_metrics(empty,outputs)
    assert retained['state'] == 'near_reported' and retained['lost_near_zones'] == 0


def test_generated_scenes_have_a_real_near_surface_point_in_field_of_view():
    from scripts.probe_tof_surfaces import scenes
    from simulation.tof_surfaces import Sphere, cast_surfaces
    for spec, target in scenes():
        point = np.array(target.center)
        if isinstance(target, Sphere):
            point[0] -= target.radius
        distance = np.linalg.norm(point)
        azimuth = np.arctan2(point[1], point[0])
        elevation = np.arctan2(point[2], np.linalg.norm(point[:2]))
        assert distance < .8 and max(abs(azimuth), abs(elevation)) < np.deg2rad(22.5)
        # An analytic ray toward this known point must reach the surface.
        assert cast_surfaces([0,0,0], [point/distance], [target])[0] == pytest.approx(distance,abs=1e-8)
