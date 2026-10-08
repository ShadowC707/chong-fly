import math

import numpy as np
import pytest

from simulation.drone_env import BoxObstacle, CylinderObstacle
from simulation.drone_interface import DownwardLaserSensor


def sensor():
    return DownwardLaserSensor(noise_std=0, mount_offset=np.zeros(3))


def pitch_rotation(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]], dtype=np.float32)


def test_vertical_beam_hits_cylinder_top_before_floor():
    result = sensor().measure(np.array([0.,0.,2.]), np.eye(3),
        cylinders=[CylinderObstacle(0,0,.4,1.)])
    assert result.surface_type == 'cylinder' and result.distance == pytest.approx(1.)
    np.testing.assert_allclose(result.hit_normal, [0,0,1])


def test_tilted_beam_crossing_top_reports_cap_not_farther_side():
    result = sensor().measure(np.array([0.,0.,2.]), pitch_rotation(.1),
        cylinders=[CylinderObstacle(0,0,.4,1.)])
    assert result.surface_type == 'cylinder'
    assert result.distance == pytest.approx(1/math.cos(.1), abs=1e-6)
    np.testing.assert_allclose(result.hit_normal, [0,0,1])


def test_box_wall_hit_reports_wall_normal_not_horizontal_surface():
    result = sensor().measure(np.array([0.,0.,1.]), pitch_rotation(-math.pi/4),
        boxes=[BoxObstacle(.4,.6,-.2,.2,.1,1.5)])
    assert result.surface_type == 'box'
    np.testing.assert_allclose(result.hit_normal, [-1,0,0])


def test_parallel_ray_outside_box_is_not_a_hit():
    result = sensor().measure(np.array([0.,0.,1.]), np.eye(3),
        boxes=[BoxObstacle(.2,.4,-.2,.2,.1,2.)])
    assert result.surface_type == 'floor'
