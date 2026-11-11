import math

import numpy as np
import pytest

from demo.body import visual_contract, visual_bound
from demo.runner import DemoRunner
from simulation.drone_env import BoxObstacle, CylinderObstacle


def test_visible_drone_including_rotors_fits_physical_envelope():
    r=DemoRunner()
    visual=visual_contract(r.env.collision_radius_m)
    assert visual_bound(visual) <= r.env.collision_radius_m
    assert visual['collision_radius_m']==r.manifest()['geometry']['collision_radius_m']
    assert visual==r.manifest()['geometry']['drone_visual']
    # Old renderer had motor offsets ±0.16 m and rotor radius 0.1 m.
    assert math.hypot(.16,.16)+.1 > 3*.1  # original 10 cm envelope


@pytest.mark.parametrize('yaw',[0.,math.pi/4,math.pi/2,math.pi])
def test_rotation_does_not_move_visible_parts_outside_collision_sphere(yaw):
    visual=visual_contract(.1)
    unit=visual['unit_geometry']
    q=DemoRunner().env.physics.euler_to_quaternion(.2,-.15,yaw)
    from simulation.drone_env import QuadcopterDynamics
    rotation=QuadcopterDynamics.quaternion_to_rotation_matrix(q)
    for center in unit['rotor_centers']:
        for angle in np.linspace(0,2*math.pi,65):
            point=.1*(np.array(center)+unit['rotor_radius']*np.array([math.cos(angle),math.sin(angle),0]))
            assert np.linalg.norm(rotation@point) < .1


@pytest.mark.parametrize('kind,start',[
    ('room',[3.8999,0,1]),('box',[.8999,0,1]),('cylinder',[.6499,0,1])])
def test_first_physical_contact_terminates_even_when_center_is_outside_obstacle(kind,start):
    r=DemoRunner(scene='room',start=start,collision_radius_m=.1)
    if kind=='box': r.env.boxes=[BoxObstacle(1,1.2,-1,1,0,2.5)]
    if kind=='cylinder': r.env.cylinders=[CylinderObstacle(1,0,.25,2.5)]
    r.env.physics.vel[:]=[.4,0,0]
    r.resume(); r.advance(5)
    assert r.terminal and r.paused and r.pause_reason=='collision'
    assert r.step_index==1 and r.rows[-1]['collision']==kind
    assert r.rows[-1]['plant_input_pwm'] is None


def test_old_visual_overlap_was_outside_physical_body():
    r=DemoRunner(scene='wall_room',start=(1.3,0,1),scene_options={'wall_x':1.5},collision_radius_m=.1)
    face=r.env.boxes[0].x_min
    assert r.start[0]+math.hypot(.16,.16)+.1 > face
    assert r.start[0]+visual_bound(visual_contract(r.env.collision_radius_m)) < face
