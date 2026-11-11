from dataclasses import asdict

import numpy as np
import pytest

from demo.runner import DemoRunner
from demo.scenes import make_environment
from simulation.collisions import contact_and_clearance


def geometry(env):
    return dict(room=asdict(env.room),boxes=[asdict(x) for x in env.boxes],
                cylinders=[asdict(x) for x in env.cylinders])


def test_random_room_has_reproducible_real_geometry_and_safe_start():
    layouts=[]
    for seed in range(6):
        a=make_environment('random_room',seed=seed,start=(-.7,0,1),options={'obstacle_count':10})
        b=make_environment('random_room',seed=seed,start=(-.7,0,1),options={'obstacle_count':10})
        assert geometry(a)==geometry(b)
        assert len(a.boxes)+len(a.cylinders)==10
        assert a.room.x_max-a.room.x_min==8
        assert len({round(x.x_max-x.x_min,3) for x in a.boxes})>1
        hit,clearance=contact_and_clearance([-.7,0,1],[-.7,0,1],a.room,a.boxes,a.cylinders,.1)
        assert hit is None and clearance>=.8  # floor is 0.9 m from body
        layouts.append(geometry(a)); a.close(); b.close()
    assert layouts[0]!=layouts[1]


@pytest.mark.parametrize('axis,dimensions',[('y',(.15,3.)),('x',(3.,.15))])
def test_wall_dimensions_orientation_and_same_sensor_collision_geometry(axis,dimensions):
    r=DemoRunner(scene='wall_room',start=(-.8,0,1),scene_options={
        'wall_x':1.,'wall_y':0.,'wall_length_m':3.,'wall_axis':axis})
    b=r.env.boxes[0]
    np.testing.assert_allclose([b.x_max-b.x_min,b.y_max-b.y_min],dimensions)
    if axis=='y':
        r.resume(); r.advance(1)
        assert r.rows[0]['min_tof_m']<2.
        hit,_=contact_and_clearance([0,0,1],[1.1,0,1],r.env.room,r.env.boxes,[],.1)
        assert hit=='box'


def test_invalid_wall_or_start_is_rejected_without_clipping_geometry():
    with pytest.raises(ValueError,match='wall'):
        DemoRunner(scene='wall_room',scene_options={'wall_x':4.,'wall_length_m':3.})
    with pytest.raises(ValueError,match='Start'):
        DemoRunner(scene='wall_room',start=(1.5,0,1))


def test_random_room_input_and_geometry_recompute_exactly():
    from demo.examples import recompute
    r=DemoRunner(scene='random_room',seed=33,scene_options={'obstacle_count':8})
    r.resume(); r.event(dict(type='mode',mode='AUTO')); r.advance(150)
    data=r.recording(); rebuilt=recompute(data)
    assert rebuilt['frames']==data['frames']
    assert rebuilt['manifest']['geometry']==data['manifest']['geometry']


def test_manual_override_does_not_wait_for_stop_but_auto_does():
    r=DemoRunner(); r.resume(); r.event(dict(type='key',code='KeyW',down=True)); r.advance(250)
    r.event(dict(type='key',code='KeyW',down=False)); r.event(dict(type='mode',mode='AUTO'))
    r.advance(250)
    assert np.linalg.norm(r.env.physics.vel[:2])>.1
    assert r.rows[-1]['plant_source']=='manual' and r.request=='AUTO'
    assert any(x['check']=='flow' for x in r.admission)
    r.event(dict(type='mode',mode='MANUAL'))
    assert r.receiver.mode=='MANUAL' and r.request=='MANUAL'
    assert np.linalg.norm(r.env.physics.vel[:2])>.1
