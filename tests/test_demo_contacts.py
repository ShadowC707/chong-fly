import math

import pytest

from demo.contacts import contact_and_clearance, step_with_contact
from demo.runner import DemoRunner
from simulation.collisions import contact_and_clearance as conservative_contact
from simulation.drone_env import RoomBoundaries, BoxObstacle, CylinderObstacle

ROOM=RoomBoundaries(-4,4,-4,4,0,3)


def test_reported_corner_has_real_gap_and_must_not_crash():
    p=[-.18555520474910736,-1.3913556337356567,.9972406625747681]
    box=BoxObstacle(.006604222538990942,.15259078910646384,-3.28,-1.5900000000000003,0,2.484178986341566)
    assert conservative_contact(p,p,ROOM,[box],[],.2)==('box',0.)
    kind,gap=contact_and_clearance(p,p,ROOM,[box],[],.2)
    assert kind is None and gap==pytest.approx(.07637805582978657)


@pytest.mark.parametrize('axis',[0,1,2])
def test_swept_box_contact_detects_crossing_even_with_clear_endpoints(axis):
    box=BoxObstacle(-.05,.05,-.05,.05,.95,1.05)
    start,end=[0,0,1],[0,0,1]
    start[axis]-=.7; end[axis]+=.7
    assert contact_and_clearance(start,end,ROOM,[box],[],.2)==('box',0.)


def test_exact_edge_tangency_and_near_miss_are_distinct():
    box=BoxObstacle(0,1,0,1,0,2)
    tangent=-.2/math.sqrt(2)
    assert contact_and_clearance([tangent,tangent,1],[tangent,tangent,1],ROOM,[box],[],.2)[0]=='box'
    assert contact_and_clearance([tangent-1e-5,tangent-1e-5,1],[tangent-1e-5,tangent-1e-5,1],ROOM,[box],[],.2)[0] is None
    assert contact_and_clearance([-.3,-.19,1],[-.19,-.3,1],ROOM,[box],[],.2)[0] is None


def test_cylinder_rim_is_rounded_by_the_sphere_not_an_expanded_flat_cap():
    cyl=CylinderObstacle(0,0,.5,1)
    p=[.65,0,1.15]
    assert conservative_contact(p,p,ROOM,[],[cyl],.2)[0]=='cylinder'
    kind,gap=contact_and_clearance(p,p,ROOM,[],[cyl],.2)
    assert kind is None and gap==pytest.approx(math.hypot(.15,.15)-.2)
    assert contact_and_clearance([-.9,0,1.1],[.9,0,1.1],ROOM,[],[cyl],.2)[0]=='cylinder'
    assert contact_and_clearance([-.9,0,1.21],[.9,0,1.21],ROOM,[],[cyl],.2)[0] is None


def test_false_box_candidate_does_not_hide_a_real_cylinder_contact():
    p=[-.19,-.19,1]
    box=BoxObstacle(0,1,0,1,0,2)
    cyl=CylinderObstacle(-.3,-.3,.1,2)
    assert contact_and_clearance(p,p,ROOM,[box],[cyl],.2)[0]=='cylinder'


def test_demo_step_filters_only_false_contact_and_preserves_plant_state():
    r=DemoRunner(scene='room',start=(-.19,-.19,1))
    r.env.boxes=[BoxObstacle(0,1,0,1,0,2)]
    r.resume(); r.advance(5)
    assert not r.terminal and not r.env.is_crashed
    assert r.rows[-1]['clearance_m']>0 and r.rows[-1]['collision'] is None
    assert r.time_s==.02


def test_non_contact_termination_remains_terminal(monkeypatch):
    r=DemoRunner(); original=r.env.step
    def excessive(*args):
        obs,reward,done,info=original(*args)
        info.update(excessive_tilt=True,crashed=True)
        return obs,reward,True,info
    monkeypatch.setattr(r.env,'step',excessive)
    _,_,done,info=step_with_contact(r.env,[1213,1500,1500,1500])
    assert done and info['crashed'] and info['excessive_tilt']


def test_new_contact_model_records_and_recomputes_while_old_model_is_supported():
    from demo.examples import recompute
    r=DemoRunner(scene='random_room',seed=2209320175)
    assert r.manifest()['contact_model']=='sphere-sweep-v1'
    r.resume(); r.event({'type':'mode','mode':'AUTO'}); r.advance(3000)
    data=r.recording()
    assert recompute(data)['frames']==data['frames']
    legacy=DemoRunner(contact_model='expanded-primitives-v1')
    legacy.resume(); legacy.advance(5)
    old=legacy.recording(); old['manifest'].pop('contact_model')
    assert recompute(old)['frames']==old['frames']
