import numpy as np
import pytest
from evaluate.sitl_neural import SceneSensors, scene_spec
from evaluate.sitl_yaw_stress import scenes, LifecycleToF, lifecycle_check


def test_stress_geometry_has_clear_initial_admission_and_no_contact():
    from demo.contacts import contact_and_clearance
    state=np.zeros(12);state[2]=1.
    for name,spec in scenes().items():
        sensors=SceneSensors('clear', spec=spec)
        hit,gap=contact_and_clearance(state[:3],state[:3],sensors.room,sensors.boxes,sensors.cylinders,.2)
        assert hit is None and gap>=.2
        _,tof,_=sensors.sample(state,0.)
        assert tof.min()*3.5>=.95


def test_explicit_old_geometry_preserves_sensor_samples():
    a=SceneSensors('obstacle_right');b=SceneSensors('clear',spec=scene_spec('obstacle_right'))
    for x in (0.,.4,.8):
        state=np.zeros(12);state[:3]=[x,.05,1.];state[3]=.2;state[8]=.1
        fa,ta,aa=a.sample(state,0.);fb,tb,ab=b.sample(state,0.)
        np.testing.assert_array_equal(fa,fb);np.testing.assert_array_equal(ta,tb)
        assert aa==ab


def test_scene_sensor_uses_off_origin_start_state():
    spec = scene_spec('clear')
    spec['start'] = [1., 0., 1.]
    sensors = SceneSensors('clear', spec=spec)
    state = np.zeros(12)
    state[:3] = spec['start']
    _, off_origin, _ = sensors.sample(state, 0.)
    state[0] = 0.
    _, at_origin, _ = sensors.sample(state, 0.)
    assert 3. < off_origin.reshape(8, 8)[3:5].min()*3.5 < 3.1
    assert at_origin.reshape(8, 8)[3:5].min() == 1.


@pytest.mark.parametrize('time,expected', [(0.,None),(1.98,None),(2.,.7),(3.38,.7),
    (3.4,2.),(3.78,2.),(3.8,.7),(13.98,.7),(14.,2.),(16.98,2.),(17.,None)])
def test_probe_is_fixed_in_time_and_does_not_mutate_truth(time,expected):
    raw=np.linspace(.1,1.,64);saved=raw.copy()
    probe=LifecycleToF();value=probe.apply(raw,time)
    np.testing.assert_array_equal(raw,saved)
    np.testing.assert_allclose(value,raw if expected is None else expected/3.5)
    assert probe.last['overridden'] == (expected is not None)


def test_lifecycle_check_requires_both_events_and_neutral_by_original_deadline():
    decisions=[dict(time_s=2.,event='started',phase='BRAKE'),
               dict(time_s=2.5,event='braked',phase='TURN'),
               dict(time_s=3.6,event='clear',phase='RECOVER'),
               dict(time_s=3.8,event='renewed_threat',phase='BRAKE'),
               dict(time_s=4.,event='braked',phase='TURN'),
               dict(time_s=6.66,event='turn_timeout',phase='FAILED')]
    t=np.arange(1000)*.02; yaw=np.full(1000,1500.)
    # Cumulative TURN before the second entry is 1.1s; original deadline is 6.9s.
    result=lifecycle_check(decisions,t,yaw)
    assert result['covered'] and result['neutral_by_deadline']
    assert result['original_deadline_s']==pytest.approx(6.9)
    yaw[(t>=6.9)&(t<7.2)]=1640.
    assert not lifecycle_check(decisions,t,yaw)['neutral_by_deadline']
    assert not lifecycle_check(decisions[:3],t,yaw)['covered']
