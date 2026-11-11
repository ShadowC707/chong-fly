import copy
import json

import numpy as np
import pytest

from demo.manual_input import ManualInput
from demo.recording import load_recording, validate_recording, replay_frame
from demo.runner import DemoRunner


def test_keys_are_bounded_opposed_repeat_safe_and_simulation_timed():
    a, b = ManualInput(), ManualInput()
    for x in (a, b):
        x.key('KeyW', True)
        x.key('KeyW', True)
    for _ in range(50): a.advance(.004)
    for _ in range(10): b.advance(.02)
    np.testing.assert_allclose(a.axes, b.axes)
    assert a.axes[1] > 0 and max(abs(a.axes)) <= 100
    a.key('KeyS', True)
    for _ in range(100): a.advance(.004)
    np.testing.assert_array_equal(a.axes, [0, 0, 0])
    a.key('ArrowLeft', True)
    for _ in range(100): a.advance(.004)
    assert a.axes[2] == -100
    a.clear()
    for _ in range(100): a.advance(.004)
    np.testing.assert_array_equal(a.axes, [0, 0, 0])


@pytest.mark.parametrize('code,axis,sign', [('KeyW',0,1),('KeyS',0,-1),
    ('KeyA',1,1),('KeyD',1,-1),('ArrowLeft',2,1),('ArrowRight',2,-1)])
def test_key_directions_in_actual_physics(code, axis, sign):
    r = DemoRunner()
    r.resume(); r.event(dict(type='key', code=code, down=True))
    r.advance(150)
    value = r.env.physics.vel[axis] if axis < 2 else r.env.physics.omega[2]
    assert sign * value > .001


def test_shadow_never_drives_plant_and_controllers_are_distinct():
    a, b = DemoRunner(), DemoRunner()
    a.resume(); b.resume(); b.event(dict(type='mode', mode='SHADOW'))
    a.advance(100); b.advance(100)
    np.testing.assert_array_equal(a.env.physics.pos, b.env.physics.pos)
    assert a.system.runtime.altitude._now is None
    assert b.system.runtime.altitude is not b.manual_altitude
    assert b.system.runtime.altitude._integral == 0
    assert b.rows[-1]['trace']['network_pwm'] is not None
    assert b.rows[-1]['trace']['issued_pwm'] is None
    assert b.rows[-1]['plant_source'] == 'manual'


def test_pause_disconnect_reset_and_no_wall_catchup():
    from demo.server import Pacer
    r = DemoRunner(); p = Pacer(r)
    r.resume(); p.poll(10); p.poll(10.02)
    n = r.step_index
    r.event(dict(type='key', code='KeyW', down=True))
    r.pause('focus_lost'); p.poll(200)
    assert r.step_index == n and not r.manual.keys and r.request == 'MANUAL'
    r.resume(); p.poll(201)
    assert r.step_index == n
    p.poll(201.02); assert r.step_index == n+5
    old = r.session_id
    r.reset(); assert r.session_id != old and r.step_index == 0 and r.paused
    assert not np.any(r.system.runtime.policy.h)
    assert r.manual_altitude._now is None


def test_determinism_does_not_depend_on_snapshot_frequency():
    a, b = DemoRunner(), DemoRunner()
    for r in (a, b):
        r.resume(); r.event(dict(type='key', code='KeyW', down=True))
    a.advance(125)
    for _ in range(25): b.advance(5); b.snapshot()
    assert a.rows == b.rows


def test_recording_rejects_corruption_and_replay_cannot_invent_sensors(tmp_path):
    r = DemoRunner(); r.resume(); r.advance(10)
    recording = r.recording()
    validate_recording(recording)
    assert replay_frame(recording, .023)['time_s'] == .02
    bad = copy.deepcopy(recording); bad['frames'][1]['time_s'] = -1
    with pytest.raises(ValueError): validate_recording(bad)
    bad = copy.deepcopy(recording); bad['frames'][0]['plant_input_pwm'][0] = 0
    with pytest.raises(ValueError): validate_recording(bad)
    p = tmp_path/'broken.json'; p.write_text('{')
    with pytest.raises(ValueError, match='recording'): load_recording(p)
    legacy = dict(version='standalone-mvp-viewer-v2', geometry={}, scene='clear',
        trace=dict(physics=[dict(time_s=0.,position_m=[0,0,1],yaw_rad=0)], control=[]))
    p.write_text(json.dumps(legacy))
    row = load_recording(p)['frames'][0]
    assert row['sensors'] is None and row['attitude_rad'] == [None,None,0]
    assert row['trace']['accepted_pwm'] is None


def test_auto_requires_dwell_acceptance_and_takeover_revokes_old_packets():
    from edge.session_receiver import CommandPacket
    from edge.flight_bridge import CommandLease
    r=DemoRunner(); r.resume(); r.event(dict(type='mode',mode='AUTO'))
    r.advance(125)
    assert all(x['trace']['issued_pwm'] is None for x in r.rows)
    r.advance(125)
    auto=[x for x in r.rows if x['plant_source']=='auto']
    assert auto and auto[0]['time_s']==.5
    assert auto[0]['plant_input_pwm']==auto[0]['manual_pwm']
    for row in auto:
        assert row['trace']['receiver_status']=='accepted'
        assert row['plant_input_pwm']==row['receiver']['accepted_pwm']
    previous=np.array(r.previous_pwm)
    r.event(dict(type='mode',mode='MANUAL'))
    assert r.receiver.gate.tripped and r.receiver.gate.pwm is None
    r.advance(1)
    assert abs(r.previous_pwm[0]-previous[0]) <= 150*.004+1e-8
    assert max(abs(np.array(r.previous_pwm[1:])-previous[1:])) <= 300*.004+1e-8
    assert r.rows[-1]['plant_source']=='manual'
    assert r.manual_altitude._now == 1.0
    old_session=r.session_id
    old=CommandPacket(old_session,49,CommandLease(49,tuple(previous),.04))
    assert r.receiver.receive(old,now_s=r.time_s).accepted_pwm is None
    r.event(dict(type='mode',mode='AUTO')); r.advance(150)
    assert all(x['plant_source']!='auto' for x in r.rows[250:])
    r.reset()
    assert r.receiver.receive(old,now_s=0.).reason=='session_mismatch'


def test_near_obstacle_denial_preserves_thresholds_and_stop_unavailable():
    r=DemoRunner(scene='obstacle_center',start=(.4,0,1))
    r.resume(); r.event(dict(type='stop'))
    assert not r.stop_requested and r.events[-1]['kind']=='stop_unavailable'
    r.event(dict(type='mode',mode='AUTO')); r.advance(175)
    assert not any(x['plant_source']=='auto' for x in r.rows)
    assert any(x['check']=='min_tof_m' for x in r.admission)


def test_sensor_capture_identity_and_tof_left_right():
    r=DemoRunner(scene='obstacle_left',start=(0,0,1)); r.resume(); r.advance(11)
    a,b=r.rows[0],r.rows[5]
    assert a['sensors']['tof']['sample_id']==b['sensors']['tof']['sample_id']
    assert b['sensors']['tof']['age_s']==.02 and b['sensors']['tof']['state']=='held'
    tof=np.array(a['sensors']['tof']['value']).reshape(8,8)
    assert tof[:,:4].min() < tof[:,4:].min()


def test_disconnect_is_detected_without_browser_blur():
    from demo.server import DemoService
    s=DemoService(); r=s.new_run(); r.resume()
    s.heartbeat('test',10.); s.poll(10.); s.poll(11.6)
    assert r.paused and r.pause_reason=='client_disconnected'
    assert not r.manual.keys


def test_busy_renderer_drops_wall_time_without_physics_jump():
    from demo.server import Pacer
    r=DemoRunner(); r.resume(); p=Pacer(r)
    p.poll(0); p.poll(2.)
    assert not r.paused and r.step_index==0
    p.poll(2.02); assert r.step_index==5


def test_automatic_pause_is_in_reproducible_input_schedule():
    r=DemoRunner(); r.resume(); r.advance(20); r.pause('client_disconnected')
    assert r.input_schedule[-1]['event']==dict(type='pause',reason='client_disconnected')
    assert r.input_schedule[-1]['step']==20


def test_controller_updates_once_per_owned_tick(monkeypatch):
    r=DemoRunner(); calls=[[],[]]
    for i,controller in enumerate((r.manual_altitude,r.system.runtime.altitude)):
        original=controller.update
        def count(sample, *, now_s, _i=i, _fn=original, **kw):
            calls[_i].append(now_s)
            return _fn(sample,now_s=now_s,**kw)
        monkeypatch.setattr(controller,'update',count)
    r.resume(); r.event(dict(type='mode',mode='SHADOW')); r.advance(100)
    assert len(calls[0])==100 and len(calls[1])==20
    assert all(len(x)==len(set(x)) for x in calls)


def test_arbitrary_input_schedule_recomputes_exact_physics():
    from demo.examples import recompute
    r=DemoRunner(); r.resume()
    r.event(dict(type='key',code='KeyW',down=True)); r.advance(30)
    r.pause('client_disconnected'); r.resume()
    r.event(dict(type='key',code='KeyW',down=False)); r.advance(20)
    recording=r.recording()
    assert recompute(recording)['frames']==recording['frames']


def test_invalid_physics_stops_without_nan_in_recording(monkeypatch):
    r=DemoRunner(); original=r.env.step
    def invalid(*args,**kwargs):
        obs,reward,_,info=original(*args,**kwargs)
        r.env.physics.pos[0]=float('nan')
        info.update(invalid_state=True,crashed=False,collision_kind='invalid_state')
        return obs,reward,True,info
    monkeypatch.setattr(r.env,'step',invalid)
    r.resume(); r.advance(10)
    assert r.terminal and r.step_index==1 and r.pause_reason=='invalid_physics'
    validate_recording(r.recording())
    assert r.events[-2]['kind']=='termination'


def test_collision_is_terminal_and_cannot_resume():
    r=DemoRunner(scene='obstacle_center',start=(.7,0,1)); r.resume()
    r.event(dict(type='key',code='KeyW',down=True)); r.advance(1000)
    assert r.terminal and r.paused and r.rows[-1]['collision']=='box'
    assert r.rows[-1]['plant_input_pwm'] is None
    assert r.rows[-1]['plant_source']=='terminated'
    for m in r.rows[-1]['sensors'].values():
        assert m['age_s']==pytest.approx(r.rows[-1]['time_s']-m['timestamp_s'])
    with pytest.raises(ValueError,match='reset'): r.resume()


def test_one_click_fly_uses_real_policy_and_receiver_without_keyboard():
    from demo.server import DemoService
    s=DemoService(); r=s.start_fly(scene='obstacle_left')
    assert not r.paused and r.request=='AUTO' and not r.manual.keys
    r.advance(125)
    assert not any(x['trace']['issued_pwm'] is not None for x in r.rows)
    r.advance(1375)
    assert not r.terminal and any(x['plant_source']=='auto' for x in r.rows)
    assert not any(x['event']['type']=='key' for x in r.input_schedule)
    auto=[x for x in r.rows if x['plant_source']=='auto']
    assert all(x['trace']['receiver_status']=='accepted' for x in auto)
    assert max(abs(x['trace']['network_pwm'][3]-1500) for x in auto)>50
    assert np.linalg.norm(r.env.physics.pos-np.array(r.start))>.1


def test_neural_observation_is_actual_hidden_state_and_never_advances_policy():
    r=DemoRunner(); r.resume(); r.advance(5)
    assert r.snapshot()['neural']['status']=='not_running'
    r.event(dict(type='mode',mode='SHADOW')); r.advance(5)
    before=r.system.runtime.policy.h.copy()
    state=r.snapshot()['neural']
    assert state['status']=='running' and len(state['values'])==306
    np.testing.assert_array_equal(state['values'],before)
    np.testing.assert_array_equal(r.system.runtime.policy.h,before)
    assert state['control_time_s']==.02
    r.pause('focus_lost')
    assert r.snapshot()['neural']['status']=='paused'
