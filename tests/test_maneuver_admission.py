from dataclasses import replace
import numpy as np
import pytest

from edge.sensors import Measurement,SensorSnapshot
from simulation.maneuver_admission import ManeuverAdmission


def frame(i, *, distance=2., flow=(0.,0.), tilt=(0.,0.)):
    t=i*.02
    return SensorSnapshot(i,t,Measurement(i,t,flow),Measurement(i,t,np.full(64,distance/3.5)),
                          Measurement(i,t,(1.,0.,0.)),Measurement(i,t,tilt))


def test_clear_scene_does_not_claim_proven_corridor():
    a=ManeuverAdmission()
    r=a.observe(frame(0),now_s=0.)
    assert r['state']=='NO_EPISODE'
    assert r['resume_allowed'] is None
    assert not r['episode_pending']


def test_front_clear_and_neutral_yaw_cannot_erase_pending_maneuver():
    a=ManeuverAdmission()
    a.observe(frame(0,distance=.7,flow=(.1,0)),now_s=0.)
    for i in range(1,101):
        r=a.observe(frame(i),now_s=i*.02)
    assert r['state']=='SURVEY_REQUIRED'
    assert r['episode_pending'] and r['resume_allowed'] is False
    assert r['episode_id']==1
    assert 'corridor_unverified' in r['reasons']


def test_move_or_tilt_interrupts_settling_evidence():
    a=ManeuverAdmission()
    for i in range(31):r=a.observe(frame(i,distance=.9),now_s=i*.02)
    assert not r['motion_quiet']
    r=a.observe(frame(31,tilt=(.1,0)),now_s=.62)
    assert r['quiet_elapsed_s']==0
    for i in range(32,63):r=a.observe(frame(i),now_s=i*.02)
    assert not r['motion_quiet']
    r=a.observe(frame(63,flow=(0,.4)),now_s=1.26)
    assert r['state']=='MOTION_UNCONFIRMED'


def test_critical_range_keeps_reflex_priority_even_when_quiet():
    a=ManeuverAdmission()
    for i in range(61):r=a.observe(frame(i,distance=.3),now_s=i*.02)
    assert r['state']=='REFLEX_REQUIRED'
    assert r['resume_allowed'] is False


def test_held_flow_does_not_create_new_motion_evidence():
    a=ManeuverAdmission()
    a.observe(frame(0,distance=.7),now_s=0.)
    f=replace(frame(1),flow=frame(0).flow)
    r=a.observe(f,now_s=.02)
    assert r['quiet_elapsed_s']==0
    assert r['sensor_states']['flow']=='held'


def test_invalid_capture_blocks_and_does_not_clear_episode():
    a=ManeuverAdmission()
    a.observe(frame(0,distance=.7),now_s=0.)
    f=replace(frame(1),tof=Measurement(1,.02,np.full(64,np.nan)))
    r=a.observe(f,now_s=.02)
    assert r['state']=='INVALID_SENSORS' and r['episode_pending']
    assert r['resume_allowed'] is False
    for i in range(2,20):r=a.observe(frame(i),now_s=i*.02)
    assert r['state']=='MOTION_UNCONFIRMED'


def test_replayed_snapshot_never_issues_permission():
    a=ManeuverAdmission();a.observe(frame(0),now_s=0.)
    assert a.observe(frame(0),now_s=.02)['state']=='INVALID_SENSORS'


def test_explicit_session_reset_is_not_automatic_resume():
    a=ManeuverAdmission();a.observe(frame(0,distance=.7),now_s=0.)
    a.reset()
    r=a.observe(frame(0),now_s=0.)
    assert r['state']=='NO_EPISODE' and r['resume_allowed'] is None


def test_late_capture_cannot_bridge_a_missing_quiet_interval():
    a=ManeuverAdmission()
    for i in range(40):a.observe(frame(i,distance=.7),now_s=i*.02)
    r=a.observe(frame(60),now_s=1.2)
    assert r['quiet_elapsed_s']==0
    assert not r['motion_quiet']


def test_changed_payload_under_same_identity_is_invalid():
    a=ManeuverAdmission();a.observe(frame(0,distance=.7),now_s=0.)
    f=replace(frame(1),flow=Measurement(0,0.,(.1,0.)))
    assert a.observe(f,now_s=.02)['state']=='INVALID_SENSORS'


def test_monitor_owns_copies_not_mutable_input_arrays():
    a=ManeuverAdmission();f=frame(0,distance=.7)
    a.observe(f,now_s=0.)
    f.tof.value[:]=1.
    reused=replace(frame(1),tof=f.tof)
    assert a.observe(reused,now_s=.02)['state']=='INVALID_SENSORS'


@pytest.mark.parametrize('kwargs',[{'quiet_flow':1.},{'quiet_dwell_s':0.},
                                  {'notice_m':float('nan')},{'critical_m':1.}])
def test_invalid_configuration_is_rejected(kwargs):
    from simulation.maneuver_admission import AdmissionConfig
    with pytest.raises(ValueError):AdmissionConfig(**kwargs)


def test_replay_adapter_uses_only_measured_channels_and_frame_identity():
    from dataclasses import asdict
    from evaluate.replay_maneuver_admission import snapshot_from_row
    original=frame(0,distance=.7)
    row=dict(sensors={name:asdict(getattr(original,name)) for name in ('flow','tof','altitude','attitude')},
             trace=dict(sequence=0,tick_time_s=0.))
    a=ManeuverAdmission().observe(snapshot_from_row(row),now_s=0.)
    row.update(position_m=[100,100,100],velocity_mps=[0,0,0],attitude_rad=[0,0,2.5],clearance_m=100.)
    b=ManeuverAdmission().observe(snapshot_from_row(row),now_s=0.)
    assert a==b


def test_same_small_flow_at_large_downrange_is_not_a_calibrated_stop():
    a=ManeuverAdmission()
    for i in range(61):
        f=replace(frame(i,distance=.9),altitude=Measurement(i,i*.02,(2.5,0.,0.)))
        r=a.observe(f,now_s=i*.02)
    assert not r['motion_quiet']
    assert r['state']=='OUTSIDE_STUDIED_RANGE'
    assert r['resume_allowed'] is False


def test_critical_front_alert_is_not_hidden_by_out_of_range_altitude():
    f=replace(frame(0,distance=.3),altitude=Measurement(0,0.,(2.5,0.,0.)))
    r=ManeuverAdmission().observe(f,now_s=0.)
    assert r['state']=='REFLEX_REQUIRED'
    assert not r['range_domain_ok'] and not r['motion_quiet']
