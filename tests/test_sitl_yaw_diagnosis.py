import numpy as np
import pytest

from evaluate.sitl_yaw_diagnosis import split_thrust, load_commands


def test_equal_mean_rpm_does_not_mean_equal_collective_thrust():
    flat=split_thrust([10,10,10,10],.1)
    split=split_thrust([12,8,8,12],.1)
    assert flat['mean_rpm_thrust_n']==split['mean_rpm_thrust_n']==40
    assert flat['split_thrust_n']==0
    assert split['split_thrust_n']==pytest.approx(1.6)
    assert split['total_thrust_n']==pytest.approx(41.6)


def test_replay_refuses_commands_that_change_between_rc_frames(tmp_path):
    p=tmp_path/'trace.npz'
    command=np.full((40,4),1500.)
    np.savez(p,time_s=np.arange(41)*.001,pwm=command)
    assert load_commands(p,.04)[1]==1000
    command[3,3]=1600
    np.savez(p,time_s=np.arange(41)*.001,pwm=command)
    with pytest.raises(ValueError,match='50 Hz'):load_commands(p,.04)


def test_replay_rejects_incomplete_or_nonuniform_recording(tmp_path):
    p=tmp_path/'trace.npz'
    np.savez(p,time_s=np.arange(21)*.001,pwm=np.full((20,4),1500.))
    with pytest.raises(ValueError,match='horizon'):load_commands(p,.04)
    t=np.arange(41)*.001;t[10]+=.0001
    np.savez(p,time_s=t,pwm=np.full((40,4),1500.))
    with pytest.raises(ValueError,match='uniform'):load_commands(p,.04)


def test_submillisecond_step_cannot_invent_a_new_rc_frame():
    from evaluate.sitl_lockstep import Session
    # Validation must happen before touching sockets or advancing counters.
    session=Session.__new__(Session)
    for us in (250,500,1000):
        with pytest.raises(ValueError,match='explicit RC freshness'):
            session.step(duration_us=us)
