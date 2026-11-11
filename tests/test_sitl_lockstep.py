import struct

import numpy as np
import pytest

from simulation.sitl_lockstep import encode_step, decode_step, infer_hover


def test_request_has_version_sequence_duration_and_explicit_rc_freshness():
    fdm=struct.pack('<18d',*([0.]*18))
    frame=encode_step(7,4000,True,fdm,[1200,1501,1502,1503],[1000]*12)
    assert len(frame)==196
    assert struct.unpack_from('<4sQII',frame)==(b'CFS1',7,4000,1)
    assert struct.unpack_from('<16H',frame,164)[:4]==(1501,1502,1200,1503)
    held=encode_step(8,4000,False,fdm,[1200,1501,1502,1503],[1000]*12)
    assert struct.unpack_from('<4sQII',held)[3]==0
    for dt in (0,3999,4010):
        with pytest.raises(ValueError): encode_step(7,dt,True,fdm,[1200,1500,1500,1500],[1000]*12)


def test_rejects_wrong_ack_and_clock_jump_even_when_motor_values_look_fine():
    packet=struct.pack('<QQII4f',7,8000,32,0,.2,.2,.2,.2)
    result=decode_step(packet,7,4000,4000)
    assert result['motor_updates']==32
    for seq,clock in ((8,4000),(7,3000)):
        with pytest.raises(ValueError):decode_step(packet,seq,clock,4000)
    with pytest.raises(ValueError):decode_step(packet[:-1],7,4000,4000)
    bad=struct.pack('<QQII4f',7,8000,32,1,.2,.2,.2,.2)
    with pytest.raises(ValueError):decode_step(bad,7,4000,4000)


def test_hover_is_inverse_of_measured_monotone_map_not_old_rc_constant():
    rows=[{'rc_pwm':x,'mean_motor':y} for x,y in [(1100,.1),(1200,.2),(1300,.3)]]
    assert infer_hover(rows,.25)==pytest.approx(1250)
    with pytest.raises(ValueError):infer_hover(rows,.4)
    with pytest.raises(ValueError):infer_hover(rows[::-1],.25)
    with pytest.raises(ValueError):infer_hover([rows[0],rows[0]],.1)


def test_saved_trace_audit_rejects_clock_gap_and_rc_updates_between_frames():
    from evaluate.report_sitl_lockstep import check_trace
    trace = {'state': np.zeros((1001, 12)), 'time_s': np.arange(1001)*.004,
             'motors': np.full((1000, 4), .2), 'pwm': np.full((1000, 4), 1500.),
             'acks': np.column_stack((np.arange(1, 1001), np.arange(1, 1001)*4000,
                                      np.full(1000, 32)))}
    report = {'completed': True, 'armed': {'armed': True, 'angle': True},
              'firmware_elapsed_us': 4000000}
    assert all(check_trace(trace, report).values())
    trace['acks'][500, 1] += 10
    assert not check_trace(trace, report)['ack_sequence_clock']
    trace['pwm'][2, 0] += 1
    assert not check_trace(trace, report)['rc_50hz_hold']
    report['completed'] = False
    assert not check_trace(trace, report)['completed']
