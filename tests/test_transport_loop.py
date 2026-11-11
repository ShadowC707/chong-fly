from dataclasses import replace
import numpy as np
import pytest

from edge.runtime import AltitudeSample
from evaluate.validate_sensor_handover import SampleSchedule
from evaluate.validate_transport_loop import CASES, PI_ORIGIN, authority, map_frame, reset_time
from simulation.receiver_fallback import ReceiverFallback
from tests.test_edge_runtime import exported


@pytest.mark.parametrize('delay', [-.001, float('nan'), float('inf')])
def test_compute_delay_rejects_invalid_values(delay):
    from evaluate.validate_transport_loop import PiComputeQueue
    with pytest.raises(ValueError):
        PiComputeQueue(delay)


def test_compute_completion_is_quantized_and_cannot_bypass_busy_job():
    from evaluate.validate_transport_loop import PiComputeQueue
    worker = PiComputeQueue(.002)
    worker.submit('first', now_s=.004)
    worker.submit('second', now_s=.004)
    assert worker.drain(.004) == []
    first, = worker.drain(.008)
    assert first['payload'] == 'first'
    assert first['started_s'] == .004
    assert first['due_s'] == pytest.approx(.006)
    assert first['completed_s'] == .008
    assert worker.drain(.008) == []
    second, = worker.drain(.012)
    assert second['payload'] == 'second'
    assert second['started_s'] == .008
    assert second['received_s'] == .004
    assert worker.drain(.016) == []


def test_zero_compute_delay_preserves_immediate_reply_and_session_identity():
    from evaluate.validate_transport_loop import PiComputeQueue
    worker = PiComputeQueue(0.)
    worker.submit({'session_id': 'old'}, now_s=.004)
    worker.submit({'session_id': 'new'}, now_s=.004)
    ready = worker.drain(.004)
    assert [r['payload']['session_id'] for r in ready] == ['old', 'new']
    assert all(r['received_s'] == r['started_s'] == r['completed_s'] == .004 for r in ready)


def test_twenty_ms_latency_does_not_pass_due_to_float_roundoff():
    from evaluate.validate_transport_loop import within_roundtrip_budget
    assert within_roundtrip_budget(.016)
    assert not within_roundtrip_budget(.02 - 1e-13)
    assert not within_roundtrip_budget(.024)


def test_physical_runner_delays_issue_and_revalidates_sensor_age(exported):
    from evaluate.validate_transport_loop import Case, physical_case
    from tests.test_system_assembly import config
    _, bundle = exported
    cfg = config(bundle, sensor_mode='multi_rate_v1', handover_mode='freeze_sync_v1')
    immediate = physical_case(cfg, Case('zero', horizon_s=.12))
    delayed = physical_case(cfg, Case('delayed', horizon_s=.12, pi_compute_delay_s=.002))
    a, b = immediate['traces'][0], delayed['traces'][0]
    assert a['issue_global_s'] == .004
    assert b['pi_received_global_s'] == .004
    assert b['issue_global_s'] == .008
    assert b['range_capture_age_s']-a['range_capture_age_s'] == pytest.approx(.004)
    sent = [r for r in delayed['link_events'] if r['direction']=='downlink' and r['kind']=='scheduled']
    assert sent[0]['sent_s'] == .008
    stale = physical_case(cfg, Case('stale', horizon_s=.16, pi_compute_delay_s=.044))
    assert stale['traces'][0]['issued_pwm'] is None
    assert stale['traces'][0]['fault'] == 'Stale, future or replayed frame'


def test_clock_mapping_keeps_capture_angles_and_held_identity():
    schedule=SampleSchedule('multi')
    a=AltitudeSample(.982,.1,-.2,0.,True)
    first=schedule.frame(0,(0.,0.),np.ones(64),a)
    held=schedule.frame(1,(.01,0.),np.ones(64),replace(a,roll_rad=.3,timestamp_s=.02))
    mapped=map_frame(held,17)
    assert mapped.altitude.sample_id==first.altitude.sample_id
    assert mapped.altitude.value==first.altitude.value
    assert mapped.altitude.timestamp_s==PI_ORIGIN
    assert mapped.attitude.value[0]==.3
    assert mapped.timestamp_s==PI_ORIGIN+.02 and mapped.sequence==17


def test_reentry_protocol_requires_explicit_manual_reset_then_shadow():
    for c in CASES:
        t=reset_time(c)
        if t is not None:
            assert authority(c,t-.1)=='MANUAL'
            assert authority(c,t)=='SHADOW'
            assert authority(c,t+1.49)=='SHADOW'
            assert authority(c,t+1.5)=='AUTO'


def test_fallback_freezes_integral_during_auto_then_damps_measured_flow():
    f=ReceiverFallback()
    for i in range(30):
        p=f.step((.1,-.1),AltitudeSample(.932,0.,0.,i*.02,True),
                 now_s=i*.02,active=False,previous_pwm=(1220.,1500.,1500.,1500.))
    assert f.altitude._integral==0.
    assert p[0]==1220. and p[1]<1500 and p[2]<1500
    p=f.step((.1,-.1),AltitudeSample(.932,0.,0.,.60,True),
             now_s=.60,active=True,previous_pwm=p)
    assert abs(p[0]-1220.)<=3.+1e-8
    assert p[3]==1500.


def test_fallback_rejects_bad_local_sensors_instead_of_inventing_height():
    f=ReceiverFallback()
    with pytest.raises(ValueError):
        f.step((float('nan'),0.),AltitudeSample(1.,0.,0.,0.,True),
               now_s=0.,active=True,previous_pwm=(1213.,1500.,1500.,1500.))


def test_real_system_lease_crosses_link_and_observation_survives_expiry(exported):
    from edge.runtime import FlightRuntime
    from edge.system import build_system, observe_receiver
    from edge.session_receiver import SessionReceiver, CommandPacket
    from edge.transport import SimulatedTransport
    from tests.test_system_assembly import config
    from tests.test_flight_bridge import frame
    _, bundle=exported
    system=build_system(config(bundle,session_id='test',sensor_mode='multi_rate_v1',
                               handover_mode='freeze_sync_v1'))
    probe=FlightRuntime(bundle)
    f=frame(0)
    manual=probe.step(f.flow_xy,f.tof_8x8,f.altitude,now_s=0.)['applied_pwm']
    rx=SessionReceiver(); rx.reset(session_id='test',now_s=1000.)
    link=SimulatedTransport()
    for i in range(40):
        t=i*.02
        mode='SHADOW' if i<30 else 'AUTO'
        rx.set_mode(mode,now_s=1000.+t)
        rx.output(now_s=1000.+t,fallback_pwm=manual)
        rx.announce_sensor(i,now_s=1000.+t)
        link.send('uplink',map_frame(frame(i),i),now_s=t)
        sent=link.drain(t+.004)[0]
        result=system.tick(sent.payload,now_s=PI_ORIGIN+t+.004,
                           request=mode,manual_pwm=manual)
        trace=result['trace']
        link.send('downlink',CommandPacket('test',i,result['lease']),now_s=t+.004)
        delivered=link.drain(t+.008)[0]
        receipt=rx.receive(delivered.payload,now_s=1000.+t+.008)
        if result['lease'] is not None:
            observed=observe_receiver(trace,session_id='test',sequence=i,
                accepted_pwm=receipt.accepted_pwm,receiver_time_s=receipt.receiver_time_s,reason=receipt.reason)
            assert observed.accepted_pwm==trace.issued_pwm
            assert trace.accepted_pwm is None
    assert observed.receiver_status=='accepted'
    out=rx.output(now_s=rx.gate.deadline,fallback_pwm=manual)
    assert out['tripped'] and out['accepted_pwm'] is None
    assert observed.accepted_pwm is not None
