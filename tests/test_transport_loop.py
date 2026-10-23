from dataclasses import replace
import numpy as np
import pytest

from edge.runtime import AltitudeSample
from evaluate.validate_sensor_handover import SampleSchedule
from evaluate.validate_transport_loop import CASES, PI_ORIGIN, authority, map_frame, reset_time
from simulation.receiver_fallback import ReceiverFallback
from tests.test_edge_runtime import exported


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
