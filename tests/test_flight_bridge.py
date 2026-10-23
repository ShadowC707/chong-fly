from types import SimpleNamespace
import numpy as np
import pytest

from edge.flight_bridge import FlightBridge, SensorFrame, CommandLease, CommandGate
from tests.test_edge_runtime import exported


class Runtime:
    def __init__(self):
        self.calls=0; self.resets=0
        self.altitude=SimpleNamespace(last_command=None,config=SimpleNamespace(target_height_m=1.))
        self.metadata={'tof_range_m':3.5}
    def reset(self): self.resets+=1
    def step(self,flow,tof,altitude,**kw):
        self.calls+=1
        if not altitude.valid: raise ValueError('invalid altitude')
        self.altitude.last_command=SimpleNamespace(height_m=1.,vertical_speed_mps=0.)
        return dict(applied_pwm=np.array([1213.,1500.,1527.,1500.]))


def frame(i,*,flow=(0.,0.),valid=True):
    from simulation.altitude_control import AltitudeSample
    t=i*.02
    return SensorFrame(i,t,flow,np.ones(64),AltitudeSample(.982,0.,0.,t,valid))


def tick(b,i,request='SHADOW',**kw):
    return b.tick(frame(i,**kw),now_s=i*.02,request=request,manual_pwm=[1213.,1500.,1500.,1500.])


def ready(b):
    for i in range(31): tick(b,i)


def test_shadow_has_no_authority_and_auto_requires_observed_dwell():
    b=FlightBridge(Runtime())
    assert tick(b,0,'AUTO')['lease'] is None
    for i in range(1,31): assert tick(b,i)['lease'] is None
    r=tick(b,31,'AUTO')
    assert r['state']=='AUTO'
    assert r['lease'].pwm==(1213.,1500.,1500.,1500.)
    r=tick(b,32,'AUTO')
    assert r['lease'].pwm[2]==1527.


def test_moving_entry_is_denied_without_claiming_authority():
    b=FlightBridge(Runtime())
    for i in range(40): r=tick(b,i,'AUTO',flow=(.05,0.))
    assert r['state']=='SHADOW' and r['lease'] is None and not r['ready']


def test_manual_override_does_not_require_healthy_sensors():
    b=FlightBridge(Runtime()); ready(b); tick(b,31,'AUTO')
    calls=b.runtime.calls
    r=b.tick(None,now_s=.64,request='MANUAL',manual_pwm=None)
    assert r['state']=='MANUAL' and r['lease'] is None and b.runtime.calls==calls


@pytest.mark.parametrize('kind',['invalid','duplicate','late','gap'])
def test_fault_revokes_authority_and_requires_explicit_reset(kind):
    b=FlightBridge(Runtime()); ready(b); tick(b,31,'AUTO')
    f=frame(32,valid=kind!='invalid'); now=.64
    if kind=='duplicate': f=frame(31)
    if kind=='late': now=.72
    if kind=='gap': f=frame(35); now=.70
    r=b.tick(f,now_s=now,request='AUTO',manual_pwm=[1213.,1500.,1500.,1500.])
    assert r['state']=='FAULT' and r['lease'] is None
    assert tick(b,36,'SHADOW')['state']=='FAULT'
    tick(b,37,'MANUAL'); b.reset_fault()
    assert tick(b,38)['state']=='SHADOW'


def test_throttle_and_attitude_command_disagreement_blocks_engagement():
    b=FlightBridge(Runtime()); ready(b)
    r=b.tick(frame(31),now_s=.62,request='AUTO',manual_pwm=[1400.,1500.,1500.,1500.])
    assert r['lease'] is None and not r['ready']


def test_receiver_lease_expires_without_pi_and_rejects_replay_after_revocation():
    gate=CommandGate()
    lease=CommandLease(1,(1213.,1500.,1500.,1500.),.04)
    def accept(lease,t):
        return gate.accept(lease,received_s=t,auto_authorized=True,
            sensor_sent_s=t,latest_sensor_sequence=lease.sequence)
    assert accept(lease,10.)
    assert gate.command(now_s=10.02,auto_authorized=True)==lease.pwm
    assert gate.command(now_s=10.041,auto_authorized=True) is None
    assert not accept(lease,10.05)
    assert not accept(CommandLease(2,lease.pwm,.04),10.06)
    gate.reset(auto_authorized=False)
    assert accept(CommandLease(2,lease.pwm,.04),10.06)
    assert gate.command(now_s=10.07,auto_authorized=False) is None
    assert not accept(CommandLease(2,lease.pwm,.04),10.08)


def test_uart_buffering_does_not_extend_lease_from_old_sensor_frame():
    gate=CommandGate(); lease=CommandLease(5,(1213.,1500.,1500.,1500.),.04)
    assert not gate.accept(lease,received_s=10.1,auto_authorized=True,
        sensor_sent_s=10.,latest_sensor_sequence=5)


def test_real_runtime_invalid_altitude_is_caught_and_fault_is_latched(exported):
    from edge.runtime import FlightRuntime
    _,bundle=exported
    runtime=FlightRuntime(bundle,coordinator_mode='both',stop_hover=True)
    b=FlightBridge(runtime)
    r=tick(b,0,valid=False)
    assert r['state']=='FAULT' and r['lease'] is None
    assert runtime.fault is not None


def test_output_throttle_slew_and_first_command_are_bounded():
    runtime=Runtime(); b=FlightBridge(runtime); ready(b)
    assert tick(b,31,'AUTO')['lease'].pwm[0]==1213.
    original=runtime.step
    def jump(*args,**kwargs):
        r=original(*args,**kwargs); r['applied_pwm'][0]=1350.; return r
    runtime.step=jump
    assert tick(b,32,'AUTO')['lease'].pwm[0]==pytest.approx(1216.)
