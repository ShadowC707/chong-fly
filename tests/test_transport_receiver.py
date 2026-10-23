import pytest

from edge.flight_bridge import CommandLease
from edge.session_receiver import SessionReceiver, CommandPacket
from edge.transport import LinkConfig, SimulatedTransport

PWM=(1213.,1500.,1500.,1500.)


def ready():
    r=SessionReceiver()
    r.reset(session_id='one',now_s=1000.)
    r.set_mode('SHADOW',now_s=1000.)
    assert r.output(now_s=1000.,fallback_pwm=PWM)['accepted_pwm'] is None
    r.set_mode('AUTO',now_s=1000.02)
    r.announce_sensor(1,now_s=1000.02)
    return r


def packet(seq=1,session='one'):
    return CommandPacket(session,seq,CommandLease(seq,PWM,.04))


def test_receiver_uses_local_sensor_clock_and_watchdog_never_auto_recovers():
    r=ready()
    event=r.receive(packet(),now_s=1000.028)
    assert event.accepted_pwm==PWM
    assert r.output(now_s=1000.059,fallback_pwm=PWM)['source']=='auto'
    out=r.output(now_s=1000.0601,fallback_pwm=PWM)
    assert out['tripped'] and out['accepted_pwm'] is None and out['source']=='fallback'
    r.announce_sensor(2,now_s=1000.08)
    assert r.receive(packet(2),now_s=1000.088).accepted_pwm is None
    with pytest.raises(ValueError): r.reset(session_id='two',now_s=1000.09)
    r.set_mode('MANUAL',now_s=1000.10)
    r.reset(session_id='two',now_s=1000.10)
    r.set_mode('SHADOW',now_s=1000.10)
    r.set_mode('AUTO',now_s=1000.12)
    r.announce_sensor(0,now_s=1000.12)
    assert r.receive(packet(0,'two'),now_s=1000.128).accepted_pwm==PWM
    assert r.receive(packet(1,'one'),now_s=1000.129).reason=='session_mismatch'
    assert not r.gate.tripped


@pytest.mark.parametrize('kind',['late','wrong_sequence','duplicate','revoke'])
def test_current_session_failure_latches(kind):
    r=ready()
    assert r.receive(packet(),now_s=1000.028).accepted_pwm==PWM
    if kind=='late':
        r.announce_sensor(2,now_s=1000.04)
        e=r.receive(packet(2),now_s=1000.061)
    elif kind=='wrong_sequence':
        r.announce_sensor(2,now_s=1000.04)
        e=r.receive(packet(),now_s=1000.048)
    elif kind=='duplicate': e=r.receive(packet(),now_s=1000.03)
    else:
        r.set_mode('MANUAL',now_s=1000.03)
        e=r.receive(packet(),now_s=1000.031)
    assert e.accepted_pwm is None and r.gate.tripped


def test_fresh_response_after_gap_trips_without_watchdog_poll():
    r=ready(); r.receive(packet(),now_s=1000.028)
    r.announce_sensor(3,now_s=1000.08)
    assert r.receive(packet(3),now_s=1000.088).accepted_pwm is None
    assert r.gate.tripped


def test_no_lease_response_revokes_active_receiver():
    r=ready(); r.receive(packet(),now_s=1000.028)
    r.announce_sensor(2,now_s=1000.04)
    assert r.receive(CommandPacket('one',2,None),now_s=1000.048).accepted_pwm is None
    assert r.gate.tripped


def test_duplicate_no_lease_response_is_also_rejected():
    r=ready()
    r.receive(CommandPacket('one',1,None),now_s=1000.028)
    assert not r.gate.tripped
    r.receive(CommandPacket('one',1,None),now_s=1000.029)
    assert r.gate.tripped


def test_transport_is_seeded_and_old_packets_survive_session_reset_in_queue():
    cfg=LinkConfig(uplink_delay_s=.004,downlink_delay_s=.004,jitter_s=.002,seed=3)
    a,b=SimulatedTransport(cfg),SimulatedTransport(cfg)
    for t in range(10):
        for link in (a,b): link.send('downlink',packet(),now_s=float(t))
    assert a.drain(20.)==b.drain(20.)
    a.send('downlink',packet(),now_s=21.,delay_s=.1)
    a.send('downlink',packet(0,'two'),now_s=21.,delay_s=.01)
    assert a.drain(21.02)[0].payload.session_id=='two'
    assert a.drain(21.11)[0].payload.session_id=='one'
    a.send('downlink',packet(),now_s=22.,drop=True)
    assert a.drain(23.)==[]
    assert a.events[-1]['kind']=='dropped'


def test_reset_cannot_reuse_session_or_clear_fault_outside_manual():
    r=ready()
    r.receive(packet(),now_s=1000.028)
    r.set_mode('MANUAL',now_s=1000.03)
    with pytest.raises(ValueError): r.reset(session_id='one',now_s=1000.03)
    r.set_mode('SHADOW',now_s=1000.04)
    with pytest.raises(ValueError): r.reset(session_id='new',now_s=1000.04)
    assert r.gate.tripped


def test_watchdog_expiry_is_exact_and_receipt_history_remains_immutable():
    r=ready()
    receipt=r.receive(packet(),now_s=1000.028)
    deadline=r.gate.deadline
    assert r.output(now_s=deadline,fallback_pwm=PWM)['accepted_pwm'] is None
    assert receipt.accepted_pwm==PWM  # historical acceptance is not current authority
    assert r.gate.tripped


def test_session_reset_does_not_reset_receiver_clock():
    r=ready(); r.set_mode('MANUAL',now_s=1000.04)
    with pytest.raises(ValueError): r.reset(session_id='new',now_s=0.)
    assert r.gate.tripped


def test_transport_copies_payload_and_can_reorder_packets():
    link=SimulatedTransport()
    value={'sequence':1,'pwm':[1213.]}
    link.send('uplink',value,now_s=0.,delay_s=.1)
    value['pwm'][0]=2000.
    link.send('uplink',{'sequence':2},now_s=.02,delay_s=.01)
    first=link.drain(.04)
    assert first[0].payload['sequence']==2
    last=link.drain(.12)[0]
    assert last.payload['pwm']==[1213.]
    assert last.due_s==.1 and last.delivered_s==.12


@pytest.mark.parametrize('kwargs',[{'jitter_s':-1},{'loss_probability':1.1},
    {'uplink_delay_s':float('nan')},{'seed':-1}])
def test_invalid_link_config(kwargs):
    with pytest.raises(ValueError): LinkConfig(**kwargs)
