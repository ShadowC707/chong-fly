"""Audit examples: freshness of a new packet does not restore expired authority."""
import pytest

from edge.flight_bridge import CommandGate, CommandLease


def accept(gate,sequence,sent,received):
    return gate.accept(CommandLease(sequence,(1213.,1500.,1500.,1500.),.04),
        received_s=received,auto_authorized=True,sensor_sent_s=sent,
        latest_sensor_sequence=sequence)


@pytest.mark.parametrize('latencies',[(.002,),(.002,.015,.004)])
def test_continuous_50hz_stream_with_bounded_latency_does_not_trip(latencies):
    gate=CommandGate()
    for i in range(1000):
        sent=1.+i*.02
        received=sent+latencies[i%len(latencies)]
        assert accept(gate,i,sent,received)
        assert gate.command(now_s=received,auto_authorized=True) is not None
    assert not gate.tripped


@pytest.mark.parametrize('poll_before_accept',[False,True])
def test_fresh_reply_after_authority_expiry_requires_reset_in_either_call_order(poll_before_accept):
    gate=CommandGate()
    assert accept(gate,1,1.,1.002)
    assert gate.deadline==pytest.approx(1.040)
    if poll_before_accept:
        assert gate.command(now_s=1.040,auto_authorized=True) is None
    # The new reply is 21ms old, but authority has been absent since 1.040.
    assert not accept(gate,2,1.020,1.041)
    assert gate.tripped
    assert gate.command(now_s=1.041,auto_authorized=True) is None
    gate.reset(auto_authorized=False)
    assert accept(gate,3,1.060,1.062)


def test_old_response_cannot_replace_newer_sensor_frame_even_inside_ttl():
    gate=CommandGate()
    assert not gate.accept(CommandLease(2,(1213.,1500.,1500.,1500.),.04),
        received_s=1.041,auto_authorized=True,sensor_sent_s=1.020,
        latest_sensor_sequence=3)


def test_late_new_packet_does_not_hide_a_missed_watchdog_poll():
    gate=CommandGate()
    assert accept(gate,1,1.,1.002)
    # The receiver was not polled for an entire second. Freshness alone must
    # not silently reinstate authority when its event loop resumes.
    assert not accept(gate,2,2.,2.002)
