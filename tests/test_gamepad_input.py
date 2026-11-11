"""Read-only PC input seam: scripted source or injected XInput, never FC I/O."""
import ctypes
import struct

import pytest

from edge.gamepad_input import GamepadState,MockGamepadSource,XInputSource,XInputState,summarize
from edge.gamepad_probe import capture


STATE=GamepadState(packet_number=9,buttons=0x1001,left_trigger=0,right_trigger=255,
    left_x=-32768,left_y=32767,right_x=-123,right_y=456)


def test_windows_xinput_binary_layout_is_16_bytes_and_preserves_signed_extremes():
    raw=struct.pack('<IHBBhhhh',9,0x1001,0,255,-32768,32767,-123,456)
    assert ctypes.sizeof(XInputState)==16
    sample=XInputState.from_buffer_copy(raw)
    assert GamepadState.from_xinput(sample)==STATE


def test_fixed_slot_and_disconnect_never_return_cached_controls():
    class Backend:
        def __init__(self): self.slots=[]
        def get_state(self,slot):
            self.slots.append(slot)
            return (0,STATE) if len(self.slots)==1 else (1167,None)
    backend=Backend()
    clock=iter([1000,2000])
    source=XInputSource(2,backend=backend,clock_us=lambda:next(clock))
    first,second=source.poll(),source.poll()
    assert first.connected and first.state==STATE
    assert not second.connected and second.state is None and second.api_status==1167
    assert backend.slots==[2,2] and (first.sample_id,second.sample_id)==(0,1)


def test_unchanged_packet_number_is_not_reported_as_controller_loss_or_rf_freshness():
    class Backend:
        def get_state(self,slot): return 0,STATE
    clock=iter([1000,21000])
    source=XInputSource(0,backend=Backend(),clock_us=lambda:next(clock))
    samples=[source.poll(),source.poll()]
    result=summarize(samples)
    assert result['connected_samples']==2 and result['packet_number_changes']==0
    assert result['rf_freshness']=='unknown' and result['physical_transport']=='unverified'
    assert samples[1].read_completed_us==21000


def test_mock_exhaustion_is_disconnected_not_an_invented_neutral():
    source=MockGamepadSource([STATE],clock_us=lambda:123)
    assert source.poll().state==STATE
    end=source.poll()
    assert not end.connected and end.state is None and end.source=='mock'


@pytest.mark.parametrize('slot',[-1,4,True])
def test_bad_slots_are_rejected_before_loading_a_driver(slot):
    with pytest.raises(ValueError): XInputSource(slot,backend=object())


@pytest.mark.parametrize('changes',[{'buttons':65536},{'left_trigger':-1},{'right_trigger':256},
    {'left_x':-32769},{'left_y':32768},{'packet_number':2**32}])
def test_out_of_range_states_are_rejected(changes):
    from dataclasses import replace
    with pytest.raises(ValueError): replace(STATE,**changes)


def test_capture_summary_is_raw_calibration_data_without_rc_mapping():
    from dataclasses import replace
    states=[replace(STATE,left_x=-100),None,replace(STATE,left_x=200)]
    source=MockGamepadSource(states,clock_us=iter([1,2,3]).__next__)
    report=capture(source,count=3,hz=50,sleep=lambda _:None,clock=lambda:0.)
    assert report['summary']['connected_samples']==2 and report['summary']['disconnected_samples']==1
    assert report['summary']['axes']['left_x']==dict(min=-100,max=200,median=50.)
    assert report['udp_sends']==0 and report['fc_attempts']==0
    assert report['calibration_profile_applied'] is False


@pytest.mark.parametrize('count,hz',[(0,50),(12001,50),(2,0),(2,201),(True,50)])
def test_capture_has_finite_bounds(count,hz):
    with pytest.raises(ValueError): capture(MockGamepadSource([]),count=count,hz=hz)
