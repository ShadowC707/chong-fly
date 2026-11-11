"""End-to-end bytes through the pinned FlightSystem, entirely on this host."""
from dataclasses import replace
from pathlib import Path
import pytest

from edge.hardware_mock import run_mock, BoundedInbox
from tests.test_esp32_host import esp_host
from tests.test_hardware_wire import sensor, SESSION
from edge.hardware_wire import ClockMapping, encode_sensor, decode_command
from edge.pi_hardware_adapter import PiHardwareAdapter
from edge.system import RunConfig


def test_mock_uses_pinned_system_and_receipts_then_expires_without_packets(esp_host):
    report=run_mock(esp_host)
    assert report['hardware_tested'] is False
    assert report['accepted_count'] > 0
    assert report['watchdog_tripped']
    assert report['fc_attempts']==0
    trace=next(t for t in report['traces'] if t['receiver_status']=='accepted')
    assert trace['accepted_pwm']==trace['issued_pwm']
    assert all(k in trace for k in ('network_pwm','coordinated_pwm','supervised_pwm'))
    assert report['roundtrip_us']['max']==10_000
    assert report['timing_source']=='synthetic_integer_microseconds'


def test_mock_twenty_ms_response_is_rejected_even_with_40ms_ttl(esp_host):
    report=run_mock(esp_host,uplink_us=12_000,service_us=4_000,downlink_us=4_000)
    assert report['accepted_count']==0
    assert report['reasons'].get('sequence_rejected',0)>0
    assert report['roundtrip_us']['max']==20_000


def test_bounded_inbox_overflow_is_visible_and_does_not_replace_old_frames():
    q=BoundedInbox(2)
    assert q.put('a') and q.put('b')
    assert not q.put('c')
    assert q.overflows==1 and len(q)==2
    assert q.get()=='a' and q.get()=='b'
    with pytest.raises(ValueError): BoundedInbox(0)


def test_pi_does_not_rebind_on_old_telemetry_or_invent_missing_manual_input():
    cfg=RunConfig.from_json(Path(__file__).resolve().parents[1]/'configs/mvp_runtime_v3.json')
    adapter=PiHardwareAdapter(cfg,session_id=SESSION,
        clock_mapping=ClockMapping(1_000_000,10.,0.,0.,11.))
    old=replace(sensor(),session_id='02000000000000000100000000000000')
    assert adapter.handle_sensor(encode_sensor(old),now_s=10.004) is None
    assert adapter.system.bridge.sequence==-1
    bad=replace(sensor(),local_mode='AUTO',manual_pwm=None)
    data,trace=adapter.handle_sensor(encode_sensor(bad),now_s=10.004)
    assert decode_command(data).lease is None
    assert trace.network_pwm is None and trace.fault is not None


def test_pi_invalid_tof_does_not_become_clear_space_or_advance_neural_state():
    cfg=RunConfig.from_json(Path(__file__).resolve().parents[1]/'configs/mvp_runtime_v3.json')
    adapter=PiHardwareAdapter(cfg,session_id=SESSION,
        clock_mapping=ClockMapping(1_000_000,10.,0.,0.,11.))
    s=sensor()
    invalid=replace(s,tof=replace(s.tof,valid=False,quality=0))
    data,trace=adapter.handle_sensor(encode_sensor(invalid),now_s=10.004)
    assert decode_command(data).lease is None and trace.network_pwm is None
    assert 'tof' in trace.fault


def test_mock_overload_has_bounded_queues_and_visible_losses(esp_host):
    report=run_mock(esp_host,service_us=100_000)
    assert report['inbox_overflows']>0
    assert report['accepted_count']==0 and report['fc_attempts']==0
    assert report['roundtrip_deadline_misses']>0
