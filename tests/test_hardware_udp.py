"""Real UDP sockets on this host; synthetic control time, no Wi-Fi or FC."""
from dataclasses import replace
from pathlib import Path
import socket
import time

import pytest

from edge.hardware_udp import UdpEndpoint, PiUdpPump
from edge.hardware_udp_probe import run_probe
from edge.hardware_wire import ClockMapping, encode_sensor, encode_receipt, ReceiptMessage
from edge.pi_hardware_adapter import PiHardwareAdapter
from edge.system import RunConfig
from tests.test_hardware_wire import SESSION, sensor
from tests.test_esp32_host import esp_host


def pair(*,budget=8):
    a=UdpEndpoint(receive_budget=budget)
    b=UdpEndpoint(receive_budget=budget)
    a.set_peer(b.address)
    b.set_peer(a.address)
    return a,b


def wait_packets(endpoint):
    deadline=time.monotonic()+2.
    while time.monotonic()<deadline:
        packets=endpoint.poll()
        if packets: return packets
        time.sleep(.001)
    raise AssertionError('No UDP datagram received on loopback')


def adapter():
    config=RunConfig.from_json(Path(__file__).resolve().parents[1]/'configs/mvp_runtime_v3.json')
    return PiHardwareAdapter(config,session_id=SESSION,
        clock_mapping=ClockMapping(1_000_000,10.,0.,0.,20.))


def test_datagram_bytes_are_exact_and_foreign_peer_cannot_become_bound():
    a,b=pair()
    stranger=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    try:
        stranger.sendto(b'foreign',b.address)
        assert a.send(b'CFHW-same-bytes')
        assert wait_packets(b)==[b'CFHW-same-bytes']
        assert b.counters['peer_rejected']==1
        with pytest.raises(ValueError): b.set_peer(stranger.getsockname())
    finally:
        a.close(); b.close(); stranger.close()


def test_receive_scan_is_bounded_including_foreign_flood():
    a,b=pair(budget=2)
    stranger=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    try:
        for _ in range(5): stranger.sendto(b'foreign',b.address)
        assert a.send(b'last')
        assert b.poll()==[]
        assert b.counters['peer_rejected']==2
        assert b.counters['scan_budget_exhausted']==1
        assert wait_packets(b)==[b'last']
        assert b.counters['peer_rejected']==5
    finally:
        a.close(); b.close(); stranger.close()


def test_oversize_datagram_is_rejected_instead_of_truncated_into_valid_frame():
    a,b=pair()
    try:
        # Bypass send validation to emulate an invalid sender on the paired port.
        a.socket.sendto(b'x'*2048,b.address)
        assert a.send(b'good')
        assert wait_packets(b)==[b'good']
        assert b.counters['oversize']==1
        with pytest.raises(ValueError): a.send(b'x'*1025)
    finally:
        a.close(); b.close()


def test_udp_preserves_duplicates_for_the_reference_replay_checks():
    a,b=pair()
    try:
        assert a.send(b'replay') and a.send(b'replay')
        assert wait_packets(b)==[b'replay',b'replay']
    finally:
        a.close(); b.close()


def test_pi_udp_ignores_bad_packets_and_old_session_without_rebinding():
    a,b=pair()
    try:
        pump=PiUdpPump(adapter(),b)
        a.send(b'bad')
        a.send(encode_sensor(replace(sensor(),session_id='02000000000000000100000000000000')))
        a.send(encode_sensor(sensor()))
        traces=pump.step(now_s=10.004)
        assert len(traces)==1 and traces[0].issued_pwm is None
        assert pump.counters['malformed']==1 and pump.counters['session_ignored']==1
        assert len(wait_packets(a))==1
        assert pump.adapter.system.bridge.sequence==0
    finally:
        a.close(); b.close()


def test_expired_clock_mapping_sends_no_command_and_does_not_change_neural_state():
    a,b=pair()
    try:
        pump=PiUdpPump(adapter(),b)
        before=pump.adapter.system.runtime.policy.h.copy()
        a.send(encode_sensor(sensor()))
        assert pump.step(now_s=20.001)==[]
        assert pump.counters['adapter_error']==1
        assert a.poll()==[]
        assert (pump.adapter.system.runtime.policy.h==before).all()
    finally:
        a.close(); b.close()


def test_full_length_corruption_is_a_decode_error_not_a_clock_error():
    a,b=pair()
    try:
        pump=PiUdpPump(adapter(),b)
        data=bytearray(encode_sensor(sensor()))
        data[-1]^=1
        a.send(bytes(data))
        assert pump.step(now_s=10.004)==[]
        assert pump.counters['malformed']==1 and pump.counters['adapter_error']==0
        assert a.poll()==[]
    finally:
        a.close(); b.close()


def test_receipt_logs_are_bounded_and_do_not_grant_authority():
    a,b=pair()
    try:
        pump=PiUdpPump(adapter(),b,receipt_capacity=2)
        for seq in range(3):
            r=ReceiptMessage(SESSION,seq,1_000_000+seq,'awaiting_command',False,
                             'SHADOW',None,None,'none','not_attempted')
            a.send(encode_receipt(r))
            assert pump.step(now_s=10.004)==[]
        assert len(pump.receipts)==2 and pump.counters['receipt_evicted']==1
        assert pump.adapter.system.bridge.state=='MANUAL'
    finally:
        a.close(); b.close()


def test_udp_loopback_crosses_real_system_and_cpp_receiver_without_fc(esp_host):
    report=run_probe(esp_host)
    assert report['transport']=='host_udp_loopback'
    assert report['hardware_tested'] is False and report['fc_attempts']==0
    assert report['accepted_count']==10
    assert report['command_clock']=='synthetic_20ms_ticks'
    assert report['watchdog_tripped']
    assert report['host_wall_roundtrip_us']['max']>=0
    assert all(r['fc_stage']=='not_attempted' for r in report['receipts'])
    accepted=[t for t in report['traces'] if t['receiver_status']=='accepted']
    assert len(accepted)==10 and all(t['accepted_pwm']==t['issued_pwm'] for t in accepted)
