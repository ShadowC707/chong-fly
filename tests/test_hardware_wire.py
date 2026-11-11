"""Host-only protocol evidence; no ESP, Pi board or flight controller attached."""
from dataclasses import replace
import json
from pathlib import Path
import struct
import zlib

import pytest

from edge.flight_bridge import CommandLease
from edge.session_receiver import CommandPacket
from edge.hardware_wire import (
    SensorMessage, WireMeasurement, ReceiptMessage, ClockMapping,
    encode_command, decode_command, encode_sensor, decode_sensor,
    encode_receipt, decode_receipt, to_snapshot,
)

SESSION = '01000000000000000100000000000000'
GOLDEN = Path(__file__).parent/'fixtures'/'esp32_wire_v1.json'


def sensor(sequence=0, created_us=1_000_000):
    def m(n, values):
        return WireMeasurement(n, created_us, True, 255, values)
    return SensorMessage(SESSION, sequence, created_us, 'SHADOW', False, False,
        (1213., 1500., 1500., 1500.), None, 'none',
        m(sequence, (0., 0.)), m(sequence, (1.,)*64),
        m(sequence, (.982, 0., 0.)), m(sequence, (0., 0.)))


def recalc(data):
    return data[:-4]+struct.pack('<I', zlib.crc32(data[:-4]))


def test_golden_command_bytes_and_exact_float_channels():
    fixtures = json.loads(GOLDEN.read_text())
    for row in fixtures['commands']:
        lease = None if row['pwm'] is None else CommandLease(row['sequence'], tuple(row['pwm']), .04)
        command = CommandPacket(SESSION, row['sequence'], lease)
        encoded = encode_command(command)
        assert encoded.hex() == row['hex']
        assert decode_command(encoded) == command


@pytest.mark.parametrize('kind', ['truncate','extra','crc','version','type','length','flags','nan','ttl','range'])
def test_command_rejects_invalid_frames(kind):
    data = bytearray(encode_command(CommandPacket(SESSION, 7, CommandLease(7,(1213.,1500.,1500.,1500.),.04))))
    if kind == 'truncate': data = data[:-1]
    if kind == 'extra': data += b'\0'
    if kind == 'crc': data[35] ^= 1
    if kind == 'version': data[4] = 2
    if kind == 'type': data[5] = 1
    if kind == 'length': data[6] ^= 1
    if kind == 'flags': data[28] = 2
    if kind == 'nan': struct.pack_into('<d',data,33,float('nan'))
    if kind == 'ttl': struct.pack_into('<I',data,29,40001)
    if kind == 'range': struct.pack_into('<d',data,33,999.)
    if kind not in ('truncate','extra','crc'): data = recalc(data)
    with pytest.raises(ValueError): decode_command(bytes(data))


def test_no_lease_has_one_canonical_representation_and_finite_widths():
    p = CommandPacket(SESSION, 0xffffffff, None)
    assert decode_command(encode_command(p)) == p
    for bad in (replace(p, sequence=2**32), replace(p, sequence=-1),
                replace(p, session_id='one'), replace(p, session_id='00'*16),
                replace(p, lease=CommandLease(1,(1213.,1500.,1500.,1500.),.04))):
        with pytest.raises(ValueError): encode_command(bad)
    data=bytearray(encode_command(p))
    data[29]=1
    with pytest.raises(ValueError): decode_command(recalc(data))


def test_sensor_roundtrip_preserves_held_capture_angles_invalid_and_stop():
    first=sensor()
    held=replace(sensor(1,1_020_000), altitude=first.altitude,
                 attitude=WireMeasurement(1,1_020_000,True,200,(.1,.2)),
                 tof=replace(first.tof,valid=False,quality=0), stop_requested=True)
    decoded=decode_sensor(encode_sensor(held))
    assert decoded==held
    mapping=ClockMapping(1_000_000,10.,0.,0.,11.)
    snapshot=to_snapshot(decoded,mapping,now_s=10.024)
    assert snapshot.altitude.timestamp_s==10.
    assert snapshot.altitude.value==(.982,0.,0.)
    assert snapshot.attitude.value==(.1,.2)
    assert not snapshot.tof.valid


def test_clock_mapping_requires_valid_bound_and_uses_oldest_possible_time():
    mapping=ClockMapping(1_000_000,10.,.001,100.,11.)
    snap=to_snapshot(sensor(1,1_020_000),mapping,now_s=10.025)
    assert snap.flow.timestamp_s==pytest.approx(10.018998)
    with pytest.raises(ValueError,match='expired'):
        to_snapshot(sensor(),mapping,now_s=11.001)
    with pytest.raises(ValueError): ClockMapping(0,10.,-.001,0.,11.)
    future=replace(sensor(),altitude=replace(sensor().altitude,capture_us=1_000_001))
    with pytest.raises(ValueError): encode_sensor(future)


def test_receipt_distinguishes_acceptance_active_selection_and_fc_attempt():
    r=ReceiptMessage(SESSION,7,1_008_000,'accepted',False,'AUTO',
        (1213.,1500.,1500.,1500.),(1213.,1500.,1500.,1500.),'auto','not_attempted',SESSION,7)
    assert decode_receipt(encode_receipt(r))==r
    expired=replace(r,receiver_us=1_040_000,reason='watchdog_expired',tripped=True,
                    accepted_pwm=None,selected_pwm=None,source='none',
                    selected_session_id=None,selected_sequence=None)
    assert decode_receipt(encode_receipt(expired))==expired
    assert r.accepted_pwm is not None


@pytest.mark.parametrize('change', [dict(tof=WireMeasurement(0,1_000_000,True,255,(1.,)*63)),
    dict(local_mode='ARM'),dict(source='auto'),dict(stop_requested=1)])
def test_sensor_rejects_shapes_flags_and_inconsistent_selection(change):
    with pytest.raises(ValueError): encode_sensor(replace(sensor(),**change))


def test_full_message_golden_packets_are_stable():
    fixtures=json.loads(GOLDEN.read_text())
    assert encode_sensor(sensor()).hex()==fixtures['sensor_hex']
    receipt=ReceiptMessage(SESSION,7,1_008_000,'accepted',False,'AUTO',
        (1213.25,1499.5,1500.125,1501.75),(1213.25,1499.5,1500.125,1501.75),
        'auto','not_attempted',SESSION,7)
    assert encode_receipt(receipt).hex()==fixtures['receipt_hex']
