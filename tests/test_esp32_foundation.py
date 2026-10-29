"""Portable application tests. No ESP timing, RF, sensor or FC qualification."""
import json
from pathlib import Path
import struct
import subprocess
import zlib

import pytest

from edge.hardware_wire import decode_sensor,decode_receipt,encode_command
from edge.flight_bridge import CommandLease
from edge.session_receiver import CommandPacket
from edge.manual_wire import ManualInput,decode_challenge,decode_manual,encode_manual

ROOT=Path(__file__).resolve().parents[1]
SESSION='01000000000000000100000000000000'
PWM=(1213.25,1499.5,1500.125,1501.75)


@pytest.fixture(scope='session')
def foundation_host(tmp_path_factory):
    out=tmp_path_factory.mktemp('foundation-host')
    result=subprocess.run(['powershell','-NoProfile','-File',str(ROOT/'firmware/esp32/build_host.ps1'),
        '-OutDir',str(out),'-Foundation'],capture_output=True,text=True,timeout=90)
    assert result.returncode==0,result.stdout+result.stderr
    return out/'foundation_host.exe'


def run(exe,actions,*,bench=True):
    result=subprocess.run([str(exe),'1',str(int(bench))],input='\n'.join(actions)+'\n',
        capture_output=True,text=True,timeout=10)
    assert result.returncode==0,result.stderr
    return [json.loads(line) for line in result.stdout.splitlines()]


def manual(seq=0,*,session=SESSION,valid=True,takeover=False):
    return encode_manual(ManualInput(session,seq,valid,takeover,True,PWM if valid else None)).hex()


def command(seq=0):
    return encode_command(CommandPacket(SESSION,seq,CommandLease(seq,PWM,.04))).hex()


def test_default_sensors_are_invalid_and_fc_output_is_absent(foundation_host):
    row=run(foundation_host,['reset 1000000','frame 0 1000000'])[-1]
    frame=decode_sensor(bytes.fromhex(row['sensor_hex']))
    assert frame.session_id==SESSION and frame.sequence==0 and frame.local_mode=='MANUAL'
    assert frame.manual_pwm is None and frame.selected_pwm is None
    for name in ('flow','tof','altitude','attitude'): assert not getattr(frame,name).valid
    challenge=decode_challenge(bytes.fromhex(row['challenge_hex']))
    assert challenge.session_id==SESSION and challenge.sequence==0 and challenge.issued_us==1000000
    assert row['fc_stage']=='not_attempted'


def test_manual_priority_and_pi_cannot_grant_auto(foundation_host):
    rows=run(foundation_host,['reset 0','frame 0 0',f'manual {manual()} 8000',f'pi {command()} 9000'])
    assert rows[-1]['mode']=='MANUAL' and rows[-1]['source']=='manual'
    assert rows[-1]['selected_pwm']==list(PWM) and rows[-1]['fc_stage']=='not_attempted'
    receipt=decode_receipt(bytes.fromhex(rows[-1]['receipt_hex']))
    assert receipt.accepted_pwm is None and receipt.selected_pwm==PWM


def test_auto_requires_local_grant_qualified_bench_and_fresh_manual(foundation_host):
    assert run(foundation_host,['reset 0','mode AUTO 0'])[-1]['mode']=='MANUAL'
    actions=['reset 0','frame 0 0',f'manual {manual()} 1000','mode AUTO 2000',f'pi {command()} 8000']
    row=run(foundation_host,actions,bench=False)[-1]
    assert row['mode']=='MANUAL' and row['source']=='manual'
    row=run(foundation_host,actions)[-1]
    assert row['mode']=='AUTO' and row['source']=='auto' and row['selected_pwm']==list(PWM)
    receipt=decode_receipt(bytes.fromhex(row['receipt_hex']))
    assert receipt.accepted_pwm==PWM and receipt.selected_session_id==SESSION


def test_manual_takeover_revokes_auto_without_mixing_sources(foundation_host):
    row=run(foundation_host,['reset 0','frame 0 0',f'manual {manual()} 1000','mode AUTO 2000',
        f'pi {command()} 8000','frame 1 20000',f'manual {manual(1,takeover=True)} 21000'])[-1]
    assert row['mode']=='MANUAL' and row['source']=='manual' and row['auto_tripped']


def test_watchdog_expiry_precedes_late_manual_and_requires_local_reset(foundation_host):
    rows=run(foundation_host,['reset 0','frame 0 0',f'manual {manual()} 1000',
        'frame 1 90000',f'manual {manual(1)} 100000','poll 100001','reset 100002'])
    assert rows[-3]['manual_tripped'] and rows[-3]['source']=='none'
    assert rows[-2]['selected_pwm'] is None
    assert rows[-1]['session_id']!=SESSION and not rows[-1]['manual_tripped']


def test_duplicate_and_wrong_session_manual_never_refresh_deadline(foundation_host):
    old='02000000000000000100000000000000'
    row=run(foundation_host,['reset 0','frame 0 0',f'manual {manual()} 1000',
        f'manual {manual()} 99000',f'manual {manual(session=old)} 99999','poll 100000'])[-1]
    assert row['source']=='none' and row['manual_tripped']


def test_previous_challenge_and_malformed_packets_do_not_create_manual(foundation_host):
    rows=run(foundation_host,['reset 0','frame 0 0','frame 1 20000',f'manual {manual()} 21000',
        'manual 00 22000'])
    assert rows[-2]['source']=='none'
    assert rows[-1]['manual_tripped'] and rows[-1]['source']=='none'


def test_held_capture_identity_and_bad_sensor_input(foundation_host):
    rows=run(foundation_host,['reset 0','sample flow 7 1000 1 255 0.1 -0.2 1000',
        'frame 0 1000','frame 1 20000',
        'sample flow 8 30000 1 255 0.3 -0.4 21000','frame 2 21000'])
    first=decode_sensor(bytes.fromhex(rows[2]['sensor_hex'])).flow
    held=decode_sensor(bytes.fromhex(rows[3]['sensor_hex'])).flow
    bad=decode_sensor(bytes.fromhex(rows[-1]['sensor_hex'])).flow
    assert first==held and first.capture_us==1000 and first.sample_id==7
    assert not bad.valid and bad.capture_us==1000 and bad.sample_id==7


def test_capture_angles_are_not_replaced_by_current_attitude(foundation_host):
    row=run(foundation_host,['reset 0','sample altitude 1 1000 1 200 0.9 0.1 -0.2 1000',
        'sample attitude 1 2000 1 200 0.4 -0.5 2000','frame 0 2000'])[-1]
    frame=decode_sensor(bytes.fromhex(row['sensor_hex']))
    assert frame.altitude.values==(.9,.1,-.2) and frame.attitude.values==(.4,-.5)


def test_link_loss_requires_manual_reset_and_never_writes_fc(foundation_host):
    row=run(foundation_host,['reset 0','frame 0 0',f'manual {manual()} 1000',
        'mode AUTO 2000',f'pi {command()} 8000','disconnect 9000'])[-1]
    assert row['source']=='none' and row['mode']=='MANUAL' and row['fc_stage']=='not_attempted'
    assert row['manual_tripped'] and row['auto_tripped']


@pytest.mark.parametrize('kind',['crc','version','flag','nan','range','short','extra','invalid_nonzero'])
def test_manual_wire_rejects_bad_frames(kind):
    raw=bytearray.fromhex(manual())
    if kind=='crc': raw[34]^=1
    if kind=='version': raw[4]=2
    if kind=='flag': raw[28]=2
    if kind=='nan': struct.pack_into('<d',raw,31,float('nan'))
    if kind=='range': struct.pack_into('<d',raw,31,999.)
    if kind=='invalid_nonzero': raw[28]=0
    if kind=='short': raw=raw[:-1]
    if kind=='extra': raw+=b'\0'
    if kind not in ('crc','short','extra'): struct.pack_into('<I',raw,len(raw)-4,zlib.crc32(raw[:-4]))
    with pytest.raises(ValueError): decode_manual(bytes(raw))


def test_invalid_manual_is_canonical_absence_not_neutral():
    message=ManualInput(SESSION,0,False,False,False,None)
    assert decode_manual(encode_manual(message))==message


@pytest.mark.parametrize('profile,error',[
    ('default',None),('wifi_missing','wifi_credentials_required'),
    ('same_peer','peer_conflict'),('same_port','local_port_conflict'),
    ('sensor','sensor_drivers_pending'),('fc','fc_adapter_pending'),
    ('reserved','reserved_pin'),('pin_conflict','pin_conflict'),
    ('untrusted_auto','trusted_link_required'),('manual_timeout','manual_timeout_bounds')])
def test_configuration_rejects_unresolved_or_conflicting_hardware(foundation_host,profile,error):
    assert run(foundation_host,[f'config {profile}'])[-1]['config_error']==error


def test_frame_sequence_never_wraps_or_reannounces_same_challenge(foundation_host):
    rows=run(foundation_host,['reset 0','frame 4294967295 0','frame 4294967295 1','frame 0 2'])
    assert rows[1]['sensor_hex']
    assert not rows[2]['sensor_hex'] and not rows[3]['sensor_hex']


def test_monotonic_clock_regression_cannot_be_cleared_by_session_reset(foundation_host):
    rows=run(foundation_host,['reset 100','frame 0 100',f'manual {manual()} 101','poll 99','reset 102'])
    assert rows[-1]['manual_tripped'] and rows[-1]['source']=='none'


def test_auto_lease_expiry_does_not_silently_switch_to_manual(foundation_host):
    row=run(foundation_host,['reset 0','frame 0 0',f'manual {manual()} 1000','mode AUTO 2000',
        f'pi {command()} 8000','poll 40000'])[-1]
    assert row['auto_tripped'] and row['source']=='none' and row['selected_pwm'] is None


def test_stalled_frame_publication_cannot_extend_pi_response_window(foundation_host):
    rows=run(foundation_host,['reset 0','frame 0 0',f'manual {manual()} 1000','mode AUTO 2000',
        f'pi {command()} 20000','mode MANUAL 21000','mode AUTO 22000','reset 23000'])
    receipt=decode_receipt(bytes.fromhex(rows[-4]['receipt_hex']))
    assert receipt.accepted_pwm is None and receipt.reason=='sequence_rejected' and receipt.tripped
    assert rows[-4]['source']=='none' and rows[-4]['auto_tripped']
    assert rows[-3]['source']=='manual' and rows[-2]['mode']=='MANUAL'
    assert rows[-1]['session_id']!=SESSION


def test_delayed_old_session_pi_cannot_trip_new_session_even_after_frame_window(foundation_host):
    old='02000000000000000100000000000000'
    delayed=encode_command(CommandPacket(old,0,CommandLease(0,PWM,.04))).hex()
    row=run(foundation_host,['reset 0','frame 0 0',f'manual {manual()} 1000','mode AUTO 2000',
        f'pi {command()} 8000',f'pi {delayed} 25000'])[-1]
    receipt=decode_receipt(bytes.fromhex(row['receipt_hex']))
    assert receipt.reason=='session_mismatch' and not receipt.tripped
    assert row['source']=='auto' and not row['auto_tripped']
