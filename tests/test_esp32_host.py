"""Compile the portable ESP receiver core for Windows; this is not an ESP build."""
import json
from pathlib import Path
import subprocess

import pytest

from edge.flight_bridge import CommandLease
from edge.session_receiver import CommandPacket, SessionReceiver
from edge.hardware_wire import encode_command
from tests.test_hardware_wire import recalc

ROOT=Path(__file__).resolve().parents[1]
SID='01000000000000000100000000000000'
PWM=(1213.,1500.,1500.,1500.)


@pytest.fixture(scope='session')
def esp_host(tmp_path_factory):
    out=tmp_path_factory.mktemp('esp-host')
    result=subprocess.run(['powershell','-NoProfile','-File',str(ROOT/'firmware/esp32/build_host.ps1'),
        '-OutDir',str(out)],capture_output=True,text=True,timeout=90)
    assert result.returncode==0,result.stdout+result.stderr
    return out/'receiver_host.exe'


def run(exe, actions, *, epoch=1, counter=0):
    result=subprocess.run([str(exe),str(epoch),str(counter)], input='\n'.join(actions)+'\n',
                          capture_output=True,text=True,timeout=10)
    assert result.returncode==0,result.stderr
    return [json.loads(line) for line in result.stdout.splitlines()]


def packet(seq=0,session=SID,lease=True):
    return encode_command(CommandPacket(session,seq,CommandLease(seq,PWM,.04) if lease else None)).hex()


@pytest.mark.parametrize('kind',['expiry','duplicate','no_lease','duplicate_no_lease','wrong_sequence',
    'old_session','manual','shadow','gap','deadline_boundary'])
def test_port_decisions_match_reference(esp_host,kind):
    ref=SessionReceiver()
    actions=['reset 1000000','mode AUTO 1000000','sensor 0 1000000']
    ref.reset(session_id=SID,now_s=1.)
    ref.set_mode('AUTO',now_s=1.)
    ref.announce_sensor(0,now_s=1.)
    empty=kind=='duplicate_no_lease'
    actions.append(f'receive {packet(lease=not empty)} 1008000')
    ref.receive(CommandPacket(SID,0,None if empty else CommandLease(0,PWM,.04)),now_s=1.008)
    if kind in ('expiry','deadline_boundary'):
        now=1.04
        actions.append('poll 1040000')
    elif kind=='manual' or kind=='shadow':
        now=1.01
        actions.append(f'mode {kind.upper()} 1010000')
        ref.set_mode(kind.upper(),now_s=now)
    else:
        now=1.048 if kind=='gap' else 1.012
        seq=1 if kind in ('no_lease','wrong_sequence','gap') else 0
        if seq:
            sent=1.04 if kind=='gap' else 1.01
            actions.append(f'sensor 1 {round(sent*1e6)}')
            ref.output(now_s=sent,fallback_pwm=PWM)
            ref.announce_sensor(1,now_s=sent)
        ref.output(now_s=now,fallback_pwm=PWM)
        session='02000000000000000100000000000000' if kind=='old_session' else SID
        response_seq=0 if kind=='wrong_sequence' else seq
        lease=kind not in ('no_lease','duplicate_no_lease')
        actions.append(f'receive {packet(response_seq,session,lease)} {round(now*1e6)}')
        event=ref.receive(CommandPacket(session,response_seq,
            CommandLease(response_seq,PWM,.04) if lease else None),now_s=now)
    selected=ref.output(now_s=now,fallback_pwm=PWM)
    actual=run(esp_host,actions)[-1]
    assert actual['tripped']==selected['tripped']
    assert actual['selected_pwm']==(None if selected['accepted_pwm'] is None else list(selected['accepted_pwm']))
    expected=event.reason if kind=='old_session' else selected['reason']
    assert actual['reason']==expected


def test_reset_new_identity_and_delayed_old_packet(esp_host):
    actions=['reset 0','mode AUTO 0','sensor 0 0',f'receive {packet()} 8000',
        'mode MANUAL 9000','reset 9000','mode AUTO 9000','sensor 0 9000',
        f'receive {packet(session="01000000000000000200000000000000")} 17000',
        f'receive {packet()} 18000']
    rows=run(esp_host,actions)
    assert rows[-1]['reason']=='session_mismatch' and not rows[-1]['tripped']
    assert rows[-1]['selected_pwm']==list(PWM)
    reboot=run(esp_host,['reset 0','mode AUTO 0','sensor 0 0',f'receive {packet()} 8000'],epoch=2)
    assert reboot[-1]['reason']=='session_mismatch' and not reboot[-1]['tripped']


def test_finite_counters_do_not_wrap_and_remote_bytes_cannot_reset(esp_host):
    rows=run(esp_host,['reset 0','mode AUTO 0','reset 1','sensor 4294967295 2',
        f'receive {packet(0xffffffff)} 3','sensor 0 4'])
    assert not rows[2]['tripped'] and rows[2]['reason']=='reset_requires_manual'
    assert rows[-1]['tripped'] and rows[-1]['reason']=='invalid_sensor_sequence'
    exhausted=run(esp_host,['reset 0'],counter=2**64-1)
    assert exhausted[-1]['tripped'] and exhausted[-1]['reason']=='session_exhausted'
    malformed=run(esp_host,['reset 0','mode AUTO 0','sensor 0 0','receive 00 1'])
    assert malformed[-1]['reason']=='malformed_packet' and malformed[-1]['tripped']


def test_cpp_decodes_shared_golden_packets(esp_host):
    fixtures=json.loads((ROOT/'tests/fixtures/esp32_wire_v1.json').read_text())
    for row in fixtures['commands']:
        result=run(esp_host,['reset 0','mode AUTO 0',f'sensor {row["sequence"]} 0',
            f'receive {row["hex"]} 8000'])[-1]
        assert result['accepted_pwm']==row['pwm']


def test_old_session_event_does_not_replace_current_reason_or_active_identity(esp_host):
    rows=run(esp_host,['reset 0','mode AUTO 0','sensor 0 0',f'receive {packet()} 8000',
        f'receive {packet(session="02000000000000000100000000000000")} 9000','poll 10000'])
    assert rows[-2]['reason']=='session_mismatch'
    assert rows[-1]['reason']=='accepted'
    assert rows[-1]['selected_session_id']==SID and rows[-1]['selected_sequence']==0


def test_clock_regression_and_uint64_deadline_overflow_fail_closed(esp_host):
    rows=run(esp_host,['reset 100','mode AUTO 100','sensor 0 100','poll 99'])
    assert rows[-1]['tripped'] and rows[-1]['reason']=='invalid_receiver_clock'
    now=2**64-100
    rows=run(esp_host,[f'reset {now}',f'mode AUTO {now}',f'sensor 0 {now}',f'receive {packet()} {now}'])
    assert rows[-1]['tripped'] and rows[-1]['reason']=='lease_rejected'


@pytest.mark.parametrize('kind',['crc','version','length','flag','ttl','nan','range','truncated','extra'])
def test_cpp_parser_rejects_bad_bytes_without_refreshing_command(esp_host,kind):
    import struct
    data=bytearray.fromhex(packet())
    if kind=='crc': data[35]^=1
    if kind=='version': data[4]=2
    if kind=='length': data[6]=0
    if kind=='flag': data[28]=2
    if kind=='ttl': struct.pack_into('<I',data,29,40001)
    if kind=='nan': struct.pack_into('<d',data,33,float('nan'))
    if kind=='range': struct.pack_into('<d',data,33,2001.)
    if kind=='truncated': data=data[:-1]
    if kind=='extra': data+=b'\0'
    if kind not in ('crc','truncated','extra'): data=recalc(data)
    row=run(esp_host,['reset 0','mode AUTO 0','sensor 0 0',f'receive {packet()} 8000',
                       f'receive {data.hex()} 9000'])[-1]
    assert row['tripped'] and row['reason']=='malformed_packet' and row['selected_pwm'] is None
