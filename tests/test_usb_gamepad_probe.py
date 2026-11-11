"""Synthetic descriptor/report host tests; not a raw GPW70 capture."""
import json
from pathlib import Path
import struct
import subprocess

import pytest

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='session')
def usb_probe_host(tmp_path_factory):
    out=tmp_path_factory.mktemp('usb-probe-host')
    result=subprocess.run(['powershell','-NoProfile','-File',str(ROOT/'firmware/esp32/build_host.ps1'),
        '-OutDir',str(out),'-Probe'],capture_output=True,text=True,timeout=90)
    assert result.returncode==0,result.stdout+result.stderr
    return out/'usb_probe_host.exe'


def run(exe,kind,raw):
    result=subprocess.run([str(exe),kind,raw.hex()],capture_output=True,text=True,timeout=10)
    assert result.returncode==0,result.stderr
    return json.loads(result.stdout)


def configuration(*,cls=0xff,sub=0x5d,protocol=1,packet_size=32):
    # One alternate-zero interface with one interrupt IN endpoint.
    return bytes.fromhex('090219000101008032')+bytes([9,4,0,0,1,cls,sub,protocol,0])+bytes(
        [7,5,0x81,3,packet_size,0,4])


def test_supported_interface_is_selected_from_explicit_descriptor(usb_probe_host):
    result=run(usb_probe_host,'config',configuration())
    assert result==dict(valid=True,found=True,interface=0,alternate=0,endpoint=0x81,packet_size=32)


@pytest.mark.parametrize('kind',['short','zero_length','overflow','bad_total','out_only','bulk','huge_packet','duplicate_in'])
def test_malformed_or_unsupported_descriptors_never_select_endpoint(usb_probe_host,kind):
    raw=bytearray(configuration())
    if kind=='short': raw=raw[:-1]
    if kind=='zero_length': raw[9]=0
    if kind=='overflow': raw[9]=255
    if kind=='bad_total': raw[2]=26
    if kind=='out_only': raw[20]=1
    if kind=='bulk': raw[21]=2
    if kind=='huge_packet': raw[22:24]=b'\0\x04'
    if kind=='duplicate_in':
        raw[13]=2
        raw+=raw[-7:]
        raw[2]=len(raw)
    assert not run(usb_probe_host,'config',raw)['found']


def test_hid_and_other_xusb_protocols_are_not_silently_treated_as_this_device(usb_probe_host):
    for raw in (configuration(cls=3,sub=0,protocol=0),configuration(protocol=0x81)):
        assert not run(usb_probe_host,'config',raw)['found']


def report(*,lx=0,ly=0,rx=0,ry=0,lt=0,rt=0,buttons=0):
    return b'\x00\x14'+struct.pack('<HBBhhhh',buttons,lt,rt,lx,ly,rx,ry)+bytes(6)


def test_xbox360_candidate_packet_decodes_exact_raw_values_without_rc_mapping(usb_probe_host):
    result=run(usb_probe_host,'report',report(lx=-32768,ly=32767,rx=-123,ry=456,lt=1,rt=255,buttons=0x1001))
    assert result==dict(valid=True,buttons=0x1001,lt=1,rt=255,lx=-32768,ly=32767,rx=-123,ry=456)


@pytest.mark.parametrize('raw',[b'',bytes(20),report()[:-1],report()+b'\0',b'\x01'+report()[1:]])
def test_bad_input_reports_do_not_become_controls(usb_probe_host,raw):
    assert run(usb_probe_host,'report',raw)==dict(valid=False)
