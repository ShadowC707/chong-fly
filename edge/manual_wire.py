"""CFMI v1 bench manual input. No pairing, authority grant or axis calibration.

Reply once to the ESP-issued challenge; no notebook clock is compared to ESP.
Transport identity and CRC are not authentication. See ESP32_FOUNDATION_UK.md.
"""
from dataclasses import dataclass
import math
import struct
import zlib

from .hardware_wire import HEADER,_uint


@dataclass(frozen=True)
class ManualChallenge:
    session_id: str
    sequence: int
    issued_us: int


@dataclass(frozen=True)
class ManualInput:
    session_id: str
    sequence: int
    valid: bool
    takeover: bool
    stop_requested: bool
    pwm: tuple | None


def _frame(kind,session,seq,body):
    try: sid=bytes.fromhex(session)
    except (TypeError,ValueError) as exc: raise ValueError('Invalid manual session') from exc
    if len(sid)!=16 or session!=sid.hex() or not any(sid[:8]) or not any(sid[8:]):
        raise ValueError('Nonzero canonical boot epoch and generation required')
    data=HEADER.pack(b'CFMI',1,kind,32+len(body),sid,_uint(seq,32))+body
    return data+struct.pack('<I',zlib.crc32(data))


def _unframe(data,kind,size):
    if not isinstance(data,bytes) or len(data)!=32+size: raise ValueError('Invalid manual length')
    magic,version,actual,length,sid,seq=HEADER.unpack_from(data)
    if (magic!=b'CFMI' or version!=1 or actual!=kind or length!=len(data)
            or not any(sid[:8]) or not any(sid[8:])
            or zlib.crc32(data[:-4])!=struct.unpack_from('<I',data,len(data)-4)[0]):
        raise ValueError('Invalid manual header/session/CRC')
    return sid.hex(),seq,data[28:-4]


def encode_challenge(message):
    return _frame(1,message.session_id,message.sequence,struct.pack('<Q',_uint(message.issued_us,64)))


def decode_challenge(data):
    sid,seq,body=_unframe(data,1,8)
    return ManualChallenge(sid,seq,struct.unpack('<Q',body)[0])


def encode_manual(message):
    flags=(message.valid,message.takeover,message.stop_requested)
    if any(type(f) is not bool for f in flags): raise ValueError('Boolean manual flags required')
    if message.valid:
        try: pwm=tuple(float(x) for x in message.pwm)
        except (TypeError,ValueError,OverflowError) as exc: raise ValueError('Invalid manual RC') from exc
        if len(pwm)!=4 or any(not math.isfinite(x) or not 1000<=x<=2000 for x in pwm):
            raise ValueError('Manual RC outside finite 1000..2000')
    else:
        if message.pwm is not None: raise ValueError('Invalid manual input has no RC')
        pwm=(0.,)*4
    return _frame(2,message.session_id,message.sequence,struct.pack('<BBB4d',*flags,*pwm))


def decode_manual(data):
    sid,seq,body=_unframe(data,2,35)
    valid,takeover,stop,*pwm=struct.unpack('<BBB4d',body)
    if max(valid,takeover,stop)>1: raise ValueError('Invalid manual flag')
    result=ManualInput(sid,seq,bool(valid),bool(takeover),bool(stop),tuple(pwm) if valid else None)
    if encode_manual(result)!=data: raise ValueError('Noncanonical manual payload')
    return result
