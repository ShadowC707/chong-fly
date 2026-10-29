"""Bench wire v1, transport independent. No device I/O or authority allocation.

Little endian, explicit finite widths, IEEE-754 binary64, CRC32/ISO-HDLC.
Sensor values already follow SensorSnapshot units/axes; no PMW calibration or
clock estimator is supplied here. See docs/ESP32_WIRE_V1_UK.md.
"""
from dataclasses import dataclass
import math
import re
import struct
import zlib

from .flight_bridge import CommandLease
from .session_receiver import CommandPacket
from .sensors import Measurement, SensorSnapshot

HEADER=struct.Struct('<4sBBH16sI')
MAX_FRAME=1024
MODES=('MANUAL','SHADOW','AUTO')
SOURCES=('none','manual','auto','fallback')
FC_STAGES=('not_attempted','write_attempted','confirmed','unknown')
REASONS=('awaiting_command','accepted','session_mismatch','sequence_rejected',
    'sender_revoked','malformed_lease','lease_rejected','watchdog_expired',
    'authority_revoked','invalid_receiver_clock','invalid_sensor_sequence',
    'malformed_packet','reset_required','reset_requires_manual','session_exhausted','invalid_mode')


def _uint(value,bits):
    if type(value) is not int or not 0<=value<2**bits:
        raise ValueError(f'Expected uint{bits}')
    return value


def _flag(value):
    if type(value) is not bool: raise ValueError('Expected boolean flag')
    return int(value)


def _values(values,count,*,rc=False):
    try: result=tuple(float(x) for x in values)
    except (TypeError,ValueError,OverflowError) as exc: raise ValueError('Invalid values') from exc
    if len(result)!=count or not all(math.isfinite(x) for x in result):
        raise ValueError('Invalid finite payload shape')
    if rc and not all(1000<=x<=2000 for x in result): raise ValueError('RC outside 1000..2000')
    return result


def _enum(value,items):
    try: return items.index(value)
    except ValueError as exc: raise ValueError('Unknown enum') from exc


def _frame(kind,session,sequence,body):
    if (not isinstance(session,str) or re.fullmatch('[0-9a-f]{32}',session) is None
            or int(session[:16],16)==0 or int(session[16:],16)==0):
        raise ValueError('Session must contain nonzero boot epoch and generation')
    length=HEADER.size+len(body)+4
    if length>MAX_FRAME: raise ValueError('Oversize frame')
    data=HEADER.pack(b'CFHW',1,kind,length,bytes.fromhex(session),_uint(sequence,32))+body
    return data+struct.pack('<I',zlib.crc32(data))


def _unframe(data,kind,body_size):
    if not isinstance(data,bytes) or len(data)!=HEADER.size+body_size+4 or len(data)>MAX_FRAME:
        raise ValueError('Truncated, extra or oversize frame')
    magic,version,actual,length,session,sequence=HEADER.unpack_from(data)
    if (magic!=b'CFHW' or version!=1 or actual!=kind or length!=len(data)
            or not any(session[:8]) or not any(session[8:])
            or zlib.crc32(data[:-4])!=struct.unpack_from('<I',data,len(data)-4)[0]):
        raise ValueError('Invalid header/version/type/length/session/CRC')
    return session.hex(),sequence,data[HEADER.size:-4]


COMMAND=struct.Struct('<BI4d')


def encode_command(packet):
    if not isinstance(packet,CommandPacket): raise ValueError('CommandPacket required')
    if packet.lease is None:
        body=COMMAND.pack(0,0,0.,0.,0.,0.)
    else:
        lease=packet.lease
        if not isinstance(lease,CommandLease) or lease.sequence!=packet.sequence:
            raise ValueError('Lease identity differs')
        ttl=lease.ttl_s*1_000_000
        if not math.isfinite(ttl) or not 1<=ttl<=40000 or abs(ttl-round(ttl))>1e-6:
            raise ValueError('TTL must be whole microseconds, 1..40000')
        body=COMMAND.pack(1,round(ttl),*_values(lease.pwm,4,rc=True))
    return _frame(2,packet.session_id,packet.sequence,body)


def decode_command(data):
    session,sequence,body=_unframe(data,2,COMMAND.size)
    present,ttl,*pwm=COMMAND.unpack(body)
    if present==0:
        if body!=bytes(COMMAND.size): raise ValueError('Noncanonical no-lease')
        lease=None
    elif present==1 and 1<=ttl<=40000:
        lease=CommandLease(sequence,_values(pwm,4,rc=True),ttl/1_000_000)
    else: raise ValueError('Invalid lease flag/TTL')
    return CommandPacket(session,sequence,lease)


@dataclass(frozen=True)
class WireMeasurement:
    sample_id: int
    capture_us: int
    valid: bool
    quality: int  # 0..255 producer diagnostic, not a universal admission threshold
    values: tuple


@dataclass(frozen=True)
class SensorMessage:
    session_id: str
    sequence: int
    created_us: int
    local_mode: str
    stop_requested: bool
    receiver_tripped: bool
    manual_pwm: tuple | None
    selected_pwm: tuple | None
    source: str
    flow: WireMeasurement
    tof: WireMeasurement
    altitude: WireMeasurement
    attitude: WireMeasurement


SENSOR_PREFIX=struct.Struct('<QBBBBB8d')
MEASUREMENT=struct.Struct('<IQBB')
CHANNELS=(('flow',2),('tof',64),('altitude',3),('attitude',2))
SENSOR_SIZE=SENSOR_PREFIX.size+sum(MEASUREMENT.size+count*8 for _,count in CHANNELS)


def _selection(selected,source):
    _enum(source,SOURCES)
    if (selected is None)!=(source=='none'): raise ValueError('Selection/source mismatch')
    return (0.,)*4 if selected is None else _values(selected,4,rc=True)


def encode_sensor(message):
    if not isinstance(message,SensorMessage): raise ValueError('SensorMessage required')
    manual=(0.,)*4 if message.manual_pwm is None else _values(message.manual_pwm,4,rc=True)
    selected=_selection(message.selected_pwm,message.source)
    body=SENSOR_PREFIX.pack(_uint(message.created_us,64),_enum(message.local_mode,MODES),
        _flag(message.stop_requested),_flag(message.receiver_tripped),_enum(message.source,SOURCES),
        int(message.manual_pwm is not None),*manual,*selected)
    for name,count in CHANNELS:
        m=getattr(message,name)
        if not isinstance(m,WireMeasurement) or m.capture_us>message.created_us:
            raise ValueError('Invalid capture time')
        values=_values(m.values,count)
        if name=='tof' and not all(0<=v<=1 for v in values): raise ValueError('ToF not normalized')
        body+=MEASUREMENT.pack(_uint(m.sample_id,32),_uint(m.capture_us,64),_flag(m.valid),
                               _uint(m.quality,8))+struct.pack('<'+str(count)+'d',*values)
    return _frame(1,message.session_id,message.sequence,body)


def decode_sensor(data):
    session,sequence,body=_unframe(data,1,SENSOR_SIZE)
    created,mode,stop,tripped,source,manual_valid,*values=SENSOR_PREFIX.unpack_from(body)
    if mode>=len(MODES) or source>=len(SOURCES) or max(stop,tripped,manual_valid)>1:
        raise ValueError('Invalid sensor flags')
    channels={}
    offset=SENSOR_PREFIX.size
    for name,count in CHANNELS:
        sid,capture,valid,quality=MEASUREMENT.unpack_from(body,offset)
        if valid>1: raise ValueError('Invalid measurement flag')
        offset+=MEASUREMENT.size
        v=struct.unpack_from('<'+str(count)+'d',body,offset)
        offset+=count*8
        channels[name]=WireMeasurement(sid,capture,bool(valid),quality,v)
    result=SensorMessage(session,sequence,created,MODES[mode],bool(stop),bool(tripped),
        tuple(values[:4]) if manual_valid else None,tuple(values[4:]) if source else None,
        SOURCES[source],**channels)
    # Re-encoding validates ranges and canonical absent-value bytes too.
    if encode_sensor(result)!=data: raise ValueError('Noncanonical sensor payload')
    return result


@dataclass(frozen=True)
class ReceiptMessage:
    session_id: str  # identity of the received command, even on session_mismatch
    sequence: int
    receiver_us: int
    reason: str
    tripped: bool
    local_mode: str
    accepted_pwm: tuple | None  # historical event, not current authority
    selected_pwm: tuple | None
    source: str
    fc_stage: str
    selected_session_id: str | None = None
    selected_sequence: int | None = None


RECEIPT=struct.Struct('<QBBBBBB16sI8d')


def encode_receipt(r):
    accepted=(0.,)*4 if r.accepted_pwm is None else _values(r.accepted_pwm,4,rc=True)
    if r.accepted_pwm is not None and (r.reason!='accepted' or r.tripped):
        raise ValueError('Inconsistent acceptance')
    if r.source=='auto' and (r.tripped or r.local_mode!='AUTO'):
        raise ValueError('Inconsistent AUTO selection')
    if r.source=='auto':
        # Reuse the same canonical identity checks as the common header.
        _frame(3,r.selected_session_id,r.selected_sequence,b'')
        selected_session=bytes.fromhex(r.selected_session_id)
        selected_sequence=r.selected_sequence
    else:
        if r.selected_session_id is not None or r.selected_sequence is not None:
            raise ValueError('Only AUTO selection has a command identity')
        selected_session=bytes(16)
        selected_sequence=0
    body=RECEIPT.pack(_uint(r.receiver_us,64),_enum(r.reason,REASONS),_flag(r.tripped),
        _enum(r.local_mode,MODES),int(r.accepted_pwm is not None),_enum(r.source,SOURCES),
        _enum(r.fc_stage,FC_STAGES),selected_session,selected_sequence,
        *accepted,*_selection(r.selected_pwm,r.source))
    return _frame(3,r.session_id,r.sequence,body)


def decode_receipt(data):
    session,sequence,body=_unframe(data,3,RECEIPT.size)
    time,reason,tripped,mode,accepted,source,fc,selected_session,selected_sequence,*values=RECEIPT.unpack(body)
    if (reason>=len(REASONS) or mode>=len(MODES) or source>=len(SOURCES)
            or fc>=len(FC_STAGES) or max(tripped,accepted)>1): raise ValueError('Invalid receipt flags')
    r=ReceiptMessage(session,sequence,time,REASONS[reason],bool(tripped),MODES[mode],
        tuple(values[:4]) if accepted else None,tuple(values[4:]) if source else None,
        SOURCES[source],FC_STAGES[fc],selected_session.hex() if source==2 else None,
        selected_sequence if source==2 else None)
    if encode_receipt(r)!=data: raise ValueError('Noncanonical receipt')
    return r


@dataclass(frozen=True)
class ClockMapping:
    """Caller-supplied bound, NOT an estimator. Mock uses an exact known clock.

    All times map to the oldest possible Pi time to avoid understating ages.
    Drift is an error bound around unit rate, not a measured correction.
    A hardware producer must establish/refresh these bounds independently.
    """
    esp_anchor_us: int
    pi_anchor_s: float
    uncertainty_s: float
    drift_bound_ppm: float
    valid_until_pi_s: float

    def __post_init__(self):
        _uint(self.esp_anchor_us,64)
        if (not all(math.isfinite(x) for x in (self.pi_anchor_s,self.uncertainty_s,
                self.drift_bound_ppm,self.valid_until_pi_s)) or self.pi_anchor_s<0
                or self.uncertainty_s<0 or not 0<=self.drift_bound_ppm<1_000_000
                or self.valid_until_pi_s<self.pi_anchor_s):
            raise ValueError('Invalid clock mapping bound')

    def timestamp(self,time_us,*,now_s):
        _uint(time_us,64)
        if not math.isfinite(now_s) or not self.pi_anchor_s<=now_s<=self.valid_until_pi_s:
            raise ValueError('Clock mapping expired or not yet valid')
        delta=(time_us-self.esp_anchor_us)/1_000_000
        return self.pi_anchor_s+delta-self.uncertainty_s-abs(delta)*self.drift_bound_ppm/1_000_000


def to_snapshot(message,mapping,*,now_s):
    def stamp(t): return mapping.timestamp(t,now_s=now_s)
    channels={name:Measurement(m.sample_id,stamp(m.capture_us),m.values,m.valid)
              for name,_ in CHANNELS for m in (getattr(message,name),)}
    return SensorSnapshot(message.sequence,stamp(message.created_us),**channels)
