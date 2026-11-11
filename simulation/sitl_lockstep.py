"""Private experimental Linux host ABI v1; never send to real hardware."""
import struct
import numpy as np
from simulation.ready_sim_contract import sitl_rc_packet


def encode_step(seq,duration_us,rc_new,fdm,pwm,aux):
    if not isinstance(seq,int) or not 1<=seq<2**64:
        raise ValueError('Invalid sequence')
    if not isinstance(duration_us,int) or not 0<duration_us<=4000 or duration_us%10:
        raise ValueError('Duration must be 10..4000 us in multiples of 10')
    if len(fdm)!=144 or not np.isfinite(struct.unpack('<18d',fdm)).all():
        raise ValueError('Invalid FDM packet')
    if not isinstance(rc_new,bool):
        raise ValueError('Explicit RC freshness flag required')
    rc=sitl_rc_packet(pwm,timestamp_s=0,aux=aux)[8:]
    return struct.pack('<4sQII',b'CFS1',seq,duration_us,int(rc_new))+fdm+rc


def decode_step(packet,seq,previous_clock_us,duration_us):
    if len(packet)!=40: raise ValueError('Expected 40-byte acknowledgement')
    echoed,clock,count,status,*motors=struct.unpack('<QQII4f',packet)
    if echoed!=seq or status: raise ValueError('Rejected or mismatched step acknowledgement')
    if previous_clock_us is not None and clock!=previous_clock_us+duration_us:
        raise ValueError('Firmware clock discontinuity')
    if not np.isfinite(motors).all() or min(motors)<0 or max(motors)>1:
        raise ValueError('Invalid normal-mode motor outputs')
    return dict(seq=echoed,clock_us=clock,motor_updates=count,motors=motors)


def infer_hover(rows,target_motor):
    x=np.array([r['rc_pwm'] for r in rows],dtype=float)
    y=np.array([r['mean_motor'] for r in rows],dtype=float)
    if len(x)<2 or not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(target_motor):
        raise ValueError('Invalid static map')
    if np.any(np.diff(x)<=0) or np.any(np.diff(y)<=0):
        raise ValueError('Static map must be strictly increasing')
    if not y[0]<=target_motor<=y[-1]:
        raise ValueError('Hover requires measured bracket, not extrapolation')
    return float(np.interp(target_motor,y,x))
