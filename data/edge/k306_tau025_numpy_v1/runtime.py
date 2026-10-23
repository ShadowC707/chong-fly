"""Standalone float32 inference: standard library + NumPy, no torch/scipy.

Inputs are normalized flow[2], normalized ToF[64], optional memory[8].
Neural state is retained until reset. Only FlightRuntime owns applied throttle.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

if __package__:
    from .altitude_control import AltitudeConfig, AltitudeHold, AltitudeSample, AltitudeTelemetryError
    from .xy_coordinator import XYConfig, XYCoordinator
else:
    from altitude_control import AltitudeConfig, AltitudeHold, AltitudeSample, AltitudeTelemetryError
    from xy_coordinator import XYConfig, XYCoordinator


FORMAT = 'chong-fly-numpy-v1'
ACTION_ORDER = ['throttle', 'roll', 'pitch', 'yaw']


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_bundle(directory):
    directory = Path(directory)
    path = directory/'metadata.json'
    if sha256(path) != (directory/'metadata.sha256').read_text().strip():
        raise ValueError('Metadata checksum mismatch')
    meta = json.loads(path.read_text(encoding='utf-8'))
    if (meta.get('format') != FORMAT or meta.get('solver') != 'exponential_euler'
            or meta.get('sensor_encoding') != 'proximity-v1'
            or meta.get('matrix_orientation') != 'source_target_h_at_W'
            or meta.get('dtype') != 'float32' or meta.get('action_order') != ACTION_ORDER
            or meta.get('dt') != .02 or meta.get('sensor_dim') not in (66,74)
            or meta.get('tof_range_m') != 3.5 or meta.get('proximity_threshold_m') != .8
            or meta.get('pwm') != dict(min=1000., mid=1500., half=500., max=2000.)):
        raise ValueError('Unsupported runtime contract')
    files = meta.get('files', {})
    required = {'model.npz','runtime.py','altitude_control.py','xy_coordinator.py','benchmark.py','requirements.txt'}
    if set(files) != required:
        raise ValueError('Invalid bundle file contract')
    for name,digest in files.items():
        if sha256(directory/name) != digest:
            raise ValueError('Hash mismatch: '+name)
    k, d = meta['hidden_size'], meta['sensor_dim']
    if type(k) is not int or not 1 <= k <= 4096:
        raise ValueError('Invalid hidden size')
    shapes = dict(sensor_gain=(d,), sensor_bias=(d,), W=(k,k), W_in=(k,d),
                  input_bias=(k,), b=(k,), A=(k,), tau=(k,), dn_weight=(4,k), dn_bias=(4,))
    with np.load(directory/'model.npz', allow_pickle=False) as stored:
        if set(stored.files) != set(shapes):
            raise ValueError('Invalid tensor names')
        arrays = {name:np.array(stored[name],copy=True,order='C') for name in shapes}
    for name,shape in shapes.items():
        a=arrays[name]
        if a.shape != shape or a.dtype != np.float32 or not np.isfinite(a).all():
            raise ValueError('Invalid tensor: '+name)
        a.flags.writeable=False
    if np.any(arrays['tau'] <= 0):
        raise ValueError('Nonpositive tau')
    return meta,arrays


class NumpyPolicy:
    def __init__(self, directory):
        self.metadata,self.arrays=load_bundle(directory)
        self.default_dt=self.metadata['dt']
        self.fraction=-np.expm1(-np.float32(self.default_dt)/self.arrays['tau'])
        self.reset_state()

    def reset_state(self):
        self.h=np.zeros(self.metadata['hidden_size'],dtype=np.float32)

    @property
    def parameter_bytes(self):
        return sum(a.nbytes for a in self.arrays.values())

    def step_np(self, flow_xy, tof_8x8, memory_ring=None, dt=None):
        dt=self.default_dt if dt is None else float(dt)
        flow=np.asarray(flow_xy,dtype=np.float32)
        tof=np.asarray(tof_8x8,dtype=np.float32)
        memory=np.ones(8,dtype=np.float32) if memory_ring is None else np.asarray(memory_ring,dtype=np.float32)
        if (not math.isfinite(dt) or not math.isclose(dt,self.default_dt,rel_tol=0,abs_tol=1e-12)
                or flow.shape != (2,) or tof.shape not in ((64,),(8,8)) or memory.shape != (8,)
                or not all(np.isfinite(a).all() for a in (flow,tof,memory))
                or np.any(tof<0) or np.any(tof>1) or np.any(memory<0) or np.any(memory>1)):
            raise ValueError('Expected finite flow[2], normalized ToF[64]/memory[8], dt=0.02')
        a=self.arrays
        depth=np.maximum(np.float32(0),np.float32(1)-tof.ravel()/np.float32(
            self.metadata['proximity_threshold_m']/self.metadata['tof_range_m']))
        parts=[flow,depth]
        if self.metadata['sensor_dim']==74:
            parts.append(np.float32(1)-memory)
        u=np.concatenate(parts)*a['sensor_gain']+a['sensor_bias']
        # W is source -> target; preserve the reference operation order.
        drive=self.h @ a['W'] + (u @ a['W_in'].T + a['input_bias']) + a['b']
        target=a['A']*np.tanh(drive)
        h=self.h+self.fraction*(target-self.h)
        logits=h @ a['dn_weight'].T+a['dn_bias']
        pwm=np.empty(4,dtype=np.float32)
        # Stable sigmoid without overflow for either sign.
        v=logits[0]
        if v>=0:
            s=np.float32(1)/(np.float32(1)+np.exp(-v))
        else:
            exp=np.exp(v); s=exp/(np.float32(1)+exp)
        pwm[0]=np.float32(1000)+s*np.float32(1000)
        pwm[1:]=np.float32(1500)+np.tanh(logits[1:])*np.float32(500)
        if not np.isfinite(h).all() or not np.isfinite(pwm).all():
            raise ValueError('Nonfinite neural state/output')
        self.h=h
        return pwm


class FlightRuntime:
    """Sensor -> network -> optional XY -> unchanged altitude controller.

    Call at 50 Hz. A telemetry fault latches and returns no command; the caller
    owns authority handover. This module never arms motors or transmits RC.
    """
    def __init__(self,directory,*,coordinator_mode='baseline',coordinator_preset='original_v1'):
        if coordinator_mode not in ('baseline','anti_rollback','turn_latch','both'):
            raise ValueError('Unknown coordinator mode')
        self.policy=NumpyPolicy(directory)
        self.metadata=self.policy.metadata
        config=self.metadata['control_contract']['altitude']['config']
        self.altitude=AltitudeHold(AltitudeConfig(**config))
        if self.altitude.contract() != self.metadata['control_contract']['altitude']:
            raise ValueError('Altitude contract mismatch')
        if coordinator_preset not in self.metadata['xy_presets']:
            raise ValueError('Unknown coordinator preset')
        self.coordinator=None if coordinator_mode=='baseline' else XYCoordinator(
            anti_rollback=coordinator_mode in ('anti_rollback','both'),
            turn_latch=coordinator_mode in ('turn_latch','both'),
            config=XYConfig(**self.metadata['xy_presets'][coordinator_preset]))
        self.fault=None

    def reset(self):
        self.policy.reset_state(); self.altitude.reset(); self.fault=None
        if self.coordinator is not None:
            self.coordinator.reset()

    def step(self,flow_xy,tof_8x8,altitude_sample,*,now_s,memory_ring=None):
        if self.fault is not None:
            raise AltitudeTelemetryError('Latched runtime fault; reset required: '+self.fault)
        try:
            raw=self.policy.step_np(flow_xy,tof_8x8,memory_ring)
            applied=raw.copy()
            decision=dict(intervened=False,reasons=[])
            if self.coordinator is not None:
                applied=self.coordinator.apply(raw,flow_xy=flow_xy,
                    tof_m=np.asarray(tof_8x8)*self.metadata['tof_range_m'],dt=self.policy.default_dt)
                decision=self.coordinator.last_decision.copy()
            applied=self.altitude.apply(applied,altitude_sample,now_s=now_s)
        except ValueError as exc:
            self.fault=str(exc)
            raise
        return dict(network_pwm=raw,applied_pwm=applied,coordinator=decision)
