"""Authority protocol reference; NumPy only. CommandGate must run on the receiver.

No UART, motor arming or physical fallback is implemented here. MANUAL means
revocation of Pi authority, never an invented replacement RC command.
"""
from dataclasses import dataclass
import math
import numpy as np

if __package__:
    from .sensors import SensorSnapshot, SensorTracker, Measurement
else:
    from sensors import SensorSnapshot, SensorTracker, Measurement


@dataclass(frozen=True)
class SensorFrame:
    sequence: int
    timestamp_s: float  # synchronized monotonic Pi clock, all navigation inputs
    flow_xy: object
    tof_8x8: object
    altitude: object


@dataclass(frozen=True)
class CommandLease:
    sequence: int  # echoes ESP sensor sequence, not an independent command counter
    pwm: tuple
    ttl_s: float


@dataclass(frozen=True)
class BridgeConfig:
    ready_dwell_s: float = .5
    max_age_s: float = .04
    min_step_s: float = .01
    max_step_s: float = .03
    max_flow: float = .015  # filtered normalized flow; nominal 0.06 m/s at 1 m
    flow_filter_tau_s: float = .12
    height_tolerance_m: float = .10
    max_vertical_speed_mps: float = .10
    max_tilt_rad: float = .15
    clear_m: float = .95
    max_throttle_difference_pwm: float = 15.
    max_attitude_difference_pwm: float = 35.
    max_yaw_difference_pwm: float = 20.
    lease_s: float = .04

    def __post_init__(self):
        if not all(math.isfinite(v) and v>0 for v in self.__dict__.values()):
            raise ValueError('Invalid bridge bounds')
        if not self.min_step_s<.02<self.max_step_s or self.lease_s>self.max_age_s:
            raise ValueError('Invalid bridge time bounds')


def pwm_array(value):
    a=np.asarray(value,dtype=float)
    if a.shape!=(4,) or not np.isfinite(a).all() or np.any(a<1000) or np.any(a>2000):
        raise ValueError('Invalid four-channel PWM')
    return a


class FlightBridge:
    def __init__(self,runtime,config=None,*,sensor_budgets=None,handover_mode='legacy'):
        if handover_mode not in ('legacy','freeze_sync_v1'):
            raise ValueError('Unknown handover mode')
        self.runtime=runtime; self.config=config or BridgeConfig()
        self.handover_mode=handover_mode
        self.sensor_tracker=None if sensor_budgets is None else SensorTracker(sensor_budgets)
        if sensor_budgets is not None and sensor_budgets.altitude_max_age_s > runtime.altitude.config.max_age_s:
            raise ValueError('altitude age budget exceeds the validated controller budget')
        self.state='MANUAL'; self.fault=None; self.sequence=-1
        self.last_time=None; self.last_frame_time=None; self.ready_since=None; self.last_output=None
        self.filtered_flow=None

    def reset_fault(self):
        if self.state!='MANUAL': raise ValueError('Reset requires MANUAL authority')
        self.runtime.reset(); self.fault=None
        self.last_time=self.last_frame_time=self.ready_since=self.last_output=None
        self.filtered_flow=None
        if self.sensor_tracker is not None: self.sensor_tracker.reset()
        # Do not reset sequence: delayed commands/frames remain invalid.

    def _result(self,reason,*,ready=False,lease=None,preview=None):
        return dict(state=self.state,ready=ready,reason=reason,fault=self.fault,
            lease=lease,preview_pwm=preview,revoke_auto=lease is None)

    def tick(self,frame,*,now_s,request,manual_pwm,stop_requested=False):
        if self.sensor_tracker is not None: self.sensor_tracker.last_report={}
        if request=='MANUAL':
            self.state='MANUAL'; self.ready_since=None; self.last_output=None
            self.filtered_flow=None
            return self._result('manual_override')
        if self.fault is not None:
            self.state='FAULT'; return self._result('reset_required')
        try:
            if request not in ('SHADOW','AUTO'): raise ValueError('Invalid authority request')
            if self.state=='MANUAL':
                self.runtime.reset(); self.last_time=self.last_frame_time=None
                if self.sensor_tracker is not None: self.sensor_tracker.reset()
            if not isinstance(frame,(SensorFrame,SensorSnapshot)): raise ValueError('Missing sensor frame')
            if isinstance(frame,SensorSnapshot) and self.sensor_tracker is None:
                raise ValueError('SensorSnapshot requires multi_rate_v1')
            c=self.config
            if (type(frame.sequence) is not int or frame.sequence<=self.sequence
                    or not math.isfinite(now_s) or not math.isfinite(frame.timestamp_s)
                    or frame.timestamp_s<0 or not 0<=now_s-frame.timestamp_s<=c.max_age_s):
                raise ValueError('Stale, future or replayed frame')
            if self.last_time is not None:
                if not c.min_step_s<=now_s-self.last_time<=c.max_step_s:
                    raise ValueError('Control deadline missed')
                if not c.min_step_s<=frame.timestamp_s-self.last_frame_time<=c.max_step_s:
                    raise ValueError('Sensor stream gap or duplicate timestamp')
            manual=pwm_array(manual_pwm)
            if self.sensor_tracker is not None:
                if isinstance(frame,SensorFrame):
                    if abs(frame.altitude.timestamp_s-frame.timestamp_s)>1e-6:
                        raise ValueError('Legacy SensorFrame requires synchronized measurements')
                    a=frame.altitude
                    frame=SensorSnapshot(frame.sequence,frame.timestamp_s,
                        Measurement(frame.sequence,frame.timestamp_s,frame.flow_xy),
                        Measurement(frame.sequence,frame.timestamp_s,frame.tof_8x8),
                        Measurement(frame.sequence,a.timestamp_s,(a.range_m,a.roll_rad,a.pitch_rad),a.valid),
                        Measurement(frame.sequence,frame.timestamp_s,(a.roll_rad,a.pitch_rad)))
                resolved=self.sensor_tracker.resolve(frame,now_s=now_s)
                attitude=resolved['attitude']
                frame=SensorFrame(frame.sequence,frame.timestamp_s,resolved['flow'],resolved['tof'],resolved['altitude'])
            else:
                attitude=(frame.altitude.roll_rad,frame.altitude.pitch_rad)
            # Validate before advancing neural state; altitude validity remains owned by runtime.
            flow=np.asarray(frame.flow_xy,float); tof=np.asarray(frame.tof_8x8,float)
            if (flow.shape!=(2,) or tof.shape not in ((64,),(8,8))
                    or not np.isfinite(flow).all() or not np.isfinite(tof).all()
                    or np.any(tof<0) or np.any(tof>1)):
                raise ValueError('Invalid navigation sensors')
            if self.sensor_tracker is None and abs(frame.altitude.timestamp_s-frame.timestamp_s)>1e-6:
                raise ValueError('Altitude and navigation snapshots are not synchronized')
            options={}
            if self.sensor_tracker is not None: options['allow_sensor_hold']=True
            if self.handover_mode=='freeze_sync_v1':
                options['integrate_altitude']=self.state=='AUTO' and request=='AUTO'
            result=self.runtime.step(flow,tof,frame.altitude,now_s=now_s,stop_requested=stop_requested,**options)
            target=pwm_array(result['applied_pwm'])
            altitude=self.runtime.altitude.last_command
            dt=.02 if self.last_time is None else now_s-self.last_time
            self.filtered_flow=(flow.copy() if self.filtered_flow is None else
                self.filtered_flow+-math.expm1(-dt/c.flow_filter_tau_s)*(flow-self.filtered_flow))
            # Only admission uses this envelope; AUTO reflexes remain allowed to exceed it.
            eligible=(np.linalg.norm(self.filtered_flow)<=c.max_flow
                and abs(altitude.height_m-1.)<=c.height_tolerance_m
                and abs(altitude.vertical_speed_mps)<=c.max_vertical_speed_mps
                and max(abs(attitude[0]),abs(attitude[1]))<=c.max_tilt_rad
                and float(tof.min())*3.5>=c.clear_m
                and abs(target[0]-manual[0])<=c.max_throttle_difference_pwm
                and np.max(abs(target[1:3]-manual[1:3]))<=c.max_attitude_difference_pwm
                and abs(target[3]-manual[3])<=c.max_yaw_difference_pwm)
            if self.handover_mode=='freeze_sync_v1':
                eligible=eligible and self.runtime.altitude.config.min_pwm<=manual[0]<=self.runtime.altitude.config.max_pwm
            if not eligible: self.ready_since=None
            elif self.ready_since is None: self.ready_since=now_s
            ready=self.ready_since is not None and now_s-self.ready_since+1e-9>=c.ready_dwell_s
            dt=.02 if self.last_time is None else now_s-self.last_time
            self.sequence=frame.sequence; self.last_time=now_s; self.last_frame_time=frame.timestamp_s
            entering=self.state!='AUTO'
            if request=='SHADOW' or (entering and not ready):
                self.state='SHADOW'; self.last_output=None
                return self._result('ready' if ready else 'engage_denied',ready=ready,preview=tuple(target))
            self.state='AUTO'
            if entering:
                applied=manual.copy()  # first issued command exactly matches operator snapshot
                if self.handover_mode=='freeze_sync_v1':
                    self.runtime.altitude.synchronize_output(float(manual[0]),now_s=now_s)
            else:
                # Explicit wrapper contract: throttle 150 PWM/s; R/P 2000; yaw 10000.
                bound=np.array([150.,2000.,2000.,10000.])*dt
                applied=np.clip(target,self.last_output-bound,self.last_output+bound)
            self.last_output=applied.copy()
            lease=CommandLease(frame.sequence,tuple(float(v) for v in applied),c.lease_s)
            return self._result('engaged' if entering else 'auto',ready=ready,lease=lease,preview=tuple(target))
        except (ValueError,TypeError,AttributeError,OverflowError) as exc:
            self.fault=str(exc); self.state='FAULT'; self.ready_since=None; self.last_output=None
            return self._result('fault')


class CommandGate:
    """ESP-side executable specification. Its clock is independent of Pi time.

    sensor_sent_s is ESP's recorded send time for latest_sensor_sequence.
    Never derive it from a timestamp supplied back by Pi.
    """
    def __init__(self):
        self.sequence=-1; self.pwm=None; self.deadline=None; self.tripped=False; self.clock=None

    def reset(self,*,auto_authorized):
        if auto_authorized: raise ValueError('Receiver reset requires revoked AUTO')
        self.pwm=self.deadline=None; self.tripped=False

    def accept(self,lease,*,received_s,auto_authorized,sensor_sent_s,latest_sensor_sequence):
        try:
            if not isinstance(lease,CommandLease): raise ValueError('Malformed lease')
            pwm=pwm_array(lease.pwm)
            if (self.tripped or not auto_authorized or type(lease.sequence) is not int
                    or lease.sequence<=self.sequence or lease.sequence!=latest_sensor_sequence
                    or not math.isfinite(received_s) or not math.isfinite(sensor_sent_s)
                    or (self.clock is not None and received_s<self.clock)
                    or not math.isfinite(lease.ttl_s) or not 0<lease.ttl_s<=.04
                    or not 0<=received_s-sensor_sent_s<lease.ttl_s
                    or (self.deadline is not None and received_s>=self.deadline)):
                raise ValueError('Expired or unauthorized lease')
            self.clock=received_s; self.sequence=lease.sequence; self.pwm=tuple(pwm)
            self.deadline=sensor_sent_s+lease.ttl_s
            return True
        except (ValueError,TypeError,AttributeError):
            self.pwm=None; self.tripped=True
            return False

    def command(self,*,now_s,auto_authorized):
        if (not auto_authorized or not math.isfinite(now_s) or self.deadline is None
                or now_s>=self.deadline or (self.clock is not None and now_s<self.clock)):
            self.pwm=None; self.tripped=True
        if math.isfinite(now_s): self.clock=now_s
        return None if self.tripped else self.pwm
