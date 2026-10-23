"""Per-channel measurement identity and zero-order hold, without timestamp fabrication.

Altitude payload is (downrange, capture_roll, capture_pitch). Current attitude
is a separate channel; it must never rewrite the angles of a held range sample.
All measurement times are already mapped to the Pi monotonic clock by the caller.
"""
from dataclasses import asdict, dataclass
import math
import numpy as np

if __package__:
    from .altitude_control import AltitudeSample
else:
    from altitude_control import AltitudeSample


@dataclass(frozen=True)
class Measurement:
    sample_id: int
    timestamp_s: float
    value: object
    valid: bool = True


@dataclass(frozen=True)
class SensorSnapshot:
    sequence: int  # transport identity, distinct from each measurement identity
    timestamp_s: float  # snapshot creation time, not the age of every measurement
    flow: Measurement
    tof: Measurement
    altitude: Measurement
    attitude: Measurement


@dataclass(frozen=True)
class SensorBudgets:
    flow_max_age_s: float = .04
    tof_max_age_s: float = .10
    altitude_max_age_s: float = .06
    attitude_max_age_s: float = .04

    def __post_init__(self):
        if any(type(v) not in (int,float) or not math.isfinite(v) or v <= 0
               for v in asdict(self).values()):
            raise ValueError('Sensor age budgets must be finite positive seconds')


class SensorTracker:
    def __init__(self, budgets=None):
        self.budgets = budgets or SensorBudgets()
        self.reset()

    def reset(self):
        self._samples = {}
        self._sequence = -1
        self._now = self._timestamp = None
        self.last_report = {}

    def resolve(self, snapshot, *, now_s):
        self.last_report = {}
        if (not isinstance(snapshot,SensorSnapshot) or type(snapshot.sequence) is not int
                or snapshot.sequence <= self._sequence or not math.isfinite(now_s)
                or not math.isfinite(snapshot.timestamp_s) or snapshot.timestamp_s < 0
                or snapshot.timestamp_s > now_s+1e-8
                or (self._now is not None and (now_s <= self._now or snapshot.timestamp_s <= self._timestamp))):
            raise ValueError('Invalid/replayed sensor snapshot')
        pending, states, ages = {}, {}, {}
        for name, shape in (('flow',(2,)), ('tof',(64,)), ('altitude',(3,)), ('attitude',(2,))):
            m = getattr(snapshot,name)
            try:
                if (not isinstance(m,Measurement) or type(m.sample_id) is not int or m.sample_id < 0
                        or not isinstance(m.valid,(bool,np.bool_)) or not m.valid
                        or not math.isfinite(m.timestamp_s) or m.timestamp_s < 0):
                    raise ValueError('invalid measurement identity/validity')
                age = now_s-m.timestamp_s
                if (m.timestamp_s > snapshot.timestamp_s+1e-8 or age < -1e-8
                        or age > getattr(self.budgets,name+'_max_age_s')+1e-8):
                    raise ValueError('stale/future measurement')
                value = np.asarray(m.value,dtype=float)
                if name=='tof' and value.shape==(8,8): value=value.ravel()
                if value.shape != shape or not np.isfinite(value).all():
                    raise ValueError('invalid measurement payload')
                if name=='tof' and (np.any(value<0) or np.any(value>1)):
                    raise ValueError('ToF must be normalized to [0,1]')
                if name=='attitude' and np.any(np.abs(value)>=math.pi/2):
                    raise ValueError('invalid current attitude')
                entry = (m.sample_id, float(m.timestamp_s), tuple(float(v) for v in value))
                previous = self._samples.get(name)
                state = 'new'
                if previous is not None:
                    if entry[0]==previous[0]:
                        if entry != previous: raise ValueError('changed payload/time under same identity')
                        state='held'
                    elif entry[0]<previous[0] or entry[1]<=previous[1]:
                        raise ValueError('regressed identity/time or new ID without new capture')
                pending[name], states[name], ages[name] = entry, state, max(0.,age)
                self.last_report[name] = dict(state=state,age_s=ages[name],sample_id=m.sample_id,
                                              timestamp_s=float(m.timestamp_s))
            except (ValueError,TypeError,AttributeError,OverflowError) as exc:
                self.last_report[name] = dict(state='invalid',reason=str(exc))
                raise ValueError(name+': '+str(exc)) from exc
        # Commit only when every channel has been validated; copy payloads, not caller arrays.
        self._samples = pending
        self._sequence, self._now, self._timestamp = snapshot.sequence, now_s, snapshot.timestamp_s
        a = pending['altitude']
        return dict(flow=pending['flow'][2],tof=pending['tof'][2],
                    altitude=AltitudeSample(*a[2],a[1],True),attitude=pending['attitude'][2],
                    states=states,ages=ages)
