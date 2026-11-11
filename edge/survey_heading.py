"""Experimental measured-heading channel and capture-time ToF association.

Independent of SensorSnapshot v1, RC, world geometry and the pinned export.
Angles are FLU radians: positive left, opposite to positive RC yaw PWM.
Error bounds are declared conservative bounds, NOT standard deviations.
All timestamps must already share one monotonic clock.
"""
from dataclasses import dataclass, replace
import math

import numpy as np

from edge.sensors import Measurement


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


@dataclass(frozen=True)
class HeadingSample:
    reference_id: str
    sample_id: int
    timestamp_s: float
    angle_rad: float  # Wrapped [-pi, pi], relative to the declared reference.
    error_bound_rad: float = 0.
    timestamp_error_s: float = 0.
    valid: bool = True


@dataclass(frozen=True)
class HeadingConfig:
    max_age_s: float = .04
    max_capture_gap_s: float = .04
    max_alignment_gap_s: float = .02
    tof_max_age_s: float = .10
    history_s: float = .25
    max_rate_rad_s: float = math.radians(240)
    max_error_rad: float = math.radians(10)
    tof_timestamp_error_s: float = 0.  # Exact simulated capture clock; hardware must qualify this.

    def __post_init__(self):
        positive=(self.max_age_s,self.max_capture_gap_s,self.max_alignment_gap_s,
                  self.tof_max_age_s,self.history_s,self.max_rate_rad_s,self.max_error_rad)
        if (any(not _finite(v) or v<=0 for v in positive)
                or not _finite(self.tof_timestamp_error_s) or self.tof_timestamp_error_s<0
                or self.history_s<self.tof_max_age_s+self.max_alignment_gap_s
                or self.max_error_rad>=math.pi/2
                or self.max_rate_rad_s*self.max_capture_gap_s+2*self.max_error_rad>=math.pi):
            raise ValueError('Invalid heading budgets or ambiguous unwrap bound')


@dataclass(frozen=True)
class UnwrappedHeading:
    sample: HeadingSample
    angle_rad: float


@dataclass(frozen=True)
class AlignedToF:
    reference_id: str
    sample_id: int
    timestamp_s: float
    normalized_ranges: tuple
    heading_rad: float
    heading_error_bound_rad: float
    heading_sample_id: int
    heading_timestamp_s: float
    state: str
    version: str = 'capture-heading-tof-v1'


class SurveyHeadingTracker:
    """Bounded history, no extrapolation masquerading as an exact pose.

    Invalid heading breaks continuity and latches a fault until explicit reset
    to a different reference. Invalid ToF rejects that observation only.
    This research policy deliberately does not conceal packet reordering.
    No result is a free-corridor certificate or a flight authorization.
    """
    def __init__(self, reference_id, config=None):
        self.config=config or HeadingConfig()
        self.reference_id=None
        self.reset(reference_id)

    def reset(self, reference_id):
        if not isinstance(reference_id,str) or not reference_id or reference_id==self.reference_id:
            raise ValueError('Reset requires a different nonempty reference ID')
        self.reference_id=reference_id
        self._history=[]
        self._tof=None
        self._now=None
        self.faulted=False

    def _clock(self, now_s):
        if (not _finite(now_s) or now_s<0
                or (self._now is not None and now_s<self._now-1e-8)):
            raise ValueError('Invalid or regressed arrival clock')
        self._now=now_s

    def _ready(self):
        if self.faulted:raise ValueError('Heading continuity lost; explicit reference reset required')

    def _sample_error(self, sample):
        return sample.error_bound_rad+self.config.max_rate_rad_s*sample.timestamp_error_s

    def add_heading(self, sample, *, now_s):
        self._ready()
        try:
            self._clock(now_s)
            c=self.config
            if (not isinstance(sample,HeadingSample) or sample.reference_id!=self.reference_id
                    or type(sample.sample_id) is not int or sample.sample_id<0
                    or type(sample.valid) is not bool or not sample.valid
                    or any(not _finite(v) for v in (sample.timestamp_s,sample.angle_rad,
                                                    sample.error_bound_rad,sample.timestamp_error_s))
                    or sample.timestamp_s<0 or not -math.pi<=sample.angle_rad<=math.pi
                    or sample.error_bound_rad<0 or sample.timestamp_error_s<0
                    or self._sample_error(sample)>c.max_error_rad):
                raise ValueError('Invalid heading identity, reference, payload or uncertainty')
            age=now_s-sample.timestamp_s
            if age < -1e-8:raise ValueError('Heading capture is ahead of arrival clock')
            if age>c.max_age_s+1e-8:raise ValueError('Heading is stale')
            angle=sample.angle_rad
            if self._history:
                previous=self._history[-1]
                p=previous.sample
                if sample.sample_id==p.sample_id:
                    if sample!=p:raise ValueError('Changed heading under held identity')
                    return previous
                gap=sample.timestamp_s-p.timestamp_s
                if sample.sample_id<p.sample_id or gap<=0 or gap>c.max_capture_gap_s+1e-8:
                    raise ValueError('Regressed heading identity/time or capture gap')
                delta=(sample.angle_rad-p.angle_rad+math.pi)%(2*math.pi)-math.pi
                bound=c.max_rate_rad_s*gap+self._sample_error(p)+self._sample_error(sample)
                if abs(delta)>bound+1e-8:raise ValueError('Heading rate exceeds declared bound')
                angle=previous.angle_rad+delta
            result=UnwrappedHeading(sample,angle)
            self._history.append(result)
            cutoff=sample.timestamp_s-c.history_s
            while len(self._history)>1 and self._history[1].sample.timestamp_s<cutoff:
                self._history.pop(0)
            return result
        except (ValueError,TypeError,AttributeError,OverflowError):
            self.faulted=True
            self._history.clear()
            self._tof=None
            raise

    def align_tof(self, sample, *, reference_id, now_s):
        self._ready()
        self._clock(now_s)
        c=self.config
        if (reference_id!=self.reference_id or not isinstance(sample,Measurement)
                or type(sample.sample_id) is not int or sample.sample_id<0
                or not isinstance(sample.valid,(bool,np.bool_)) or not sample.valid
                or not _finite(sample.timestamp_s) or sample.timestamp_s<0
                or sample.timestamp_s>now_s+1e-8
                or now_s-sample.timestamp_s>c.tof_max_age_s+1e-8):
            raise ValueError('Invalid/stale/future ToF or wrong reference')
        if not self._history or now_s-self._history[-1].sample.timestamp_s>c.max_age_s+1e-8:
            raise ValueError('No fresh heading for current observation')
        values=np.asarray(sample.value,dtype=float)
        if (values.shape not in ((64,),(8,8)) or not np.isfinite(values).all()
                or (values<0).any() or (values>1).any()):
            raise ValueError('ToF must contain 64 normalized finite ranges')
        ranges=tuple(float(v) for v in values.ravel())
        if self._tof is not None:
            old=self._tof
            if sample.sample_id==old.sample_id:
                if sample.timestamp_s!=old.timestamp_s or ranges!=old.normalized_ranges:
                    raise ValueError('Changed ToF under held identity')
                return replace(old,state='held')
            if sample.sample_id<old.sample_id or sample.timestamp_s<=old.timestamp_s:
                raise ValueError('Regressed ToF identity or capture time')
        source=next((h for h in reversed(self._history)
                     if h.sample.timestamp_s<=sample.timestamp_s+1e-8),None)
        if source is None:raise ValueError('No preceding heading at ToF capture')
        gap=max(0.,sample.timestamp_s-source.sample.timestamp_s)
        error=self._sample_error(source.sample)+c.max_rate_rad_s*(gap+c.tof_timestamp_error_s)
        if gap>c.max_alignment_gap_s+1e-8 or error>c.max_error_rad+1e-8:
            raise ValueError('ToF-heading alignment gap/uncertainty exceeds budget')
        self._tof=AlignedToF(self.reference_id,sample.sample_id,sample.timestamp_s,ranges,
            source.angle_rad,error,source.sample.sample_id,source.sample.timestamp_s,'new')
        return self._tof
