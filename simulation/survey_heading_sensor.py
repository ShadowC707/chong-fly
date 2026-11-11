"""Synthetic sensor boundary; truth enters here, never the survey consumer.

Bounded uniform angle error plus fixed drift is an explicit test assumption,
not a calibrated IMU model. The caller models delivery latency/loss separately.
"""
import math

import numpy as np

from edge.survey_heading import HeadingSample


class SimulatedHeadingSensor:
    def __init__(self, reference_id, *, seed=20261104, initial_yaw_rad=0.,
                 noise_bound_rad=math.radians(.25), drift_rad_s=math.radians(.1),
                 timestamp_error_s=.001):
        if (type(seed) is not int or seed<0 or not math.isfinite(drift_rad_s)
                or not math.isfinite(noise_bound_rad) or noise_bound_rad<0
                or not math.isfinite(timestamp_error_s) or timestamp_error_s<0):
            raise ValueError('Invalid synthetic heading sensor configuration')
        self.seed=seed
        self.noise_bound_rad=noise_bound_rad
        self.drift_rad_s=drift_rad_s
        self.timestamp_error_s=timestamp_error_s
        self.reset(reference_id,initial_yaw_rad=initial_yaw_rad,timestamp_s=0.)

    def reset(self, reference_id, *, initial_yaw_rad, timestamp_s):
        if (not isinstance(reference_id,str) or not reference_id
                or reference_id==getattr(self,'reference_id',None)
                or not math.isfinite(initial_yaw_rad) or not math.isfinite(timestamp_s)
                or timestamp_s<0):
            raise ValueError('Invalid new sensor reference')
        self.reference_id=reference_id
        self.initial_yaw_rad=initial_yaw_rad
        self.origin_time_s=timestamp_s
        self.last_capture_s=None
        self.sample_id=0
        self.rng=np.random.default_rng(self.seed)

    def capture(self, true_yaw_rad, *, timestamp_s):
        if (not math.isfinite(true_yaw_rad) or not math.isfinite(timestamp_s)
                or timestamp_s<self.origin_time_s
                or (self.last_capture_s is not None and timestamp_s<=self.last_capture_s)):
            raise ValueError('Invalid capture clock or angle; reuse sample for hold')
        elapsed=timestamp_s-self.origin_time_s
        error=self.drift_rad_s*elapsed+self.rng.uniform(-self.noise_bound_rad,self.noise_bound_rad)
        angle=(true_yaw_rad-self.initial_yaw_rad+error+math.pi)%(2*math.pi)-math.pi
        sample=HeadingSample(self.reference_id,self.sample_id,timestamp_s,float(angle),
            self.noise_bound_rad+abs(self.drift_rad_s)*elapsed,self.timestamp_error_s)
        self.sample_id+=1
        self.last_capture_s=timestamp_s
        return sample
