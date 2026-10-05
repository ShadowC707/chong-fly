"""Range-based outer altitude loop, separate from the inner attitude/rate PID.

Uses timestamped downrange and FLU roll/pitch only. This is a simulation/reference
implementation; RC gain/hover calibration and a hardware authority arbiter are
required before flight. Losing telemetry raises a latched fault, not a motor kill.
"""
from dataclasses import dataclass, asdict
import math

import numpy as np

ALTITUDE_VERSION = 'range-altitude-v1'


class AltitudeTelemetryError(ValueError):
    """No autonomous command is available; the caller must revoke authority."""


@dataclass(frozen=True)
class AltitudeSample:
    range_m: float
    roll_rad: float
    pitch_rad: float
    timestamp_s: float
    valid: bool


@dataclass(frozen=True)
class AltitudeConfig:
    target_height_m: float = 1.
    hover_pwm: float = 1213.
    kp: float = 4.                    # acceleration per metre of height error
    kd: float = 3.                    # acceleration per m/s of vertical velocity
    ki: float = .8
    integral_limit: float = .5
    velocity_filter_tau_s: float = .12
    max_acceleration_mps2: float = 3.
    gravity_mps2: float = 9.81
    min_pwm: float = 1100.
    max_pwm: float = 1400.
    slew_pwm_per_s: float = 150.
    min_range_m: float = .08
    max_range_m: float = 3.5
    min_tilt_cosine: float = .7
    emitter_below_center_m: float = .018
    max_age_s: float = .06
    max_sample_gap_s: float = .10
    max_vertical_speed_mps: float = 3.
    range_jump_slack_m: float = .03

    def __post_init__(self):
        if not all(math.isfinite(v) for v in asdict(self).values()):
            raise ValueError('Altitude config values must be finite')
        if (not 0 < self.min_range_m < self.target_height_m < self.max_range_m
                or not 1000 <= self.min_pwm < self.hover_pwm < self.max_pwm <= 2000
                or not 0 < self.min_tilt_cosine <= 1
                or not 0 < self.max_acceleration_mps2 < self.gravity_mps2
                or min(self.kp, self.kd, self.ki, self.integral_limit, self.emitter_below_center_m,
                       self.range_jump_slack_m) < 0
                or min(self.velocity_filter_tau_s, self.slew_pwm_per_s, self.max_age_s,
                       self.max_sample_gap_s, self.max_vertical_speed_mps) <= 0):
            raise ValueError('Invalid altitude control bounds/gains')


@dataclass(frozen=True)
class AltitudeCommand:
    throttle_pwm: float
    height_m: float
    vertical_speed_mps: float
    limited: bool


def sample_from_observation(obs):
    """Adapter reads sensor fields only, never world position/velocity/hover truth.

    sim_time is the timestamp of the cached simulation sensor snapshot. Hardware
    must supply measurement timestamps and a synchronized attitude estimate.
    """
    try:
        return AltitudeSample(float(obs.laser_distance), float(obs.attitude_euler[0]),
                              float(obs.attitude_euler[1]), float(obs.sim_time), obs.laser_valid)
    except (AttributeError, TypeError, IndexError, ValueError) as exc:
        raise AltitudeTelemetryError('Missing/malformed range-attitude snapshot') from exc


class AltitudeHold:
    def __init__(self, config=None):
        self.config = config or AltitudeConfig()
        self.reset()

    def reset(self):
        self._time = self._now = self._height = self._pwm = None
        self._velocity = self._integral = 0.
        self.fault = None
        self.last_command = None

    def _reject(self, reason):
        self.fault = reason
        raise AltitudeTelemetryError(reason)

    def update(self, sample, *, now_s):
        if self.fault is not None:
            raise AltitudeTelemetryError(f'Latched altitude fault; reset required: {self.fault}')
        c = self.config
        if not isinstance(sample, AltitudeSample): self._reject('Missing altitude sample')
        values = (sample.range_m, sample.roll_rad, sample.pitch_rad, sample.timestamp_s, now_s)
        try:
            finite = all(math.isfinite(v) for v in values)
        except (TypeError, ValueError):
            finite = False
        if not finite or not isinstance(sample.valid, (bool, np.bool_)) or not sample.valid:
            self._reject('Invalid range/attitude telemetry')
        age = now_s - sample.timestamp_s
        if sample.timestamp_s < 0 or age < -1e-8 or age > c.max_age_s:
            self._reject('Stale or future altitude telemetry')
        if not c.min_range_m <= sample.range_m <= c.max_range_m:
            self._reject('Altitude range outside operating bounds')
        cosine = math.cos(sample.roll_rad)*math.cos(sample.pitch_rad)
        if cosine < c.min_tilt_cosine or abs(sample.roll_rad) >= math.pi/2 or abs(sample.pitch_rad) >= math.pi/2:
            self._reject('Excessive tilt for range altitude control')
        height = (sample.range_m + c.emitter_below_center_m)*cosine
        dt = 0. if self._time is None else sample.timestamp_s - self._time
        if self._time is not None:
            if dt <= 0 or dt > c.max_sample_gap_s or now_s <= self._now:
                self._reject('Nonmonotonic or interrupted altitude telemetry')
            delta = height-self._height
            if abs(delta) > c.range_jump_slack_m+c.max_vertical_speed_mps*dt:
                self._reject('Abrupt range change: surface discontinuity or invalid sample')
            alpha = -math.expm1(-dt/c.velocity_filter_tau_s)
            velocity = self._velocity + alpha*(delta/dt-self._velocity)
        else:
            velocity = 0.  # no derivative kick on engagement
        error = c.target_height_m-height
        integral = float(np.clip(self._integral+error*dt, -c.integral_limit, c.integral_limit))
        acceleration = c.kp*error-c.kd*velocity+c.ki*integral
        bounded_acceleration = float(np.clip(acceleration, -c.max_acceleration_mps2, c.max_acceleration_mps2))
        # Reference motor model: thrust proportional to squared normalized RC.
        # Hover PWM is an explicit calibration, never env.physics.hover_throttle.
        pwm_raw = 1000+(c.hover_pwm-1000)*math.sqrt((1+bounded_acceleration/c.gravity_mps2)/cosine)
        pwm = float(np.clip(pwm_raw, c.min_pwm, c.max_pwm))
        if self._pwm is not None:
            limit = c.slew_pwm_per_s*(now_s-self._now)
            pwm = float(np.clip(pwm, self._pwm-limit, self._pwm+limit))
        limited = acceleration != bounded_acceleration or not math.isclose(pwm, pwm_raw, abs_tol=1e-8)
        # Freeze integration while pushing farther into any output/acceleration limit.
        pushing_high = error > 0 and (acceleration > bounded_acceleration or pwm_raw > pwm)
        pushing_low = error < 0 and (acceleration < bounded_acceleration or pwm_raw < pwm)
        if not (pushing_high or pushing_low): self._integral = integral
        self._time, self._now, self._height = sample.timestamp_s, now_s, height
        self._velocity, self._pwm = velocity, pwm
        self.last_command = AltitudeCommand(pwm, height, velocity, limited)
        return self.last_command

    def apply(self, navigation_pwm, sample, *, now_s):
        result = np.asarray(navigation_pwm, dtype=float).copy()
        if result.shape != (4,) or not np.isfinite(result).all() or np.any(result < 1000) or np.any(result > 2000):
            raise ValueError('Navigation command must be four finite PWM values in [1000, 2000]')
        result[0] = self.update(sample, now_s=now_s).throttle_pwm
        return result

    def contract(self):
        return {'version': ALTITUDE_VERSION, 'throttle_owner': 'altitude_controller',
                'navigation_channels': ['roll', 'pitch', 'yaw'], 'config': asdict(self.config)}
