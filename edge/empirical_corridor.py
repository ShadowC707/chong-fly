"""Simulation research policy with measured motion history and explicit risk limits.

PROCEED_LIMITED is an empirical permission, not a free-volume certificate.
The translation error model must be validated against each simulation suite.
"""
from collections import deque
from dataclasses import dataclass
import math

import numpy as np

from edge.sensors import SensorTracker
from edge.survey_heading import SurveyHeadingTracker
from edge.survey_memory import point_ray_directions, rotation


@dataclass(frozen=True)
class CorridorConfig:
    history_s: float = 4.
    max_frames: int = 64
    speed_limit_mps: float = .22
    clearance_radius_m: float = .30
    drift_m_per_s: float = .02
    distance_error_fraction: float = .10
    maximum_error_m: float = .15

    def __post_init__(self):
        values = (self.history_s, self.speed_limit_mps, self.clearance_radius_m,
                  self.drift_m_per_s, self.distance_error_fraction, self.maximum_error_m)
        if any(not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError('Invalid corridor limits')
        if type(self.max_frames) is not int or self.max_frames < 1:
            raise ValueError('Invalid frame capacity')


class EmpiricalCorridor:
    def __init__(self, reference_id, config=None, *, screen_hazards=True):
        self.config = config or CorridorConfig()
        self.screen_hazards = screen_hazards
        self.reset(reference_id)

    def reset(self, reference_id):
        if not isinstance(reference_id, str) or not reference_id or reference_id == getattr(self, 'reference_id', None):
            raise ValueError('Reset requires a new nonempty reference')
        self.reference_id = reference_id
        self.heading = SurveyHeadingTracker(reference_id)
        self.sensors = SensorTracker()
        self.frames = deque()
        self.position = np.zeros(3)
        self.velocity = np.zeros(2)
        self.distance = 0.
        self.last_s = None
        self.fault = None

    def update(self, snapshot, heading_sample, *, now_s):
        if self.fault:
            return dict(state='INVALID', reason=self.fault)
        try:
            self.sensors.resolve(snapshot, now_s=now_s)
            heading = self.heading.add_heading(heading_sample, now_s=now_s)
            frame = self.heading.align_tof(snapshot.tof, reference_id=heading_sample.reference_id, now_s=now_s)
            dt = 0. if self.last_s is None else now_s - self.last_s
            if dt < 0 or dt > .04 + 1e-8:
                raise ValueError('Motion history interrupted')
            roll, pitch = snapshot.attitude.value
            downrange = snapshot.altitude.value[0]
            if not .08 <= downrange <= 3.5 or max(abs(v) for v in snapshot.flow.value) >= .98:
                raise ValueError('Range or flow outside the experiment domain')
            measured_velocity = np.asarray(snapshot.flow.value) * 4. * downrange
            self.velocity += (1. - math.exp(-dt / .12)) * (measured_velocity - self.velocity)
            # Use unfiltered increments to avoid accumulating filter lag into pose.
            transform = rotation(roll, pitch, heading.angle_rad)
            increment = transform @ np.r_[measured_velocity * dt, 0.]
            self.position += increment
            self.distance += float(np.linalg.norm(increment))
            self.last_s = now_s
            if frame.state == 'new':
                if abs(frame.timestamp_s - now_s) > 1e-8:
                    raise ValueError('This adapter requires capture-time pose delivery')
                ranges = np.asarray(frame.normalized_ranges) * 3.5
                directions = point_ray_directions() @ transform.T
                hits = self.position + directions[ranges < 3.49] * ranges[ranges < 3.49, None]
                self.frames.append((now_s, self.position.copy(), self.distance, heading.angle_rad, hits))
            while self.frames and (now_s - self.frames[0][0] > self.config.history_s + 1e-8
                                   or len(self.frames) > self.config.max_frames):
                self.frames.popleft()
            speed = float(np.linalg.norm(self.velocity))
            # Empirical stopping screen retains tilt and 100 ms delivery reserve.
            horizon = .06 + 1.5 * speed + 1.5 * abs(pitch)
            direction = np.r_[self.velocity / speed, 0.] if speed > .03 else np.array([1., 0., 0.])
            endpoint = direction * horizon
            hazards = 0
            usable = 0 if self.screen_hazards else len(self.frames)
            for captured, origin, distance, angle, hits in (self.frames if self.screen_hazards else ()):
                age = now_s - captured
                error = .01 + self.config.drift_m_per_s * age + self.config.distance_error_fraction * (self.distance - distance)
                if error > self.config.maximum_error_m:
                    continue
                usable += 1
                points = (hits - self.position) @ transform
                fraction = np.clip(points @ endpoint / (endpoint @ endpoint), 0., 1.)
                gap = np.linalg.norm(points - fraction[:, None] * endpoint, axis=1)
                hazards += int(np.count_nonzero(gap < self.config.clearance_radius_m + error))
            state, reason = 'PROCEED_LIMITED', 'empirical_forward_permission'
            if hazards:
                state, reason = 'RESTRICT', 'remembered_obstacle'
            elif speed > self.config.speed_limit_mps:
                state, reason = 'SLOW', 'speed_budget'
            elif not usable or max(abs(roll), abs(pitch)) > math.radians(10):
                state, reason = 'UNKNOWN', 'motion_or_observation_unqualified'
            return dict(state=state, reason=reason, frames=len(self.frames), usable_frames=usable,
                        hazards=hazards, speed_mps=speed, horizon_m=horizon,
                        position_m=self.position.tolist(), accumulated_distance_m=self.distance,
                        heading_rad=heading.angle_rad, tof_state=frame.state,
                        tof_capture_s=frame.timestamp_s)
        except (ValueError, TypeError) as exc:
            self.fault = str(exc)
            self.frames.clear()
            return dict(state='INVALID', reason=self.fault)
