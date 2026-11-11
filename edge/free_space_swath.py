"""Observed free sectors in a local measured-motion reference.

Simulation assumption: adjacent ToF columns bound a filled sector out to the
smaller return. This interpolation must be tested for thin/occluded objects.
The bootstrap disk is an explicit preflight clearance assumption.
"""
from collections import deque
from dataclasses import dataclass
import math

import numpy as np

from edge.empirical_corridor import CorridorConfig, EmpiricalCorridor
from edge.survey_memory import rotation


@dataclass(frozen=True)
class SwathConfig:
    history_s: float = 8.
    bootstrap_radius_m: float = 1.35
    clearance_radius_m: float = .30
    body_radius_m: float = .20
    grid_m: float = .02
    range_error_m: float = .02
    drift_m_per_s: float = .001
    random_walk_m_sqrt_s: float = .01
    distance_error_fraction: float = .03

    def __post_init__(self):
        values = (self.history_s, self.clearance_radius_m, self.body_radius_m, self.grid_m,
                  self.range_error_m, self.drift_m_per_s, self.random_walk_m_sqrt_s, self.distance_error_fraction)
        if any(not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError('Invalid swath limits')
        if not math.isfinite(self.bootstrap_radius_m) or self.bootstrap_radius_m < 0:
            raise ValueError('Invalid preflight clearance')
        if self.body_radius_m > self.clearance_radius_m:
            raise ValueError('The protected footprint must contain the airframe')


def motion_endpoint(velocity, action, attitude):
    """Empirical stopping reserve plus requested outward acceleration (FLU).

    A counter-command may brake inside an unknown sector; it must not shorten
    the momentum reserve used to screen the already moving vehicle.
    """
    velocity = np.asarray(velocity, dtype=float)
    roll, pitch = attitude
    tilt = 1.5 * np.array([pitch, -roll])
    command = .06 * np.clip(np.array([action[2]-1500., 1500.-action[1]]) / 20., -1., 1.)
    tilt = np.where(tilt * velocity >= 0., tilt, 0.)
    command = np.where((command * velocity >= 0.) | (np.abs(velocity) < .02), command, 0.)
    return velocity * 1.2 + tilt + command


class FreeSpaceSwath(EmpiricalCorridor):
    def __init__(self, reference_id, config=None):
        self.swath_config = config or SwathConfig()
        super().__init__(reference_id, CorridorConfig(history_s=self.swath_config.history_s, max_frames=128),
                         screen_hazards=False)

    def reset(self, reference_id):
        super().reset(reference_id)
        self.sectors = deque()
        self.bootstrap_origin = self.position.copy()
        self.bootstrap_time_s = None
        self.last_report = {}
        self.hit_points = np.empty((0, 2))
        self.hit_errors = np.empty(0)

    def error(self, age, distance):
        c = self.swath_config
        return (.020 + c.drift_m_per_s * age + c.random_walk_m_sqrt_s * np.sqrt(age)
                + c.distance_error_fraction * distance)

    def update(self, snapshot, heading_sample, *, now_s):
        report = super().update(snapshot, heading_sample, now_s=now_s)
        self.last_report = report
        if report['state'] == 'INVALID':
            self.sectors.clear()
            return report
        report['heading_error_rad'] = (heading_sample.error_bound_rad
            + self.heading.config.max_rate_rad_s * heading_sample.timestamp_error_s)
        if self.bootstrap_time_s is None:
            self.bootstrap_time_s = now_s
        roll, pitch = snapshot.attitude.value
        report['observation_qualified'] = max(abs(roll), abs(pitch)) <= math.radians(10)
        if report['tof_state'] == 'new' and report['observation_qualified']:
            # Conservative horizontal reach across the vertical sensor rows.
            horizontal = np.asarray(snapshot.tof.value).reshape(8, 8).min(axis=0) * 3.5
            horizontal *= math.cos(math.pi / 8 + max(abs(roll), abs(pitch)))
            azimuths = np.linspace(-math.pi/8, math.pi/8, 8)
            normals = np.c_[-np.sin(azimuths), np.cos(azimuths), np.zeros(8)]
            normals = normals @ rotation(roll, pitch, 0.).T
            # Intersection of each column plane with the local horizontal
            # plane; roll/pitch must not silently rotate coverage into a gap.
            edges = np.arctan2(-normals[:, 0], normals[:, 1])
            self.sectors.append((now_s, self.position[:2].copy(), self.distance,
                                 report['heading_rad'], horizontal[::-1], edges,
                                 report['heading_error_rad']))
        while self.sectors and (now_s - self.sectors[0][0] > self.swath_config.history_s + 1e-8
                               or len(self.sectors) > self.config.max_frames):
            self.sectors.popleft()
        self.hit_points = np.concatenate([f[4][:, :2] for f in self.frames])
        self.hit_errors = np.repeat(
            self.error(now_s - np.array([f[0] for f in self.frames]),
                       self.distance - np.array([f[2] for f in self.frames])),
            [len(f[4]) for f in self.frames])
        return report

    def assess(self, endpoint_body, *, now_s, include_current=False):
        if (self.fault or not self.sectors or not self.last_report.get('observation_qualified', False)
                or not math.isfinite(now_s)
                or self.last_s is None or abs(now_s - self.last_s) > 1e-8):
            return dict(covered=False, reason='invalid_observation', uncovered_cells=0)
        c = self.swath_config
        endpoint = np.asarray(endpoint_body, dtype=float)
        if endpoint.shape != (2,) or not np.isfinite(endpoint).all():
            raise ValueError('A finite body-plane endpoint is required')
        radius = c.clearance_radius_m
        diagonal = c.grid_m / math.sqrt(2.)
        lo, hi = np.minimum(0., endpoint) - radius, np.maximum(0., endpoint) + radius
        x, y = np.meshgrid(np.arange(lo[0], hi[0] + c.grid_m, c.grid_m),
                           np.arange(lo[1], hi[1] + c.grid_m, c.grid_m))
        points = np.c_[x.ravel(), y.ravel()]
        length2 = float(endpoint @ endpoint)
        fraction = np.clip(points @ endpoint / max(length2, 1e-12), 0., 1.)
        points = points[np.linalg.norm(points - fraction[:, None] * endpoint, axis=1) <= radius + diagonal]
        if not include_current:
            # Incremental swept volume: only the expansion beyond the current
            # protected footprint needs new angular coverage. Requiring the
            # whole rear disk again would prohibit straight flight once old
            # rear-facing observations expire. Known hits veto the FULL tube
            # below, including the footprint that this mask omits.
            points = points[(np.linalg.norm(points, axis=1) + diagonal > radius)
                            & (points @ endpoint + diagonal * math.sqrt(length2) > 0.)]
        yaw = self.last_report['heading_rad']
        transform = rotation(0., 0., yaw)[:2, :2]
        world = self.position[:2] + points @ transform.T
        local = (self.hit_points - self.position[:2]) @ transform
        portion = np.clip(local @ endpoint / max(length2, 1e-12), 0., 1.)
        gap = np.linalg.norm(local - portion[:, None] * endpoint, axis=1)
        hazards = int(np.count_nonzero(gap < radius + self.hit_errors + c.range_error_m))
        if hazards:
            return dict(covered=False, reason='remembered_obstacle', hazards=hazards,
                        uncovered_cells=0, endpoint_body_m=endpoint.tolist())
        bootstrap_error = self.error(now_s - self.bootstrap_time_s, self.distance)
        covered = np.linalg.norm(world - self.bootstrap_origin[:2], axis=1) + diagonal + bootstrap_error <= c.bootstrap_radius_m
        # The current airframe volume is the continuity seed from collision-free
        # flight. The additional clearance margin still requires observations.
        covered |= np.linalg.norm(points, axis=1) + diagonal <= c.body_radius_m
        remaining = np.flatnonzero(~covered)
        if len(remaining):
            captures = np.array([f[0] for f in self.sectors])
            origins = np.array([f[1] for f in self.sectors])
            distances = np.array([f[2] for f in self.sectors])
            angles = np.array([f[3] for f in self.sectors])
            ranges = np.array([f[4] for f in self.sectors])
            edges = np.array([f[5] for f in self.sectors])
            heading_error = np.array([f[6] for f in self.sectors]) + self.last_report['heading_error_rad']
            delta = world[None, remaining, :] - origins[:, None, :]
            cosine, sine = np.cos(angles)[:, None], np.sin(angles)[:, None]
            local_x = delta[:, :, 0]*cosine + delta[:, :, 1]*sine
            local_y = -delta[:, :, 0]*sine + delta[:, :, 1]*cosine
            rho = np.hypot(local_x, local_y)
            bearing = np.arctan2(local_y, local_x)
            error = (self.error(now_s-captures, self.distance-distances) + diagonal)[:, None]
            angular_error = np.arcsin(np.minimum(1., error / np.maximum(rho, 1e-9))) + heading_error[:, None]
            left, right = bearing - angular_error, bearing + angular_error
            inside = (left >= edges[:, :1]) & (right <= edges[:, -1:]) & (rho > error)
            reach = np.full(rho.shape, np.inf)
            for index in range(7):
                overlaps = (left <= edges[:, index+1:index+2]) & (right >= edges[:, index:index+1])
                reach = np.where(overlaps, np.minimum(reach, np.minimum(ranges[:, index], ranges[:, index+1])[:, None]), reach)
            inside &= rho + error + c.range_error_m <= reach
            covered[remaining] |= inside.any(axis=0)
        return dict(covered=bool(covered.all()), reason='observed_swath' if covered.all() else 'outside_observed_swath',
                    uncovered_cells=int((~covered).sum()), total_cells=len(points),
                    retained_frames=len(self.sectors), endpoint_body_m=endpoint.tolist())
