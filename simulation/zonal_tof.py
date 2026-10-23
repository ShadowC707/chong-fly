"""Research-only equal-angular zonal sampling, not a calibrated VL53L5CX.

The existing intersection engine is shared; production ToFRaycaster is unchanged.
Distances are capped ray ranges, not optical signal strengths or target likelihoods.
"""
from dataclasses import asdict, dataclass
from numbers import Integral

import numpy as np

from simulation.drone_env import ToFRaycaster


@dataclass(frozen=True)
class ZonalConfig:
    samples_per_axis: int = 4
    rule: str = 'supported'
    group_span_m: float = .05
    min_coverage: float = .125

    def __post_init__(self):
        if (isinstance(self.samples_per_axis, bool)
                or not isinstance(self.samples_per_axis, Integral)
                or not 1 <= self.samples_per_axis <= 32):
            raise ValueError('samples_per_axis must be an integer in [1, 32]')
        if self.rule not in ('mean', 'minimum', 'supported'):
            raise ValueError('Unknown aggregation rule')
        if not np.isfinite(self.group_span_m) or self.group_span_m <= 0:
            raise ValueError('group_span_m must be finite and positive')
        if not np.isfinite(self.min_coverage) or not 0 < self.min_coverage <= 1:
            raise ValueError('min_coverage must be in (0, 1]')


def aggregate_ranges(values, config, max_range=3.5):
    """Return (range_m, support), with max range denoting no selected return.

    Mean includes the range cap. Mean/minimum support is the fraction below cap;
    supported-rule support is the selected group's fraction of ALL samples.
    Groups are disjoint and anchored at their nearest depth (no gap chaining).
    Support is a geometric diagnostic, never a calibrated confidence estimate.
    """
    x = np.asarray(values, dtype=np.float64)
    if not np.isfinite(max_range) or max_range <= 0:
        raise ValueError('max_range must be finite and positive')
    if (x.ndim != 1 or x.size == 0 or not np.isfinite(x).all()
            or (x < 0).any() or (x > max_range).any()):
        raise ValueError('Expected nonempty finite 1-D ranges within sensor range')
    hits = x[x < max_range]
    support = float(hits.size / x.size)
    if config.rule == 'mean':
        return float(x.mean()), support
    if config.rule == 'minimum':
        return float(x.min()), support
    hits.sort()
    start = 0
    while start < hits.size:
        end = int(np.searchsorted(hits, hits[start] + config.group_span_m, side='right'))
        coverage = (end - start) / x.size
        if coverage >= config.min_coverage:
            return float(hits[start:end].mean()), float(coverage)
        start = end
    return float(max_range), 0.


class ZonalToF:
    """8x8 scalar output; N midpoint rays inside each equal-angular zone."""
    geometry_version = 'equal-angular-zonal-research-v1'
    rows = cols = 8

    def __init__(self, config=None, *, fov_h_deg=45., fov_v_deg=45., max_range=3.5):
        self.config = config if config is not None else ZonalConfig()
        for fov in (fov_h_deg, fov_v_deg):
            if not np.isfinite(fov) or not 0 < fov < 180:
                raise ValueError('FOV must be finite and in (0, 180) degrees')
        if not np.isfinite(max_range) or max_range <= 0:
            raise ValueError('max_range must be finite and positive')
        self.max_range = float(max_range)
        self.fov_h_deg, self.fov_v_deg = float(fov_h_deg), float(fov_v_deg)
        n = self.config.samples_per_axis
        # Array axes: zone row, zone column, subrow, subcolumn.
        row, col, subrow, subcol = np.indices((8, 8, n, n))
        az = np.deg2rad(fov_h_deg * (.5 - (col + (subcol + .5)/n)/8))
        el = np.deg2rad(fov_v_deg * (.5 - (row + (subrow + .5)/n)/8))
        self.subray_directions = np.stack((np.cos(el)*np.cos(az),
            np.cos(el)*np.sin(az), np.sin(el)), axis=-1).reshape(8, 8, n*n, 3).astype(np.float32)
        self._caster = ToFRaycaster(max_range=max_range)
        self._caster.body_ray_dirs = self.subray_directions.reshape(-1, 3).copy()
        self._caster.num_rays = len(self._caster.body_ray_dirs)
        self.last_ranges_m = None
        self.last_support = None

    def contract(self):
        return dict(geometry_version=self.geometry_version, **asdict(self.config),
            rows=8, cols=8, fov_h_deg=self.fov_h_deg, fov_v_deg=self.fov_v_deg,
            max_range_m=self.max_range, sampling='equal-angular subcell midpoints',
            mount_pitch_deg=0., output='normalized slant range, row-major', research_only=True)

    def cast_rays(self, drone_pos, drone_rot, room, cylinders=None, boxes=None):
        # Float64 multiplication preserves the normalized engine's [0, 1] cap.
        samples = self._caster.cast_rays(drone_pos, drone_rot, room, cylinders, boxes)
        self.last_ranges_m = samples.astype(np.float64).reshape(64, -1)*self.max_range
        result = np.array([aggregate_ranges(x, self.config, self.max_range)
                           for x in self.last_ranges_m])
        self.last_support = result[:, 1].reshape(8, 8)
        return (result[:, 0]/self.max_range).astype(np.float32)
