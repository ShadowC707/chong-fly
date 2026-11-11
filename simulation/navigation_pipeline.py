"""One navigation pipeline: network A -> [1] braking without rollback -> [2] side veto -> XY base.

Both stages are of kind 'limits': per channel the result lies between 1500 and the network
command (same side, never larger). They cannot generate a command, so generated_share is 0.
Every stage writes one record per tick into last_decision['stages'].

The XY base (turn_latch, STOP) runs last, on the limited command, exactly as in baseline A+XY.
Hazard memory is session-wide: a bridge reset restarts the control state, not the memory.

Memory aging (MemoryConfig.aging):
- 'time' (pipeline_v1): the last 4 s / 64 ToF frames, FIFO. Round 1 card: a box seen 7 s before the
  contact was erased by this rule while the drone hovered beside it (category b).
- 'transport' (pipeline_v1_r1): time alone never erases a point. Each point carries its accumulated
  transport error (path travelled and heading turned since capture) and is removed only when that
  error exceeds r_veto. Points are thinned on a voxel grid (the newest observation of a voxel wins),
  so new frames of a hovering drone do not push an old obstacle out of a FIFO.
"""
from collections import deque
from dataclasses import dataclass, field, asdict
import json
import math
from pathlib import Path

import numpy as np

from edge.empirical_corridor import CorridorConfig, EmpiricalCorridor
from edge.survey_memory import rotation
from .xy_coordinator import XYCoordinator

LIMITS, GENERATES = 'limits', 'generates'
NEUTRAL = 1500.
ROLL, PITCH, YAW = 1, 2, 3


@dataclass(frozen=True)
class BrakeConfig:
    v_stop: float = .0105
    flow_filter_tau_s: float = .06
    law: str = 'smoothstep'  # 'smoothstep' (v1, r1) or 'proportional' (r2)
    gain_pwm_per_flow: float = 500.

    def __post_init__(self):
        values = (self.v_stop, self.flow_filter_tau_s, self.gain_pwm_per_flow)
        if self.law not in ('smoothstep', 'proportional') or not all(math.isfinite(v) and v > 0 for v in values):
            raise ValueError('Invalid brake stage configuration')


@dataclass(frozen=True)
class VetoConfig:
    r_veto_m: float = .30
    side_min_deg: float = 22.5
    side_max_deg: float = 157.5
    history_s: float = 4.
    max_frames: int = 64
    base_error_m: float = .02
    drift_m_per_s: float = .001
    random_walk_m_sqrt_s: float = .01
    distance_error_fraction: float = .03

    def __post_init__(self):
        values = (self.r_veto_m, self.history_s, self.base_error_m, self.drift_m_per_s,
                  self.random_walk_m_sqrt_s, self.distance_error_fraction)
        if (not all(math.isfinite(v) and v > 0 for v in values) or type(self.max_frames) is not int
                or self.max_frames < 1 or not 0 <= self.side_min_deg < self.side_max_deg <= 180):
            raise ValueError('Invalid veto stage configuration')


@dataclass(frozen=True)
class MemoryConfig:
    aging: str = 'time'
    voxel_m: float = .05
    max_points: int = 6000
    turn_error_rad_per_rad: float = .03
    heading_error_rad: float = .005

    def __post_init__(self):
        values = (self.voxel_m, self.turn_error_rad_per_rad, self.heading_error_rad)
        if (self.aging not in ('time', 'transport') or not all(math.isfinite(v) and v > 0 for v in values)
                or type(self.max_points) is not int or self.max_points < 64):
            raise ValueError('Invalid memory configuration')


@dataclass(frozen=True)
class RetreatConfig:
    """Round 3: a braking/backward Pitch may pass the k*flow cap only along the drone's own corridor."""
    enabled: bool = False
    corridor_s: float = 7.
    tolerance_m: float = .05
    margin_m: float = .05
    coast_s_per_mps: float = 4.1
    flow_scale_mps: float = 3.88
    step_m: float = .02
    forward_min_mps: float = -.03

    def __post_init__(self):
        values = (self.corridor_s, self.tolerance_m, self.margin_m, self.coast_s_per_mps,
                  self.flow_scale_mps, self.step_m)
        if type(self.enabled) is not bool or not all(math.isfinite(v) and v > 0 for v in values):
            raise ValueError('Invalid retreat configuration')


@dataclass(frozen=True)
class PipelineConfig:
    brake: BrakeConfig = field(default_factory=BrakeConfig)
    veto: VetoConfig = field(default_factory=VetoConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    retreat: RetreatConfig = field(default_factory=RetreatConfig)
    anti_rollback: bool = False
    turn_latch: bool = True

    @classmethod
    def from_json(cls, path):
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        brake, veto = data['stage1_brake'], data['stage2_veto']
        model = veto['error_model']
        memory = {k: v for k, v in data.get('memory', {}).items() if k in MemoryConfig.__dataclass_fields__}
        retreat = {k: v for k, v in data.get('stage1_retreat', {}).items() if k in RetreatConfig.__dataclass_fields__}
        return cls(
            brake=BrakeConfig(v_stop=brake['v_stop'], flow_filter_tau_s=brake['flow_filter_tau_s'],
                              **{k: brake[k] for k in ('law', 'gain_pwm_per_flow') if k in brake}),
            veto=VetoConfig(r_veto_m=veto['r_veto_m'], side_min_deg=veto['side_min_deg'],
                side_max_deg=veto['side_max_deg'], history_s=veto['history_s'], max_frames=veto['max_frames'],
                base_error_m=model['base_m'], drift_m_per_s=model['drift_m_per_s'],
                random_walk_m_sqrt_s=model['random_walk_m_sqrt_s'],
                distance_error_fraction=model['distance_error_fraction']),
            memory=MemoryConfig(**memory),
            retreat=RetreatConfig(**retreat),
            anti_rollback=data['coordinator']['anti_rollback'], turn_latch=data['coordinator']['turn_latch'])


def within_limits(network, result):
    """True when every channel of `result` is between 1500 and `network` (same side, not larger)."""
    network, result = np.asarray(network, float), np.asarray(result, float)
    low, high = np.minimum(network, NEUTRAL), np.maximum(network, NEUTRAL)
    return bool(np.all((result >= low-1e-9) & (result <= high+1e-9))
                and np.all(result[0:1] == network[0:1]))


def proportional_brake_stage(network, filtered_flow, gain):
    """Stage 1, round 2. A braking Pitch is limited to gain * forward flow: the full network brake
    passes at speed, the allowed brake shrinks with the remaining forward motion and is zero at rest
    or when moving back, so no residual tilt is left to roll the drone backwards."""
    after = np.asarray(network, float).copy()
    if after[PITCH] >= NEUTRAL:
        return after, dict(active=False, reason='not_braking')
    cap = gain*max(float(filtered_flow), 0.)
    limited = NEUTRAL-cap > after[PITCH]
    if limited:
        after[PITCH] = NEUTRAL-cap
    return after, dict(active=bool(limited), reason='brake_above_speed' if limited else 'brake_within_speed',
                       cap_pwm=cap)


def retreat_check(trail_local, remembered, back_speed_mps, config, r_veto_m):
    """Round 3. Backward motion is allowed only along the drone's own recent corridor.

    `trail_local` are positions the body occupied (not while moving back) during the last
    corridor_s, in the current yaw frame relative to the body (FLU). The check segment runs straight
    back over L = margin + coast_s_per_mps * back speed (the distance the drone coasts once the stage
    removes the backward command). Every sample of it must lie within tolerance of the trail, and no
    remembered threat point behind the drone may lie within r_veto + its error of the segment.
    An empty memory behind is not a permission: only the trail is."""
    length = config.margin_m+config.coast_s_per_mps*max(back_speed_mps, 0.)
    samples = np.unique(np.r_[np.arange(0., length, config.step_m), length])  # always ends at L
    info = dict(retreat_length_m=float(length))
    if trail_local is None or not len(trail_local):
        return False, dict(info, retreat_block='no_corridor', corridor_m=0.)
    points = np.c_[-samples, np.zeros((len(samples), 2))]
    dist = np.min(np.linalg.norm(points[:, None, :]-trail_local[None, :, :], axis=2), axis=1)
    inside = dist <= config.tolerance_m
    corridor = float(samples[np.argmin(inside)-1]) if not inside.all() else float(samples[-1])
    if not inside[0]:
        corridor = 0.
    info['corridor_m'] = corridor
    if not inside.all():
        return False, dict(info, retreat_block='outside_corridor')
    if remembered is not None:
        local, error, _ = remembered
        behind = local[:, 0] < 0.
        if behind.any():
            p = local[behind]
            nearest = np.c_[np.clip(p[:, 0], -length, 0.), np.zeros((len(p), 2))]
            gap = np.linalg.norm(p-nearest, axis=1)-error[behind]
            info['memory_gap_m'] = float(gap.min())
            if gap.min() < r_veto_m:
                return False, dict(info, retreat_block='remembered_threat')
    return True, info


def brake_stage(network, filtered_flow, v_stop):
    """Stage 1. A braking Pitch (< 1500) passes while the drone still moves forward faster than
    v_stop and fades to 1500 once the motion is gone, so braking never turns into a rollback."""
    after = np.asarray(network, float).copy()
    if after[PITCH] >= NEUTRAL:
        return after, dict(active=False, reason='not_braking')
    x = float(np.clip(filtered_flow/v_stop, 0., 1.))
    gain = x*x*(3.-2.*x)
    after[PITCH] = NEUTRAL+(after[PITCH]-NEUTRAL)*gain
    return after, dict(active=gain < 1., reason='motion_gone' if gain < 1. else 'moving_forward', gain=gain)


def veto_stage(network, left_gap_m, right_gap_m, r_veto_m):
    """Stage 2. Roll or Yaw leading toward a side whose remembered obstacle is closer than r_veto
    is limited to 1500. Right is Roll/Yaw above 1500, left below (FLU, positive left)."""
    after = np.asarray(network, float).copy()
    blocked = {'left': left_gap_m < r_veto_m, 'right': right_gap_m < r_veto_m}
    vetoed = []
    for channel, name in ((ROLL, 'roll'), (YAW, 'yaw')):
        offset = after[channel]-NEUTRAL
        if (offset > 0 and blocked['right']) or (offset < 0 and blocked['left']):
            after[channel] = NEUTRAL
            vetoed.append(name)
    reason = 'side_obstacle_' + '_'.join(vetoed) if vetoed else (
        'side_obstacle_not_in_command_direction' if any(blocked.values()) else 'no_remembered_side_obstacle')
    finite = lambda v: float(v) if math.isfinite(v) else None  # JSON has no infinity
    return after, dict(active=bool(vetoed), reason=reason, vetoed=vetoed,
                       left_gap_m=finite(left_gap_m), right_gap_m=finite(right_gap_m))


class ThreatMemory:
    """Obstacle points of the last ToF frames carried by flow odometry and measured heading."""

    def __init__(self, reference_id, config):
        self.config = config
        self.observer = EmpiricalCorridor(reference_id,
            CorridorConfig(history_s=config.history_s, max_frames=config.max_frames), screen_hazards=False)
        self.report = {}
        self.resets = 0

    @property
    def reference_id(self):
        return self.observer.reference_id

    @property
    def faulted(self):
        return self.observer.fault is not None

    def reset(self, reference_id):
        self.observer.reset(reference_id)
        self.report = {}
        self.resets += 1

    def observe(self, snapshot, heading_sample, *, now_s):
        self.report = self.observer.update(snapshot, heading_sample, now_s=now_s)
        return self.report

    def error(self, age, distance):
        c = self.config
        return (c.base_error_m+c.drift_m_per_s*age+c.random_walk_m_sqrt_s*np.sqrt(age)
                +c.distance_error_fraction*distance)

    @property
    def size(self):
        return sum(len(f[4]) for f in self.observer.frames)

    def remembered(self, now_s):
        """Remembered points in the current yaw frame (FLU, no roll/pitch), per-point transport error
        and capture time; None when the memory has no valid pose or no points."""
        o = self.observer
        if o.fault or not o.frames or 'heading_rad' not in self.report:
            return None
        transform = rotation(0., 0., self.report['heading_rad'])
        parts = [((hits-o.position) @ transform, np.full(len(hits), self.error(now_s-captured, o.distance-distance)),
                  np.full(len(hits), captured)) for captured, _, distance, _, hits in o.frames if len(hits)]
        if not parts:
            return None
        return tuple(np.concatenate(x) for x in zip(*parts))

    def side_gaps(self, now_s):
        """Smallest (3D distance - transport error) to remembered side points, left and right."""
        c = self.config
        points = self.remembered(now_s)
        if points is None:
            return math.inf, math.inf
        local, error, _ = points
        bearing = np.degrees(np.arctan2(local[:, 1], local[:, 0]))
        side = (np.abs(bearing) > c.side_min_deg) & (np.abs(bearing) < c.side_max_deg)
        gap = np.linalg.norm(local, axis=1)-error
        gaps = {}
        for name, mask in (('left', side & (local[:, 1] > 0)), ('right', side & (local[:, 1] < 0))):
            gaps[name] = float(gap[mask].min()) if mask.any() else math.inf
        return gaps['left'], gaps['right']


class TransportMemory(ThreatMemory):
    """Voxel-thinned obstacle points aged by accumulated transport error, never by time alone.

    error = base + distance_fraction * path since capture
            + (turn_error * |heading turned since capture| + heading_error) * current range.
    A point is removed when its error exceeds r_veto: it can no longer place an obstacle inside
    the veto radius. The observer keeps only the newest frame; the store owns the history, and
    over capacity the least certain points go first, never the oldest by age.
    """

    def __init__(self, reference_id, config, memory_config):
        self.memory_config = memory_config
        self.config = config
        self.observer = EmpiricalCorridor(reference_id,
            CorridorConfig(history_s=1., max_frames=1), screen_hazards=False)
        self.report = {}
        self.resets = 0
        self._clear()

    def _clear(self):
        self.points = np.zeros((0, 3))
        self.path0 = np.zeros(0)
        self.turn0 = np.zeros(0)
        self.captured = np.zeros(0)
        self.turn = 0.
        self.heading = None
        self.removed_by_error = 0
        self.removed_by_capacity = 0

    def reset(self, reference_id):
        super().reset(reference_id)
        self._clear()

    @property
    def size(self):
        return len(self.points)

    def observe(self, snapshot, heading_sample, *, now_s):
        report = self.observer.update(snapshot, heading_sample, now_s=now_s)
        self.report = report
        o = self.observer
        if o.fault or 'heading_rad' not in report:
            return report
        heading = report['heading_rad']
        if self.heading is not None:
            self.turn += abs(math.remainder(heading-self.heading, math.tau))
        self.heading = heading
        if report.get('tof_state') == 'new' and o.frames and o.frames[-1][0] == now_s and len(o.frames[-1][4]):
            self._insert(o.frames[-1][4], now_s)
        self._prune()
        report['memory_points'] = len(self.points)
        return report

    def transport_error(self, ranges):
        c, m = self.config, self.memory_config
        return (c.base_error_m+c.distance_error_fraction*(self.observer.distance-self.path0)
                +(m.turn_error_rad_per_rad*(self.turn-self.turn0)+m.heading_error_rad)*ranges)

    def _insert(self, hits, now_s):
        n = len(hits)
        points = np.concatenate([self.points, hits])
        path0 = np.r_[self.path0, np.full(n, self.observer.distance)]
        turn0 = np.r_[self.turn0, np.full(n, self.turn)]
        captured = np.r_[self.captured, np.full(n, now_s)]
        keys = np.floor(points/self.memory_config.voxel_m).astype(np.int64)
        # np.unique keeps the first occurrence; on the reversed array that is the newest point.
        _, first = np.unique(keys[::-1], axis=0, return_index=True)
        keep = np.sort(len(points)-1-first)
        self.points, self.path0, self.turn0, self.captured = points[keep], path0[keep], turn0[keep], captured[keep]

    def _prune(self):
        if not len(self.points):
            return
        error = self.transport_error(np.linalg.norm(self.points-self.observer.position, axis=1))
        keep = error <= self.config.r_veto_m
        self.removed_by_error += int((~keep).sum())
        excess = int(keep.sum())-self.memory_config.max_points
        if excess > 0:
            order = np.argsort(np.where(keep, error, np.inf), kind='stable')
            keep[:] = False
            keep[order[:self.memory_config.max_points]] = True
            self.removed_by_capacity += excess
        self.points, self.path0, self.turn0, self.captured = (
            self.points[keep], self.path0[keep], self.turn0[keep], self.captured[keep])

    def remembered(self, now_s):
        o = self.observer
        if o.fault or 'heading_rad' not in self.report or not len(self.points):
            return None
        local = (self.points-o.position) @ rotation(0., 0., self.report['heading_rad'])
        return local, self.transport_error(np.linalg.norm(local, axis=1)), self.captured.copy()


class NavigationPipeline(XYCoordinator):
    """Drop-in XY coordinator whose avoidance logic is the two-stage limiting pipeline."""

    def __init__(self, *, reference_id, pipeline_config=None, **options):
        self.pipeline_config = pipeline_config or PipelineConfig()
        c = self.pipeline_config
        self.memory = (TransportMemory(reference_id, c.veto, c.memory) if c.memory.aging == 'transport'
                       else ThreatMemory(reference_id, c.veto))
        self.now_s = None
        self.trail = deque()
        super().__init__(**options)

    def reset(self):
        super().reset()
        self.pipeline_flow = None

    def contract(self):
        return super().contract() | dict(pipeline_version='navigation-pipeline-v1',
            pipeline=asdict(self.pipeline_config), stage_kinds=dict(brake_no_rollback=LIMITS, side_veto=LIMITS),
            order='network A -> brake_no_rollback -> side_veto -> XY base (turn_latch, STOP) -> altitude/STOP/arbiter',
            memory='flow-odometry + measured heading hit points, session-wide; veto only from remembered side hits; '
                   + ('aged by accumulated path and turn error, voxel-thinned, removed when error > r_veto'
                      if self.pipeline_config.memory.aging == 'transport' else 'last 4 s / 64 frames FIFO'))

    def observe(self, snapshot, heading_sample, *, now_s):
        self.now_s = now_s
        if heading_sample is None:
            return None
        report = self.memory.observe(snapshot, heading_sample, now_s=now_s)
        r, o = self.pipeline_config.retreat, self.memory.observer
        if r.enabled:
            if self.memory.faulted:
                self.trail.clear()
            elif float(o.velocity[0]) >= r.forward_min_mps:  # space entered while moving back is not corridor
                self.trail.append((now_s, o.position.copy()))
            while self.trail and now_s-self.trail[0][0] > r.corridor_s+1e-9:
                self.trail.popleft()
        return report

    def _retreat(self):
        c = self.pipeline_config
        m = self.memory
        if m.faulted or 'heading_rad' not in m.report or self.now_s is None:
            return False, dict(retreat_block='no_pose')
        trail = (np.array([p for _, p in self.trail]) if self.trail else None)
        local = None if trail is None else (trail-m.observer.position) @ rotation(0., 0., m.report['heading_rad'])
        back = max(-float(self.pipeline_flow), 0.)*c.retreat.flow_scale_mps
        return retreat_check(local, m.remembered(self.now_s), back, c.retreat, c.veto.r_veto_m)

    def apply(self, network_pwm, *, flow_xy, tof_m, dt, stop_requested=False):
        network = np.asarray(network_pwm, dtype=float).copy()
        flow = np.asarray(flow_xy, dtype=float)
        valid = (network.shape == (4,) and flow.shape == (2,) and np.isfinite(network).all()
                 and np.isfinite(flow).all() and math.isfinite(dt) and dt > 0)
        records, command = [], network.copy()
        if valid:
            c = self.pipeline_config
            alpha = -math.expm1(-dt/c.brake.flow_filter_tau_s)
            self.pipeline_flow = float(flow[0]) if self.pipeline_flow is None else float(
                self.pipeline_flow+alpha*(flow[0]-self.pipeline_flow))
            if stop_requested:
                for name in ('brake_no_rollback', 'side_veto'):
                    records.append(self._record(name, dict(active=False, reason='stop_requested'), network, network))
            else:
                after, info = (proportional_brake_stage(command, self.pipeline_flow, c.brake.gain_pwm_per_flow)
                               if c.brake.law == 'proportional'
                               else brake_stage(command, self.pipeline_flow, c.brake.v_stop))
                info['filtered_flow'] = self.pipeline_flow
                if c.retreat.enabled and info['active']:
                    permitted, extra = self._retreat()
                    info.update(extra)
                    if permitted:
                        after = command.copy()
                        info.update(active=False, reason='retreat_along_corridor')
                records.append(self._record('brake_no_rollback', info, command, after))
                command = after
                left, right = (self.memory.side_gaps(self.now_s) if self.now_s is not None
                               else (math.inf, math.inf))
                after, info = veto_stage(command, left, right, c.veto.r_veto_m)
                if self.memory.faulted:
                    info['reason'] = 'memory_fault'
                    info['memory_fault'] = self.memory.observer.fault
                records.append(self._record('side_veto', info, command, after))
                command = after
        result = super().apply(command, flow_xy=flow_xy, tof_m=tof_m, dt=dt, stop_requested=stop_requested)
        if valid:
            reasons = [r['stage'] for r in records if r['active']]
            self.last_decision['stages'] = records
            self.last_decision['reasons'] = list(self.last_decision['reasons'])+reasons
            self.last_decision['intervened'] = bool(self.last_decision['reasons'])
            self.last_decision['stage_limit_violation'] = not within_limits(network, command)
            self.last_decision['memory_resets'] = self.memory.resets
            self.last_decision['memory_points'] = self.memory.size
        return result

    @staticmethod
    def _record(stage, info, before, after):
        info = dict(info)
        active, reason = bool(info.pop('active')), info.pop('reason')
        return dict(stage=stage, kind=LIMITS, active=active, reason=reason, before=before.tolist(),
                    after=after.tolist(), **info)
