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
    # Round 5: the heading-turn term models a gyro scale-factor error. The simulated heading does not
    # drift, so the calibrated profile switches it off; it stays available for hardware.
    turn_term: bool = True

    def __post_init__(self):
        values = (self.voxel_m, self.turn_error_rad_per_rad, self.heading_error_rad)
        if (self.aging not in ('time', 'transport') or type(self.turn_term) is not bool
                or not all(math.isfinite(v) and v > 0 for v in values)
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
    # Round 3b: 'speed' (r3, L from the lagged speed only) or 'tilt' (coast predicted from tilt and speed,
    # the backward command limited so the predicted coast fits the free corridor length).
    prediction: str = 'speed'
    beta_mps2_per_deg: float = .1737
    gamma_per_s: float = .2565
    tilt_tau_s: float = .2595
    tilt_gain_deg_per_pwm_tick: float = .005388
    lag_s: float = .08
    max_check_m: float = 2.
    # Round 3b, second attempt: credit forward speed and forward tilt (signed linear model). The first
    # attempt (False) ignored them and so limited the network's ordinary braking from forward motion.
    signed: bool = False

    def __post_init__(self):
        values = (self.corridor_s, self.tolerance_m, self.margin_m, self.coast_s_per_mps,
                  self.flow_scale_mps, self.step_m, self.beta_mps2_per_deg, self.gamma_per_s, self.tilt_tau_s,
                  self.tilt_gain_deg_per_pwm_tick, self.lag_s, self.max_check_m)
        if (type(self.enabled) is not bool or type(self.signed) is not bool
                or self.prediction not in ('speed', 'tilt')
                or not all(math.isfinite(v) and v > 0 for v in values)):
            raise ValueError('Invalid retreat configuration')

    def signed_coast_m(self, back_speed_mps, back_tilt_deg, next_back_tilt_deg=None):
        """Signed linear model: net backward displacement if the backward command stops after the
        next tick. Speed crosses zero at most once, so the final displacement is the farthest point back."""
        s = back_speed_mps+(self.beta_mps2_per_deg*back_tilt_deg-self.gamma_per_s*back_speed_mps)*self.lag_s
        tilt = back_tilt_deg if next_back_tilt_deg is None else next_back_tilt_deg
        return (s+self.beta_mps2_per_deg*tilt*self.tilt_tau_s)/self.gamma_per_s

    def signed_max_back_pwm(self, free_m, back_speed_mps, back_tilt_deg):
        """Closed form of the largest backward command with signed_coast <= free_m - margin."""
        s = back_speed_mps+(self.beta_mps2_per_deg*back_tilt_deg-self.gamma_per_s*back_speed_mps)*self.lag_s
        tilt_max = ((free_m-self.margin_m)*self.gamma_per_s-s)/(self.beta_mps2_per_deg*self.tilt_tau_s)
        a = math.exp(-.02/self.tilt_tau_s)
        return max((tilt_max-a*back_tilt_deg)/self.tilt_gain_deg_per_pwm_tick, 0.)

    def coast_m(self, back_speed_mps, back_tilt_deg):
        """Distance the drone still travels backwards if the backward command is removed now.

        Linear model of round 2: tilt decays with tau, accel = beta*tilt - gamma*v, so the remaining
        displacement is (v + beta*tilt*tau)/gamma. The lagged speed is advanced by lag_s of the
        current acceleration. Forward tilt is not credited (it would only shorten the coast)."""
        v, tilt = max(back_speed_mps, 0.), max(back_tilt_deg, 0.)
        v += max(self.beta_mps2_per_deg*tilt-self.gamma_per_s*v, 0.)*self.lag_s
        return (v+self.beta_mps2_per_deg*tilt*self.tilt_tau_s)/self.gamma_per_s

    def max_back_pwm(self, free_m, back_speed_mps, back_tilt_deg):
        """Largest backward command whose tilt after one tick keeps the predicted coast within
        free_m - margin; 0 when even removing the command now does not fit."""
        room = free_m-self.margin_m
        v = max(back_speed_mps, 0.)
        a = math.exp(-.02/self.tilt_tau_s)
        tilt_now = max(back_tilt_deg, 0.)
        # coast(v, tilt_next) <= room, solved for tilt_next (coast is increasing in tilt)
        lo, hi = 0., 90.
        if self.coast_m(v, lo) > room:
            return 0.
        for _ in range(40):
            mid = (lo+hi)/2
            lo, hi = (mid, hi) if self.coast_m(v, mid) <= room else (lo, mid)
        return max((lo-a*tilt_now)/self.tilt_gain_deg_per_pwm_tick, 0.)


@dataclass(frozen=True)
class HoldConfig:
    """Round 4: on a neutral axis, hold zero velocity near known obstacles or after a blocked retreat."""
    enabled: bool = False
    gain_pwm_per_flow: float = 400.
    window_s: float = 1.
    cap_pwm: float = 10.
    neutral_pwm: float = 10.
    near_m: float = .5
    after_block_s: float = 1.

    def __post_init__(self):
        values = (self.gain_pwm_per_flow, self.window_s, self.cap_pwm, self.neutral_pwm, self.near_m,
                  self.after_block_s)
        if type(self.enabled) is not bool or not all(math.isfinite(v) and v > 0 for v in values):
            raise ValueError('Invalid hold configuration')


def hold_stage(command, mean_flow, config, *, trigger):
    """Stage 3 (kind generates). On each of Pitch and Roll that is neutral (within neutral_pwm of
    1500) the stage adds a correction against the averaged flow, as the XY STOP law does
    (Pitch -gain*flow_x, Roll +gain*flow_y), clipped to cap_pwm. Nothing happens without a trigger."""
    after = np.asarray(command, float).copy()
    if not trigger:
        return after, dict(active=False, reason='no_trigger')
    if mean_flow is None:
        return after, dict(active=False, reason='flow_window_filling', trigger=trigger)
    corrections = {}
    for channel, name, value in ((PITCH, 'pitch', -config.gain_pwm_per_flow*mean_flow[0]),
                                 (ROLL, 'roll', config.gain_pwm_per_flow*mean_flow[1])):
        if abs(after[channel]-NEUTRAL) <= config.neutral_pwm:
            delta = float(np.clip(value, -config.cap_pwm, config.cap_pwm))
            after[channel] += delta
            corrections[name] = delta
    active = any(abs(v) > 1e-9 for v in corrections.values())
    return after, dict(active=active, reason=('hold_'+trigger) if active else 'axes_not_neutral',
                       trigger=trigger, corrections=corrections, mean_flow=[float(v) for v in mean_flow])


@dataclass(frozen=True)
class PipelineConfig:
    brake: BrakeConfig = field(default_factory=BrakeConfig)
    veto: VetoConfig = field(default_factory=VetoConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    retreat: RetreatConfig = field(default_factory=RetreatConfig)
    hold: HoldConfig = field(default_factory=HoldConfig)
    anti_rollback: bool = False
    turn_latch: bool = True

    @classmethod
    def from_json(cls, path):
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        brake, veto = data['stage1_brake'], data['stage2_veto']
        model = veto['error_model']
        memory = {k: v for k, v in data.get('memory', {}).items() if k in MemoryConfig.__dataclass_fields__}
        retreat = {k: v for k, v in data.get('stage1_retreat', {}).items() if k in RetreatConfig.__dataclass_fields__}
        hold = {k: v for k, v in data.get('stage3_hold', {}).items() if k in HoldConfig.__dataclass_fields__}
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
            hold=HoldConfig(**hold),
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


def retreat_room(trail_local, remembered, config, r_veto_m):
    """Round 3b. Free straight-back length: up to the first sample outside the own corridor or
    the first sample within r_veto + error of a remembered threat behind the drone, at most
    max_check_m. An empty memory is never room by itself: without a corridor the room is 0."""
    samples = np.arange(0., config.max_check_m+config.step_m/2, config.step_m)
    if trail_local is None or not len(trail_local):
        return 0., dict(retreat_block='no_corridor', corridor_m=0.)
    points = np.c_[-samples, np.zeros((len(samples), 2))]
    dist = np.min(np.linalg.norm(points[:, None, :]-trail_local[None, :, :], axis=2), axis=1)
    outside = np.flatnonzero(dist > config.tolerance_m)
    corridor = float(samples[outside[0]-1]) if len(outside) and outside[0] > 0 else (
        0. if len(outside) else float(samples[-1]))
    room, block = corridor, 'outside_corridor'
    if remembered is not None:
        local, error, _ = remembered
        behind = local[:, 0] < 0.
        if behind.any():
            p, e = local[behind], error[behind]
            # threat reached at the first sample whose distance to the point is below r_veto + error
            d = np.linalg.norm(points[:, None, :]-p[None, :, :], axis=2)-e[None, :]
            hit = np.flatnonzero((d < r_veto_m).any(axis=1))
            if len(hit):
                threat = float(samples[hit[0]-1]) if hit[0] > 0 else 0.
                if threat < room:
                    room, block = threat, 'remembered_threat'
    return room, dict(corridor_m=corridor, room_m=room, room_limit=block)


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
                +(m.turn_error_rad_per_rad*(self.turn-self.turn0)*m.turn_term+m.heading_error_rad)*ranges)

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
        self.tilt_deg = None
        self.flow_window = None
        self.hold_until = None
        super().__init__(**options)

    def reset(self):
        super().reset()
        self.pipeline_flow = None

    def contract(self):
        return super().contract() | dict(pipeline_version='navigation-pipeline-v1',
            pipeline=asdict(self.pipeline_config), stage_kinds=dict(brake_no_rollback=LIMITS, side_veto=LIMITS,
                **({'velocity_hold': GENERATES} if self.pipeline_config.hold.enabled else {})),
            order='network A -> brake_no_rollback -> side_veto -> XY base (turn_latch, STOP) -> altitude/STOP/arbiter',
            memory='flow-odometry + measured heading hit points, session-wide; veto only from remembered side hits; '
                   + ('aged by accumulated path and turn error, voxel-thinned, removed when error > r_veto'
                      if self.pipeline_config.memory.aging == 'transport' else 'last 4 s / 64 frames FIFO'))

    def observe(self, snapshot, heading_sample, *, now_s):
        self.now_s = now_s
        if heading_sample is None:
            return None
        report = self.memory.observe(snapshot, heading_sample, now_s=now_s)
        try:
            self.tilt_deg = math.degrees(float(snapshot.attitude.value[1]))  # +: nose down, forward
        except (AttributeError, TypeError, IndexError):
            self.tilt_deg = None
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
        if c.retreat.prediction == 'tilt':
            if getattr(self, 'tilt_deg', None) is None:
                return 0., dict(retreat_block='no_attitude')
            room, info = retreat_room(local, m.remembered(self.now_s), c.retreat, c.veto.r_veto_m)
            back_tilt = -self.tilt_deg
            if c.retreat.signed:
                back = -float(self.pipeline_flow)*c.retreat.flow_scale_mps  # signed: forward speed credits
                allowed = c.retreat.signed_max_back_pwm(room, back, back_tilt)
                coast = c.retreat.signed_coast_m(back, back_tilt)
            else:
                allowed = c.retreat.max_back_pwm(room, back, back_tilt)
                coast = c.retreat.coast_m(back, back_tilt)
            return allowed, dict(info, back_speed_mps=back, back_tilt_deg=back_tilt,
                                 predicted_coast_m=coast, allowed_back_pwm=allowed)
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
            if c.hold.enabled:
                if self.flow_window is None:
                    self.flow_window = deque(maxlen=max(1, round(c.hold.window_s/dt)))
                self.flow_window.append(flow.copy())
            self.pipeline_flow = float(flow[0]) if self.pipeline_flow is None else float(
                self.pipeline_flow+alpha*(flow[0]-self.pipeline_flow))
            if stop_requested:
                for name in ('brake_no_rollback', 'side_veto'):
                    records.append(self._record(name, dict(active=False, reason='stop_requested'), network, network))
                if c.hold.enabled:
                    records.append(self._record('velocity_hold', dict(active=False, reason='stop_requested'),
                                                network, network, kind=GENERATES))
            else:
                after, info = (proportional_brake_stage(command, self.pipeline_flow, c.brake.gain_pwm_per_flow)
                               if c.brake.law == 'proportional'
                               else brake_stage(command, self.pipeline_flow, c.brake.v_stop))
                info['filtered_flow'] = self.pipeline_flow
                if c.retreat.enabled and info['active'] and c.retreat.prediction == 'tilt':
                    allowed, extra = self._retreat()
                    info.update(extra)
                    # the larger of the k*flow brake and the corridor retreat, never beyond the network
                    after[PITCH] = max(command[PITCH], NEUTRAL-max(info['cap_pwm'], allowed))
                    if after[PITCH] <= command[PITCH]+1e-9:
                        after[PITCH] = command[PITCH]
                        info.update(active=False, reason='retreat_along_corridor')
                    elif allowed > info['cap_pwm']:
                        info.update(reason='retreat_speed_limited')
                elif c.retreat.enabled and info['active']:
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
                if c.hold.enabled:
                    command = self._hold(records, command)
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
    def _record(stage, info, before, after, kind=LIMITS):
        info = dict(info)
        active, reason = bool(info.pop('active')), info.pop('reason')
        return dict(stage=stage, kind=kind, active=active, reason=reason, before=before.tolist(),
                    after=after.tolist(), **info)

    def _hold(self, records, command):
        h, c = self.pipeline_config.hold, self.pipeline_config
        brake = records[0]
        # A blocked retreat: the network asks to move back, stage 1 limits it and the drone is not
        # braking from forward motion (filtered forward flow at or below v_stop).
        if (brake['active'] and brake['before'][PITCH] < NEUTRAL
                and self.pipeline_flow is not None and self.pipeline_flow <= c.brake.v_stop):
            self.hold_until = self.now_s+h.after_block_s if self.now_s is not None else None
        nearest = math.inf
        if self.now_s is not None:
            remembered = self.memory.remembered(self.now_s)
            if remembered is not None:
                nearest = float(np.min(np.linalg.norm(remembered[0], axis=1)))
        trigger = ('near_obstacle' if nearest <= h.near_m else
                   'after_blocked_retreat' if (self.hold_until is not None and self.now_s is not None
                                               and self.now_s <= self.hold_until) else None)
        full = len(self.flow_window) == self.flow_window.maxlen
        mean = np.mean(self.flow_window, axis=0) if full else None
        after, info = hold_stage(command, mean, h, trigger=trigger)
        info['nearest_memory_m'] = nearest if math.isfinite(nearest) else None
        records.append(self._record('velocity_hold', info, command, after, kind=GENERATES))
        return after
