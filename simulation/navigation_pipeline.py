"""One navigation pipeline: network A -> [1] braking without rollback -> [2] side veto -> XY base.

Both stages are of kind 'limits': per channel the result lies between 1500 and the network
command (same side, never larger). They cannot generate a command, so generated_share is 0.
Every stage writes one record per tick into last_decision['stages'].

The XY base (turn_latch, STOP) runs last, on the limited command, exactly as in baseline A+XY.
Hazard memory is session-wide: a bridge reset restarts the control state, not the memory.
"""
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

    def __post_init__(self):
        if not all(math.isfinite(v) and v > 0 for v in asdict(self).values()):
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
class PipelineConfig:
    brake: BrakeConfig = field(default_factory=BrakeConfig)
    veto: VetoConfig = field(default_factory=VetoConfig)
    anti_rollback: bool = False
    turn_latch: bool = True

    @classmethod
    def from_json(cls, path):
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        brake, veto = data['stage1_brake'], data['stage2_veto']
        model = veto['error_model']
        return cls(
            brake=BrakeConfig(v_stop=brake['v_stop'], flow_filter_tau_s=brake['flow_filter_tau_s']),
            veto=VetoConfig(r_veto_m=veto['r_veto_m'], side_min_deg=veto['side_min_deg'],
                side_max_deg=veto['side_max_deg'], history_s=veto['history_s'], max_frames=veto['max_frames'],
                base_error_m=model['base_m'], drift_m_per_s=model['drift_m_per_s'],
                random_walk_m_sqrt_s=model['random_walk_m_sqrt_s'],
                distance_error_fraction=model['distance_error_fraction']),
            anti_rollback=data['coordinator']['anti_rollback'], turn_latch=data['coordinator']['turn_latch'])


def within_limits(network, result):
    """True when every channel of `result` is between 1500 and `network` (same side, not larger)."""
    network, result = np.asarray(network, float), np.asarray(result, float)
    low, high = np.minimum(network, NEUTRAL), np.maximum(network, NEUTRAL)
    return bool(np.all((result >= low-1e-9) & (result <= high+1e-9))
                and np.all(result[0:1] == network[0:1]))


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

    def side_gaps(self, now_s):
        """Smallest (3D distance - transport error) to remembered side points, left and right."""
        o, c = self.observer, self.config
        if o.fault or not o.frames or 'heading_rad' not in self.report:
            return math.inf, math.inf
        transform = rotation(0., 0., self.report['heading_rad'])
        gaps = {'left': math.inf, 'right': math.inf}
        for captured, _, distance, _, hits in o.frames:
            if not len(hits):
                continue
            local = (hits-o.position) @ transform  # world -> body (FLU)
            error = self.error(now_s-captured, o.distance-distance)
            bearing = np.degrees(np.arctan2(local[:, 1], local[:, 0]))
            side = (np.abs(bearing) > c.side_min_deg) & (np.abs(bearing) < c.side_max_deg)
            gap = np.linalg.norm(local, axis=1)-error
            for name, mask in (('left', side & (local[:, 1] > 0)), ('right', side & (local[:, 1] < 0))):
                if mask.any():
                    gaps[name] = min(gaps[name], float(gap[mask].min()))
        return gaps['left'], gaps['right']


class NavigationPipeline(XYCoordinator):
    """Drop-in XY coordinator whose avoidance logic is the two-stage limiting pipeline."""

    def __init__(self, *, reference_id, pipeline_config=None, **options):
        self.pipeline_config = pipeline_config or PipelineConfig()
        self.memory = ThreatMemory(reference_id, self.pipeline_config.veto)
        self.now_s = None
        super().__init__(**options)

    def reset(self):
        super().reset()
        self.pipeline_flow = None

    def contract(self):
        return super().contract() | dict(pipeline_version='navigation-pipeline-v1',
            pipeline=asdict(self.pipeline_config), stage_kinds=dict(brake_no_rollback=LIMITS, side_veto=LIMITS),
            order='network A -> brake_no_rollback -> side_veto -> XY base (turn_latch, STOP) -> altitude/STOP/arbiter',
            memory='flow-odometry + measured heading hit points, session-wide; veto only from remembered side hits')

    def observe(self, snapshot, heading_sample, *, now_s):
        self.now_s = now_s
        if heading_sample is None:
            return None
        return self.memory.observe(snapshot, heading_sample, now_s=now_s)

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
                after, info = brake_stage(command, self.pipeline_flow, c.brake.v_stop)
                info['filtered_flow'] = self.pipeline_flow
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
        return result

    @staticmethod
    def _record(stage, info, before, after):
        info = dict(info)
        active, reason = bool(info.pop('active')), info.pop('reason')
        return dict(stage=stage, kind=LIMITS, active=active, reason=reason, before=before.tolist(),
                    after=after.tolist(), **info)
