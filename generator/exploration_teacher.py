"""Experimental, stateful scan supervisor around the frozen v4 reflex.

This generates research actions, not learned connectome behavior. No dataset
factory selects it implicitly. Throttle remains an unused interface placeholder;
the shared rollout must supply AltitudeHold.
"""
from dataclasses import asdict, dataclass

import numpy as np

from configs.flight_config import CONTROL_DT, SENSOR_DIM, PWM_MID
from generator.generate_reflex_dataset import ExpertReflexPolicy
from generator.reflex_contract import BRAKE_FIRST_TEACHER_VERSION as TEACHER_VERSION


class ScanBlocked(RuntimeError):
    """Stop a research episode when settling fails; never force a scan."""


@dataclass(frozen=True)
class ScanConfig:
    interval_s: float = 8.
    duration_s: float = 1.5
    yaw_offset_pwm: float = 80.
    settle_flow: float = .03
    settle_s: float = .20
    abort_flow: float = .08
    brake_timeout_s: float = 3.
    recovery_s: float = .4

    def __post_init__(self):
        if any(not np.isfinite(v) or v <= 0 for v in asdict(self).values()):
            raise ValueError('Scan parameters must be positive and finite')
        if not self.settle_flow < self.abort_flow <= 1:
            raise ValueError('Require settle_flow < abort_flow <= 1')
        if self.yaw_offset_pwm > 200 or self.brake_timeout_s <= self.settle_s:
            raise ValueError('Invalid scan yaw offset or settling timeout')


class ExplorationTeacher:
    version = 'exploration-teacher-v1'
    training_ready = False

    def __init__(self, config=None):
        self.config = config if config is not None else ScanConfig()
        self.reflex = ExpertReflexPolicy(noise_std_pwm=0)
        self.teacher_parameters = dict(scan=asdict(self.config), reflex_version=TEACHER_VERSION,
                               reflex=dict(self.reflex.parameters))
        self.reset()

    def reset(self):
        self.reflex.reset()
        self.mode = 'move'
        self.elapsed_s = self._phase_s = self._since_scan_s = 0.
        self._action_time_s = 0.
        self._clear_s = self._settled_s = 0.
        self._scan_sign = 1.
        self.scan_reason = None
        self.completed_scans = self.interrupted_scans = 0
        self.events = []
        self.diagnostics = {}

    reset_state = reset

    def _transition(self, mode, reason):
        if self.mode != mode:
            self.events.append(dict(time_s=self._action_time_s, from_mode=self.mode,
                                    to=mode, reason=reason))
            self.mode = mode
            self._phase_s = self._settled_s = 0.

    def _damped(self, obs, cruise=0., yaw=0.):
        r = self.reflex
        return np.clip(np.array([r.pwm_hover,
            r.pwm_level_roll + r.flow_gain_pwm*obs[1],
            PWM_MID + cruise - r.flow_gain_pwm*obs[0],
            r.pwm_neutral_yaw + yaw], dtype=np.float32), r.pwm_min, r.pwm_max)

    def step_np(self, flow, tof, memory_ring, dt=CONTROL_DT):
        return self.step(np.concatenate([np.ravel(flow), np.ravel(tof),
                                        np.ravel(memory_ring)]), dt=dt)

    def step(self, obs_74, dt=CONTROL_DT):
        obs = np.asarray(obs_74, dtype=np.float32)
        if (obs.shape != (SENSOR_DIM,) or not np.isfinite(obs).all()
                or np.any(abs(obs[:2]) > 1) or np.any(obs[2:] < 0)
                or np.any(obs[2:] > 1)):
            raise ValueError('Expected finite normalized [flow(2), ToF(64), memory(8)]')
        if not np.isfinite(dt) or not 0 < dt <= .1:
            raise ValueError('dt must be finite and within (0, .1] seconds')
        if self.mode == 'blocked':
            raise ScanBlocked('Unable to settle before scan; reset required')

        self._action_time_s = self.elapsed_s
        self.elapsed_s += dt
        self._since_scan_s += dt
        r, c = self.reflex, self.config
        distance = float(obs[2:66].min()) * r.max_sensor_range_m
        flow = float(np.linalg.norm(obs[:2]))
        clear = distance >= r.distance_threshold_m + r.release_margin_m
        self._clear_s = self._clear_s + dt if clear else 0.
        clear_held = self._clear_s + 1e-9 >= r.clear_hold_s
        self.diagnostics = dict(min_distance_m=distance, flow_norm=flow,
            tof_spread_m=float(np.ptp(obs[2:66])) * r.max_sensor_range_m)
        reflex_action = r.step(obs, dt=dt)

        if distance < r.distance_threshold_m:
            if self.mode == 'scan':
                self.interrupted_scans += 1
            self._scan_sign = float(r.latched_turn or self._scan_sign)
            self.scan_reason = 'after_avoidance'
            self._transition('avoid', 'near_obstacle')

        if self.mode == 'avoid':
            self._settled_s = (self._settled_s + dt
                if clear and flow <= r.brake_release_flow else 0.)
            if not (clear_held and self._settled_s + 1e-9 >= r.brake_settle_s):
                self._phase_s += dt
                return reflex_action
            self._transition('brake', 'after_avoidance')

        if self.mode == 'move' and self._since_scan_s + 1e-9 >= c.interval_s:
            self.scan_reason = 'periodic'
            self._scan_sign = 1.
            self._transition('brake', self.scan_reason)

        if self.mode == 'scan':
            if not clear or flow >= c.abort_flow:
                self.interrupted_scans += 1
                self._transition('brake', 'scan_interrupted')
            elif self._phase_s + 1e-9 >= c.duration_s:
                self.completed_scans += 1
                self._since_scan_s = 0.
                self._transition('recover', 'scan_complete')

        if self.mode == 'recover' and not clear:
            self._transition('brake', 'clearance_lost')

        if self.mode == 'brake':
            self._settled_s = (self._settled_s + dt
                if clear_held and flow <= c.settle_flow else 0.)
            if self._settled_s + 1e-9 >= c.settle_s:
                self._transition('scan', self.scan_reason)
            elif self._phase_s + dt + 1e-9 >= c.brake_timeout_s:
                self._transition('blocked', 'settle_timeout')
                raise ScanBlocked('Unable to settle before scan; reset required')

        if self.mode == 'move':
            action = reflex_action
        elif self.mode == 'scan':
            action = self._damped(obs, yaw=self._scan_sign*c.yaw_offset_pwm)
        elif self.mode == 'recover':
            fraction = min(1., (self._phase_s + dt)/c.recovery_s)
            action = self._damped(obs, cruise=fraction*(r.pwm_cruise_pitch-PWM_MID))
            if fraction >= 1.:
                self._transition('move', 'cruise_restored')
        else:
            action = self._damped(obs)
        self._phase_s += dt
        return action
