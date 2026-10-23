"""Optional sensor-only XY experiment; no memory routing or altitude ownership.

This is not a hardware safety arbiter. It limits requested backward tilt and
remembers a turn sign. Opt-in search/stop behaviors use sensors, never world truth.
"""
from dataclasses import asdict, dataclass
import math

import numpy as np


@dataclass(frozen=True)
class XYConfig:
    flow_filter_tau_s: float = .06
    stopped_flow: float = .02
    forward_flow: float = .06
    threat_m: float = .8
    clear_m: float = .95
    clear_dwell_s: float = .8
    acquire_yaw_pwm: float = 20.

    def __post_init__(self):
        if (not all(math.isfinite(v) for v in asdict(self).values())
                or not -1 <= self.stopped_flow < self.forward_flow <= 1
                or not 0 < self.threat_m < self.clear_m
                or min(self.flow_filter_tau_s, self.clear_dwell_s, self.acquire_yaw_pwm) <= 0):
            raise ValueError('Invalid XY coordinator configuration')


@dataclass(frozen=True)
class BehaviorConfig:
    # Conditional Pareto: density proportional to t**(-alpha-1), not clipped draws.
    levy_alpha: float = 1.8
    levy_min_s: float = 1.5
    levy_max_s: float = 6.
    search_duration_s: float = .28
    search_amplitude_pwm: float = 110.
    search_clear_m: float = 1.15
    search_preempt_m: float = .95
    reflex_preempt_pwm: float = 25.
    side_tie_m: float = .05
    stop_critical_m: float = .50
    stop_gain_pwm: float = 400.
    stop_limit_pwm: float = 100.
    stop_transition_s: float = .12
    hover_flow: float = .0075  # nominal 0.03 m/s at 1 m, max_flow_rate=4
    hover_dwell_s: float = .5

    def __post_init__(self):
        if (not all(math.isfinite(v) and v > 0 for v in asdict(self).values())
                or not self.levy_min_s < self.levy_max_s
                or not self.stop_critical_m < self.search_preempt_m < self.search_clear_m
                or self.search_duration_s >= self.levy_min_s
                or max(self.search_amplitude_pwm, self.reflex_preempt_pwm, self.stop_limit_pwm) > 500):
            raise ValueError('Invalid behavior configuration')


class XYCoordinator:
    def __init__(self, *, anti_rollback=False, turn_latch=False, config=None,
                 levy_explore=False, stop_hover=False, behavior_config=None, seed=42):
        if any(type(x) is not bool for x in (anti_rollback, turn_latch, levy_explore, stop_hover)):
            raise ValueError('Coordinator features must be explicit booleans')
        if type(seed) is not int or seed < 0:
            raise ValueError('Seed must be a nonnegative integer')
        self.anti_rollback, self.turn_latch = anti_rollback, turn_latch
        self.config = config or XYConfig()
        self.levy_explore, self.stop_hover = levy_explore, stop_hover
        self.behavior_config = behavior_config or BehaviorConfig()
        self.seed = seed
        self.reset()

    def reset(self):
        self.filtered_flow = None
        self.turn_sign = 0
        self.clear_elapsed = 0.
        self.last_decision = {}
        self.filtered_flow_y = None
        self.rng = np.random.default_rng(self.seed)
        self.wait_target = self._sample_interval() if self.levy_explore else 0.
        self.wait_elapsed = 0.
        self.pulse_elapsed = None
        self.search_sign = 0
        self.stop_command = None
        self.previous_pr = None
        self.hover_elapsed = 0.

    def _sample_interval(self):
        b = self.behavior_config
        u = self.rng.random()
        return b.levy_min_s * (1-u*(1-(b.levy_min_s/b.levy_max_s)**b.levy_alpha))**(-1/b.levy_alpha)

    def _reset_search(self):
        self.wait_elapsed, self.pulse_elapsed, self.search_sign = 0., None, 0
        if self.levy_explore:
            self.wait_target = self._sample_interval()

    def contract(self):
        contract = dict(version='xy-coordinator-v1', anti_rollback=self.anti_rollback,
                    turn_latch=self.turn_latch, config=asdict(self.config),
                    inputs='normalized flow[2], ToF[64] in metres, control dt',
                    output_order=['throttle', 'roll', 'pitch', 'yaw'])
        if self.levy_explore or self.stop_hover:
            contract.update(version='xy-coordinator-v2', levy_explore=self.levy_explore,
                stop_hover=self.stop_hover, seed=self.seed, behavior_config=asdict(self.behavior_config),
                priority='critical STOP evasion > STOP/HOVER > reflex/latch > search',
                search_distribution='conditional Pareto CDF; density t**(-alpha-1)',
                stop_semantics='velocity damping; HOVER is a flow estimate, not position lock')
        return contract

    def apply(self, network_pwm, *, flow_xy, tof_m, dt, stop_requested=False):
        if type(stop_requested) is not bool or (stop_requested and not self.stop_hover):
            raise ValueError('STOP requires an enabled stop_hover feature and boolean request')
        action = np.asarray(network_pwm, dtype=float).copy()
        flow = np.asarray(flow_xy, dtype=float)
        ranges = np.asarray(tof_m, dtype=float).ravel()
        if (not math.isfinite(dt) or dt <= 0 or action.shape != (4,)
                or flow.shape != (2,) or ranges.shape != (64,)
                or not all(np.isfinite(x).all() for x in (action, flow, ranges))
                or np.any(action < 1000) or np.any(action > 2000) or np.any(ranges < 0)):
            raise ValueError('Invalid XY command, sensors or control dt')
        c, reasons = self.config, []
        self.filtered_flow = (float(flow[0]) if self.filtered_flow is None else
            self.filtered_flow + -math.expm1(-dt/c.flow_filter_tau_s)*(flow[0]-self.filtered_flow))
        self.filtered_flow_y = (float(flow[1]) if self.filtered_flow_y is None else
            self.filtered_flow_y + -math.expm1(-dt/c.flow_filter_tau_s)*(flow[1]-self.filtered_flow_y))
        brake_gain = 1.
        if self.anti_rollback and action[2] < 1500.:
            x = float(np.clip((self.filtered_flow-c.stopped_flow)/(c.forward_flow-c.stopped_flow), 0., 1.))
            brake_gain = x*x*(3.-2.*x)
            action[2] = 1500. + (action[2]-1500.)*brake_gain
            if brake_gain < 1.:
                reasons.append('anti_rollback')

        if self.turn_latch:
            distance = float(ranges.min())
            self.clear_elapsed = self.clear_elapsed + dt if distance >= c.clear_m else 0.
            if self.clear_elapsed + 1e-9 >= c.clear_dwell_s:
                self.turn_sign = 0
            yaw = action[3]-1500.
            if not self.turn_sign and distance < c.threat_m and abs(yaw) >= c.acquire_yaw_pwm:
                self.turn_sign = 1 if yaw > 0 else -1
            if self.turn_sign and yaw*self.turn_sign < 0:
                action[3] = 1500. + self.turn_sign*abs(yaw)
                reasons.append('turn_latch')
        self.last_decision = dict(intervened=bool(reasons), reasons=reasons,
            filtered_flow_x=float(self.filtered_flow), brake_gain=brake_gain,
            turn_sign=self.turn_sign, clear_elapsed_s=self.clear_elapsed)
        if self.levy_explore or self.stop_hover:
            action = self._behavior(action, float(network_pwm[3]), ranges, dt, stop_requested)
        self.previous_pr = action[1:3].copy()
        return action

    def _behavior(self, action, network_yaw, ranges, dt, stop_requested):
        b, d = self.behavior_config, self.last_decision
        distance = float(ranges.min())
        d.update(state='CRUISE', search_pwm=0., search_event=None,
                 stop_requested=stop_requested, filtered_flow_y=float(self.filtered_flow_y))
        if stop_requested:
            if self.pulse_elapsed is not None or self.wait_elapsed > 0:
                d['search_event'] = 'preempted_by_stop'
                self._reset_search()
            if distance < b.stop_critical_m:
                self.stop_command = None
                self.hover_elapsed = 0.
                d['state'] = 'STOP_EVASION'
                d['reasons'].append('stop_yields_to_critical_threat')
            else:
                target = 1500. + np.clip(b.stop_gain_pwm*np.array(
                    [self.filtered_flow_y, -self.filtered_flow]), -b.stop_limit_pwm, b.stop_limit_pwm)
                if self.stop_command is None:
                    self.stop_command = (self.previous_pr.copy() if self.previous_pr is not None
                                         else action[1:3].copy())
                self.stop_command += -math.expm1(-dt/b.stop_transition_s)*(target-self.stop_command)
                action[1:3] = self.stop_command
                action[3] = 1500.
                still = math.hypot(self.filtered_flow, self.filtered_flow_y) <= b.hover_flow
                self.hover_elapsed = self.hover_elapsed + dt if still else 0.
                d['state'] = 'HOVER' if self.hover_elapsed + 1e-9 >= b.hover_dwell_s else 'STOP'
                d['reasons'].append('stop_hover')
        else:
            self.stop_command = None
            self.hover_elapsed = 0.
            if self.levy_explore:
                preempt = (distance < b.search_preempt_m
                           or abs(network_yaw-1500.) >= b.reflex_preempt_pwm or self.turn_sign != 0)
                if preempt:
                    if self.pulse_elapsed is not None or self.wait_elapsed > 0:
                        d['search_event'] = 'preempted'
                        self._reset_search()
                elif self.pulse_elapsed is None:
                    # Require a continuously clear interval before starting a pulse.
                    self.wait_elapsed = self.wait_elapsed + dt if distance >= b.search_clear_m else 0.
                    if self.wait_elapsed + 1e-9 >= self.wait_target:
                        image = ranges.reshape(8,8)
                        difference = float(image[:,4:].mean()-image[:,:4].mean())
                        self.search_sign = (int(self.rng.choice([-1,1])) if abs(difference) <= b.side_tie_m
                                            else (1 if difference > 0 else -1))
                        self.pulse_elapsed = 0.
                        d['search_event'] = 'started'
                if self.pulse_elapsed is not None:
                    if self.pulse_elapsed + 1e-9 >= b.search_duration_s:
                        self._reset_search()
                        d['search_event'] = 'completed'
                    else:
                        pulse = self.search_sign*b.search_amplitude_pwm*math.sin(
                            math.pi*self.pulse_elapsed/b.search_duration_s)**2
                        action[3] = 1500. + pulse
                        self.pulse_elapsed += dt
                        d.update(state='SEARCH', search_pwm=pulse)
                        d['reasons'].append('levy_search')
        d.update(intervened=bool(d['reasons']), search_wait_s=self.wait_elapsed,
                 search_interval_s=self.wait_target)
        return action
