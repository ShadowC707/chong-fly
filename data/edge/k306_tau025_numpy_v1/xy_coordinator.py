"""Optional sensor-only XY experiment; no memory routing or altitude ownership.

This is not a hardware safety arbiter. It limits requested backward tilt and
remembers a turn sign; it never manufactures a yaw pulse or reads world truth.
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


class XYCoordinator:
    def __init__(self, *, anti_rollback=False, turn_latch=False, config=None):
        if type(anti_rollback) is not bool or type(turn_latch) is not bool:
            raise ValueError('Coordinator features must be explicit booleans')
        self.anti_rollback, self.turn_latch = anti_rollback, turn_latch
        self.config = config or XYConfig()
        self.reset()

    def reset(self):
        self.filtered_flow = None
        self.turn_sign = 0
        self.clear_elapsed = 0.
        self.last_decision = {}

    def contract(self):
        return dict(version='xy-coordinator-v1', anti_rollback=self.anti_rollback,
                    turn_latch=self.turn_latch, config=asdict(self.config),
                    inputs='normalized flow[2], ToF[64] in metres, control dt',
                    output_order=['throttle', 'roll', 'pitch', 'yaw'])

    def apply(self, network_pwm, *, flow_xy, tof_m, dt):
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
        return action
