"""Experimental episode release / flow-gated yaw over the pinned XY controller.

No change to weights or legacy defaults. Admission requires closed-loop paired
evaluation; sensor flow is not an absolute velocity or a safety guarantee.
"""
from dataclasses import asdict, dataclass
import math

import numpy as np

from .xy_coordinator import XYCoordinator


@dataclass(frozen=True)
class ManeuverConfig:
    release_s: float = .20
    neutral_pwm: float = 20.
    flow_full: float = .05
    flow_stop: float = .12
    critical_m: float = .50
    emergency_m: float = .30

    def __post_init__(self):
        if (any(type(v) not in (float, int) or not math.isfinite(v) for v in asdict(self).values())
                or self.release_s <= 0 or not 0 < self.neutral_pwm <= 500
                or not 0 < self.flow_full < self.flow_stop <= 1
                or not 0 < self.emergency_m < self.critical_m):
            raise ValueError('Invalid maneuver contract')


def _smooth_fraction(x):
    x = float(np.clip(x, 0., 1.))
    return x*x*(3.-2.*x)


class ManeuverCoordinator(XYCoordinator):
    def __init__(self, *, release_on_neutral=True, gate_turn_by_flow=True,
                 maneuver_config=None, **kwargs):
        if type(release_on_neutral) is not bool or type(gate_turn_by_flow) is not bool:
            raise ValueError('Maneuver features must be explicit booleans')
        if kwargs.get('levy_explore', False):
            raise ValueError('Maneuver experiment excludes search pulses')
        self.release_on_neutral = release_on_neutral
        self.gate_turn_by_flow = gate_turn_by_flow
        self.maneuver_config = maneuver_config or ManeuverConfig()
        if not isinstance(self.maneuver_config, ManeuverConfig):
            raise ValueError('Expected ManeuverConfig')
        super().__init__(**kwargs)
        if self.maneuver_config.critical_m >= self.config.threat_m:
            raise ValueError('Critical turn override must be inside the threat range')

    def reset(self):
        super().reset()
        self.neutral_elapsed_s = 0.

    def contract(self):
        contract = super().contract()
        contract.update(version='xy-maneuver-experiment-v1',
            maneuver=dict(release_on_neutral=self.release_on_neutral,
                          gate_turn_by_flow=self.gate_turn_by_flow,
                          config=asdict(self.maneuver_config)),
            admission='experimental_not_flight_admission')
        return contract

    def apply(self, network_pwm, *, flow_xy, tof_m, dt, stop_requested=False):
        # Base implementation owns validation, filtering, STOP and anti-rollback.
        action = super().apply(network_pwm, flow_xy=flow_xy, tof_m=tof_m,
                               dt=dt, stop_requested=stop_requested)
        c, d = self.maneuver_config, self.last_decision
        distance = float(np.min(tof_m))
        yaw = float(network_pwm[3])-1500.
        released = False
        if self.release_on_neutral and self.turn_latch:
            completed = distance >= self.config.threat_m and abs(yaw) <= c.neutral_pwm
            self.neutral_elapsed_s = self.neutral_elapsed_s+dt if completed else 0.
            if self.turn_sign and self.neutral_elapsed_s+1e-9 >= c.release_s:
                self.turn_sign = 0
                released = True
                # Undo only this tick's obsolete sign reflection; STOP keeps yaw ownership.
                if not stop_requested:
                    action[3] = network_pwm[3]
                d['reasons'] = [reason for reason in d['reasons'] if reason != 'turn_latch']
                d['reasons'].append('maneuver_sign_released')
        fraction = 1.
        speed = math.hypot(self.filtered_flow, self.filtered_flow_y)
        if self.gate_turn_by_flow and d.get('state') not in ('STOP', 'HOVER'):
            flow_fraction = _smooth_fraction((c.flow_stop-speed)/(c.flow_stop-c.flow_full))
            critical_fraction = _smooth_fraction((c.critical_m-distance)/(c.critical_m-c.emergency_m))
            fraction = max(flow_fraction, critical_fraction)
            previous = float(action[3])
            action[3] = 1500.+(previous-1500.)*fraction
            if action[3] != previous:
                d['reasons'].append('turn_flow_gate')
        d.update(turn_sign=self.turn_sign, maneuver_release_event=released,
                 maneuver_neutral_s=self.neutral_elapsed_s, turn_flow_fraction=fraction,
                 turn_flow_norm=speed, intervened=bool(d['reasons']))
        return action
