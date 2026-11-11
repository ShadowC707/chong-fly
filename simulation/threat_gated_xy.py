"""Opt-in restoration of the neural brake near a visible threat."""
import numpy as np

from .xy_coordinator import XYCoordinator


class ThreatGatedXYCoordinator(XYCoordinator):
    threat_m = .85
    minimum_forward_flow = -.005

    def contract(self):
        return super().contract() | dict(threat_gated_brake='restore-neural-pitch-v1',
            threat_gated_brake_m=self.threat_m, threat_gated_brake_flow=self.minimum_forward_flow)

    def apply(self, network_pwm, *, flow_xy, tof_m, dt, stop_requested=False):
        action = super().apply(network_pwm, flow_xy=flow_xy, tof_m=tof_m, dt=dt,
                               stop_requested=stop_requested)
        decision = self.last_decision
        if (not stop_requested and 'anti_rollback' in decision['reasons']
                and self.filtered_flow > self.minimum_forward_flow and np.min(tof_m) < self.threat_m):
            action[2] = network_pwm[2]
            self.previous_pr = action[1:3].copy()
            decision['reasons'].remove('anti_rollback')
            decision['reasons'].append('threat_gated_brake')
            decision['brake_gain'] = 1.0
        return action
