"""Short-lived yaw restraint following a visible edge avoidance response."""
import numpy as np

from .phased_recovery import PocketEscapeCoordinator


class SideClearanceCoordinator(PocketEscapeCoordinator):
    edge_support_m = 1.0
    acquire_pwm = 50.0
    loss_dwell_s = .50
    expiry_s = 1.5
    flow_integral_limit = .06

    def reset(self):
        super().reset()
        self._clear_edge()

    def _clear_edge(self):
        self.edge_side = 0
        self.edge_age_s = 0.0
        self.edge_forward_integral = 0.0

    def _enter(self, phase):
        super()._enter(phase)
        self._clear_edge()

    def contract(self):
        return super().contract() | dict(side_clearance='recent-edge-yaw-restraint-v2',
            edge_support_m=self.edge_support_m, edge_acquire_pwm=self.acquire_pwm,
            edge_loss_dwell_s=self.loss_dwell_s, edge_reverse_pwm=self.config.acquire_yaw_pwm,
            edge_expiry_s=self.expiry_s, edge_flow_integral_limit=self.flow_integral_limit)

    def _cruise_yaw(self, target, network_pwm, tof_m, dt):
        columns = np.asarray(tof_m).reshape(8, 8).min(axis=0)
        column = int(columns.argmin())
        side = -1 if column < 2 else 1 if column >= 6 else 0
        if self.edge_side:
            self.edge_age_s += dt
            self.edge_forward_integral += max(self.filtered_flow, 0.0) * dt
            if self.edge_age_s >= self.expiry_s or self.edge_forward_integral >= self.flow_integral_limit:
                self._clear_edge()
        if (side and self.edge_side in (0, side) and columns[column] < self.edge_support_m
                and side*(network_pwm[3]-1500.) <= -self.acquire_pwm and self.filtered_flow > 0.):
            self._clear_edge()
            self.edge_side = side
        if (self.edge_side*(target-1500.) >= self.config.acquire_yaw_pwm
                and self.edge_age_s >= self.loss_dwell_s):
            protected, opposite = (columns[4:], columns[:4]) if self.edge_side > 0 else (columns[:4], columns[4:])
            if protected.min() >= self.edge_support_m and opposite.min() < self.edge_support_m:
                self.last_decision['reasons'].append('side_clearance_yaw')
                return 1500.
        return target
