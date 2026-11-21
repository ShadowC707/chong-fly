"""Bounded exit from a stationary FAILED phase on top of phased_side_guard (author decision, step 9).

Without it FAILED is left only when the front stays clear for 0.5 s, which a drone hovering in front of
an obstacle never sees: it can hold in place for the rest of the flight. Here a FAILED phase that has
been stationary for wait_retry_s (as WAIT_CLEAR) starts a new BRAKE -> TURN episode. The first retry
continues the direction of the turn that timed out (the headings already swept did not clear); each
further retry reverses it. The budget equals the WAIT_CLEAR one (max_retries) and is restored by the
same rule (progress_reset_s of clear moving CRUISE). After the budget is spent the drone stays in FAILED
with its flow damping (the hold of phased_side_guard) and an operator event; no further retries until the
budget is restored. FAILED is still left on a clear front, as in phased_side_guard. Native STOP is not
used here (v1 did; step 9 showed the drone drifting to 0.048 m in it). A retreat along the own corridor
is not available: phased_side_guard has no odometry memory.
"""
import math

import numpy as np

from .phased_coordinator import PhasedCoordinator
from .side_clearance import SideClearanceCoordinator


class FailedExitCoordinator(SideClearanceCoordinator):
    def reset(self):
        super().reset()
        self.failed_retry_count = 0
        self.failed_retry_elapsed = 0.0
        self.operator_event = None
        self.failed_exhausted = False

    def contract(self):
        return super().contract() | dict(failed_exit='bounded-failed-retry-v2',
            failed_retry_s=self.wait_retry_s, failed_max_retries=self.max_retries,
            failed_retry_direction='first retry continues the timed-out turn, then alternates',
            failed_retry_budget_reset='same as WAIT_CLEAR: explicit reset or 5 s clear moving CRUISE',
            failed_exhausted='stay in FAILED (its flow damping, exit on clear front), operator event, '
                             'no retries until the budget is restored')

    def apply(self, network_pwm, *, flow_xy, tof_m, dt, stop_requested=False):
        action = super().apply(network_pwm, flow_xy=flow_xy, tof_m=tof_m, dt=dt, stop_requested=stop_requested)
        d = self.last_decision
        if self.enabled and not stop_requested and not self.navigation_blocked:
            c = self.phased_config
            if self.progress_elapsed + 1e-9 >= self.progress_reset_s:
                self.failed_retry_count = 0
                self.failed_exhausted = False
                self.operator_event = None
            speed = math.hypot(self.filtered_flow, self.filtered_flow_y)
            stalled = self.phase == 'FAILED' and speed <= c.quiet_flow and not self.failed_exhausted
            self.failed_retry_elapsed = self.failed_retry_elapsed + dt if stalled else 0.0
            if self.failed_retry_elapsed + 1e-9 >= self.wait_retry_s:
                self.failed_retry_elapsed = 0.0
                if self.failed_retry_count >= self.max_retries:
                    self.failed_exhausted = True
                    self.operator_event = 'failed_retry_budget_exhausted'
                    d['recovery_event'] = 'failed_retry_budget_exhausted'
                else:
                    if self.direction == 0:
                        # plain larger-side rule, without the pocket-escape repeat override
                        self.direction = PhasedCoordinator._choose_direction(self, np.asarray(tof_m).reshape(8, 8))
                    elif self.failed_retry_count > 0:
                        self.direction = -self.direction
                    self.next_tie = -self.direction
                    self.repeat_episode = False
                    self.failed_retry_count += 1
                    self.armed = False
                    self.episode_active = True
                    self.episode_elapsed = self.turn_elapsed = 0.0
                    self.rearm_elapsed = self.cooldown_elapsed = 0.0
                    self._enter('BRAKE')
                    # The output of this tick is still the FAILED damping; the new phase starts next tick.
                    d['recovery_event'] = 'failed_retry_scheduled'
        else:
            self.failed_retry_elapsed = 0.0
        d.update(failed_retry_count=self.failed_retry_count, failed_retry_wait_s=self.failed_retry_elapsed,
                 failed_exhausted=self.failed_exhausted, operator_event=self.operator_event,
                 navigation_blocked=self.navigation_blocked,
                 recovery_next_phase=self.phase)
        return action
