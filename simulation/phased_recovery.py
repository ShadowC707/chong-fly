"""Experimental bounded retry of a stationary WAIT_CLEAR, never world-pose navigation."""
import math
import numpy as np

from .phased_yaw import SmoothPhasedCoordinator


class RecoveryPhasedCoordinator(SmoothPhasedCoordinator):
    wait_retry_s = 1.0
    max_retries = 3
    progress_reset_s = 5.0

    def __init__(self, **kwargs):
        if kwargs.get('stop_hover') is not True:
            raise ValueError('Bounded recovery requires native stop_hover fallback')
        super().__init__(**kwargs)

    def reset(self):
        super().reset()
        self.retry_count = 0
        self.wait_retry_elapsed = 0.0
        self.progress_elapsed = 0.0
        self.navigation_blocked = False

    def contract(self):
        return super().contract() | dict(
            wait_recovery='bounded-wait-retry-v1', wait_retry_s=self.wait_retry_s,
            max_retries=self.max_retries, progress_reset_s=self.progress_reset_s,
            retry_budget_reset='Explicit reset or 5s continuous clear moving CRUISE; not episode completion',
            exhausted='Latched navigation_blocked; native STOP with existing critical priority',
            retry_scope='WAIT_CLEAR only, after damping; FAILED timeouts are not retried',
            recovery_admission='Experiment only; neither escape nor position holding guaranteed')

    def apply(self, network_pwm, *, flow_xy, tof_m, dt, stop_requested=False):
        action = super().apply(network_pwm, flow_xy=flow_xy, tof_m=tof_m, dt=dt,
                               stop_requested=stop_requested or self.navigation_blocked)
        d = self.last_decision
        if self.enabled and not stop_requested and not self.navigation_blocked:
            c = self.phased_config
            speed = math.hypot(self.filtered_flow, self.filtered_flow_y)
            progress = self.phase == 'CRUISE' and d['phased_clear'] and speed > c.quiet_flow
            self.progress_elapsed = self.progress_elapsed + dt if progress else 0.0
            if self.progress_elapsed + 1e-9 >= self.progress_reset_s:
                self.retry_count = 0
            waiting = self.phase == 'WAIT_CLEAR' and speed <= c.quiet_flow
            self.wait_retry_elapsed = self.wait_retry_elapsed + dt if waiting else 0.0
            if self.wait_retry_elapsed + 1e-9 >= self.wait_retry_s:
                self.wait_retry_elapsed = 0.0
                if self.retry_count >= self.max_retries:
                    self.navigation_blocked = True
                    d['recovery_event'] = 'retry_budget_exhausted'
                else:
                    image = np.asarray(tof_m).reshape(8, 8)
                    self.direction = self._choose_direction(image)
                    self.next_tie = -self.direction
                    self.retry_count += 1
                    self.armed = False
                    self.episode_active = True
                    self.episode_elapsed = self.turn_elapsed = 0.0
                    self.rearm_elapsed = self.cooldown_elapsed = 0.0
                    self._enter('BRAKE')
                    # The current output is still the WAIT_CLEAR damping result.
                    # The new phase starts next tick, through the existing yaw ramp.
                    d['recovery_event'] = 'wait_retry_scheduled'
        else:
            self.wait_retry_elapsed = self.progress_elapsed = 0.0
        d.update(recovery_retry_count=self.retry_count,
                 recovery_wait_s=self.wait_retry_elapsed,
                 recovery_progress_s=self.progress_elapsed,
                 recovery_next_phase=self.phase,
                 navigation_blocked=self.navigation_blocked)
        return action


class PocketEscapeCoordinator(RecoveryPhasedCoordinator):
    repeat_window_s = 3.5
    repeat_dwell_scale = 1.75

    def reset(self):
        super().reset()
        self.since_completed_s = math.inf
        self.completed_direction = 0
        self.repeat_episode = False

    def contract(self):
        return super().contract() | dict(pocket_escape='consistent-direction-v1',
            pocket_repeat_window_s=self.repeat_window_s, pocket_dwell_scale=self.repeat_dwell_scale)

    def _choose_direction(self, image):
        self.repeat_episode = self.since_completed_s < self.repeat_window_s - 1e-9
        if self.repeat_episode:
            return self.completed_direction
        return super()._choose_direction(image)

    def _clear_dwell_s(self):
        dwell = super()._clear_dwell_s()
        return dwell * self.repeat_dwell_scale if self.repeat_episode else dwell

    def apply(self, network_pwm, *, flow_xy, tof_m, dt, stop_requested=False):
        if stop_requested:
            self.since_completed_s = math.inf
            self.completed_direction = 0
            self.repeat_episode = False
        self.since_completed_s += dt
        action = super().apply(network_pwm, flow_xy=flow_xy, tof_m=tof_m, dt=dt,
                               stop_requested=stop_requested)
        decision = self.last_decision
        if decision.get('phased_event') == 'completed':
            self.since_completed_s = 0.0
            self.completed_direction = self.direction
        if self.repeat_episode and (decision.get('phased_event') == 'started'
                or decision.get('recovery_event') == 'wait_retry_scheduled'):
            decision['reasons'].append('consistent_escape')
        decision['pocket_clear_dwell_s'] = self._clear_dwell_s()
        return action
