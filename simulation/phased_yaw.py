"""Opt-in, reference-PWM yaw shaping; no changes to the pinned neural bundle.

Cosine edges preserve values (not derivatives) across interruptions. A separate
slew bound handles live targets and remains in CRUISE. STOP/critical bypass it.
"""
import math
import numpy as np
from .phased_coordinator import PhasedCoordinator


class YawRamp:
    duration_s = .24
    rate = 140.*math.pi/(2*.24)

    def __init__(self):
        self.synchronize(1500.)

    def synchronize(self, value):
        if not math.isfinite(value) or not 1000 <= value <= 2000:
            raise ValueError('Invalid reference yaw')
        self.output = self.anchor = float(value)
        self.target = float(value)
        self.key = None
        self.elapsed = 0.
        self.duration = self.duration_s

    def duration_for(self, target):
        return max(self.duration_s, math.pi*abs(target-self.output)/(2*self.rate))

    def step(self, target, key, dt, *, duration=None):
        if not math.isfinite(dt) or dt <= 0 or not math.isfinite(target) or not 1000 <= target <= 2000:
            raise ValueError('Invalid yaw ramp input')
        if key != self.key:
            self.key, self.anchor, self.elapsed = key, self.output, 0.
            self.duration = max(duration or self.duration_s, self.duration_for(target))
        else:
            self.elapsed += dt
        self.target = float(target)
        w = (1-math.cos(math.pi*min(self.elapsed/self.duration, 1.)))/2
        desired = self.anchor+w*(self.target-self.anchor)
        self.output += float(np.clip(desired-self.output, -self.rate*dt, self.rate*dt))
        return self.output


class SmoothPhasedCoordinator(PhasedCoordinator):
    def reset(self):
        super().reset()
        self.yaw = YawRamp()
        self.previous_yaw = 1500.
        self.phase_epoch = 0
        self.recover_tail_s = .24
        self.timeout_reserve_s = 0.

    def contract(self):
        return super().contract() | dict(yaw_shaping='cosine-bounded-v1',
            yaw_ramp_s=self.yaw.duration_s, yaw_slew_pwm_s=self.yaw.rate,
            recovery='Neutral tail, then live native target; slew persists into CRUISE',
            yaw_priority='STOP and idle critical bypass; revoke outside coordinator is immediate',
            timeout_tail='Reserve neutral decay inside existing timeout; no extra turn after deadline',
            handover='Explicit issued yaw synchronization; no assumption of receiver acceptance')

    def _enter(self, phase):
        super()._enter(phase)
        self.phase_epoch += 1
        if phase == 'RECOVER':
            self.recover_tail_s = self.yaw.duration_for(1500.)

    def _timeout_reason(self):
        # One control step of reserve covers discrete deadline detection. This
        # changes only the opt-in variant; the baseline method remains exact.
        c = self.phased_config
        reserve = self.yaw.duration_for(1500.)+self.control_dt
        checks = ((self.episode_active, self.episode_elapsed, c.episode_timeout_s, 'episode_timeout'),
                  (self.phase == 'TURN', self.turn_elapsed, c.turn_timeout_s, 'turn_timeout'),
                  (self.phase == 'BRAKE', self.elapsed, c.brake_timeout_s, 'brake_timeout'))
        for enabled, elapsed, limit, reason in checks:
            if enabled and elapsed+reserve+1e-9 >= limit:
                self.timeout_reserve_s = reserve
                return reason
        return None

    def synchronize_issued_yaw(self, value):
        self.yaw.synchronize(float(value))
        self.previous_yaw = float(value)

    def _cruise_yaw(self, target, network_pwm, tof_m, dt):
        return target

    def apply(self, network_pwm, *, flow_xy, tof_m, dt, stop_requested=False):
        self.control_dt = dt
        action = super().apply(network_pwm, flow_xy=flow_xy, tof_m=tof_m,
                               dt=dt, stop_requested=stop_requested)
        unshaped = float(action[3])
        bypass = ('disabled' if not self.enabled else 'native_stop' if stop_requested
                  else 'idle_critical' if self.phase == 'CRITICAL' else None)
        key = (self.phase_epoch, self.phase)
        target = unshaped
        duration = None
        if bypass:
            self.yaw.synchronize(unshaped)
        else:
            if self.phase == 'CRUISE':
                target = self._cruise_yaw(target, network_pwm, tof_m, dt)
            if self.phase == 'RECOVER':
                if self.elapsed <= self.recover_tail_s+1e-9:
                    target = 1500.
                    key += ('neutral_tail',)
                else:
                    target = self.last_native_yaw
                    key += ('handoff',)
                    duration = max(self.phased_config.recover_s-self.recover_tail_s, .02)
            action[3] = self.yaw.step(target, key, dt, duration=duration)
        self.previous_yaw = float(action[3])
        changed = bool(abs(action[3]-unshaped) > 1e-9)
        d = self.last_decision
        d['yaw_smoothing'] = dict(anchor=self.yaw.anchor, target=target, output=self.previous_yaw,
            elapsed_s=self.yaw.elapsed, duration_s=self.yaw.duration, bypass=bypass,
            timeout_reserve_s=self.timeout_reserve_s, phase_epoch=self.phase_epoch,
            unshaped_yaw=unshaped, changed=changed)
        if changed:
            d['reasons'].append('yaw_shaping')
            if 'phased_owned_channels' in d: d['phased_owned_channels'][3] = True
            if 'phased_changed_channels' in d:
                d['phased_changed_channels'][3] = bool(abs(action[3]-self.last_native_yaw) > 1e-9)
            d['intervened'] = True
        return action
