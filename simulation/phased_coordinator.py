"""Opt-in Brake/Turn/Clear research overlay. No world pose or deployment claim."""
from dataclasses import asdict, dataclass
import math
import numpy as np
from .xy_coordinator import XYCoordinator


@dataclass(frozen=True)
class PhasedConfig:
    trigger_m: float = .85
    critical_m: float = .50
    clear_min_m: float = 1.05
    clear_mean_m: float = 1.30
    clear_dwell_s: float = .20
    quiet_flow: float = .015
    quiet_dwell_s: float = .12
    brake_gain: float = 400.
    brake_limit_pwm: float = 100.
    yaw_pwm: float = 140.
    brake_timeout_s: float = 2.
    turn_timeout_s: float = 4.
    episode_timeout_s: float = 8.
    recover_s: float = .50
    side_tie_m: float = .05
    rearm_clear_s: float = .50
    cooldown_s: float = .50

    def __post_init__(self):
        if (not all(math.isfinite(v) and v>0 for v in asdict(self).values())
            or not self.critical_m<self.trigger_m<self.clear_min_m<=self.clear_mean_m
            or max(self.brake_limit_pwm,self.yaw_pwm)>500):
            raise ValueError('Invalid phased contract')


class PhasedCoordinator(XYCoordinator):
    def __init__(self, *, enabled=True, phased_config=None, **kwargs):
        if type(enabled) is not bool or kwargs.get('levy_explore',False):
            raise ValueError('Explicit enabled flag; no Levy in phased experiment')
        self.enabled=enabled
        self.phased_config=phased_config or PhasedConfig()
        super().__init__(**kwargs)

    def reset(self):
        super().reset()
        self.phase='CRUISE'; self.direction=0; self.next_tie=1
        self.elapsed=0.;self.episode_elapsed=0.;self.quiet_elapsed=0.;self.open_elapsed=0.
        self.completed=0;self.failures=0
        self.armed=True;self.episode_active=False;self.turn_elapsed=0.
        self.rearm_elapsed=0.;self.cooldown_elapsed=0.

    def contract(self):
        return super().contract() | dict(phased_enabled=self.enabled,
            phased_version='brake-turn-clear-v2', phased_config=asdict(self.phased_config),
            priority='external authority/sensor validity > native STOP > active phased damping/turn (including critical) > idle critical evasion > native XY',
            side_rule='larger clipped-to-2m half mean; within .05m alternate +1/-1 starting +1',
            timeout='Continuous episode and cumulative turn budgets; FAILED damping until clear .5s plus cooldown .5s; no critical rearm',
            admission='experimental only; flow damping is not position hold')

    def _enter(self, phase):
        self.phase=phase;self.elapsed=0.;self.quiet_elapsed=0.;self.open_elapsed=0.

    def _choose_direction(self, image):
        c = self.phased_config
        difference = float(np.minimum(image[:, 4:], 2.).mean() - np.minimum(image[:, :4], 2.).mean())
        return (1 if difference > 0 else -1) if abs(difference) > c.side_tie_m else self.next_tie

    def _clear_dwell_s(self):
        return self.phased_config.clear_dwell_s

    def _timeout_reason(self):
        c=self.phased_config
        if self.episode_active and self.episode_elapsed+1e-9>=c.episode_timeout_s:
            return 'episode_timeout'
        if self.phase=='TURN' and self.turn_elapsed+1e-9>=c.turn_timeout_s:
            return 'turn_timeout'
        if self.phase=='BRAKE' and self.elapsed+1e-9>=c.brake_timeout_s:
            return 'brake_timeout'
        return None

    def apply(self, network_pwm, *, flow_xy, tof_m, dt, stop_requested=False):
        try:
            base=super().apply(network_pwm,flow_xy=flow_xy,tof_m=tof_m,dt=dt,stop_requested=stop_requested)
        except ValueError:
            self.reset()
            raise
        self.last_native_yaw=float(base[3])
        if not self.enabled: return base
        c=self.phased_config; d=self.last_decision; action=base.copy()
        image=np.asarray(tof_m).reshape(8,8); distance=float(image.min())
        event=None; owned=np.zeros(4,dtype=bool)
        clear=distance>=c.clear_min_m and float(image.mean())>=c.clear_mean_m
        speed=math.hypot(self.filtered_flow,self.filtered_flow_y)
        damping=1500+np.clip(c.brake_gain*np.array([self.filtered_flow_y,-self.filtered_flow]),
                            -c.brake_limit_pwm,c.brake_limit_pwm)
        if stop_requested:
            if self.phase!='STOP':
                event='cancelled' if self.episode_active else None
                self.armed=False; self.episode_active=False
                self.rearm_elapsed=0.;self.cooldown_elapsed=0.
                self._enter('STOP')
        else:
            if self.phase=='STOP': self._enter('WAIT_CLEAR')
            if not self.episode_active and not self.armed:
                self.cooldown_elapsed+=dt
                self.rearm_elapsed=self.rearm_elapsed+dt if clear else 0.
                if (self.rearm_elapsed+1e-9>=c.rearm_clear_s
                        and self.cooldown_elapsed+1e-9>=c.cooldown_s):
                    self.armed=True
                    if self.phase in ('FAILED','WAIT_CLEAR'):
                        self._enter('RECOVER');event='clear_after_failure'
            if self.phase=='CRITICAL' and distance>=c.critical_m: self._enter('CRUISE')
            if self.phase=='CRUISE' and distance<c.trigger_m:
                if not self.armed:
                    self._enter('WAIT_CLEAR')
                elif distance<c.critical_m:
                    self._enter('CRITICAL')
                else:
                    self.direction=self._choose_direction(image)
                    self.next_tie=-self.direction
                    self.armed=False;self.episode_active=True
                    self.episode_elapsed=0.;self.turn_elapsed=0.
                    self.rearm_elapsed=0.;self.cooldown_elapsed=0.
                    self._enter('BRAKE');event='started'
            if self.phase in ('BRAKE','TURN','RECOVER'):
                self.elapsed+=dt
                if self.episode_active: self.episode_elapsed+=dt
                self.open_elapsed=self.open_elapsed+dt if clear else 0.
                if self.phase=='TURN': self.turn_elapsed+=dt
                timeout=self._timeout_reason()
                if timeout:
                    self._enter('FAILED');self.failures+=1;event=timeout
                    self.episode_active=False;self.armed=False
                    self.rearm_elapsed=0.;self.cooldown_elapsed=0.
                elif self.phase=='BRAKE':
                    self.quiet_elapsed=self.quiet_elapsed+dt if speed<=c.quiet_flow else 0.
                    if self.quiet_elapsed+1e-9>=c.quiet_dwell_s:
                        self._enter('TURN');event='braked'
                elif self.phase=='TURN' and self.open_elapsed+1e-9>=self._clear_dwell_s():
                    self._enter('RECOVER');self.turn_sign=0;event='clear'
                elif self.phase=='RECOVER':
                    if distance<c.trigger_m:
                        self._enter('BRAKE' if self.episode_active else 'WAIT_CLEAR')
                        self.armed=False;self.rearm_elapsed=0.;event='renewed_threat'
                    elif self.elapsed+1e-9>=c.recover_s:
                        self._enter('CRUISE')
                        if self.episode_active:
                            self.completed+=1;event='completed'
                            self.armed=False;self.rearm_elapsed=0.;self.cooldown_elapsed=0.
                        self.episode_active=False
            if self.phase in ('BRAKE','TURN','FAILED','WAIT_CLEAR','RECOVER'):
                owned[1:]=True
                action[1:3]=damping
                action[3]=1500+c.yaw_pwm*self.direction if self.phase=='TURN' else 1500.
                if self.phase=='RECOVER':
                    f=float(np.clip(self.elapsed/c.recover_s,0,1)); f=f*f*(3-2*f)
                    action[1:3]=(1-f)*damping+f*base[1:3]
                    action[3]=1500+f*(base[3]-1500)
        changed=np.abs(action-base)>1e-9
        if owned.any(): d['reasons'].append('phased_'+self.phase.lower())
        d.update(phased_phase=self.phase,phased_event=event,phased_direction=self.direction,
            phased_owned_channels=owned.tolist(),phased_changed_channels=changed.tolist(),
            phased_completed=self.completed,phased_failures=self.failures,
            phased_elapsed_s=self.elapsed,phased_flow_norm=speed,
            phased_episode_s=self.episode_elapsed,phased_turn_s=self.turn_elapsed,
            phased_armed=self.armed,phased_rearm_clear_s=self.rearm_elapsed,
            phased_critical=distance<c.critical_m,
            phased_tof_column_min_m=image.min(axis=0).tolist(),
            phased_tof_mean_m=float(image.mean()),phased_clear=clear,
            phased_native_clear=distance>=self.config.clear_m,
            intervened=bool(d['reasons']))
        self.previous_pr=action[1:3].copy()
        return action
