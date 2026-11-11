"""Shared empirical swath overlay; observations in, no world/scene access.

The native coordinator owns phases and yaw. Observation memory belongs to a
whole session, including manual flight; bridge handover resets must not erase
hazards or re-grant the initial preflight disk at a different position.
"""
from dataclasses import asdict, dataclass, replace
import numpy as np

from edge.free_space_swath import FreeSpaceSwath, SwathConfig, motion_endpoint
from edge.sensors import SensorSnapshot


@dataclass(frozen=True)
class SwathControlConfig:
    resume_forward_m: float = .34
    turn_timeout_s: float = 12.
    episode_timeout_s: float = 16.
    exploration_cell_m: float = .5
    exploration_targets_m: tuple = (.5, 1.)
    preference_release_turn_s: float = 9.


class VisitMemory:
    """Session-local odometry preference, never a clearance permission."""
    def __init__(self, config):
        self.config = config
        self.cells = set()

    def observe(self, position):
        self.cells.add(tuple(np.floor(position[:2] / self.config.exploration_cell_m).astype(int)))

    def new_targets(self, position, heading):
        ahead = position[:2] + np.array(self.config.exploration_targets_m)[:, None] * np.array([np.cos(heading), np.sin(heading)])
        return sum(tuple(cell) not in self.cells for cell in np.floor(ahead / self.config.exploration_cell_m).astype(int))


class SwathNavigation:
    def __init__(self, coordinator, reference_id, *, swath_config=None, control_config=None):
        if hasattr(coordinator, 'swath_navigation'):
            raise ValueError('Swath overlay already installed')
        self.config = control_config or SwathControlConfig()
        self.observer = FreeSpaceSwath(reference_id, swath_config or SwathConfig())
        self.coordinator = coordinator
        self.memory = VisitMemory(self.config)
        self.native_apply = coordinator.apply
        self.native_contract = coordinator.contract
        self.native_dwell = coordinator._clear_dwell_s
        coordinator.phased_config = replace(coordinator.phased_config,
            turn_timeout_s=self.config.turn_timeout_s, episode_timeout_s=self.config.episode_timeout_s)
        self.resume_covered = False
        self.verified_preflight = False
        self.blocked_reason = None
        self.now_s = None
        self.attitude = (0., 0.)
        self.decision = dict(state='INVALID', reason='missing_observation')
        coordinator._clear_dwell_s = lambda: self.native_dwell() if self.resume_covered else 1000.
        coordinator.apply = self.apply
        coordinator.contract = self.contract
        coordinator.swath_navigation = self

    def confirm_preflight(self):
        if self.now_s is not None:
            raise ValueError('Preflight clearance can only be confirmed before observation starts')
        self.verified_preflight = True

    def contract(self):
        return self.native_contract() | dict(swath_version='observed-expansion-v1',
            swath_config=asdict(self.observer.swath_config), swath_control=asdict(self.config),
            speed_cap_applied=False, coverage='incremental expansion; known hits veto full tube',
            heading_input='explicit capture-time HeadingSample; no pose/map input',
            reset_scope='New session resets memory/fault/budget; bridge reset only resets native control',
            visit_memory='50cm flow-odometry cells for session; preference only, relax after 9s of turn',
            timeout_policy='Native timeout or retry exhaustion latches navigation blocked until new session',
            bootstrap='Explicit preflight clearance required; no automatic re-grant on handover',
            admission='Standalone empirical experiment; not MVP/SITL/hardware admission')

    def observe(self, frame, heading, *, now_s):
        self.now_s = now_s
        if not self.verified_preflight or not isinstance(frame, SensorSnapshot):
            self.decision = dict(state='INVALID', reason='unverified_preflight_or_missing_snapshot')
        else:
            self.attitude = frame.attitude.value
            self.decision = self.observer.update(frame, heading, now_s=now_s)
            self.memory.observe(self.observer.position)
        if self.decision['state'] == 'INVALID':
            self.blocked_reason = self.decision['reason']
        return self.decision

    def _record(self, action, native, reason=None):
        c = self.coordinator
        self.decision.update(blocked=bool(self.blocked_reason or c.navigation_blocked),
            blocked_reason=self.blocked_reason, native_phase=c.phase,
            native_pwm=native.tolist(), guarded_pwm=action.tolist())
        if reason:
            c.last_decision['reasons'].append(reason)
            c.last_decision['intervened'] = True
        c.last_decision.update(swath=dict(self.decision), phased_phase=c.phase,
                              navigation_blocked=bool(self.blocked_reason or c.navigation_blocked))
        return action

    def apply(self, network_pwm, *, flow_xy, tof_m, dt, stop_requested=False):
        c = self.coordinator
        invalid = self.decision.get('state') == 'INVALID'
        if not invalid and not self.blocked_reason and not stop_requested:
            resume = self.observer.assess([self.config.resume_forward_m, 0.], now_s=self.now_s)
            self.resume_covered = resume['covered']
            self.decision['resume_swath'] = resume
            new_targets = self.memory.new_targets(self.observer.position, self.observer.last_report['heading_rad'])
            self.decision['exploration_new_targets'] = new_targets
            if not new_targets and c.turn_elapsed < self.config.preference_release_turn_s:
                self.resume_covered = False
            if not self.resume_covered and c.phase == 'TURN':
                c.open_elapsed = 0.
        action = self.native_apply(network_pwm, flow_xy=flow_xy, tof_m=tof_m, dt=dt,
            stop_requested=stop_requested or bool(self.blocked_reason) or invalid)
        native = action.copy()
        event = c.last_decision.get('phased_event')
        if event in ('turn_timeout', 'episode_timeout', 'brake_timeout'):
            self.blocked_reason = event
        if c.navigation_blocked:
            self.blocked_reason = self.blocked_reason or 'retry_budget_exhausted'
        if stop_requested or self.blocked_reason or invalid:
            self.resume_covered = False
            return self._record(action, native, 'swath_blocked' if self.blocked_reason else None)
        state = self.decision['state']
        endpoint = motion_endpoint(self.observer.velocity, action, self.attitude)
        check = self.observer.assess(endpoint, now_s=self.now_s)
        self.decision.update(swath=check, observer_state=state, reason=check['reason'],
            state='PROCEED_LIMITED' if check['covered'] else 'OUTSIDE_SWATH')
        if not check['covered']:
            forward = self.observer.assess([max(0., endpoint[0]), 0.], now_s=self.now_s)
            damping = 1500. + np.clip(400.*np.array([c.filtered_flow_y, -c.filtered_flow]), -100., 100.)
            action[1] = damping[0]
            if not forward['covered']:
                action[2] = damping[1]
            if not forward['covered'] and c.phase in ('CRUISE', 'RECOVER'):
                if c.phase == 'CRUISE':
                    c.direction = c._choose_direction(np.asarray(tof_m).reshape(8, 8))
                c.episode_elapsed = c.turn_elapsed = 0.
                c.episode_active = True
                c.armed = False
                c._enter('BRAKE')
                action[3] = 1500.
                c.synchronize_issued_yaw(1500.)
            c.previous_pr = action[1:3].copy()
        return self._record(action, native, 'swath_outside' if not check['covered'] else None)
