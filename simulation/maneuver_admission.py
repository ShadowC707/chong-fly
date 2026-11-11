"""Passive maneuver-completion assessment from the existing sensor contract.

This observer never emits PWM or grants flight authority. It deliberately has
no successful corridor admission: the current snapshot lacks measured heading
change and there is no validated body-corridor observation contract yet.
"""
from dataclasses import asdict,dataclass
import math

import numpy as np

from edge.sensors import SensorTracker,SensorBudgets


@dataclass(frozen=True)
class AdmissionConfig:
    notice_m: float = .95
    critical_m: float = .50
    flow_tau_s: float = .12
    quiet_flow: float = .0075
    quiet_tilt_rad: float = math.radians(2.)
    quiet_dwell_s: float = 1.
    min_downrange_m: float = .90
    max_downrange_m: float = 1.10

    def __post_init__(self):
        values=asdict(self)
        if (any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in values.values())
                or not self.critical_m<self.notice_m<=3.5 or self.quiet_flow>=1
                or self.quiet_tilt_rad>=math.pi/2
                or self.min_downrange_m>=self.max_downrange_m):
            raise ValueError('Invalid maneuver admission configuration')


class ManeuverAdmission:
    def __init__(self,config=None,budgets=None):
        self.config=config or AdmissionConfig()
        self.budgets=budgets or SensorBudgets()
        self.reset()

    def reset(self):
        """New observed session only; never use reset to authorize a maneuver."""
        self.tracker=SensorTracker(self.budgets)
        self.episode_pending=False
        self.episode_id=0
        self.episode_started_s=None
        self.last_threat_s=None
        self.filtered_flow=None
        self.flow_capture_s=None
        self.quiet_since_s=None
        self.quiet_elapsed_s=0.

    def contract(self):
        return dict(version='maneuver-admission-observer-v1',config=asdict(self.config),
            sensor_budgets=asdict(self.budgets),output='advisory_state_only',
            grants_authority=False,can_certify_corridor=False,
            missing=['capture-aligned measured heading change','validated observed body corridor'],
            quiet_semantics='filtered-flow/tilt evidence; not true velocity or position hold')

    def observe(self,snapshot,*,now_s):
        c=self.config
        try:
            data=self.tracker.resolve(snapshot,now_s=now_s)
            flow=np.asarray(data['flow'],float)
            if np.any(abs(flow)>1):raise ValueError('Flow outside normalized range')
        except (ValueError,TypeError) as exc:
            self.filtered_flow=None;self.flow_capture_s=None
            self.quiet_since_s=None;self.quiet_elapsed_s=0.
            return self._result('INVALID_SENSORS',['invalid_sensor_evidence',str(exc)],
                                motion_quiet=False,resume_allowed=False)
        distance=float(min(data['tof'])*3.5)
        if distance<c.notice_m:
            if not self.episode_pending:
                self.episode_id+=1
                self.episode_started_s=now_s
            self.episode_pending=True
            # Retain measurement time, not repeated arrival time for held ToF.
            self.last_threat_s=snapshot.tof.timestamp_s
        tilt_ok=bool(np.max(abs(np.asarray(data['attitude'])))<=c.quiet_tilt_rad)
        altitude=data['altitude']
        range_domain_ok=(c.min_downrange_m<=altitude.range_m<=c.max_downrange_m
                         and max(abs(altitude.roll_rad),abs(altitude.pitch_rad))<=c.quiet_tilt_rad)
        if not tilt_ok or not range_domain_ok:
            self.quiet_since_s=None;self.quiet_elapsed_s=0.
        if data['states']['flow']=='new':
            capture=snapshot.flow.timestamp_s
            gap=None if self.flow_capture_s is None else capture-self.flow_capture_s
            if gap is None or gap>self.budgets.flow_max_age_s+1e-8:
                self.filtered_flow=flow.copy()
                self.quiet_since_s=None;self.quiet_elapsed_s=0.
            else:
                self.filtered_flow += -math.expm1(-gap/c.flow_tau_s)*(flow-self.filtered_flow)
            self.flow_capture_s=capture
            quiet=tilt_ok and range_domain_ok and np.linalg.norm(self.filtered_flow)<=c.quiet_flow
            if quiet:
                if self.quiet_since_s is None:self.quiet_since_s=capture
                self.quiet_elapsed_s=capture-self.quiet_since_s
            else:
                self.quiet_since_s=None;self.quiet_elapsed_s=0.
        motion_quiet=self.quiet_elapsed_s+1e-8>=c.quiet_dwell_s
        if distance<c.critical_m:
            state,reasons='REFLEX_REQUIRED',['critical_front_range','corridor_unverified']
        elif not range_domain_ok:
            state,reasons='OUTSIDE_STUDIED_RANGE',['flow_scale_not_covered','corridor_unverified']
        elif not self.episode_pending:
            state,reasons='NO_EPISODE',['no_observed_maneuver','corridor_not_assessed']
        elif not motion_quiet:
            state,reasons='MOTION_UNCONFIRMED',['motion_not_settled','corridor_unverified']
        elif distance<c.notice_m:
            state,reasons='OBSTACLE_UNRESOLVED',['front_obstacle_remains','corridor_unverified']
        else:
            state,reasons='SURVEY_REQUIRED',['motion_quiet_only','corridor_unverified','heading_change_missing']
        return self._result(state,reasons,motion_quiet=motion_quiet,
            resume_allowed=False if self.episode_pending or not range_domain_ok else None,
            min_tof_m=distance,filtered_flow_norm=float(np.linalg.norm(self.filtered_flow)),
            tilt_ok=tilt_ok,range_domain_ok=range_domain_ok,
            sensor_states=data['states'],sensor_ages_s=data['ages'])

    def _result(self,state,reasons,**kwargs):
        return dict(state=state,reasons=reasons,episode_pending=self.episode_pending,
            episode_id=self.episode_id,episode_started_s=self.episode_started_s,
            last_threat_s=self.last_threat_s,quiet_elapsed_s=self.quiet_elapsed_s,
            mode='observer_only',**kwargs)
