"""Experimental static-hazard memory; unknown is never treated as free space.

Only the existing centered, forward point-ray simulator geometry is supported.
No world position/map, PWM, neural input or corridor authorization is produced.
Displacement/attitude uncertainty must be supplied by a qualified source;
this module cannot infer a guaranteed displacement bound from optical flow.
"""
from collections import OrderedDict
from dataclasses import dataclass, replace
import math

import numpy as np

from edge.survey_heading import AlignedToF, UnwrappedHeading


def finite(x):
    return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)


def rotation(roll,pitch,yaw):
    """Body FLU to reference, Rz(yaw) Ry(pitch) Rx(roll)."""
    cr,sr,cp,sp,cy,sy=math.cos(roll),math.sin(roll),math.cos(pitch),math.sin(pitch),math.cos(yaw),math.sin(yaw)
    return np.array([[cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr],
                     [sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr],[-sp,cp*sr,cp*cr]])


def point_ray_directions():
    az=np.tile(np.linspace(math.pi/8,-math.pi/8,8),8)
    el=np.repeat(np.linspace(math.pi/8,-math.pi/8,8),8)
    return np.stack((np.cos(el)*np.cos(az),np.cos(el)*np.sin(az),np.sin(el)),axis=1)


@dataclass(frozen=True)
class PoseEvidence:
    reference_id: str
    timestamp_s: float
    roll_rad: float = 0.
    pitch_rad: float = 0.
    attitude_error_rad: float = 0.  # Bound on residual orientation error, not per-axis std.
    origin_displacement_bound_m: float | None = None  # Unknown by default; NOT inferred as zero.
    valid: bool = True


@dataclass(frozen=True)
class MemoryConfig:
    ttl_s: float = 2.
    max_frames: int = 32
    range_error_m: float = .01
    hazard_range_m: float = 1.15
    max_displacement_m: float = .05
    max_tilt_rad: float = math.radians(2)
    max_angle_error_rad: float = math.radians(10)
    max_heading_age_s: float = .04
    max_yaw_rate_rad_s: float = math.radians(240)
    max_tof_age_s: float = .10

    def __post_init__(self):
        positive=(self.ttl_s,self.hazard_range_m,self.max_displacement_m,self.max_tilt_rad,
                  self.max_angle_error_rad,self.max_heading_age_s,self.max_yaw_rate_rad_s,self.max_tof_age_s)
        if (any(not finite(x) or x<=0 for x in positive)
                or not finite(self.range_error_m) or self.range_error_m<0
                or type(self.max_frames) is not int or not 1<=self.max_frames<=1024
                or self.hazard_range_m>=3.5 or self.max_tilt_rad>=math.pi/2
                or self.max_angle_error_rad>=math.pi/2):
            raise ValueError('Invalid observation memory configuration')


class LocalObservationMemory:
    geometry_version='point-rays-solid-v2'

    def __init__(self,reference_id,config=None):
        self.config=config or MemoryConfig()
        self.reference_id=None
        self.reset(reference_id)

    def reset(self,reference_id):
        if not isinstance(reference_id,str) or not reference_id or reference_id==self.reference_id:
            raise ValueError('A new nonempty reference is required')
        self.reference_id=reference_id
        self._frames=OrderedDict()
        self._last=None
        self._now=None
        self.faulted=False
        self.fault_reason=None
        self.expired_frames=self.evicted_frames=0

    def invalidate(self,reason):
        """Caller must propagate heading/reference faults, even without a new ToF."""
        if not self.faulted:self.fault_reason=str(reason)
        self.faulted=True
        self._frames.clear()

    def _clock(self,now):
        if self.faulted:raise ValueError(self.fault_reason or 'Memory invalidated')
        if not finite(now) or now<0 or (self._now is not None and now<self._now-1e-8):
            raise ValueError('Invalid observation clock')
        self._now=now

    def _pose(self,p,t):
        c=self.config
        if (not isinstance(p,PoseEvidence) or p.reference_id!=self.reference_id
                or type(p.valid) is not bool or not p.valid
                or any(not finite(x) for x in (p.timestamp_s,p.roll_rad,p.pitch_rad,
                     p.attitude_error_rad,p.origin_displacement_bound_m))
                or abs(p.timestamp_s-t)>1e-8
                or not 0<=p.origin_displacement_bound_m<=c.max_displacement_m
                or not 0<=p.attitude_error_rad<=c.max_angle_error_rad
                or max(abs(p.roll_rad),abs(p.pitch_rad))+p.attitude_error_rad>c.max_tilt_rad+1e-8):
            raise ValueError('Unknown/unqualified displacement, tilt, reference or capture pose')

    def _expire(self,now):
        for key,(f,_,_) in list(self._frames.items()):
            if now-f.timestamp_s>self.config.ttl_s+1e-8:
                del self._frames[key]
                self.expired_frames+=1

    def remember(self,frame,capture_pose,*,geometry_version,now_s):
        """Consume a validated capture-aligned ToF, with same-time pose evidence."""
        try:
            self._clock(now_s)
            c=self.config
            if (geometry_version!=self.geometry_version or not isinstance(frame,AlignedToF)
                    or frame.version!='capture-heading-tof-v1' or frame.reference_id!=self.reference_id
                    or type(frame.sample_id) is not int or frame.sample_id<0
                    or frame.state not in ('new','held')
                    or any(not finite(v) for v in (frame.timestamp_s,frame.heading_rad,frame.heading_error_bound_rad))
                    or frame.timestamp_s<0 or frame.timestamp_s>now_s+1e-8
                    or now_s-frame.timestamp_s>c.max_tof_age_s+1e-8
                    or not 0<=frame.heading_error_bound_rad<=c.max_angle_error_rad):
                raise ValueError('Invalid aligned ToF or unsupported sensor geometry')
            self._pose(capture_pose,frame.timestamp_s)
            ranges=np.asarray(frame.normalized_ranges,dtype=float)
            if ranges.shape!=(64,) or not np.isfinite(ranges).all() or (ranges<0).any() or (ranges>1).any():
                raise ValueError('Invalid normalized ranges')
            # Copy even if a caller violates the tuple annotation with a mutable list.
            frame=replace(frame,normalized_ranges=tuple(float(x) for x in ranges),state='new')
            if self._last is not None:
                old,old_pose=self._last
                if frame.sample_id==old.sample_id:
                    if frame!=old or capture_pose!=old_pose:raise ValueError('Changed held observation')
                    self._expire(now_s)
                    return 'held'
                if frame.sample_id<old.sample_id or frame.timestamp_s<=old.timestamp_s:
                    raise ValueError('Regressed observation identity or time')
            transform=rotation(capture_pose.roll_rad,capture_pose.pitch_rad,frame.heading_rad)
            distances=ranges*3.5
            points=[]
            for zone,(distance,direction) in enumerate(zip(distances,point_ray_directions())):
                # A capped/no-return value is not a free-space observation.
                if distance>=3.5 or distance-c.range_error_m>c.hazard_range_m:continue
                error_angle=frame.heading_error_bound_rad+capture_pose.attitude_error_rad
                radius=(c.range_error_m+2*(distance+c.range_error_m)*math.sin(error_angle/2)
                        +capture_pose.origin_displacement_bound_m)
                points.append((zone,transform@(direction*distance),float(distance),radius))
            self._expire(now_s)
            self._frames[frame.sample_id]=(frame,capture_pose,points)
            self._last=(frame,capture_pose)
            while len(self._frames)>c.max_frames:
                self._frames.popitem(last=False);self.evicted_frames+=1
            return 'new'
        except (ValueError,TypeError,AttributeError,OverflowError) as exc:
            self.invalidate(exc)
            raise ValueError(str(exc)) from exc

    def project(self,current_heading,current_pose,*,now_s):
        """Return possible hazard bearings, never free/occupied certainty or RC."""
        try:
            self._clock(now_s);self._pose(current_pose,now_s)
            c=self.config
            h=current_heading
            if not isinstance(h,UnwrappedHeading):raise ValueError('Measured heading unavailable')
            s=h.sample
            if (s.reference_id!=self.reference_id or not s.valid
                    or any(not finite(v) for v in (h.angle_rad,s.timestamp_s,s.error_bound_rad,s.timestamp_error_s))
                    or s.error_bound_rad<0 or s.timestamp_error_s<0
                    or not -1e-8<=now_s-s.timestamp_s<=c.max_heading_age_s+1e-8):
                raise ValueError('Invalid/stale heading or reference')
            heading_error=s.error_bound_rad+c.max_yaw_rate_rad_s*(s.timestamp_error_s+max(0.,now_s-s.timestamp_s))
            if heading_error>c.max_angle_error_rad:raise ValueError('Current heading uncertainty too large')
            self._expire(now_s)
            transform=rotation(current_pose.roll_rad,current_pose.pitch_rad,h.angle_rad).T
            observations=[]
            sector_centers=np.linspace(-math.pi,math.pi,72,endpoint=False)+math.pi/72
            marked=np.zeros(72,dtype=bool)
            for f,p,points in self._frames.values():
                for zone,reference_point,distance,capture_radius in points:
                    point=transform@reference_point
                    total_displacement=p.origin_displacement_bound_m+current_pose.origin_displacement_bound_m
                    max_distance=distance+c.range_error_m+total_displacement
                    radius=(capture_radius+current_pose.origin_displacement_bound_m
                        +2*max_distance*math.sin((heading_error+current_pose.attitude_error_rad)/2))
                    rho=float(np.linalg.norm(point[:2]))
                    bearing=math.atan2(point[1],point[0])
                    width=math.pi if radius>=rho else math.asin(radius/rho)
                    delta=np.arctan2(np.sin(sector_centers-bearing),np.cos(sector_centers-bearing))
                    marked|=abs(delta)<=width+math.pi/72+1e-10
                    observations.append(dict(sample_id=f.sample_id,zone=zone,capture_s=f.timestamp_s,
                        age_s=now_s-f.timestamp_s,bearing_rad=bearing,bearing_halfwidth_rad=width,
                        position_error_bound_m=radius,horizontal_range_min_m=max(0.,rho-radius),
                        horizontal_range_max_m=rho+radius,
                        outside_front_fov=abs(bearing)-width>math.pi/8))
            return self._report('HAZARDS_REMEMBERED' if observations else 'UNKNOWN',observations,marked)
        except (ValueError,TypeError,AttributeError,OverflowError) as exc:
            self.invalidate(exc)
            return self._report('INVALID',[],np.zeros(72,dtype=bool))

    def _report(self,state,observations,marked):
        return dict(version='local-hazard-memory-v1',reference_id=self.reference_id,state=state,
            reason=self.fault_reason,observations=observations,
            sector_states=['hazard_possible' if x else 'unknown' for x in marked],
            sector_width_deg=5.,sector_first_center_deg=-177.5,
            expired_frames=self.expired_frames,evicted_frames=self.evicted_frames,
            needs_resurvey=bool(self.faulted or self.expired_frames or self.evicted_frames or not observations),
            resume_allowed=None,static_scene_assumption=True,
            motion_bounds_source='external; not inferred from flow',changes_rc=False)
