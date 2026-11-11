from dataclasses import replace
import math

import numpy as np
import pytest

from edge.survey_heading import AlignedToF, HeadingSample, UnwrappedHeading
from edge.survey_memory import LocalObservationMemory, PoseEvidence, MemoryConfig, point_ray_directions


def frame(i=0,t=0.,yaw=0.,distance=.8):
    ranges=np.ones(64);ranges[28]=distance/3.5
    return AlignedToF('scan',i,t,tuple(ranges),math.radians(yaw),0.,i,t,'new')


def pose(t=0.,**kwargs):
    kwargs.setdefault('origin_displacement_bound_m',0.)  # Explicit stationary test fixture.
    return PoseEvidence('scan',t,**kwargs)


def heading(t=0.,yaw=0.,error=0.):
    angle=math.radians(yaw)
    return UnwrappedHeading(HeadingSample('scan',round(t*50),t,
        (angle+math.pi)%(2*math.pi)-math.pi,error),angle)


def remember(memory,f=None,p=None,now=0.):
    return memory.remember(f or frame(),p or pose(),geometry_version='point-rays-solid-v2',now_s=now)


@pytest.mark.parametrize('yaw',[-60,60,179,-179])
def test_turn_keeps_obstacle_outside_front_fov(yaw):
    memory=LocalObservationMemory('scan');remember(memory)
    report=memory.project(heading(.5,yaw),pose(.5),now_s=.5)
    point=report['observations'][0]
    expected=math.radians(-22.5/7-yaw)
    assert abs(math.atan2(math.sin(point['bearing_rad']-expected),math.cos(point['bearing_rad']-expected)))<1e-9
    assert point['outside_front_fov']
    assert report['state']=='HAZARDS_REMEMBERED'
    assert set(report['sector_states'])<= {'unknown','hazard_possible'}
    assert report['resume_allowed'] is None


def test_held_frame_does_not_refresh_age_or_duplicate_observations():
    memory=LocalObservationMemory('scan',MemoryConfig(ttl_s=.2));f=frame();remember(memory,f)
    assert remember(memory,replace(f,state='held'),now=.08)=='held'
    report=memory.project(heading(.21),pose(.21),now_s=.21)
    assert report['observations']==[] and report['expired_frames']==1
    assert set(report['sector_states'])=={'unknown'}


def test_clear_new_frame_does_not_erase_previous_hazard():
    memory=LocalObservationMemory('scan');remember(memory)
    remember(memory,frame(1,.02,distance=3.5),pose(.02),now=.02)
    report=memory.project(heading(.02),pose(.02),now_s=.02)
    assert len(report['observations'])==1
    assert report['observations'][0]['capture_s']==0.


def test_displacement_bound_widens_uncertainty_without_guessing_direction():
    a,b=LocalObservationMemory('scan'),LocalObservationMemory('scan')
    remember(a);remember(b,p=pose(origin_displacement_bound_m=.02))
    x=a.project(heading(.02),pose(.02),now_s=.02)['observations'][0]
    y=b.project(heading(.02),pose(.02,origin_displacement_bound_m=.03),now_s=.02)['observations'][0]
    assert y['position_error_bound_m']==pytest.approx(x['position_error_bound_m']+.05)
    assert y['bearing_halfwidth_rad']>x['bearing_halfwidth_rad']
    assert y['horizontal_range_min_m']<x['horizontal_range_min_m']


@pytest.mark.parametrize('kwargs',[{'origin_displacement_bound_m':.051},
    {'roll_rad':math.radians(3)}, {'pitch_rad':math.radians(3)}, {'valid':False},
    {'origin_displacement_bound_m':None}])
def test_unqualified_motion_or_tilt_invalidates_memory_until_reset(kwargs):
    memory=LocalObservationMemory('scan');remember(memory)
    report=memory.project(heading(.02),pose(.02,**kwargs),now_s=.02)
    assert report['state']=='INVALID' and report['observations']==[]
    assert set(report['sector_states'])=={'unknown'}
    assert memory.project(heading(.04),pose(.04),now_s=.04)['state']=='INVALID'
    with pytest.raises(ValueError):memory.reset('scan')
    memory.reset('new-scan')
    assert not memory.faulted


def test_missing_heading_stale_heading_or_wrong_reference_cannot_project():
    for h in (None,heading(0),replace(heading(.1),sample=replace(heading(.1).sample,reference_id='other'))):
        memory=LocalObservationMemory('scan');remember(memory)
        report=memory.project(h,pose(.1),now_s=.1)
        assert report['state']=='INVALID'


@pytest.mark.parametrize('change',['geometry','time','reference','payload','heading_error'])
def test_bad_capture_or_modified_held_frame_latches_fault(change):
    memory=LocalObservationMemory('scan');remember(memory)
    f=frame();p=pose();geometry='point-rays-solid-v2'
    if change=='geometry':geometry='equal-angular-zonal-research-v1'
    if change=='time':p=pose(.01)
    if change=='reference':f=replace(f,reference_id='other')
    if change=='payload':f=frame(distance=.5)
    if change=='heading_error':f=replace(f,heading_error_bound_rad=-.1)
    with pytest.raises(ValueError):memory.remember(f,p,geometry_version=geometry,now_s=.02)
    assert memory.faulted


def test_capacity_eviction_is_explicit_and_never_marks_free_space():
    memory=LocalObservationMemory('scan',MemoryConfig(max_frames=2))
    for i in range(3):remember(memory,frame(i,i*.02),pose(i*.02),now=i*.02)
    report=memory.project(heading(.04),pose(.04),now_s=.04)
    assert report['evicted_frames']==1 and report['needs_resurvey']
    assert len(report['observations'])==2
    assert 'free' not in report['sector_states']


def test_near_zero_return_is_all_direction_uncertainty_not_clear():
    memory=LocalObservationMemory('scan');remember(memory,frame(distance=0))
    report=memory.project(heading(),pose(),now_s=0)
    assert set(report['sector_states'])=={'hazard_possible'}


def test_geometry_matches_existing_simulator_exact_profile():
    from simulation.drone_env import ToFRaycaster
    assert np.allclose(point_ray_directions(),ToFRaycaster().body_ray_dirs,atol=1e-7,rtol=0)


@pytest.mark.parametrize('kwargs',[{'ttl_s':0},{'max_frames':0},{'max_frames':True},
                                  {'range_error_m':-1},{'max_displacement_m':float('nan')}])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):MemoryConfig(**kwargs)


def test_unknown_displacement_is_not_assumed_stationary():
    memory=LocalObservationMemory('scan')
    with pytest.raises(ValueError):remember(memory,p=PoseEvidence('scan',0.))


def test_geometric_error_bounds_cover_perturbed_poses_and_translation():
    from simulation.drone_env import QuadcopterDynamics as Dynamics
    rng=np.random.default_rng(20261104)
    def rot(roll,pitch,yaw):
        return Dynamics.quaternion_to_rotation_matrix(Dynamics.euler_to_quaternion(roll,pitch,yaw)).astype(float)
    def displacement(bound):
        v=rng.normal(size=3);return v/np.linalg.norm(v)*rng.uniform(0,bound)
    for _ in range(150):
        rc,pc,rn,pn=np.deg2rad(rng.uniform(-1,1,4))
        yc,yn=np.deg2rad(rng.uniform(-180,180,2))
        ae,he=math.radians(.2),math.radians(.5)
        memory=LocalObservationMemory('scan')
        f=replace(frame(yaw=math.degrees(yc)),heading_error_bound_rad=he)
        remember(memory,f,pose(roll_rad=rc,pitch_rad=pc,attitude_error_rad=ae,origin_displacement_bound_m=.02))
        report=memory.project(heading(.02,math.degrees(yn),he),
            pose(.02,roll_rad=rn,pitch_rad=pn,attitude_error_rad=ae,origin_displacement_bound_m=.03),now_s=.02)
        item=report['observations'][0]
        er,ep,enr,enp=np.deg2rad(rng.uniform(-.1,.1,4))
        true_capture=rot(rc+er,pc+ep,yc+rng.uniform(-he,he))
        true_current=rot(rn+enr,pn+enp,yn+rng.uniform(-he,he))
        reference=displacement(.02)+true_capture@point_ray_directions()[28]*(.8+rng.uniform(-.01,.01))
        point=true_current.T@(reference-displacement(.03))
        bearing=math.atan2(point[1],point[0]);rho=np.linalg.norm(point[:2])
        delta=math.atan2(math.sin(bearing-item['bearing_rad']),math.cos(bearing-item['bearing_rad']))
        assert abs(delta)<=item['bearing_halfwidth_rad']+1e-6
        assert item['horizontal_range_min_m']-1e-6<=rho<=item['horizontal_range_max_m']+1e-6


def test_upstream_heading_fault_invalidates_even_with_no_new_tof():
    memory=LocalObservationMemory('scan');remember(memory)
    memory.invalidate('heading continuity lost')
    assert memory.project(heading(.02),pose(.02),now_s=.02)['state']=='INVALID'


def test_first_fault_reason_is_not_overwritten_by_followup_failures():
    memory=LocalObservationMemory('scan');remember(memory)
    memory.invalidate('first uncertainty failure')
    memory.invalidate('later transport failure')
    report=memory.project(heading(.02),pose(.02),now_s=.02)
    assert report['reason']=='first uncertainty failure'


def test_raycaster_experiment_retains_out_of_view_box_and_expires_to_unknown():
    from evaluate.validate_survey_memory import run_case
    rows,summary=run_case('left_60')
    assert summary['at_1s']['tof_hits']==0
    assert summary['at_1s']['remembered_points']>0
    assert summary['at_1s']['fully_outside_fov']==summary['at_1s']['remembered_points']
    assert summary['endpoint_checks']>0
    assert rows[-1]['state']=='UNKNOWN' and set(rows[-1]['sector_states'])=={'unknown'}
