"""Controlled rotation by a static box; tests hazard memory, not flight safety.

Truth is confined to the raycaster/synthetic sensors and geometric QA. The
displacement bound is a declared fixture assumption checked against truth;
there is no qualified airborne estimator of that bound yet.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import json
import math
from pathlib import Path

import numpy as np

from edge.runtime import sha256
from edge.sensors import Measurement
from edge.survey_heading import SurveyHeadingTracker
from edge.survey_memory import LocalObservationMemory, PoseEvidence, MemoryConfig
from simulation.drone_env import ToFRaycaster, RoomBoundaries, BoxObstacle, QuadcopterDynamics
from simulation.survey_heading_sensor import SimulatedHeadingSensor

ROOT=Path(__file__).resolve().parents[1]
CASES=('left_60','right_60','bounded_shift_3cm','shift_exceeds_5cm','tilt_exceeds_2deg','heading_gap')
SEED=20261104


def run_case(name,duration_s=3.):
    if name not in CASES:raise ValueError('Unknown fixed case')
    memory=LocalObservationMemory(name)
    heading_tracker=SurveyHeadingTracker(name)
    sensor=SimulatedHeadingSensor(name,seed=SEED)
    caster=ToFRaycaster()
    assert caster.geometry_version==memory.geometry_version
    room=RoomBoundaries(-4,4,-4,4,0,3)
    box=BoxObstacle(.8,.95,-.08,.08,.5,1.5)
    queue=[];latest=None;tof=None;last_tof_epoch=-1;capture_poses={};hit_points={}
    trace=[];checks=0;max_excess=0.;heading_fault=None
    for step in range(round(duration_s/.004)+1):
        t=step*.004
        yaw=math.radians((-60 if name=='right_60' else 60)*min(t,1.))
        shift=(.03 if name=='bounded_shift_3cm' else .08 if name=='shift_exceeds_5cm' else 0.)*min(t,1.)
        roll=math.radians(3 if name=='tilt_exceeds_2deg' and t>=.6 else 0.)
        pos=np.array([shift,0.,1.])
        rot=QuadcopterDynamics.quaternion_to_rotation_matrix(
            QuadcopterDynamics.euler_to_quaternion(roll,0.,yaw)).astype(float)
        # This is a fixture envelope, not an estimate derived from world data.
        motion_bound=(.03 if name=='bounded_shift_3cm' else .08*min(t,1.) if name=='shift_exceeds_5cm' else 0.)
        assert np.linalg.norm(pos-[0.,0.,1.])<=motion_bound+1e-10
        pose=PoseEvidence(name,t,roll_rad=roll,origin_displacement_bound_m=motion_bound)
        if step%5==0:
            sample=sensor.capture(yaw,timestamp_s=t)
            if not (name=='heading_gap' and sample.sample_id in (15,16)):
                queue.append((t+.012,sample))
            epoch=int(math.floor(t*15+1e-8))
            if epoch!=last_tof_epoch:
                last_tof_epoch=epoch
                ranges=caster.cast_rays(pos,rot,room,boxes=[box])
                tof=Measurement(epoch,t,tuple(float(x) for x in ranges))
                capture_poses[epoch]=pose
                for zone,r in enumerate(ranges):
                    if r*3.5<=1.16:
                        hit_points[(epoch,zone)]=pos+rot@caster.body_ray_dirs[zone].astype(float)*(float(r)*3.5)
        while queue and queue[0][0]<=t+1e-8:
            _,sample=queue.pop(0)
            if not heading_tracker.faulted:
                try:latest=heading_tracker.add_heading(sample,now_s=t)
                except ValueError as exc:
                    heading_fault=str(exc);memory.invalidate(heading_fault);latest=None
        if step%5:continue
        if latest is None and not memory.faulted:
            trace.append(dict(time_s=t,state='WAITING_FOR_HEADING',observations=[],tof_hits=int(sum(v*3.5<=1.16 for v in tof.value)),
                              sector_states=['unknown']*72,checks=0))
            continue
        if not memory.faulted:
            try:
                aligned=heading_tracker.align_tof(tof,reference_id=name,now_s=t)
                memory.remember(aligned,capture_poses[tof.sample_id],geometry_version=caster.geometry_version,now_s=t)
            except ValueError as exc:memory.invalidate(exc)
        report=memory.project(latest,pose,now_s=t)
        for item in report['observations']:
            true_point=rot.T@(hit_points[(item['sample_id'],item['zone'])]-pos)
            bearing=math.atan2(true_point[1],true_point[0])
            delta=abs(math.atan2(math.sin(bearing-item['bearing_rad']),math.cos(bearing-item['bearing_rad'])))
            excess=delta-item['bearing_halfwidth_rad']
            max_excess=max(max_excess,excess)
            assert excess<1e-6
            rho=np.linalg.norm(true_point[:2])
            assert item['horizontal_range_min_m']-1e-6<=rho<=item['horizontal_range_max_m']+1e-6
            checks+=1
        assert 'free' not in report['sector_states'] and report['resume_allowed'] is None
        trace.append(dict(time_s=t,**report,tof_hits=int(sum(v*3.5<=1.16 for v in tof.value)),checks=checks))
    at_one=min(trace,key=lambda row:abs(row['time_s']-1.))
    summary=dict(name=name,states=dict(Counter(row['state'] for row in trace)),
        first_invalid_s=next((row['time_s'] for row in trace if row['state']=='INVALID'),None),
        first_invalid_reason=next((row['reason'] for row in trace if row['state']=='INVALID'),None),
        first_no_front_hits_with_memory_s=next((row['time_s'] for row in trace
            if row['tof_hits']==0 and row['observations']),None),
        at_1s=dict(tof_hits=at_one['tof_hits'],remembered_points=len(at_one['observations']),
            fully_outside_fov=sum(p['outside_front_fov'] for p in at_one['observations'])),
        last_remembered_s=max((row['time_s'] for row in trace if row['observations']),default=None),
        endpoint_checks=checks,all_endpoint_bearings_and_ranges_within_bound=True,
        max_bearing_bound_excess_rad=max_excess,final_state=trace[-1]['state'],
        final_reason=trace[-1].get('reason'),ever_marked_free=False,granted_resume=False,
        heading_fault=heading_fault)
    return trace,summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    sources=('edge/survey_heading.py','edge/survey_memory.py','simulation/survey_heading_sensor.py',
             'simulation/drone_env.py','evaluate/validate_survey_memory.py')
    protocol=dict(scope=__doc__,cases=CASES,seed=SEED,duration_s=3.,physics_model='prescribed poses; no closed-loop flight',
        yaw_deg_s=60.,yaw_stop_s=1.,heading_hz=50,tof_hz=15,heading_delay_s=.012,
        noise_bound_deg=.25,drift_deg_s=.1,geometry='point-rays-solid-v2',
        fixture_box_m=[.8,.95,-.08,.08,.5,1.5],config=asdict(MemoryConfig()),
        source_sha256={name:sha256(ROOT/name) for name in sources})
    (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    traces={};summaries=[]
    for name in CASES:
        trace,summary=run_case(name);traces[name]=trace;summaries.append(summary)
    (args.out/'report.json').write_text(json.dumps(summaries,indent=2),encoding='utf-8')
    (args.out/'traces.json').write_text(json.dumps(traces,indent=2),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,2,figsize=(13,10),layout='constrained',sharex=True,sharey=True)
    for ax,(name,rows),summary in zip(axes.flat,traces.items(),summaries):
        # All retained return centers, with the uncertainty band of one initial return.
        xs=[];ys=[];ts=[];lower=[];upper=[]
        for row in rows:
            for item in row['observations']:
                xs.append(row['time_s']);ys.append(math.degrees(item['bearing_rad']))
                if item['sample_id']==0 and item['zone']==28:
                    ts.append(row['time_s']);lower.append(math.degrees(item['bearing_rad']-item['bearing_halfwidth_rad']))
                    upper.append(math.degrees(item['bearing_rad']+item['bearing_halfwidth_rad']))
        ax.axhspan(-22.5,22.5,color='gray',alpha=.12,label='Front horizontal FOV')
        ax.scatter(xs,ys,s=2,alpha=.12,color='tab:blue',label='Remembered return centers')
        if ts:ax.fill_between(ts,lower,upper,color='tab:orange',alpha=.5,label='One return uncertainty')
        if summary['first_invalid_s'] is not None:
            ax.axvspan(summary['first_invalid_s'],3.,color='tab:red',alpha=.10,label='Invalid: unknown')
        ax.set(title=name,xlabel='Time (s)',ylabel='Bearing in current body (deg)',ylim=(-90,90))
        ax.grid(alpha=.2)
        ax.text(.98,.96,f"t=1 s: ToF {summary['at_1s']['tof_hits']}, memory {summary['at_1s']['remembered_points']}",
                ha='right',va='top',transform=ax.transAxes,fontsize=9)
    axes[0,0].legend(fontsize=7,loc='lower left')
    fig.suptitle('Static hazard memory during prescribed rotation\nOutside FOV is remembered; expiry / invalid evidence become UNKNOWN, never FREE')
    fig.savefig(args.out/'comparison.png',dpi=150);plt.close(fig)
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
