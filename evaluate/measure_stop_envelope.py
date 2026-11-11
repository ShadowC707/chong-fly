"""Measure the unchanged STOP with explicit synthetic initial physical states.

The pinned full demo chain admits AUTO normally before a simulation-only state
perturbation. No authority checks are bypassed. This is a braking test bench,
not evidence that every perturbed state is reachable from the warmup history.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SEEDS=(20261021,20261022,20261023)
WARMUP_S=1.
HORIZON_S=8.
SPEED_LIMIT=.03
DWELL_S=1.
RADIUS_M=.2
CASES={
    'still':dict(velocity=[0,0,0]),
    'forward_01':dict(velocity=[.1,0,0]),
    'forward_02':dict(velocity=[.2,0,0]),
    'forward_04':dict(velocity=[.4,0,0]),
    'forward_06':dict(velocity=[.6,0,0]),
    'backward_02':dict(velocity=[-.2,0,0]),
    'left_02':dict(velocity=[0,.2,0]),
    'right_02':dict(velocity=[0,-.2,0]),
    'diagonal_left':dict(velocity=[.2,.2,0]),
    'diagonal_right':dict(velocity=[.2,-.2,0]),
    'pitch_forward_5':dict(velocity=[.2,0,0],pitch_deg=5),
    'pitch_backward_5':dict(velocity=[.2,0,0],pitch_deg=-5),
    'roll_left_5':dict(velocity=[.2,.2,0],roll_deg=-5),
    'roll_right_5':dict(velocity=[.2,-.2,0],roll_deg=5),
}
DELAY_CASES=('forward_02','forward_04','forward_06','diagonal_left')


def stopping_metrics(time,position,velocity,*,radius=RADIUS_M):
    t=np.asarray(time,float);p=np.asarray(position,float);v=np.asarray(velocity,float)
    if (t.ndim!=1 or len(t)<2 or p.shape!=(len(t),3) or v.shape!=p.shape
            or not all(np.isfinite(x).all() for x in (t,p,v)) or np.any(np.diff(t)<=0)
            or not np.isfinite(radius) or radius<0):
        raise ValueError('Invalid physical stopping trace')
    t=t-t[0];speed=np.linalg.norm(v[:,:2],axis=1)
    low=speed<SPEED_LIMIT
    below=np.flatnonzero(low)
    last_high=np.flatnonzero(~low)
    start=int(last_high[-1]+1) if len(last_high) else 0
    settled=start<len(t) and t[-1]-t[start]>=DWELL_S-1e-8
    excursion=np.linalg.norm(p[:,:2]-p[0,:2],axis=1)
    end_mask=t>=t[-1]-DWELL_S-1e-8
    return dict(horizon_s=float(t[-1]),initial_speed_mps=float(speed[0]),
        first_below_s=float(t[below[0]]) if len(below) else None,
        settled=bool(settled),settled_from_s=float(t[start]) if settled else None,
        settle_observed_s=float(t[start]+DWELL_S) if settled else None,
        final_second_max_speed_mps=float(speed[end_mask].max()),
        max_excursion_m=float(excursion.max()),
        max_excursion_time_s=float(t[int(excursion.argmax())]),
        swept_body_radius_m=float(radius+excursion.max()),
        final_displacement_m=float(excursion[-1]),
        path_length_m=float(np.linalg.norm(np.diff(p[:,:2],axis=0),axis=1).sum()),
        max_abs_x_m=float(abs(p[:,0]-p[0,0]).max()),max_abs_y_m=float(abs(p[:,1]-p[0,1]).max()),
        max_height_error_m=float(abs(p[:,2]-1.).max()),peak_speed_mps=float(speed.max()))


def run_trial(task):
    from demo.runner import DemoRunner
    from demo.recording import plain
    from edge.runtime import sha256
    name,seed,delay,out=task;spec=CASES[name]
    ident=f'{name}_seed{seed}_delay{round(delay*1000):03d}'
    folder=Path(out)/ident;folder.mkdir()
    r=DemoRunner(scene='clear',start=(0,0,1),seed=seed,collision_radius_m=RADIUS_M,
                 max_duration_s=WARMUP_S+HORIZON_S,run_id='stop-'+ident)
    try:
        assert not r.env.boxes and not r.env.cylinders
        r.resume();r.event(dict(type='mode',mode='AUTO'));r.advance(round(WARMUP_S/.004))
        if r.previous_source!='auto' or r.terminal:
            raise RuntimeError('Warmup did not admit AUTO; do not bypass admission')
        before=dict(position=r.env.physics.pos.copy(),velocity=r.env.physics.vel.copy(),
            quaternion=r.env.physics.quat.copy(),angular_velocity=r.env.physics.omega.copy(),
            motor_rpms=r.env.physics.motor_rpms.copy(),previous_pwm=r.previous_pwm,
            previous_pr=r.system.runtime.coordinator.previous_pr.copy(),
            filtered_flow=[r.system.runtime.coordinator.filtered_flow,r.system.runtime.coordinator.filtered_flow_y])
        # Defined test input: retain real warm actuator/controller states; change
        # only physical pose/velocity at a tick where all sensors capture anew.
        r.env.physics.pos[:]=[0,0,1]
        r.env.physics.vel[:]=spec['velocity']
        angles=np.radians([spec.get('roll_deg',0),spec.get('pitch_deg',0),0])
        r.env.physics.quat[:]=r.env.physics.euler_to_quaternion(*angles)
        r.env.physics.omega[:]=0
        r.obs=r.env._sample_observation(0.)
        manifest=r.manifest()
        manifest['experiment']=dict(version='stop-envelope-v1',name=name,seed=seed,
            injected_at_s=WARMUP_S,stop_delay_s=delay,initial_state=spec,
            warm_state=plain(before),physics_params=asdict(r.env.params),
            script_sha256=sha256(__file__),
            scope='Synthetic physical state perturbation after normal AUTO admission; no hardware qualification.')
        (folder/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        for step in range(round(HORIZON_S/.004)):
            if step==round(delay/.004):r.event(dict(type='stop'))
            r.advance()
            if r.terminal:break
        rows=[row for row in r.rows if row['time_s']>=WARMUP_S-1e-8]
        t=np.array([row['time_s']-WARMUP_S for row in rows])
        p=np.array([row['position_m'] for row in rows]);v=np.array([row['velocity_mps'] for row in rows])
        if r.time_s-WARMUP_S>t[-1]+1e-8:
            t=np.r_[t,r.time_s-WARMUP_S]
            p=np.vstack([p,r.env.physics.pos]);v=np.vstack([v,r.env.physics.vel])
        metrics=stopping_metrics(t,p,v)
        request_index=int(np.argmin(abs(t-delay)))
        after=stopping_metrics(t[request_index:],p[request_index:],v[request_index:])
        first=rows[0]
        fresh={key:abs(sample['timestamp_s']-WARMUP_S)<1e-8 for key,sample in first['sensors'].items()}
        assert all(fresh.values()), 'Initial sensor capture did not align with injected state'
        assert all(row['plant_source']=='auto' for row in rows), 'Lost AUTO in calibration'
        states={state:sum(row['trace'].get('behavior_state')==state for row in rows)
                for state in ('CRUISE','STOP','HOVER','STOP_EVASION')}
        if states['STOP_EVASION']:
            raise RuntimeError('Scene is not a free-space STOP test')
        report=dict(id=ident,case=name,seed=seed,delay_s=delay,metrics=metrics,after_request=after,
            behavior_counts=states,all_initial_sensors_fresh=fresh,
            events=r.events,collision=next((row['collision'] for row in rows if row['collision']),None))
        np.savez_compressed(folder/'trace.npz',time=t,pos=p,velocity=v,
            command_time=[row['time_s']-WARMUP_S for row in rows],
            pwm=[row['plant_input_pwm'] for row in rows],flow=[row['flow_xy'] for row in rows])
        (folder/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        return report
    finally:
        r.env.close()


def main():
    from edge.runtime import sha256
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--workers',type=int,choices=(1,2),default=2)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    tasks=[(name,seed,0.,str(args.out)) for name in CASES for seed in SEEDS]
    tasks += [(name,seed,.1,str(args.out)) for name in DELAY_CASES for seed in SEEDS]
    protocol=dict(version='stop-envelope-v1',cases=CASES,seeds=SEEDS,delayed_cases=DELAY_CASES,
        extra_stop_request_delay_s=.1,trials=len(tasks),warmup_s=WARMUP_S,horizon_s=HORIZON_S,
        radius_m=RADIUS_M,speed_threshold_mps=SPEED_LIMIT,dwell_s=DWELL_S,
        sensor_profile='50/30/15 Hz',target_height_m=1,parameter_search=False,
        config_sha256=sha256(ROOT/'configs/mvp_runtime_v3.json'),script_sha256=sha256(__file__),
        interpretation='Finite sampled envelope, not a certified upper bound. Delay is external STOP request latency, not packet loss.',
        initialization='Normal 1 s AUTO warmup, then explicit physical pose/velocity perturbation; controller and actuator histories retained.')
    (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    reports=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(run_trial,task) for task in tasks]
        for future in as_completed(futures):
            result=future.result();reports.append(result)
            (args.out/'report.json').write_text(json.dumps(sorted(reports,key=lambda r:r['id']),indent=2),encoding='utf-8')
            m=result['metrics']
            print(result['id'],'excursion',round(m['max_excursion_m'],3),'settled_from',m['settled_from_s'],flush=True)


if __name__=='__main__':main()
