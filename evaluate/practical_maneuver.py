"""Eight fixed closed-loop trials with passive same-input model/teacher observers."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse
from copy import deepcopy
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
CASES=('068fe108','89d81a9f','9d5431c1','b0a9f212')


class ObservedPolicy:
    def __init__(self,policies,teacher,driver):
        if driver not in policies:raise ValueError('Unknown driver')
        self.policies,self.teacher,self.driver=policies,teacher,driver
        self.records=[];self.segment=-1;self.reset_state()

    def __getattr__(self,key):return getattr(self.policies[self.driver],key)

    def reset_state(self):
        for p in self.policies.values():p.reset_state()
        self.teacher.reset();self.segment+=1

    def step_np(self,flow_xy,tof_8x8,memory_ring=None,dt=None):
        dt=self.default_dt if dt is None else dt
        memory=np.ones(8,np.float32) if memory_ring is None else np.asarray(memory_ring,np.float32)
        x=np.concatenate([flow_xy,tof_8x8,memory]).astype(np.float32)
        outputs={name:p.step_np(x[:2],x[2:66],x[66:],dt=dt).copy() for name,p in self.policies.items()}
        outputs['teacher']=self.teacher.step(x,dt=dt).copy()
        active=float(x[2:66].min())*3.5<.8
        phase=('threat' if active else 'held_brake' if self.teacher._escape_sign is not None else
               'recovery' if self.teacher._cruise_recovery<1 else 'cruise')
        self.records.append(dict(X=x,outputs=outputs,segment=self.segment,phase=phase,dt=dt))
        return outputs[self.driver].copy()


def mask_episodes(rows,mask,*,end_s):
    if len(rows)!=len(mask):raise ValueError('Unaligned episode mask')
    def contiguous(i,j):
        return rows[i]['segment']==rows[j]['segment'] and abs(rows[j]['time_s']-rows[i]['time_s']-.02)<1e-7
    events=[];i=0
    while i<len(rows):
        if not mask[i]:i+=1;continue
        start=i
        while i+1<len(rows) and mask[i+1] and contiguous(i,i+1):i+=1
        end=min(end_s,rows[i]['time_s']+.02)
        events.append(dict(start_s=rows[start]['time_s'],end_s=end,duration_s=end-rows[start]['time_s'],
            left_censored=start==0 or not contiguous(start-1,start),
            right_censored=i+1==len(rows) or not contiguous(i,i+1),segment=rows[start]['segment']))
        i+=1
    return events


def decision_metrics(rows,end_s,driver):
    auto=np.array([r['plant_source']=='auto' for r in rows])
    phase=np.array([r['phase'] for r in rows]);teacher=np.array([r['outputs']['teacher'] for r in rows])
    held=auto&(phase=='held_brake')&(teacher[:,2]<1495)
    released=auto&np.isin(phase,['recovery','cruise'])&(teacher[:,2]>=1515)
    predictions={name:np.array([r['outputs'][name] for r in rows]) for name in ('before','candidate')}
    report=dict(held_brake_episodes=mask_episodes(rows,held,end_s=end_s),same_input={})
    for name,p in predictions.items():
        report['same_input'][name]=dict(held_frames=int(held.sum()),
            held_pitch_mae_pwm=float(np.abs(p[held,2]-teacher[held,2]).mean()) if held.any() else None,
            premature_forward_frames=int((held&(p[:,2]>1510)).sum()),
            release_forward_target_frames=int(released.sum()),
            continued_brake_frames=int((released&(p[:,2]<1495)).sum()))
    report['premature_forward_episodes']=mask_episodes(rows,held&(predictions[driver][:,2]>1510),end_s=end_s)
    report['continued_brake_episodes']=mask_episodes(rows,released&(predictions[driver][:,2]<1495),end_s=end_s)
    return report


def trial(case,path,driver,runs,dest):
    from demo.runner import DemoRunner
    from demo.recording import plain
    from edge.system import FlightSystem
    from edge.runtime import NumpyPolicy,sha256
    from evaluate.compare_maneuver_repair import summarize
    from generator.generate_reflex_dataset import make_teacher
    m=json.loads(path.read_text(encoding='utf-8'))['manifest']
    r=DemoRunner(config=runs[driver]/'runtime_config.json',scene=m['scene'],seed=m['seed'],
        start=m['start']['position_m'],yaw=m['start']['yaw_rad'],scene_options=m.get('scene_options'),
        collision_radius_m=.2,max_duration_s=45.,contact_model='sphere-sweep-v1',run_id=case+'-'+driver)
    try:
        for key in ('room','boxes','cylinders'):assert r.manifest()['geometry'][key]==m['geometry'][key]
        policies={name:r.system.runtime.policy if name==driver else NumpyPolicy(p/'numpy') for name,p in runs.items()}
        observer=ObservedPolicy(policies,make_teacher('geometry-reflex-v5',noise_std_pwm=0,pwm_cruise_pitch=1530.),driver)
        r.system.runtime.policy=observer
        manifest=deepcopy(r.system.manifest)
        manifest['experiment']=dict(version='passive-maneuver-observation-v1',driver=driver,
            policy_outputs='Driver unchanged; other model and v5/1530 observe exactly the same inputs/resets',script_sha256=sha256(__file__))
        r.system=FlightSystem(r.system.config,r.system.runtime,manifest);r._manifest=plain(r._build_manifest())
        dest.mkdir();(dest/'manifest.json').write_text(json.dumps(r.manifest(),indent=2),encoding='utf-8')
        r.resume();r.event(dict(type='mode',mode='AUTO'))
        for _ in range(11250):
            n=len(observer.records);previous_rows=len(r.rows);now=r.time_s
            r.advance()
            if len(observer.records)>n:
                row=next(row for row in r.rows[previous_rows:] if abs(row['time_s']-now)<1e-8)
                observer.records[-1].update(time_s=now,plant_source=row['plant_source'],
                    position_m=row['position_m'],clearance_m=row['clearance_m'])
            if r.terminal:break
        metrics=summarize(r.rows,r.time_s)
        t=np.array([row['time_s'] for row in r.rows]);dt=np.diff(t,append=r.time_s)
        auto=np.array([row['plant_source']=='auto' for row in r.rows])
        vel=np.array([row['velocity_mps'] for row in r.rows]);pos=np.array([row['position_m'] for row in r.rows])
        yaw=np.array([row['attitude_rad'][2] for row in r.rows]);speed=np.linalg.norm(vel[:,:2],axis=1)
        metrics['mean_auto_speed_mps']=float((speed*dt*auto).sum()/metrics['auto_s']) if metrics['auto_s'] else None
        metrics['moving_auto_s']=float(dt[auto&(speed>.03)].sum())
        metrics['stationary_auto_s']=float(dt[auto&(speed<=.03)].sum())
        metrics['auto_distance_m']=float((speed*dt*auto).sum())
        metrics['valid_autonomous_trial']=metrics['auto_s']>0
        controls=r.rows[::5]
        trace=dict(time=np.array([row['time_s'] for row in controls]),pos=np.array([row['position_m'] for row in controls]),
            velocity=np.array([row['velocity_mps'] for row in controls]),
            network=np.array([row['trace'].get('network_pwm') or [np.nan]*4 for row in controls]),
            coordinated=np.array([row['trace'].get('coordinated_pwm') or [np.nan]*4 for row in controls]),
            applied=np.array([row.get('plant_input_pwm') or [np.nan]*4 for row in controls]),
            min_tof=np.array([row['min_tof_m'] for row in controls]),clearance=np.array([row['clearance_m'] for row in controls]))
        np.savez_compressed(dest/'trace.npz',**trace)
        np.savez_compressed(dest/'physical.npz',time=t,pos=pos,velocity=vel,yaw=yaw,auto=auto,
            clearance=[row['clearance_m'] for row in r.rows])
        if driver=='before':
            with np.load(ROOT/'output/neural_flow_balance_v1/before'/case/'trace.npz') as old:
                for key in ('pos','network'):np.testing.assert_array_equal(trace[key],old[key])
        records=observer.records
        np.savez_compressed(dest/'decisions.npz',X=np.array([v['X'] for v in records]),time=[v['time_s'] for v in records],
            segment=[v['segment'] for v in records],phase=[v['phase'] for v in records],auto=[v['plant_source']=='auto' for v in records],
            pos=[v['position_m'] for v in records],clearance=[v['clearance_m'] for v in records],
            **{name:np.array([v['outputs'][name] for v in records]) for name in (*runs,'teacher')})
        report=dict(case=case,driver=driver,metrics=metrics,events=r.events,
            baseline_exact=driver=='before',decisions=decision_metrics(records,r.time_s,driver),source_sha256=sha256(path))
        (dest/'report.json').write_text(json.dumps(plain(report),indent=2,allow_nan=False),encoding='utf-8')
        return report
    finally:r.env.close()


def run(out):
    from edge.runtime import sha256
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    source=ROOT/'output/balanced_maneuver_v1'
    runs=dict(before=source/'before',candidate=source/'plastic')
    paths={case:next((ROOT/'flight simulations/changed contract').glob('*'+case+'*.json')) for case in CASES}
    inputs={str(p):sha256(p) for p in paths.values()}
    for run in runs.values():
        for p in (run/'checkpoint.pt',run/'runtime_config.json',*list((run/'numpy').glob('*'))):
            if p.is_file():inputs[str(p)]=sha256(p)
    configs=[json.loads((p/'runtime_config.json').read_text()) for p in runs.values()]
    assert {k:v for k,v in configs[0].items() if k not in ('bundle_dir','expected_metadata_sha256')}=={k:v for k,v in configs[1].items() if k not in ('bundle_dir','expected_metadata_sha256')}
    fixed=json.loads((source/'protocol.json').read_text())['fixed_baseline_sha256']
    assert all(sha256(ROOT/p)==v for p,v in fixed.items())
    protocol=dict(version='practical-maneuver-v1',cases=CASES,drivers=list(runs),horizon_s=45,
        collision_radius_m=.2,contact_model='sphere-sweep-v1',same_runtime_configuration_except_bundle=True,
        input_sha256=inputs,script_sha256=sha256(__file__),no_training=True,no_parameter_search=True,
        teacher='Passive v5/1530 observer; never drives or supplies network inputs',
        criteria='Compare contacts, clearance, AUTO exposure, motion and retreat; new collision in baseline-completing scene is regression',
        limitations='Four known layouts, one seed each; candidate has different cruise behavior. Teacher mismatch is descriptive, not proof of collision causality.',
        promotion=False)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    results=[]
    for case,path in paths.items():
        for driver in runs:
            report=trial(case,path,driver,runs,out/(case+'_'+driver));results.append(report)
            (out/'report.json').write_text(json.dumps(results,indent=2,allow_nan=False),encoding='utf-8')
            print(case,driver,report['metrics'],flush=True)
    assert all(sha256(Path(p))==v for p,v in inputs.items())
    assert all(sha256(ROOT/p)==v for p,v in fixed.items())
    (out/'validation.json').write_text(json.dumps(dict(completed_trials=8,baseline_exact=4,
        input_and_active_baseline_unchanged=True,no_training=True,automatic_promotion=False),indent=2),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    run(p.parse_args().out)
