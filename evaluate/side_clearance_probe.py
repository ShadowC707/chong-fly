"""One fixed diagnostic brake, paired with exact historical control. Not a policy.

Only coordinator Pitch is changed at 6<=t<7, before supervision/receiver.
No world state, teacher action, time-window fitting, or promotion.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json
import numpy as np
from evaluate.mvp_selection import ROOT, CONFIGS, write

REFERENCE=ROOT/'output/mvp_selection_v1'
CASE='9d5431c1_mirror'
START,END,RAMP=6.,7.,.12


def pitch_probe(action, flow, now, phase, stop):
    result=np.array(action,copy=True,dtype=float)
    if stop or phase!='CRUISE' or not START<=now<END: return result,0.
    x=float(np.clip(min(now-START,END-now)/RAMP,0.,1.))
    weight=x*x*(3-2*x)
    target=min(float(action[2]),1500.-np.clip(400.*max(0.,flow),0.,100.))
    result[2]+=weight*(target-result[2])
    return result,weight


class PitchProbe:
    def __init__(self, base, enabled):
        self.base,self.enabled,self.now_s=base,enabled,0.
    def __getattr__(self,key): return getattr(self.base,key)
    def contract(self):
        return self.base.contract() | dict(diagnostic_pitch=dict(enabled=self.enabled,
            window_s=[START,END],ramp_s=RAMP,gain=400.,limit_pwm=100.,scope='CRUISE only; explicit clock intervention, not deployable'))
    def apply(self,*args,**kwargs):
        action=self.base.apply(*args,**kwargs)
        native=action.copy();weight=0.
        if self.enabled:
            action,weight=pitch_probe(action,self.base.filtered_flow,self.now_s,self.base.phase,kwargs.get('stop_requested',False))
        changed=bool(action[2]!=native[2])
        self.base.last_decision['pitch_probe']=dict(time_s=self.now_s,weight=weight,changed=changed,
            native_pitch=float(native[2]),output_pitch=float(action[2]))
        if changed:
            self.base.previous_pr=action[1:3].copy()
            self.base.last_decision['reasons'].append('diagnostic_pitch')
            self.base.last_decision['intervened']=True
        return action


def trial(out,enabled,scene):
    from demo.runner import DemoRunner
    from demo.yaw_experiment import install,YawHandoverSystem
    from demo.recording import plain
    from simulation.drone_env import RoomBoundaries,BoxObstacle,CylinderObstacle,QuadcopterDynamics
    from evaluate.compare_maneuver_repair import summarize
    from edge.runtime import sha256
    mode='brake' if enabled else 'control'
    dest=out/mode;dest.mkdir()
    r=DemoRunner(config=CONFIGS['candidate'],scene='room',seed=scene['seed'],start=scene['start'],
        yaw=scene['yaw'],run_id='side-clearance-'+mode,collision_radius_m=.2,max_duration_s=45.,contact_model='sphere-sweep-v1')
    try:
        r.env.room=RoomBoundaries(**scene['room'])
        r.env.boxes=[BoxObstacle(**v) for v in scene['boxes']]
        r.env.cylinders=[CylinderObstacle(**v) for v in scene['cylinders']]
        r.obs=r.env._sample_observation(0.)
        r.env.demo_geometry_spec=dict(version='mvp-selection-v1',scene=scene)
        r.system=install(r.system)
        probe=PitchProbe(r.system.runtime.coordinator,enabled)
        r.system.runtime.coordinator=probe
        m=r.system.manifest
        m['experiment']['diagnostic_pitch']=probe.contract()['diagnostic_pitch']
        m['resolved']['coordinator']=probe.contract()
        m['executed_modules'][Path(__file__).name]=dict(path=__file__,sha256=sha256(__file__))
        r.system=YawHandoverSystem(r.system.config,r.system.runtime,m)
        r._manifest=plain(r._build_manifest());write(dest/'manifest.json',r.manifest())
        r.resume();r.event(dict(type='mode',mode='AUTO'))
        decisions=[]
        for step in range(11250):
            probe.now_s=r.time_s
            r.advance()
            if step%5==0:
                decisions.append(dict(time_s=r.rows[-1]['time_s'],**plain(probe.last_decision)))
            if r.terminal: break
        rows=r.rows;controls=[v for v in rows if abs(v['time_s']/.02-round(v['time_s']/.02))<1e-7 and v['plant_input_pwm'] is not None]
        physical=dict(time=[v['time_s'] for v in rows],pos=[v['position_m'] for v in rows],
            velocity=[v['velocity_mps'] for v in rows],yaw=[v['attitude_rad'][2] for v in rows],
            clearance=[v['clearance_m'] for v in rows],auto=[v['plant_source']=='auto' for v in rows])
        trace=dict(time=[v['time_s'] for v in controls],auto=[v['plant_source']=='auto' for v in controls],min_tof=[v['min_tof_m'] for v in controls],
            **{name:np.array([v['trace'].get(key) or [np.nan]*4 for v in controls]) for name,key in
              [('network','network_pwm'),('coordinated','coordinated_pwm'),('supervised','supervised_pwm'),('issued','issued_pwm'),('accepted','accepted_pwm')]})
        np.savez_compressed(dest/'physical.npz',**physical)
        np.savez_compressed(dest/'trace.npz',**trace)
        # Full attitude allows object visibility attribution, absent in old compact traces.
        np.savez_compressed(dest/'sensors.npz',time=trace['time'],
            pos=[v['position_m'] for v in controls],attitude=[v['attitude_rad'] for v in controls],
            tof=[v['sensors']['tof']['value'] for v in controls],tof_timestamp=[v['sensors']['tof']['timestamp_s'] for v in controls])
        write(dest/'decisions.json',decisions)
        old_folder=REFERENCE/(CASE+'_candidate_smooth')
        for filename,arrays in (('physical.npz',physical),('trace.npz',trace)):
            with np.load(old_folder/filename) as old:
                now=np.asarray(arrays['time']);mask=now<START if enabled else np.ones(len(now),bool)
                for key,value in arrays.items():
                    np.testing.assert_array_equal(np.asarray(value)[mask],old[key][:len(now)][mask],err_msg=mode+':'+key)
        metrics=summarize(rows,r.time_s)
        pos=np.asarray(physical['pos']);gap=np.asarray(physical['clearance']);times=np.asarray(physical['time']);i=int(gap.argmin())
        # Terminal collision row is not a command/authority sample.
        auto=np.asarray(trace['auto']);first=np.flatnonzero(auto)
        metrics.update(minimum=dict(time_s=float(times[i]),position_m=pos[i].tolist()),
            continuous_auto=bool(len(first) and auto[first[0]:].all()),
            intervention_controls=sum(d.get('pitch_probe',{}).get('changed',False) for d in decisions),
            window_min_clearance_m=float(gap[(times>=START)&(times<=16)].min()),
            max_pitch_change_pwm=max(abs(d.get('pitch_probe',{}).get('native_pitch',0)-d.get('pitch_probe',{}).get('output_pitch',0)) for d in decisions),
            fault=any(e['kind'] in ('bridge_fault','simulation_fault') for e in r.events))
        result=dict(mode=mode,metrics=metrics,baseline_exact=not enabled,pre_intervention_exact=True,events=r.events)
        write(dest/'report.json',result)
        return result
    finally:r.env.close()


def visibility_audit(out,scene):
    from simulation.drone_env import RoomBoundaries,BoxObstacle,CylinderObstacle,QuadcopterDynamics,ToFRaycaster
    # Reconstruct ray hits at actual capture pose, never combine held ToF with new attitude.
    ray=ToFRaycaster();room=RoomBoundaries(**scene['room'])
    boxes=[BoxObstacle(**v) for v in scene['boxes']];cylinders=[CylinderObstacle(**v) for v in scene['cylinders']]
    result=[]
    with np.load(out/'control/sensors.npz') as a:
        for target in (5.,5.2,5.5,6.,6.5,9.06,9.08,10.94):
            i=int(np.argmin(abs(a['time']-target)));capture=float(a['tof_timestamp'][i])
            # Sensor is computed after the preceding 4ms physics step. Exact pose is
            # the next row's pre-step state at that capture timestamp.
            j=int(np.argmin(abs(a['time']-capture)))
            if abs(a['time'][j]-capture)>1e-8: raise ValueError('Capture pose unavailable')
            pos=a['pos'][j];rot=QuadcopterDynamics.quaternion_to_rotation_matrix(QuadcopterDynamics.euler_to_quaternion(*a['attitude'][j]))
            full=ray.cast_rays(pos,rot,room,cylinders,boxes)
            # Stored Euler -> quaternion -> rotation roundtrip is not bit exact.
            # Record discrepancy; 1 mm is an analysis-only reconstruction bound,
            # never a change to collision, clearance or admission tolerances.
            reconstruction_error_m=float(np.max(abs(full-a['tof'][i]))*3.5)
            if reconstruction_error_m>.001: raise ValueError('Ray reconstruction differs by more than 1mm')
            wall=ray.cast_rays(pos,rot,room,[],[])
            values=[];names=[]
            for kind,objects in (('box',boxes),('cylinder',cylinders)):
                for k,obj in enumerate(objects):
                    values.append(ray.cast_rays(pos,rot,room,[obj] if kind=='cylinder' else [],[obj] if kind=='box' else []));names.append(f'{kind}_{k}')
            values=np.asarray(values)
            owner=[]
            for col in (0,7):
                pixel=col+8*int(np.argmin(full.reshape(8,8)[:,col]))
                ids=np.flatnonzero((abs(values[:,pixel]-full[pixel])<=1e-6)&(values[:,pixel]<wall[pixel]-1e-6))
                owner.append(dict(column=col,distance_m=float(full[pixel]*3.5),objects=[names[k] for k in ids]))
            idx=names.index('cylinder_0')
            visible=(abs(values[idx]-full)<=1e-6)&(values[idx]<wall-1e-6)
            result.append(dict(time_s=float(a['time'][i]),capture_s=capture,edges=owner,
                reconstruction_error_m=reconstruction_error_m,critical_cylinder_visible_pixels=int(visible.sum())))
    write(out/'visibility_audit.json',result)


def run(out):
    from edge.runtime import sha256
    out.mkdir(parents=True,exist_ok=False)
    old=json.loads((REFERENCE/'protocol.json').read_text())
    if not all(sha256(ROOT/p)==h for p,h in old['input_sha256'].items()): raise ValueError('Reference sources changed')
    hashes=dict(old['input_sha256']);hashes[str(Path(__file__).relative_to(ROOT))]=sha256(__file__)
    for name in ('physical.npz','trace.npz'):
        p=REFERENCE/(CASE+'_candidate_smooth')/name;hashes[str(p.relative_to(ROOT))]=sha256(p)
    protocol=dict(version='side-clearance-pitch-probe-v1',scene=CASE,window_s=[START,END],ramp_s=RAMP,
        intervention='Pitch only: min(native,1500-clip(400*max(filtered_flow_x,0),0,100)); CRUISE only; STOP untouched',
        source='Existing 400 gain / 100 PWM damping bound, .12s existing behavior transition; no grid search',
        reason='6s begins reaction to opposite-side obstacle while forward speed rebuilds; fixed 1s diagnostic',
        verdict='Useful corrective lead only if full 45s no collision/fault, clearance>=.10m, no reverse regression and continuous AUTO',
        limits='Time-scheduled counterfactual, not sensor policy or deployment. One attempt, no window/parameter retries.',
        modes=['control','brake'],input_sha256=hashes)
    write(out/'protocol.json',protocol);print('Protocol frozen',flush=True)
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(trial,out,enabled,old['scenes'][CASE]) for enabled in (False,True)]
        results=[f.result() for f in futures]
    write(out/'report.json',results)
    a,b=[r['metrics'] for r in results]
    checks=dict(complete=b['completed'],clearance=b['min_clearance_m']>=.1,no_fault=not b['fault'],
        continuous_auto=b['continuous_auto'],no_reverse_regression=b['reverse_distance_body_m']<=a['reverse_distance_body_m']+1e-9)
    unchanged=all(sha256(ROOT/p)==h for p,h in hashes.items())
    write(out/'decision.json',dict(checks=checks,useful_corrective_lead=all(checks.values()),sources_unchanged=unchanged,no_promotion=True))
    if not unchanged:raise ValueError('Sources changed')
    print(json.dumps(results,indent=2),flush=True)
    render_report(out,old['scenes'][CASE])


def render_report(out,scene):
    from edge.runtime import sha256
    protocol=json.loads((out/'protocol.json').read_text())
    for p,h in protocol['input_sha256'].items():
        source=ROOT/p
        if sha256(source)!=h:
            archived=out/'executed_source'/source.name
            if source!=Path(__file__) or not archived.exists() or sha256(archived)!=h:
                raise ValueError('Modified experiment source without verified execution snapshot: '+p)
    visibility_audit(out,scene)
    authority={}
    for mode in ('control','brake'):
        with np.load(out/mode/'trace.npz') as a:
            ids=np.flatnonzero(a['auto'])
            authority[mode]=bool(len(ids) and a['auto'][ids[0]:].all())
    write(out/'reporting_audit.json',dict(report_source_sha256=sha256(__file__),
        continuous_auto_on_command_rows=authority,
        note='Original brake report continuous_auto=false included terminal collision row; no authority loss before collision. Full-horizon/clearance rejection unchanged.',
        physical_runs_repeated=False))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle,Circle
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for b in scene['boxes']:
        axes[0,0].add_patch(Rectangle((b['x_min'],b['y_min']),b['x_max']-b['x_min'],b['y_max']-b['y_min'],color='.85'))
    for c in scene['cylinders']:
        axes[0,0].add_patch(Circle((c['center_x'],c['center_y']),c['radius'],color='.85'))
    for mode in ('control','brake'):
        with np.load(out/mode/'physical.npz') as a,np.load(out/mode/'trace.npz') as b:
            axes[0,0].plot(a['pos'][:,0],a['pos'][:,1],label=mode)
            axes[0,1].plot(a['time'],a['clearance']*100,label=mode)
            axes[1,0].plot(b['time'],b['accepted'][:,2],label=mode)
            axes[1,1].plot(b['time'],b['network'][:,3],label=mode)
    axes[0,0].set(title='Trajectory',xlabel='x (m)',ylabel='y (m)',aspect='equal')
    axes[0,1].set(title='Body clearance',xlabel='time (s)',ylabel='cm');axes[0,1].axhline(10,color='red',ls='--')
    axes[1,0].set(title='Accepted Pitch',xlabel='time (s)',ylabel='PWM',xlim=(4,16))
    axes[1,1].set(title='Network Yaw (not directly overridden)',xlabel='time (s)',ylabel='PWM',xlim=(4,16))
    for ax in (axes[0,1],axes[1,0],axes[1,1]):ax.axvspan(START,END,color='grey',alpha=.15)
    for ax in axes.flat:ax.legend()
    fig.savefig(out/'comparison.png',dpi=150);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--report-only',action='store_true')
    args=p.parse_args()
    if args.report_only:
        scene=json.loads((REFERENCE/'protocol.json').read_text())['scenes'][CASE]
        render_report(args.out,scene)
    else: run(args.out)
