"""Fixed 4-start x 3-controller decision experiment on historical demo physics.

No training, promotion, parameter search, SITL changes, or forced teacher actions.
Protocol and geometric direction labels are written before any flight.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from demo.profiles import CANDIDATE_A_CONFIG, ORIGINAL_CONFIG, PHASED_PROFILES

ROOT = Path(__file__).resolve().parents[1]
CASES = ('9d5431c1', 'b0a9f212')
MODES = ('original_xy', 'candidate_xy', 'candidate_smooth')
CONFIGS = {'original': ORIGINAL_CONFIG, 'candidate': CANDIDATE_A_CONFIG}
MVP_ROOMS = ROOT/'data/scenarios/reiteration_v1/mvp_rooms.json'
HORIZON = 45.


def write(path, value):
    from demo.recording import plain
    path.write_text(json.dumps(plain(value), indent=2, allow_nan=False)+'\n', encoding='utf-8')


def mirror_scene(scene):
    s = deepcopy(scene)
    s['start'][1] *= -1
    s['yaw'] *= -1
    s['expected_sign'] *= -1
    for b in [s['room'], *s['boxes']]:
        b['y_min'], b['y_max'] = -b['y_max'], -b['y_min']
    for c in s['cylinders']: c['center_y'] *= -1
    return s


def mirror_input(x):
    x = np.array(x, copy=True)
    x[1] *= -1
    x[2:66] = x[2:66].reshape(8,8)[:, ::-1].ravel()
    # These k306 policies mask memory inputs; preserve their neutral ones.
    return x


def first_turn(t, pwm, auto, minimum, expected):
    """First >=50 PWM yaw of one sign for three consecutive 20-ms AUTO ticks.

    A turn before ToF<1m is retained (not silently discarded); latency is signed.
    Missing/ambiguous outcomes never count as success.
    """
    t, pwm, auto, minimum = map(np.asarray, (t, pwm, auto, minimum))
    threat = np.flatnonzero(auto & (minimum < 1.))
    onset = float(t[threat[0]]) if len(threat) else None
    yaw = pwm[:,3]-1500
    for i in range(max(0,len(t)-2)):
        if (auto[i:i+3].all() and np.allclose(np.diff(t[i:i+3]),.02,rtol=0,atol=1e-8)
                and np.isfinite(yaw[i:i+3]).all() and (np.abs(yaw[i:i+3])>=50).all()
                and (np.sign(yaw[i:i+3])==np.sign(yaw[i])).all()):
            sign = int(np.sign(yaw[i]))
            return dict(time_s=float(t[i]),sign=sign,pwm=float(pwm[i,3]),
                        threat_onset_s=onset,latency_s=None if onset is None else float(t[i]-onset),
                        correct=sign==expected,expected_sign=expected)
    return dict(time_s=None,sign=0,pwm=None,threat_onset_s=onset,latency_s=None,
                correct=False,expected_sign=expected)


class PassivePair:
    """Driver remains exact. Two independent networks observe its inputs/resets."""
    def __init__(self, driver, original, candidate):
        self.driver = driver
        self.observers = dict(original=original,candidate=candidate)
        self.records = []
        self.segment = -1
        self.reset_state()

    def __getattr__(self, key): return getattr(self.driver,key)

    def reset_state(self):
        self.driver.reset_state()
        for p in self.observers.values(): p.reset_state()
        self.segment += 1

    def step_np(self, flow_xy, tof_8x8, memory_ring=None, dt=None):
        dt = self.default_dt if dt is None else dt
        memory = np.ones(8,np.float32) if memory_ring is None else memory_ring
        x = np.concatenate([flow_xy,np.asarray(tof_8x8).ravel(),memory]).astype(np.float32)
        result = self.driver.step_np(flow_xy,tof_8x8,memory_ring,dt=dt)
        outputs = {name:p.step_np(flow_xy,tof_8x8,memory_ring,dt=dt).copy()
                   for name,p in self.observers.items()}
        self.records.append(dict(X=x,segment=self.segment,dt=dt,**outputs))
        return result


def label_scene(s):
    """Straight, level sensor-only approach, not a model trajectory or teacher."""
    from simulation.drone_env import RoomBoundaries, BoxObstacle, CylinderObstacle, ToFRaycaster, QuadcopterDynamics
    rotation = QuadcopterDynamics.quaternion_to_rotation_matrix(
        QuadcopterDynamics.euler_to_quaternion(0,0,s['yaw']))
    ray = ToFRaycaster()
    for travel in np.arange(0.,3.,.01):
        pos = np.array(s['start']) + rotation[:,0]*travel
        d = ray.cast_rays(pos,rotation,RoomBoundaries(**s['room']),
            [CylinderObstacle(**c) for c in s['cylinders']],
            [BoxObstacle(**b) for b in s['boxes']]).reshape(8,8)*3.5
        if d.min()<1.:
            left,right = [float(np.minimum(a,2.).mean()) for a in (d[:,:4],d[:,4:])]
            if abs(right-left)<=.05: raise ValueError('Preflight label is ambiguous')
            return dict(position_m=pos.tolist(),min_m=float(d.min()),left_mean_m=left,
                        right_mean_m=right,expected_sign=1 if right>left else -1)
    raise ValueError('No visible threat on preregistered straight approach')


def make_protocol(out, phased_profile='phased_smooth'):
    from edge.runtime import sha256
    from edge.system import RunConfig
    configs = {k:RunConfig.from_json(p) for k,p in CONFIGS.items()}
    shared = [{k:v for k,v in c.to_dict().items() if k not in ('bundle_dir','expected_metadata_sha256')}
              for c in configs.values()]
    if shared[0]!=shared[1]: raise ValueError('Non-policy run configurations differ')
    fixture = json.loads(MVP_ROOMS.read_text(encoding='utf-8'))
    inputs = {str(MVP_ROOMS.relative_to(ROOT)): sha256(MVP_ROOMS)}
    scenes = {}
    for case in CASES:
        s = dict(fixture['cases'][case]['scene'])
        label = label_scene(s); s['expected_sign']=label['expected_sign']
        for suffix, scene in (('native',s),('mirror',mirror_scene(s))):
            check = label_scene(scene)
            if check['expected_sign']!=scene['expected_sign']: raise ValueError('Mirror label mismatch')
            scenes[case+'_'+suffix] = dict(scene,geometric_label=check)
    for c in configs.values():
        for p in Path(c.bundle_dir).iterdir():
            if p.is_file(): inputs[str(p.relative_to(ROOT))]=sha256(p)
    # Pin all local Python inputs and configuration, not just the new script.
    for folder in ('demo','simulation','edge','configs','evaluate'):
        for p in (ROOT/folder).glob('*.py'): inputs[str(p.relative_to(ROOT))]=sha256(p)
    for p in CONFIGS.values(): inputs[str(p.relative_to(ROOT))]=sha256(p)
    protocol = dict(version='mvp-selection-v1',scenes=scenes,modes=MODES,horizon_s=HORIZON,
        scenario_fixture=fixture['version'], source_recordings={k:v['source'] for k,v in fixture['cases'].items()},
        phased_profile=phased_profile,
        tasks=[dict(id=s+'_'+mode,scene=s,mode=mode,phased_profile=phased_profile)
               for s in scenes for mode in MODES],
        physics='Historical DemoRunner standalone plant; NOT SITL/Betaflight',
        collision='sphere-sweep-v1 radius 0.2m',sensor_schedule='multi 50/30/15 Hz; existing seeded noise unchanged',
        control_dt_s=.02,physics_dt_s=.004,stop='No scheduled STOP; same 45s historical horizon',
        label_rule='First straight level ToF<1m at 1cm increments; clipped-2m half-mean gap >.05m; +PWM turns right',
        first_turn='First |Yaw-1500|>=50 for three contiguous AUTO controls, same sign; no threshold retuning',
        criteria=dict(direction='Accepted first turn matches preflight side in all 4 starts',
            safety='All 4 complete 45s with no collision/fault, clearance>=.10m and continuous AUTO after engagement',
            retreat='Sum reverse distance (body forward speed < -0.03m/s) strictly below original; individual regressions retained'),
        same_input='Passive original/A on identical driver inputs, dt and reset; offline mirrored replay resets at same boundaries',
        scope='Known-room MVP selection, not unseen-environment generalization or hardware admission',
        no_training=True,no_search=True,no_promotion=True,input_sha256=inputs)
    write(out/'protocol.json',protocol)
    return protocol


def trial(out, task, scene):
    from demo.runner import DemoRunner
    from demo.recording import plain
    from demo.yaw_experiment import install
    from edge.runtime import NumpyPolicy
    from edge.system import RunConfig
    from simulation.drone_env import RoomBoundaries,BoxObstacle,CylinderObstacle
    from evaluate.compare_maneuver_repair import summarize
    selected = 'original' if task['mode']=='original_xy' else 'candidate'
    r = DemoRunner(config=CONFIGS[selected],scene='room',seed=scene['seed'],start=scene['start'],
        yaw=scene['yaw'],run_id=task['id'],collision_radius_m=.2,max_duration_s=HORIZON,contact_model='sphere-sweep-v1')
    dest = Path(out)/task['id']; dest.mkdir()
    try:
        r.env.room=RoomBoundaries(**scene['room'])
        r.env.boxes=[BoxObstacle(**v) for v in scene['boxes']]
        r.env.cylinders=[CylinderObstacle(**v) for v in scene['cylinders']]
        hit,gap=r.contact(r.start,r.start,r.env.room,r.env.boxes,r.env.cylinders,.2)
        if hit or gap<.2: raise ValueError('Invalid initial clearance')
        r.obs=r.env._sample_observation(0.)
        r.env.demo_geometry_spec=dict(version='mvp-selection-v1',scene=scene)
        if task['mode']=='candidate_smooth':
            r.system=install(r.system, profile=task.get('phased_profile', 'phased_smooth'))
        observers={k:NumpyPolicy(RunConfig.from_json(p).bundle_dir) for k,p in CONFIGS.items()}
        observer=PassivePair(r.system.runtime.policy,**observers)
        r.system.runtime.policy=observer
        r._manifest=plain(r._build_manifest())
        write(dest/'manifest.json',r.manifest())
        r.resume(); r.event(dict(type='mode',mode='AUTO'))
        decisions=[]
        for step in range(round(HORIZON/.004)):
            old=len(observer.records); row_start=len(r.rows)
            r.advance()
            if len(observer.records)>old:
                row=r.rows[row_start]
                record=observer.records[-1]
                record.update(time_s=row['time_s'],auto=row['plant_source']=='auto')
                np.testing.assert_array_equal(record[selected],row['trace']['network_pwm'],err_msg='Passive driver parity')
                decisions.append(dict(time_s=row['time_s'],auto=record['auto'],
                    **plain(r.system.runtime.coordinator.last_decision)))
            if r.terminal: break
        rows=r.rows; controls=[v for v in rows if abs(v['time_s']/.02-round(v['time_s']/.02))<1e-7 and v['plant_input_pwm'] is not None]
        metrics=summarize(rows,r.time_s)
        t=np.array([v['time_s'] for v in controls]); auto=np.array([v['plant_source']=='auto' for v in controls])
        minimum=np.array([v['min_tof_m'] for v in controls])
        arrays={name:np.array([v['trace'].get(key) or [np.nan]*4 for v in controls]) for name,key in
            [('network','network_pwm'),('coordinated','coordinated_pwm'),('supervised','supervised_pwm'),('issued','issued_pwm'),('accepted','accepted_pwm')]}
        turns={k:first_turn(t,v,auto,minimum,scene['expected_sign']) for k,v in arrays.items()}
        owned=np.array([d.get('phased_owned_channels',[False]*4) for d in decisions])
        da=np.array([d['auto'] for d in decisions])
        metrics.update(ownership_any_fraction=float(owned[da].any(axis=1).mean()) if da.any() else 0.,
            ownership_by_channel=owned[da].mean(axis=0).tolist() if da.any() else [0.]*4,
            coordinator_changed_fraction=float(np.any(abs(arrays['network'][auto,1:]-arrays['coordinated'][auto,1:])>1e-6,axis=1).mean()) if auto.any() else 0.,
            max_height_error_m=float(max(abs(v['position_m'][2]-1.) for v in rows)),
            timeout_count=sum(d.get('phased_event') in ('turn_timeout','episode_timeout','brake_timeout') for d in decisions))
        engage=np.flatnonzero(auto)
        continuous=bool(len(engage) and auto[engage[0]:].all())
        faults=[e for e in r.events if e['kind'] in ('simulation_fault','bridge_fault')]
        checks=dict(direction=turns['accepted']['correct'],full_horizon=metrics['completed'],
            clearance_10cm=metrics['min_clearance_m']>=.1,continuous_auto=continuous,no_fault=not faults)
        physical=dict(time=[v['time_s'] for v in rows],pos=[v['position_m'] for v in rows],
            velocity=[v['velocity_mps'] for v in rows],yaw=[v['attitude_rad'][2] for v in rows],
            clearance=[v['clearance_m'] for v in rows],auto=[v['plant_source']=='auto' for v in rows])
        np.savez_compressed(dest/'physical.npz',**physical)
        np.savez_compressed(dest/'trace.npz',time=t,auto=auto,min_tof=minimum,**arrays)
        write(dest/'decisions.json',decisions)
        rec=observer.records
        np.savez_compressed(dest/'same_input.npz',X=[v['X'] for v in rec],time=[v['time_s'] for v in rec],
            auto=[v['auto'] for v in rec],segment=[v['segment'] for v in rec],dt=[v['dt'] for v in rec],
            original=[v['original'] for v in rec],candidate=[v['candidate'] for v in rec])
        # Same histories, mirrored inputs; memory channels must be inactive.
        for p in observers.values():
            if np.any(p.arrays['W_in'][:,66:] != 0): raise ValueError('Mirror requires inactive memory channels')
        mirrored={k:[] for k in observers}; segment=None
        for row in rec:
            if row['segment']!=segment:
                for p in observers.values(): p.reset_state()
                segment=row['segment']
            x=mirror_input(row['X'])
            for k,p in observers.items(): mirrored[k].append(p.step_np(x[:2],x[2:66],x[66:],dt=row['dt']).copy())
        np.savez_compressed(dest/'mirrored_replay.npz',**mirrored)
        replay={}
        rt=np.array([v['time_s'] for v in rec]);ra=np.array([v['auto'] for v in rec]);rm=np.array([v['X'][2:66].min()*3.5 for v in rec])
        for k in observers:
            replay[k]=dict(native=first_turn(rt,np.array([v[k] for v in rec]),ra,rm,scene['expected_sign']),
                mirrored=first_turn(rt,np.array(mirrored[k]),ra,rm,-scene['expected_sign']))
        result=dict(**task,metrics=metrics,first_turn=turns,same_input=replay,checks=checks,
                    passed_direction_and_safety=all(checks.values()),events=r.events,passive_driver_exact=True)
        write(dest/'report.json',result)
        return result
    finally: r.env.close()


def render(out, protocol, results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle,Circle
    fig,axes=plt.subplots(2,2,figsize=(13,12),layout='constrained')
    for ax,(name,s) in zip(axes.flat,protocol['scenes'].items()):
        for b in s['boxes']: ax.add_patch(Rectangle((b['x_min'],b['y_min']),b['x_max']-b['x_min'],b['y_max']-b['y_min'],color='.85'))
        for c in s['cylinders']: ax.add_patch(Circle((c['center_x'],c['center_y']),c['radius'],color='.85'))
        for mode in MODES:
            row=next(r for r in results if r['scene']==name and r['mode']==mode)
            with np.load(out/row['id']/'physical.npz') as a:
                line,=ax.plot(a['pos'][:,0],a['pos'][:,1],label=mode)
                ax.scatter(*a['pos'][-1,:2],c=line.get_color(),marker='x')
        ax.scatter(*s['start'][:2],c='black',marker='o');ax.set(xlim=(-4,4),ylim=(-4,4),aspect='equal',title=name,xlabel='x (m)',ylabel='y (m)');ax.legend(fontsize=8)
    fig.savefig(out/'comparison.png',dpi=150);plt.close(fig)


def run(out, workers, phased_profile='phased_smooth'):
    from edge.runtime import sha256
    out.mkdir(parents=True,exist_ok=False)
    protocol=make_protocol(out, phased_profile)
    print('Protocol frozen:',out/'protocol.json',flush=True)
    results=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(trial,out,t,protocol['scenes'][t['scene']]):t for t in protocol['tasks']}
        for future in as_completed(futures):
            result=future.result();results.append(result)
            print(result['id'],json.dumps(dict(checks=result['checks'],clearance=result['metrics']['min_clearance_m'],reverse=result['metrics']['reverse_distance_body_m'])),flush=True)
    results.sort(key=lambda r:r['id']);write(out/'report.json',results)
    unchanged=all(sha256(ROOT/p)==v for p,v in protocol['input_sha256'].items())
    totals={mode:sum(r['metrics']['reverse_distance_body_m'] for r in results if r['mode']==mode) for mode in MODES}
    decision={mode:dict(direction_and_safety_passes=sum(r['passed_direction_and_safety'] for r in results if r['mode']==mode),
        reverse_distance_sum_m=totals[mode],reverse_better_than_original=totals[mode]<totals['original_xy']) for mode in MODES}
    for mode,v in decision.items(): v['accepted']=v['direction_and_safety_passes']==4 and v['reverse_better_than_original']
    write(out/'decision.json',dict(input_unchanged=unchanged,modes=decision,automatic_promotion=False))
    if not unchanged: raise ValueError('Experiment inputs changed during series')
    render(out,protocol,results)
    print(json.dumps(decision,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--workers',type=int,choices=(1,2,3),default=3)
    p.add_argument('--phased-profile', choices=PHASED_PROFILES,
                   default='phased_smooth')
    args=p.parse_args();run(args.out,args.workers,args.phased_profile)
