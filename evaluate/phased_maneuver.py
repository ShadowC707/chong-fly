"""One preregistered 6-start x 2-controller experiment, no fitting or tuning."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from demo.runner import DemoRunner
from demo.recording import plain
from edge.runtime import sha256
from edge.system import FlightSystem
from evaluate.compare_maneuver_repair import CASES, summarize
from evaluate.maneuver_robustness import SOURCE, ROOT, start_spec
from simulation.phased_coordinator import PhasedCoordinator, PhasedConfig


def write(path,value):
    path.write_text(json.dumps(plain(value),indent=2,allow_nan=False)+'\n',encoding='utf-8')


def install(system):
    old=system.runtime.coordinator
    new=PhasedCoordinator(anti_rollback=old.anti_rollback,turn_latch=old.turn_latch,
        config=old.config,stop_hover=old.stop_hover,behavior_config=old.behavior_config,seed=old.seed)
    m=deepcopy(system.manifest)
    m['version']='flight-system-phased-experiment-v2'
    m['experiment']=dict(research_only=True,overlay='brake-turn-clear-v2',active_bundle_unchanged=True)
    p=ROOT/'simulation/phased_coordinator.py'
    m['executed_modules'][p.name]=dict(path=str(p),sha256=sha256(p))
    m['resolved']['coordinator']=new.contract()
    m['limitations'].append('Experimental phased overlay; not deployment admission.')
    system.runtime.coordinator=new
    return FlightSystem(system.config,system.runtime,m)


def trial(out,task):
    custom=task.get('scene_spec')
    if custom:
        r=DemoRunner(config=SOURCE/'runtime_config.json',scene='room',seed=custom['seed'],
            start=custom['start'],yaw=custom['yaw'],collision_radius_m=.2,max_duration_s=45.,
            contact_model='sphere-sweep-v1',run_id=task['id'])
        from simulation.drone_env import BoxObstacle, CylinderObstacle
        r.env.boxes=[BoxObstacle(**v) for v in custom['boxes']]
        r.env.cylinders=[CylinderObstacle(**v) for v in custom['cylinders']]
        hit,gap=r.contact(r.start,r.start,r.env.room,r.env.boxes,r.env.cylinders,.2)
        if hit or gap<.2: raise ValueError('Invalid held-out starting clearance')
        r.obs=r.env._sample_observation(0.)
        r.env.demo_geometry_spec=dict(version='explicit-holdout-v1',scene_spec=custom)
    else:
        m=json.loads((SOURCE/'flights'/task['case']/'manifest.json').read_text())
        r=DemoRunner(config=SOURCE/'runtime_config.json',**start_spec(m,task['offset']),
            collision_radius_m=.2,max_duration_s=45.,contact_model='sphere-sweep-v1',run_id=task['id'])
    dest=out/task['id'];dest.mkdir()
    try:
        if not custom:
            for k in ('room','boxes','cylinders'): assert r.manifest()['geometry'][k]==m['geometry'][k]
        if task['mode']=='phased': r.system=install(r.system)
        r._manifest=plain(r._build_manifest());write(dest/'manifest.json',r.manifest())
        r.resume();r.event(dict(type='mode',mode='AUTO'))
        decisions=[]
        for step in range(11250):
            r.advance()
            if step%5==0:
                decisions.append(dict(time_s=r.rows[-1]['time_s'],auto=r.rows[-1]['plant_source']=='auto',
                    **plain(r.system.runtime.coordinator.last_decision)))
            if r.terminal: break
        rows=r.rows;t=np.array([v['time_s'] for v in rows]);dt=np.diff(t,append=r.time_s)
        pos=np.array([v['position_m'] for v in rows]);vel=np.array([v['velocity_mps'] for v in rows])
        yaw=np.array([v['attitude_rad'][2] for v in rows]);gap=np.array([v['clearance_m'] for v in rows])
        auto=np.array([v['plant_source']=='auto' for v in rows]);speed=np.linalg.norm(vel[:,:2],axis=1)
        physical=dict(time=t,pos=pos,velocity=vel,yaw=yaw,clearance=gap,auto=auto)
        np.savez_compressed(dest/'physical.npz',**physical)
        reference=SOURCE/'flights'/task['case']/'physical.npz' if task['offset']==0 else (
            ROOT/'output/maneuver_robustness_v1'/f'candidate_{task["offset"]:+d}'/'physical.npz')
        exact=None
        if task['mode']=='baseline' and not custom:
            with np.load(reference) as old:
                for key,value in physical.items(): np.testing.assert_array_equal(value,old[key],err_msg=task['id']+':'+key)
            exact=True
        controls=rows[::5]
        np.savez_compressed(dest/'trace.npz',time=[v['time_s'] for v in controls],
            **{name:np.array([v['trace'].get(key) or [np.nan]*4 for v in controls]) for name,key in
                [('network','network_pwm'),('coordinated','coordinated_pwm'),('supervised','supervised_pwm'),
                 ('issued','issued_pwm'),('accepted','accepted_pwm')]},
            applied=[v.get('plant_input_pwm') or [np.nan]*4 for v in controls],
            min_tof=[v['min_tof_m'] for v in controls])
        write(dest/'decisions.json',decisions)
        metrics=summarize(rows,r.time_s);i=int(gap.argmin())
        owned=np.array([v.get('phased_owned_channels',[False]*4) for v in decisions]); da=np.array([v['auto'] for v in decisions])
        durations=np.diff([v['time_s'] for v in decisions]+[r.time_s])
        denom=durations[da].sum()
        metrics.update(below_5cm_s=float(dt[auto&(gap<.05)].sum()),moving_auto_s=float(dt[auto&(speed>.03)].sum()),
            stationary_auto_s=float(dt[auto&(speed<=.03)].sum()),
            ownership_fraction_by_channel=((owned*durations[:,None]*da[:,None]).sum(axis=0)/denom).tolist() if denom else [0]*4,
            ownership_any_fraction=float(durations[da&owned.any(axis=1)].sum()/denom) if denom else 0.,
            completed_maneuvers=sum(v.get('phased_event')=='completed' for v in decisions),
            timeout_events=sum(v.get('phased_event') in ('brake_timeout','turn_timeout','episode_timeout') for v in decisions),
            failed_auto_s=float(sum(d for v,d in zip(decisions,durations) if v['auto'] and v.get('phased_phase')=='FAILED')),
            minimum=dict(time_s=float(t[i]),position_m=pos[i].tolist()))
        result=dict(**task,metrics=metrics,baseline_exact=exact,events=r.events)
        write(dest/'report.json',result)
        print(task['id'],json.dumps(metrics),flush=True)
        return result
    finally: r.env.close()


def report(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle,Circle
    rows=json.loads((out/'report.json').read_text())
    fig,axes=plt.subplots(2,3,figsize=(15,10),layout='constrained')
    lines=['# Фазовий координатор: зафіксований експеримент','',
        'Кандидат A; 45 с; радіус 0.2 м; шість пар; жодного підбору після результатів.','',
        '| Старт | Режим | Контакт | Зазор, см | <5 / <10 см, с | Шлях, м | Відкат, м | Висота max, см | Керує фазовий шар, % AUTO | Timeout |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    starts=list(dict.fromkeys((r['case'],r['offset']) for r in rows))
    for ax,(case,offset) in zip(axes.flat,starts):
        m=json.loads((SOURCE/'flights'/case/'manifest.json').read_text())
        for b in m['geometry']['boxes']:
            ax.add_patch(Rectangle((b['x_min'],b['y_min']),b['x_max']-b['x_min'],b['y_max']-b['y_min'],color='.86'))
        for c in m['geometry']['cylinders']: ax.add_patch(Circle((c['center_x'],c['center_y']),c['radius'],color='.86'))
        for r in [r for r in rows if r['case']==case and r['offset']==offset]:
            v=r['metrics']
            with np.load(out/r['id']/'physical.npz') as a:
                line,=ax.plot(*a['pos'][:,:2].T,label=r['mode'])
                ix=int(a['clearance'].argmin());ax.scatter(*a['pos'][ix,:2],marker='x',color=line.get_color())
                ax.scatter(*a['pos'][0,:2],s=20,color='green')
            lines.append(f'| {case} {offset:+d} | {r["mode"]} | {v["collision"] or "немає"} | {100*v["min_clearance_m"]:.2f} | {v["below_5cm_s"]:.2f} / {v["below_10cm_s"]:.2f} | {v["distance_m"]:.2f} | {v["reverse_distance_body_m"]:.3f} | {100*v["max_height_error_m"]:.2f} | {100*v["ownership_any_fraction"]:.1f} | {v["timeout_events"]} |')
        ax.set(xlim=(-4,4),ylim=(-4,4),title=f'{case} {offset:+d} deg',xlabel='X (m)',ylabel='Y (m)');ax.set_aspect('equal');ax.legend()
    fig.savefig(out/'comparison.png',dpi=150);plt.close(fig)
    lines+=['','Хрестик — мінімальний зазор. Timeout із зависанням не є успішним маневром. Відомі сцени, без незалежного holdout і без допуску до польоту.']
    (out/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def run(out):
    out.mkdir(parents=True,exist_ok=False)
    starts=[(c,0) for c in CASES]+[('9d5431c1',-5),('9d5431c1',5)]
    tasks=[dict(id=f'{case}_{offset:+d}_{mode}',case=case,offset=offset,mode=mode) for case,offset in starts for mode in ('baseline','phased')]
    inputs=[SOURCE/'checkpoint.pt',SOURCE/'runtime_config.json',*[p for p in (SOURCE/'numpy').iterdir() if p.is_file()],
        *[p for p in (ROOT/'data/edge/k306_tau025_integrated_v1').iterdir() if p.is_file()],
        *[SOURCE/'flights'/c/'manifest.json' for c in CASES]]
    hashes={str(p):sha256(p) for p in inputs}
    sources={str(p.relative_to(ROOT)):sha256(p) for folder in ('simulation','edge','demo','evaluate') for p in (ROOT/folder).glob('*.py')}
    write(out/'protocol.json',dict(version='phased-coordinator-v2',tasks=tasks,config=asdict(PhasedConfig()),
        horizon_s=45.,radius_m=.2,source=str(SOURCE),input_sha256=hashes,source_sha256=sources,
        parameter_search=False,no_promotion=True,stop_requested=False,
        authority='Existing FlightBridge and receiver; MANUAL reset; STOP priority unchanged; active maneuver retains control in critical range',
        acceptance='No new contact; report clearance, under-5/10cm time, motion, rollback, height, ownership and timeout. No idle-survival success.'))
    snapshot=out/'executed_source';snapshot.mkdir()
    for name in ('simulation/phased_coordinator.py','evaluate/phased_maneuver.py'):
        (snapshot/Path(name).name).write_bytes((ROOT/name).read_bytes())
    results=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(trial,out,task) for task in tasks]
        for future in as_completed(futures):
            results.append(future.result())
            results.sort(key=lambda r: next(i for i,t in enumerate(tasks) if t['id']==r['id']))
            write(out/'report.json',results)
    assert all(sha256(Path(p))==h for p,h in hashes.items())
    assert all(sha256(ROOT/p)==h for p,h in sources.items())
    write(out/'validation.json',dict(completed_trials=len(results),all_six_baselines_exact=all(r['baseline_exact'] for r in results if r['mode']=='baseline'),
        inputs_and_active_unchanged=True,executed_sources_unchanged=True))
    report(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--report-only',action='store_true')
    a=p.parse_args();report(a.out) if a.report_only else run(a.out)
