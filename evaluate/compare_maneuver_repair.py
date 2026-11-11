"""Finite paired experiment on saved demo geometries; no model fitting/export.

Example: python -m evaluate.compare_maneuver_repair --out output/maneuver_repair_v1
The protocol is written before trials. Historical playback is checked first.
New trials use the same 0.2 m sphere and a 45 s horizon without scheduled STOP.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import gc
import json
from pathlib import Path

import numpy as np

from demo.examples import recompute
from demo.recording import plain
from demo.runner import DemoRunner
from edge.runtime import sha256
from edge.system import FlightSystem
from simulation.maneuver_coordinator import ManeuverCoordinator, ManeuverConfig

ROOT = Path(__file__).resolve().parents[1]
CASES = ('068fe108', '89d81a9f', '9d5431c1', 'b0a9f212')
VARIANTS = ('baseline', 'release_only', 'flow_only', 'both')
HORIZON_S = 45.


def install_experiment(system, variant):
    """Explicit experimental assembly; never claim this is the pinned bundle.

The base system has already verified its pinned implementation. Record the
additional implementation and new resolved controller before constructing a
fresh bridge, so every command trace refers to the experimental manifest.
"""
    if variant not in VARIANTS:
        raise ValueError('Unknown maneuver variant')
    if variant == 'baseline':
        return system
    old = system.runtime.coordinator
    if old is None or old.levy_explore:
        raise ValueError('Experiment requires non-search XY coordinator')
    new = ManeuverCoordinator(anti_rollback=old.anti_rollback, turn_latch=old.turn_latch,
        config=old.config, stop_hover=old.stop_hover, behavior_config=old.behavior_config,
        seed=old.seed, release_on_neutral=variant in ('release_only','both'),
        gate_turn_by_flow=variant in ('flow_only','both'))
    manifest = system.manifest
    path = ROOT/'simulation/maneuver_coordinator.py'
    manifest['version'] = 'flight-system-maneuver-experiment-v1'
    manifest['experiment'] = dict(variant=variant, research_only=True,
                                  active_bundle_unchanged=True)
    manifest['executed_modules']['maneuver_coordinator.py'] = dict(path=str(path),sha256=sha256(path))
    manifest['resolved']['coordinator'] = new.contract()
    manifest['limitations'].append('Experimental coordinator overlay; not pinned deployment admission.')
    system.runtime.coordinator = new
    return FlightSystem(system.config,system.runtime,manifest)


def verify_history(path):
    data=json.loads(Path(path).read_text(encoding='utf-8'))
    result=recompute(data)
    exact=result['frames']==data['frames']
    report=dict(source=str(Path(path).relative_to(ROOT)),sha256=sha256(path),
                frames=len(data['frames']),frames_exact=exact,
                geometry_exact=result['manifest']['geometry']==data['manifest']['geometry'])
    if not exact or not report['geometry_exact']:
        raise ValueError('Historical replay differs: '+str(report))
    return report


def summarize(rows, end_s):
    t=np.array([r['time_s'] for r in rows]);dt=np.diff(t,append=end_s)
    p=np.array([r['position_m'] for r in rows])
    v=np.array([r['velocity_mps'] for r in rows])
    yaw=np.array([r['attitude_rad'][2] for r in rows])
    auto=np.array([r['plant_source']=='auto' for r in rows])
    forward=v[:,0]*np.cos(yaw)+v[:,1]*np.sin(yaw)
    reverse=auto & (forward < -.03)
    clearance=np.array([r['clearance_m'] for r in rows])
    collision=next((r['collision'] for r in rows if r['collision']),None)
    return dict(duration_s=float(end_s),collision=collision,
        completed=bool(collision is None and end_s>=HORIZON_S-1e-8),
        auto_s=float(dt[auto].sum()),min_clearance_m=float(clearance.min()),
        below_10cm_s=float(dt[auto & (clearance<.1)].sum()),
        reverse_s=float(dt[reverse].sum()),
        reverse_distance_body_m=float((-forward[reverse]*dt[reverse]).sum()),
        reverse_fraction_auto=float(dt[reverse].sum()/dt[auto].sum()) if auto.any() else None,
        distance_m=float(np.linalg.norm(np.diff(p[:,:2],axis=0),axis=1).sum()),
        max_height_error_m=float(abs(p[:,2]-1.).max()),
        final_speed_mps=float(np.linalg.norm(v[-1,:2])),
        latch_flip_s=float(sum(d for r,d in zip(rows,dt) if r['plant_source']=='auto'
                           and 'turn_latch' in r['trace'].get('coordinator_reasons',[]))))


def trial(task):
    path,variant,out=task
    data=json.loads(Path(path).read_text(encoding='utf-8'));m=data['manifest']
    prefix=next(k for k in CASES if k in Path(path).name)
    directory=Path(out)/(prefix+'_'+variant);directory.mkdir()
    r=DemoRunner(scene=m['scene'],seed=m['seed'],start=m['start']['position_m'],
        yaw=m['start']['yaw_rad'],scene_options=m.get('scene_options'),
        collision_radius_m=.2,max_duration_s=HORIZON_S,contact_model='sphere-sweep-v1',
        run_id='maneuver-'+prefix+'-'+variant)
    for key in ('room','boxes','cylinders'):
        if r.manifest()['geometry'][key] != m['geometry'][key]:
            raise ValueError('Scene regeneration differs: '+key)
    r.system=install_experiment(r.system,variant)
    r._manifest=plain(r._build_manifest())
    (directory/'manifest.json').write_text(json.dumps(r.manifest(),indent=2),encoding='utf-8')
    del data
    r.resume();r.event(dict(type='mode',mode='AUTO'))
    decisions=[]
    for step in range(round(HORIZON_S/.004)):
        r.advance()
        if step%5==0:
            decisions.append(dict(time_s=step*.004,**plain(r.system.runtime.coordinator.last_decision)))
        if r.terminal: break
    report=dict(case=prefix,variant=variant,source_sha256=sha256(path),
                metrics=summarize(r.rows,r.time_s),events=r.events)
    ix=[];previous=None
    for i,row in enumerate(r.rows):
        tick=row['trace'].get('tick_time_s')
        if tick!=previous:
            ix.append(i);previous=tick
    controls=[r.rows[i] for i in ix]
    np.savez_compressed(directory/'trace.npz',
        time=[x['time_s'] for x in controls],pos=[x['position_m'] for x in controls],
        velocity=[x['velocity_mps'] for x in controls],yaw=[x['attitude_rad'][2] for x in controls],
        network=[x['trace'].get('network_pwm') or [np.nan]*4 for x in controls],
        applied=[x.get('plant_input_pwm') or [np.nan]*4 for x in controls],
        min_tof=[x['min_tof_m'] for x in controls],clearance=[x['clearance_m'] for x in controls])
    (directory/'decisions.json').write_text(json.dumps(decisions,indent=2),encoding='utf-8')
    (directory/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    r.env.close()
    gc.collect()
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=2,choices=(1,2))
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    paths=[next((ROOT/'flight simulations/changed contract').glob('*'+case+'*.json')) for case in CASES]
    protocol=dict(version='maneuver-repair-paired-v1',cases=[str(p.relative_to(ROOT)) for p in paths],
        source_sha256={str(p.relative_to(ROOT)):sha256(p) for p in paths},
        variants=VARIANTS,horizon_s=HORIZON_S,collision_radius_m=.2,contact_model='sphere-sweep-v1',
        stop_requested=False,levy_explore=False,parameter_search=False,
        maneuver_config=asdict(ManeuverConfig()),
        script_sha256=sha256(__file__),repair_sha256=sha256(ROOT/'simulation/maneuver_coordinator.py'),
        primary='contacts, completed horizon, clearance; reverse time compared only alongside exposure',
        promotion='No new collision in a baseline-completing case; no automatic deployment from these four scenes.')
    (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    checks=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for check in pool.map(verify_history,paths):
            checks.append(check)
            print('Historical replay exact:',Path(check['source']).name,flush=True)
        (args.out/'historical_replay.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
        futures=[pool.submit(trial,(str(p),v,str(args.out))) for p in paths for v in VARIANTS]
        reports=[]
        for future in as_completed(futures):
            result=future.result();reports.append(result)
            (args.out/'report.json').write_text(json.dumps(sorted(reports,key=lambda r:(r['case'],r['variant'])),indent=2),encoding='utf-8')
            print(result['case'],result['variant'],result['metrics'],flush=True)


if __name__=='__main__':
    main()
