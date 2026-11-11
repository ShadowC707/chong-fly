"""First paired neural/XY behavior probe: legacy plant vs actual patched SITL.

Direct pinned runtime, ideal synchronous sensors, no authority/transport claim.
Obstacles terminate a swept-sphere trajectory; no impact response is simulated.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
from contextlib import closing
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from simulation.sitl_flight import (CASES, DT, HORIZON, RADIUS, STOP_AT, FlightView, SceneSensors,
                                    actuator_command, flight_steps, scene_spec, validate_duration)


def trial(out,source,case,backend,axis_contract='native_rc',*,physics_dt_s=DT,integration_profile=None,yaw_shaping=False,
          scene_override=None,tof_probe=None,target_twr=None,calibration_rows=None,
          horizon_s=HORIZON,stop_at_s=STOP_AT):
    validate_duration(horizon_s,stop_at_s)
    steps=flight_steps(out,source,case,backend,axis_contract,physics_dt_s=physics_dt_s,
        integration_profile=integration_profile,yaw_shaping=yaw_shaping,scene_override=scene_override,
        tof_probe=tof_probe,target_twr=target_twr,calibration_rows=calibration_rows,
        horizon_s=horizon_s,stop_at_s=stop_at_s)
    with closing(steps):
        next(steps)
        try:
            while True: steps.send({})
        except StopIteration as end:
            return end.value


def report(out,results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig,axes=plt.subplots(2,2,figsize=(11,10),constrained_layout=True)
    lines=['# Незмінна мережа + фазовий координатор: перший SITL дослід','',
           '| Сцена | Контур | Повний час | Контакт | Зазор, см | Шлях, м | Висота max, см | STOP max, м/с | Власність фаз, % |',
           '|---|---|---|---|---:|---:|---:|---:|---:|']
    for ax,case in zip(axes.flat,CASES):
        for box in scene_spec(case)['boxes']:
            ax.add_patch(Rectangle((box['x_min'],box['y_min']),box['x_max']-box['x_min'],box['y_max']-box['y_min'],color='.8'))
        for backend in ('legacy','sitl'):
            name=case+'_'+backend;r=next(v for v in results if v['case']==case and v['backend']==backend)
            with np.load(out/name/'trace.npz') as a:
                s=a['state'];line,=ax.plot(s[:,0],s[:,1],label=backend)
                ax.scatter(*s[-1,:2],color=line.get_color(),marker='x' if r['collision'] else 'o')
            stop=r['stop_last_second_max_speed_mps']
            lines.append(f"| {case} | {backend} | {r['completed']} | {r['collision'] or '—'} | {100*r['min_clearance_m']:.2f} | {r['path_m']:.2f} | {100*r['max_height_error_m']:.2f} | {stop if stop is not None else 'n/a'} | {100*r['coordinator_owned_fraction']:.1f} |")
        ax.scatter(0,0,color='green',s=20)
        ax.set(xlim=(-4,4),ylim=(-4,4),title=case,xlabel='X (m)',ylabel='Y (m)')
        ax.set_aspect('equal');ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Same pinned network and phased coordinator | ideal sensors at 50 Hz\n'
                 'Legacy dynamics + approximate FC vs PyBullet + patched Betaflight')
    fig.savefig(out/'comparison.png',dpi=145);plt.close(fig)
    (out/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def run(out,source,resolution,axis_contract='native_rc'):
    from evaluate.sitl_lockstep import BIN
    from edge.system import RunConfig
    if not json.loads((resolution/'comparison.json').read_text())['passed']:
        raise ValueError('Resolution experiment has not passed')
    prior=json.loads((source/'report.json').read_text())
    if hashlib.sha256(BIN.read_bytes()).hexdigest()!=prior['firmware_sha256']:
        raise ValueError('Firmware differs from validated build')
    out.mkdir(parents=True,exist_ok=False)
    cfg=RunConfig.from_json('configs/demo_candidate_a.json')
    files=[Path('configs/demo_candidate_a.json'),Path('configs/ready_sim_reference_v1.json'),Path('configs/sitl_virtual_v1.cli'),
           *Path(cfg.bundle_dir).glob('*')]
    files += [p for folder in ('simulation','edge','demo','evaluate','configs') for p in Path(folder).glob('*.py')]
    files=[p for p in files if p.is_file()]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    protocol=dict(cases=[scene_spec(c) | dict(name=c) for c in CASES],backends=['legacy','sitl'],
        horizon_s=HORIZON,stop_at_s=STOP_AT,dt_s=DT,control_dt_s=.02,axis_contract=axis_contract,
        sensors='ideal 50Hz synchronous; no noise, packet loss or hold',
        scope='Direct runtime behavior. No FlightBridge/CommandGate/transport validation.',
        limits='Report ALL collisions, minimum clearance, height, path, STOP < .03 m/s throughout final second, phase ownership and errors; no tuning.',
        firmware_sha256=prior['firmware_sha256'],input_source_sha256=hashes)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    for name in ('evaluate/sitl_neural.py','evaluate/sitl_lockstep.py','evaluate/ready_sim_spike.py',
                 'simulation/sitl_rc_units.py','simulation/sitl_flight.py','simulation/sitl_plant.py',
                 'simulation/sitl_process.py','simulation/sitl_control.py'):
        dest=out/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(name,dest)
    results=[]
    for case in CASES:
        for backend in ('legacy','sitl'):
            r=trial(out/(case+'_'+backend),source,case,backend,axis_contract);results.append(r)
            (out/'report.json').write_text(json.dumps(results,indent=2)+'\n')
            print(json.dumps(r),flush=True)
    unchanged=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==value for p,value in hashes.items())
    (out/'validation.json').write_text(json.dumps(dict(sources_unchanged=unchanged,
        runs=len(results),completed=sum(r['completed'] for r in results),
        contacts=sum(r['collision'] is not None for r in results),errors=sum(r['error'] is not None for r in results),
        deployment_admitted=False),indent=2)+'\n')
    report(out,results)
    if not unchanged or any(r['error'] for r in results):raise SystemExit(1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--resolution',type=Path,required=True)
    parser.add_argument('--axis-contract',choices=('native_rc','reference_units'),default='native_rc')
    args=parser.parse_args();run(args.out,args.source,args.resolution,args.axis_contract)


if __name__=='__main__':main()
