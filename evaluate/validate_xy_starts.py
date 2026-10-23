"""Frozen dense-arena avoidance validation: eight starts, two sensor-noise seeds."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path

import numpy as np
import torch

from configs.flight_config import SENSORS
from evaluate.watch_mvp import DEFAULT_RUN, ROOT, run_view
from evaluate.compare_xy_behaviors import summarize
from generator.source_contract import file_sha256
from simulation.xy_coordinator import XYConfig, BehaviorConfig
from simulation.xy_scenes import DensityStartArena


STARTS=(
    dict(name='cylinder_front',position_m=[0.,0.,1.],yaw_rad=0.,purpose='Direct approach to east cylinder'),
    dict(name='box_front',position_m=[0.,0.,1.],yaw_rad=math.pi,purpose='Direct approach to west box'),
    dict(name='cylinder_oblique',position_m=[.8,-1.,1.],yaw_rad=math.atan2(1.,1.5),purpose='Oblique east-cylinder approach'),
    dict(name='box_oblique',position_m=[-.8,-1.,1.],yaw_rad=math.atan2(1.,-1.5),purpose='Oblique west-box approach'),
    dict(name='gap_ne',position_m=[0.,0.,1.],yaw_rad=math.pi/4,purpose='Between east/north objects toward NE interior cylinder'),
    dict(name='gap_sw',position_m=[0.,0.,1.],yaw_rad=-3*math.pi/4,purpose='Between west/south objects toward SW interior box'),
    dict(name='corner_ne',position_m=[2.9,2.9,1.],yaw_rad=math.pi/4,purpose='NE room corner, initial clearance validated'),
    dict(name='corner_sw',position_m=[-2.9,-2.9,1.],yaw_rad=-3*math.pi/4,purpose='SW room corner, initial clearance validated'),
)
SEEDS=(20261021,20261022)
MIN_CLEARANCE_M=.05
CLOSE_THRESHOLD_M=.10
MAX_CLOSE_EPISODE_S=.5


def seed_probe():
    env=DensityStartArena('arena_dense',start_pos=[0.,0.,1.],start_yaw=0.)
    try:
        def sample(seed):
            obs=env.reset(seed=seed)
            state=np.r_[env.physics.pos,env.physics.quat,env.physics.vel,env.physics.omega]
            return obs,state
        a,sa=sample(SEEDS[0]); b,sb=sample(SEEDS[0]); c,sc=sample(SEEDS[1])
        return dict(same_seed_exact=bool(np.array_equal(a,b)),
            different_seed_flow_differs=bool(not np.array_equal(a[:2],c[:2])),
            different_seed_tof_equal=bool(np.array_equal(a[2:],c[2:])),
            same_physical_state=bool(np.array_equal(sa,sb) and np.array_equal(sa,sc)),
            flow_noise_std=SENSORS.flow_noise_std,laser_noise_std_m=SENSORS.laser_noise_std,
            source='DroneSimulationEnv.reset seeds self.rng; laser and flow consume this rng; ToF raycasting has no random draws')
    finally: env.close()


def proximity_metrics(physics):
    """Endpoint sample represents its preceding physics interval; do not bridge safe gaps."""
    result={}
    for key,threshold in [('below_005',MIN_CLEARANCE_M),('below_010',CLOSE_THRESHOLD_M)]:
        events=[]; active=None; elapsed=0.
        for previous,row in zip(physics,physics[1:]):
            dt=row['time_s']-previous['time_s']
            clearance=row['clearance_m']
            if dt<=0 or not np.isfinite(clearance): raise ValueError('Invalid physical clearance trace')
            if clearance<threshold:
                if active is None: active=previous['time_s']
                elapsed+=dt
            elif active is not None:
                events.append(dict(start_s=active,end_s=previous['time_s'],duration_s=elapsed))
                active=None; elapsed=0.
        if active is not None:
            events.append(dict(start_s=active,end_s=physics[-1]['time_s'],duration_s=elapsed))
        result[key]=dict(threshold_m=threshold,total_s=sum(e['duration_s'] for e in events),
            episodes=len(events),longest_s=max((e['duration_s'] for e in events),default=0.),events=events)
    return result


def admission(row):
    checks=dict(complete_no_collision=bool(row['complete'] and not row['crashed'] and row.get('feasible',True)),
        minimum_clearance=row['min_clearance_m']>=MIN_CLEARANCE_M,
        no_prolonged_close_approach=row['proximity']['below_010']['longest_s']<=MAX_CLOSE_EPISODE_S+1e-9,
        stop=bool((row['stop'] or {}).get('speed_target_passed',False)))
    return dict(passed=all(checks.values()),checks=checks)


def plot_starts(out_dir,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle,Rectangle
    for seed in SEEDS:
        fig,axes=plt.subplots(4,2,figsize=(12,20),layout='constrained')
        selected=[r for r in rows if r['seed']==seed]
        for ax,row in zip(axes.flat,selected):
            report=json.loads((out_dir/row['report_file']).read_text(encoding='utf-8'))
            for b in report['geometry']['boxes']:
                ax.add_patch(Rectangle((b['x_min'],b['y_min']),b['x_max']-b['x_min'],b['y_max']-b['y_min'],color='#bbc5d0'))
            for c in report['geometry']['cylinders']:
                ax.add_patch(Circle((c['center_x'],c['center_y']),c['radius'],color='#c8b795'))
            trace=report['trace']['physics']; p=np.array([x['position_m'] for x in trace]); t=np.array([x['time_s'] for x in trace])
            before=t<35
            ax.plot(p[before,0],p[before,1],color='#2364aa',label='Avoidance')
            ax.plot(p[~before,0],p[~before,1],color='#d04b24',label='STOP')
            ax.scatter(*p[0,:2],c='green',label='Start'); ax.scatter(*p[-1,:2],c='red',marker='x',label='End')
            yaw=row['start']['yaw_rad']; ax.arrow(*p[0,:2],.4*math.cos(yaw),.4*math.sin(yaw),width=.025,color='green')
            event=row['min_clearance_event']
            if event:
                ax.scatter(*event['position_m'][:2],c='purple',marker='D',label=f"Min @ {event['time_s']:.3f}s")
            hazard=row['proximity']['below_010']
            ax.set(title=f"{row['start']['name']} | {'PASS' if row['admission']['passed'] else 'FAIL'}\n"
                f"clearance {row['min_clearance_m']:.3f} m | <10cm {hazard['total_s']:.3f}s / {hazard['episodes']} episodes",
                xlim=(-4,4),ylim=(-4,4),aspect='equal',xlabel='World X (m)',ylabel='World Y (m)')
            ax.grid(alpha=.2); ax.legend(fontsize=7)
        fig.suptitle(f'Frozen k306 + original XY both + STOP | no search | seed {seed}',fontsize=14)
        fig.savefig(out_dir/f'trajectories_seed_{seed}.png',dpi=130); plt.close(fig)


def run_experiment(run_dir,out_dir):
    if out_dir.exists(): raise FileExistsError('Use a new output directory')
    probe=seed_probe()
    if not all(probe[k] for k in ('same_seed_exact','different_seed_flow_differs','different_seed_tof_equal','same_physical_state')):
        raise AssertionError('Seed/noise contract changed; review protocol before running')
    starts=[]
    for s in STARTS:
        env=DensityStartArena('arena_dense',start_pos=s['position_m'],start_yaw=s['yaw_rad'])
        try: starts.append(dict(**s,specification=env.specification()))
        finally: env.close()
    digest=file_sha256(run_dir/'checkpoint.pt')
    plan=dict(version='xy-start-validation-v1',scene='arena_dense',starts=starts,seeds=list(SEEDS),seed_probe=probe,
        run_count=len(starts)*len(SEEDS),seconds=40.,stop_at_s=35.,checkpoint_sha256=digest,
        control=dict(coordinator_mode='both',levy_explore=False,stop_hover=True,explore=False,
            xy_config=asdict(XYConfig()),behavior_config=asdict(BehaviorConfig())),
        admission=dict(complete_seconds=40.,collision_free=True,min_clearance_m=MIN_CLEARANCE_M,
            close_threshold_m=CLOSE_THRESHOLD_M,max_contiguous_close_s=MAX_CLOSE_EPISODE_S,
            stop='Every physical sample in [39,40] s has XY speed <0.03 m/s',
            status='Predeclared engineering screening thresholds, not hardware safety limits'),
        limitations=['Eight selected cold starts, two noise seeds; no probability-of-safety estimate.',
            'Zero initial translation/rotation velocity and recurrent state; no moving-entry validation.',
            'No target waypoint: avoidance/turnaround may pass; passage completion is not required.',
            'Coverage is visited 0.5m cells, not mapped room area.',
            'Finite simulation only; no tuning or hardware admission.'],
        code_sha256={str(p.relative_to(ROOT)):file_sha256(p) for folder in ('simulation','core','configs')
            for p in sorted((ROOT/folder).rglob('*.py'))})
    for name in ('evaluate/validate_xy_starts.py','evaluate/watch_mvp.py','evaluate/compare_xy_behaviors.py',
                 'optimizer/rollout.py','optimizer/benchmark_candidate.py', 'artifacts/research.py', 'artifacts/io.py', 'core/invariants.py'):
        plan['code_sha256'][name]=file_sha256(ROOT/name)
    out_dir.mkdir(parents=True)
    (out_dir/'plan.json').write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8')
    rows=[]
    for s in starts:
        for seed in SEEDS:
            stem=f"{s['name']}_{seed}"
            report=run_view(run_dir,scene='arena_dense',steps=2000,seed=seed,coordinator_mode='both',
                levy_explore=False,stop_at_s=35.,plot_out=out_dir/f'{stem}.png',
                arena_start=dict(start_pos=s['position_m'],start_yaw=s['yaw_rad']))
            row=summarize(report)
            row.update(start=s,seed=seed,report_file=f'{stem}.json',
                min_clearance_event=report['metrics']['min_clearance_event'],
                proximity=proximity_metrics(report['trace']['physics']))
            row['admission']=admission(row)
            report['start_validation']=row
            (out_dir/f'{stem}.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
            rows.append(row)
            (out_dir/'comparison.json').write_text(json.dumps(dict(plan=plan,runs=rows),indent=2,allow_nan=False)+'\n',encoding='utf-8')
            print(json.dumps(dict(start=s['name'],seed=seed,crashed=row['crashed'],duration_s=row['duration_s'],
                clearance=row['min_clearance_m'],close_time=row['proximity']['below_010']['total_s'],
                admission=row['admission'])),flush=True)
    if file_sha256(run_dir/'checkpoint.pt')!=digest: raise AssertionError('Checkpoint changed')
    for name,expected in plan['code_sha256'].items():
        if file_sha256(ROOT/name)!=expected: raise AssertionError(f'Code changed during experiment: {name}')
    plot_starts(out_dir,rows)
    lines=['# Frozen XY start validation','',
        '| Start | Seed | Complete / crash | Min cm | Min time / XY | <5cm s | <10cm s / episodes / longest s | STOP | Pass |',
        '|---|---:|---|---:|---|---:|---|---|---|']
    for r in rows:
        e=r['min_clearance_event']; h=r['proximity']['below_010']; low=r['proximity']['below_005']; stop=r['stop'] or {}
        event=f"{e['time_s']:.3f}s ({e['position_m'][0]:.3f}, {e['position_m'][1]:.3f})" if e else 'N/A'
        lines.append(f"| {r['start']['name']} | {r['seed']} | {r['complete']} / {r['crashed']} | {100*r['min_clearance_m']:.2f} | "
            f"{event} | {low['total_s']:.3f} | {h['total_s']:.3f} / {h['episodes']} / {h['longest_s']:.3f} | "
            f"{stop.get('speed_target_passed',False)} | {r['admission']['passed']} |")
    lines+=['',*plan['limitations']]
    (out_dir/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=DEFAULT_RUN)
    parser.add_argument('--out-dir',type=Path,required=True)
    args=parser.parse_args(); torch.set_num_threads(1)
    run_experiment(args.run_dir,args.out_dir)


if __name__=='__main__': main()
