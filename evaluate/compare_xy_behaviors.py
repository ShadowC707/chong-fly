"""Fixed 3-density x 2-behavior experiment, exact weights, STOP at 35/40 s."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from evaluate.watch_mvp import DEFAULT_RUN, run_view, ROOT
from generator.source_contract import file_sha256
from simulation.xy_coordinator import XYCoordinator
from simulation.xy_scenes import DENSITY_SCENES, DensityArena


def summarize(report):
    rows = report['trace']['physics']
    t = np.array([r['time_s'] for r in rows])
    p = np.array([r['position_m'] for r in rows])
    speed = np.linalg.norm([r['body_velocity_mps'][:2] for r in rows], axis=1)
    m = report['metrics']
    stop_at = report['stop_at_s']
    flight = p[t < stop_at]
    # Fixed 0.5 m bins; this is trajectory footprint, not mapped/sensed area.
    cells = np.unique(np.clip(np.floor((flight[:,:2]+4)/.5),0,15).astype(int), axis=0)
    quadrants = {(int(x>0),int(y>0)) for x,y in flight[:,:2] if abs(x)>.1 and abs(y)>.1}
    controls = report['trace']['control']
    events = [r['coordinator'].get('search_event') for r in controls]
    stop = t >= stop_at
    final_window = t >= 39.
    complete_stop = bool(m['horizon_completed'] and stop.any() and final_window.any())
    stop_metrics = None
    if stop.any():
        start = p[np.flatnonzero(stop)[0],:2]
        stop_metrics = dict(final_speed_mps=float(speed[-1]),
            speed_at_request_mps=float(speed[stop][0]),
            peak_speed_after_request_mps=float(speed[stop].max()),
            path_length_after_request_m=float(np.linalg.norm(np.diff(p[stop,:2],axis=0),axis=1).sum()),
            final_second_max_speed_mps=float(speed[final_window].max()) if complete_stop else None,
            final_second_mean_speed_mps=float(speed[final_window].mean()) if complete_stop else None,
            final_drift_from_stop_m=float(np.linalg.norm(p[-1,:2]-start)),
            max_displacement_from_stop_m=float(np.linalg.norm(p[stop,:2]-start,axis=1).max()),
            max_height_error_m=float(abs(p[stop,2]-report['target_height_m']).max()),
            speed_target_passed=bool(complete_stop and np.all(speed[final_window] < .03)),
            # Use only finished, contiguous final time window, not a lucky last sample.
            final_second_position_span_m=float(np.linalg.norm(np.ptp(p[final_window,:2],axis=0)))
                if complete_stop else None,
            evasion_control_steps=sum(r['coordinator'].get('state')=='STOP_EVASION' for r in controls))
    return dict(scene=report['scene'], mode='levy + STOP' if report['levy_explore'] else 'both + STOP',
        checkpoint_sha256=report['checkpoint_sha256'], complete=m['horizon_completed'],
        feasible=m['feasible'], crashed=m['crashed'], failure_reason=m['failure_reason'],
        duration_s=m['survival_time_s'], min_clearance_m=m['min_clearance_m'],
        visited_cells_05m=len(cells), trajectory_area_m2=len(cells)*.25,
        room_coverage_percent=100*len(cells)/256, quadrants=len(quadrants),
        search_starts=events.count('started'), search_completions=events.count('completed'),
        search_preemptions=(sum(r['coordinator']['search_cancelled_phase']=='pulse' for r in controls)
            if controls and all('search_cancelled_phase' in r['coordinator'] for r in controls) else None),
        search_wait_resets=(sum(r['coordinator']['search_cancelled_phase']=='waiting' for r in controls)
            if controls and all('search_cancelled_phase' in r['coordinator'] for r in controls) else None),
        search_cancellations_legacy=sum(e in ('preempted','preempted_by_stop') for e in events),
        stop=stop_metrics)


def plot_grid(reports, rows, output, *, columns=2):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle
    nrows=len(reports)//columns
    fig, axes = plt.subplots(nrows,columns,figsize=(6*columns,5*nrows+1),layout='constrained',squeeze=False)
    for ax,report,row in zip(axes.flat,reports,rows):
        for b in report['geometry']['boxes']:
            ax.add_patch(Rectangle((b['x_min'],b['y_min']), b['x_max']-b['x_min'],
                                  b['y_max']-b['y_min'], color='#bec8d2'))
        for c in report['geometry']['cylinders']:
            ax.add_patch(Circle((c['center_x'],c['center_y']),c['radius'],color='#caba9f'))
        physics = report['trace']['physics']
        p = np.array([r['position_m'] for r in physics])
        t = np.array([r['time_s'] for r in physics])
        before = t < report['stop_at_s']
        ax.plot(p[before,0],p[before,1],color='#2364aa',lw=1.4,label='Cruise / avoidance')
        ax.plot(p[~before,0],p[~before,1],color='#d04b24',lw=2,label='STOP requested')
        ax.scatter(*p[0,:2],color='green',s=35,zorder=5,label='Start')
        ax.scatter(*p[-1,:2],color='red',marker='X',s=55,zorder=5,label='End')
        if row.get('min_clearance_event'):
            event=row['min_clearance_event']
            ax.scatter(*event['position_m'][:2],color='#89278d',marker='D',s=40,zorder=7,
                       label=f"Min clearance @ {event['time_s']:.3f}s")
            ax.add_patch(Rectangle((-2.8,-2.8),5.6,5.6,fill=False,ls=':',edgecolor='#777'))
        starts = [r['time_s'] for r in report['trace']['control']
                  if r['coordinator'].get('search_event')=='started']
        if starts:
            ax.scatter(np.interp(starts,t,p[:,0]),np.interp(starts,t,p[:,1]),marker='*',
                       color='#df9e19',edgecolors='#8d5900',s=100,zorder=6,label='Search pulse starts')
        status = 'complete' if row['feasible'] else 'FAILED'
        stop = row['stop']
        speed = f"{stop['final_speed_mps']:.4f} m/s" if stop else 'not reached'
        interior=f" | interior {100*row['interior_fraction_before_stop']:.1f}%" if 'interior_fraction_before_stop' in row else ''
        ax.set(title=f"{row['scene']} | {row['mode']} | {status}\n"
                     f"clearance {row['min_clearance_m']:.3f} m | cells {row['visited_cells_05m']}/256"
                     f"{interior}\nfinal speed {speed}", xlabel='World X (m)',ylabel='World Y (m)',
               xlim=(-4,4),ylim=(-4,4),aspect='equal')
        ax.grid(alpha=.2); ax.legend(loc='best',fontsize=7)
    fig.suptitle('Fixed k306 checkpoint | 40 s standalone simulation | STOP at 35 s\n'
                 'Same scene and seed per pair; search modifies yaw only; no OU input perturbations',fontsize=13)
    fig.savefig(output,dpi=150); plt.close(fig)
    fig, axes = plt.subplots(nrows,1,figsize=(11,3*nrows+1),layout='constrained',squeeze=False)
    axes=axes[:,0]
    for i,ax in enumerate(axes):
        for report in reports[columns*i:columns*i+columns]:
            physics=report['trace']['physics']
            t=np.array([r['time_s'] for r in physics])
            speed=np.linalg.norm([r['body_velocity_mps'][:2] for r in physics],axis=1)
            ax.plot(t,speed,label=report.get('experiment_variant',
                                          'levy + STOP' if report['levy_explore'] else 'both + STOP'))
        ax.axhline(.03,color='black',ls='--',label='0.03 m/s target')
        ax.axvline(35,color='gray',ls=':'); ax.set(xlim=(34,40),ylabel='XY speed (m/s)',title=DENSITY_SCENES[i])
        ax.grid(alpha=.2); ax.legend()
    axes[-1].set_xlabel('Time (s)')
    fig.savefig(output.with_name('stop_speed.png'),dpi=150); plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=DEFAULT_RUN)
    parser.add_argument('--out-dir',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=20261021)
    args=parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError('Use a new experiment directory')
    torch.set_num_threads(1)
    specs=[]
    for name in DENSITY_SCENES:
        env=DensityArena(name)
        try:
            specs.append(env.specification())
        finally:
            env.close()
    plan=dict(version='xy-behaviors-density-v1', seed=args.seed, seconds=40.,stop_at_s=35.,
        checkpoint_sha256=file_sha256(args.run_dir/'checkpoint.pt'), scenes=specs,
        contracts=[XYCoordinator(anti_rollback=True,turn_latch=True,levy_explore=levy,
                                 stop_hover=True,seed=args.seed).contract() for levy in (False,True)],
        metrics=dict(coverage='Visited 0.5 m XY cells / 256 whole-room cells before STOP; not sensor mapping.',
            quadrants='Before STOP; abs(x)>0.1 and abs(y)>0.1 to avoid axis jitter.',
            stop_pass='Complete run and every physics sample in [39,40] s has XY speed < 0.03 m/s.'),
        limitations=['One fixed seed, three nested geometries; no success-rate estimate.',
            'Surface gaps >=1.3 m; subtract drone diameter for centerline clearance.',
            'STOP uses optical-flow damping, not a position reference; HOVER label is sensor-estimated.',
            'Forward ToF does not observe all sides or guarantee safe emergency escape.'],
        code_sha256={name:file_sha256(ROOT/name) for name in ('evaluate/compare_xy_behaviors.py',
            'simulation/xy_coordinator.py','simulation/xy_scenes.py','optimizer/rollout.py')})
    args.out_dir.mkdir(parents=True)
    (args.out_dir/'plan.json').write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8')
    reports, rows=[],[]
    for name in DENSITY_SCENES:
        for levy in (False,True):
            report=run_view(args.run_dir,scene=name,steps=2000,seed=args.seed,coordinator_mode='both',
                levy_explore=levy,stop_at_s=35.,plot_out=args.out_dir/f'{name}_{"levy" if levy else "control"}.png')
            reports.append(report); rows.append(summarize(report))
            (args.out_dir/'comparison.json').write_text(json.dumps(dict(plan=plan,runs=rows),indent=2)+'\n',encoding='utf-8')
            print(json.dumps(rows[-1]),flush=True)
    plot_grid(reports,rows,args.out_dir/'comparison.png')
    lines=['# XY behavior density experiment','',
        '| Scene | Mode | Crash | Clearance m | Area % | Quadrants | Search pulses | Final speed m/s | Last-second max m/s | STOP pass |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---|']
    for r in rows:
        s=r['stop'] or {}
        def fmt(value):
            return 'N/A' if value is None else f'{value:.4f}'
        lines.append(f"| {r['scene']} | {r['mode']} | {r['crashed']} | {r['min_clearance_m']:.3f} | "
            f"{r['room_coverage_percent']:.2f} | {r['quadrants']}/4 | {r['search_starts']} | "
            f"{fmt(s.get('final_speed_mps'))} | {fmt(s.get('final_second_max_speed_mps'))} | {s.get('speed_target_passed',False)} |")
    lines+=['',*plan['limitations']]
    (args.out_dir/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    main()
