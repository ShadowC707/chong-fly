"""Fixed 2x4, 40-second XY experiment; no tuning or training during the run.

python -m evaluate.compare_xy_coordinator --out-dir data/diagnostics/xy_factorial_v1
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from evaluate.watch_mvp import DEFAULT_RUN, COORDINATOR_MODES, run_view
from generator.source_contract import file_sha256
from simulation.xy_coordinator import XYCoordinator


def summarize(report):
    physics, controls = report['trace']['physics'], report['trace']['control']
    positions = np.array([r['position_m'] for r in physics])
    times = np.array([r['time_s'] for r in physics])
    velocities = np.array([r['body_velocity_mps'] for r in physics])
    dt = np.diff(times)
    backwards = np.maximum(0., -velocities[1:, 0])*dt
    largest = current = 0.
    for distance in backwards:
        current = current + distance if distance > 0 else 0.
        largest = max(largest, current)
    corner = (abs(positions[:, 0]) > 2.5) & (abs(positions[:, 1]) > 2.5)
    active, outside, exits, first_entry = False, 0., 0, None
    for i in range(1, len(times)):
        if corner[i]:
            active, outside = True, 0.
            if first_entry is None:
                first_entry = float(times[i])
        elif active:
            outside += dt[i-1]
            if outside + 1e-9 >= 1.:
                exits += 1
                active, outside = False, 0.
    raw = np.array([r['network_pwm'] for r in controls]).reshape(-1, 4)
    applied = np.array([r['applied_pwm'] for r in controls]).reshape(-1, 4)
    m = report['metrics']
    return dict(scene=report['scene'], mode=report['coordinator_mode'],
        completed=m['horizon_completed'], feasible=m['feasible'], seconds=m['survival_time_s'],
        crashed=m['crashed'], failure_reason=m['failure_reason'], min_clearance_m=m['min_clearance_m'],
        corner_time_s=float(np.sum(dt*corner[1:])), first_corner_entry_s=first_entry,
        corner_exits_1s=exits, ends_in_corner=bool(corner[-1]),
        backward_distance_m=float(backwards.sum()), max_continuous_backward_m=float(largest),
        max_actual_height_error_m=float(abs(positions[:, 2]-1.).max()),
        yaw_bursts=m['yaw_bursts']['count'],
        yaw_envelope_error_pwm=float(np.max(abs(abs(raw[:, 3]-1500.)-abs(applied[:, 3]-1500.)))) if len(raw) else None,
        neutral_yaw_violations=int(np.sum((raw[:, 3] == 1500.) & (applied[:, 3] != 1500.))) if len(raw) else 0,
        intervention_steps=m.get('coordinator', {}).get('intervention_steps', 0),
        intervention_reasons=m.get('coordinator', {}).get('reason_counts', {}),
        checkpoint_sha256=report['checkpoint_sha256'])


def plot_comparison(reports, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle
    fig, axes = plt.subplots(2, 4, figsize=(16, 11), layout='constrained')
    fig.get_layout_engine().set(h_pad=.14, hspace=.12)
    for ax, report in zip(axes.flat, reports):
        g = report['geometry']
        for b in g['boxes']:
            ax.add_patch(Rectangle((b['x_min'], b['y_min']), b['x_max']-b['x_min'],
                                  b['y_max']-b['y_min'], facecolor='#ccd1d9'))
        for c in g['cylinders']:
            ax.add_patch(Circle((c['center_x'], c['center_y']), c['radius'], color='#d8c5ac'))
        p = np.array([r['position_m'] for r in report['trace']['physics']])
        ax.plot(p[:, 0], p[:, 1], lw=1.3)
        ax.scatter(*p[0, :2], color='green', s=25)
        ax.scatter(*p[-1, :2], color='red', marker='x', s=40)
        for x in (-4, 2.5):
            for y in (-4, 2.5):
                ax.add_patch(Rectangle((x,y), 1.5, 1.5, color='orange', alpha=.07))
        row = summarize(report)
        ax.set_title(f"{row['scene']} | {row['mode']}\n"
                     f"{'complete' if row['feasible'] else 'FAILED'} {row['seconds']:.1f}s; "
                     f"clearance {row['min_clearance_m']:.2f}m\ncorner residence {row['corner_time_s']:.1f}s", fontsize=9)
        ax.set(xlim=(-4.1,4.1), ylim=(-4.1,4.1), xlabel='X (m)', ylabel='Y (m)')
        ax.set_aspect('equal'); ax.grid(alpha=.2)
    fig.suptitle('k306 tau x0.25 | XY factorial experiment | same seed, no OU, no retraining\n'
                 'Shading: corner measurement regions; green: start; red: final position', fontsize=12)
    try:
        fig.savefig(output, dpi=150)
    finally:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261009)
    parser.add_argument('--scene-pair', choices=('arena', 'corner'), default='arena')
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    scenes = ['arena', 'arena_mirror'] if args.scene_pair == 'arena' else ['corner_approach', 'corner_approach_mirror']
    plan = dict(version='xy-factorial-v1', scenes=scenes,
        modes=list(COORDINATOR_MODES), seed=args.seed, control_steps=2000, duration_s=40.,
        run_dir=str(args.run_dir.resolve()), checkpoint_sha256=file_sha256(args.run_dir/'checkpoint.pt'),
        contracts={mode: XYCoordinator(anti_rollback=mode in ('anti_rollback','both'),
            turn_latch=mode in ('turn_latch','both')).contract() for mode in COORDINATOR_MODES},
        mirror=('Geometry reflected in Y; same initial state and seed.' if args.scene_pair == 'arena' else
                'Geometry, initial position, yaw and velocity reflected in Y; fresh recurrent state, same seed.')
                + ' No mirroring of policy inputs or outputs.',
        metric_definitions=dict(corner='abs(x)>2.5 m AND abs(y)>2.5 m in the fixed [-4,4] arena',
            corner_exit='After entry, remain outside all corner regions for >=1 second',
            backward_distance='Integral max(0,-body_forward_velocity) at physics ticks; diagnostic ground truth only'),
        limitations=['One seed, two geometries; no reliability or hardware safety claim.',
            'The central cylinder is symmetric: mirroring other obstacles does not force opposite first turns.',
            'Coordinator preserves yaw magnitude for the same input, not future closed-loop trajectories.',
            'Leaving a corner region does not prove successful exploration or completion of a turn.'])
    (args.out_dir/'plan.json').write_text(json.dumps(plan, indent=2)+'\n', encoding='utf-8')
    reports, rows = [], []
    for scene in plan['scenes']:
        for mode in COORDINATOR_MODES:
            report = run_view(args.run_dir, scene=scene, steps=2000, seed=args.seed,
                coordinator_mode=mode, plot_out=args.out_dir/f'{scene}_{mode}.png')
            reports.append(report)
            rows.append(summarize(report))
            (args.out_dir/'comparison.json').write_text(json.dumps(dict(plan=plan, runs=rows), indent=2)+'\n', encoding='utf-8')
            print(json.dumps(rows[-1]), flush=True)
    plot_comparison(reports, args.out_dir/'comparison.png')
    lines = ['# XY factorial experiment', '', '| Scene | Mode | Complete | Clearance m | Corner s | Exits | Backward m | Max backward bout m | Interventions |',
             '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['scene']} | {r['mode']} | {r['feasible']} | {r['min_clearance_m']:.3f} | "
                     f"{r['corner_time_s']:.2f} | {r['corner_exits_1s']} | {r['backward_distance_m']:.3f} | "
                     f"{r['max_continuous_backward_m']:.3f} | {r['intervention_steps']} |")
    lines += ['', *plan['limitations']]
    (args.out_dir/'comparison.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
