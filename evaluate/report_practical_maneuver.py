"""Summarize the saved eight-trial comparison, without rerunning simulation."""
import argparse
import json
from pathlib import Path

import numpy as np


def window_metrics(a, end_s):
    """Integrate left-endpoint physics samples on a shared wall-clock horizon."""
    dt = np.maximum(0, np.minimum(np.r_[a['time'][1:], end_s], end_s) - a['time'])
    auto_dt = dt * a['auto']
    speed = np.linalg.norm(a['velocity'][:, :2], axis=1)
    forward = (a['velocity'][:, 0] * np.cos(a['yaw'])
               + a['velocity'][:, 1] * np.sin(a['yaw']))
    return dict(end_s=float(end_s), auto_s=float(auto_dt.sum()),
                distance_m=float((speed * auto_dt).sum()),
                reverse_s=float(auto_dt[forward < -.03].sum()),
                below_10cm_s=float(dt[a['clearance'] < .1].sum()))


def report(directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    directory = Path(directory)
    results = json.loads((directory / 'report.json').read_text(encoding='utf-8'))
    cases = json.loads((directory / 'protocol.json').read_text())['cases']
    assert len(results) == 8
    fig, axes = plt.subplots(2, 2, figsize=(12, 11), layout='constrained')
    analysis = dict(common_window={}, contact_context={})
    lines = ['# Fixed closed-loop comparison', '',
             '| Room | Driver | End (s) | Contact | Mean AUTO speed (m/s) | Reverse / AUTO (%) |',
             '|---|---|---:|---|---:|---:|']
    for ax, case in zip(axes.flat, cases):
        pair = [next(r for r in results if r['case'] == case and r['driver'] == name)
                for name in ('before', 'candidate')]
        end = min(r['metrics']['duration_s'] for r in pair)
        analysis['common_window'][case] = {}
        g = json.loads((directory / (case + '_before') / 'manifest.json').read_text())['geometry']
        for b in g['boxes']:
            ax.add_patch(Rectangle((b['x_min'], b['y_min']), b['x_max'] - b['x_min'],
                                   b['y_max'] - b['y_min'], color='.83'))
        for c in g['cylinders']:
            ax.add_patch(Circle((c['center_x'], c['center_y']), c['radius'], color='.83'))
        for r, color in zip(pair, ('#2764b5', '#d45123')):
            name, m = r['driver'], r['metrics']
            path = directory / (case + '_' + name)
            with np.load(path / 'physical.npz') as a:
                analysis['common_window'][case][name] = window_metrics(a, end)
                pos = a['pos']
                ax.plot(*pos[:, :2].T, color=color, lw=1.3,
                        label=f'{name}: {m["duration_s"]:.2f} s')
                ax.scatter(*pos[-1, :2], c=color, marker='x' if m['collision'] else 'o', s=45)
                ax.scatter(*pos[0, :2], c='black', marker='s', s=18)
                if m['collision']:
                    with np.load(path / 'decisions.npz') as d, np.load(path / 'trace.npz') as tr:
                        # Last policy call and corresponding physical/command samples, not the terminal row.
                        t = float(d['time'][-1])
                        i = np.searchsorted(a['time'], t, side='right') - 1
                        j = np.searchsorted(tr['time'], t, side='right') - 1
                        analysis['contact_context'][case + '_' + name] = dict(
                            time_s=t, collision_s=m['duration_s'], phase=str(d['phase'][-1]),
                            min_tof_m=float(d['X'][-1, 2:66].min() * 3.5),
                            forward_mps=float(a['velocity'][i, 0] * np.cos(a['yaw'][i])
                                              + a['velocity'][i, 1] * np.sin(a['yaw'][i])),
                            before_pitch=float(d['before'][-1, 2]),
                            candidate_pitch=float(d['candidate'][-1, 2]),
                            teacher_pitch=float(d['teacher'][-1, 2]),
                            coordinated_pitch=float(tr['coordinated'][j, 2]),
                            applied_pitch=float(tr['applied'][j, 2]))
            lines.append(f'| {case} | {name} | {m["duration_s"]:.3f} | {m["collision"] or "none"} | '
                         f'{m["mean_auto_speed_mps"]:.3f} | {100*m["reverse_fraction_auto"]:.1f} |')
        ax.set(xlim=(-4, 4), ylim=(-4, 4), xlabel='X (m)', ylabel='Y (m)', title=case)
        ax.set_aspect('equal'); ax.grid(alpha=.2); ax.legend(fontsize=9)
    fig.suptitle('Baseline vs trained candidate | same runtime, starts and geometry\n'
                 'Body radius 0.20 m; X = contact; circle = completed 45 s; square = start')
    fig.savefig(directory / 'comparison.png', dpi=150)
    plt.close(fig)
    lines += ['', 'Different speeds and durations: longer survival is not proof of better decisions.',
              'Common-window metrics use the same wall-clock horizon, with AUTO exposure reported separately.',
              'Passive teacher/model observations are descriptive; they do not establish collision causality.']
    (directory / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (directory / 'analysis.json').write_text(json.dumps(analysis, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    report(parser.parse_args().directory)
