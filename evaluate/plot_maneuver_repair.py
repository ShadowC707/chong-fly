"""Plot the completed, fixed maneuver-repair experiment (no new simulation)."""
import argparse
import json
from pathlib import Path

import numpy as np


def plot(directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    directory=Path(directory)
    report=json.loads((directory/'report.json').read_text(encoding='utf-8'))
    protocol=json.loads((directory/'protocol.json').read_text(encoding='utf-8'))
    cases=sorted({r['case'] for r in report})
    if len(report)!=len(cases)*len(protocol['variants']):
        raise ValueError('Experiment is incomplete')
    fig,axes=plt.subplots(2,2,figsize=(13,12),layout='constrained')
    colors=dict(baseline='#2764b5',release_only='#e28926',flow_only='#379754',both='#9741aa')
    lines=['# Fixed maneuver-repair experiment','',
           '| Scene | Variant | Duration (s) | Collision | Min gap (cm) | Reverse / AUTO (%) |',
           '|---|---|---:|---|---:|---:|']
    for ax,case in zip(axes.flat,cases):
        manifest=json.loads((directory/(case+'_baseline')/'manifest.json').read_text())
        g=manifest['geometry']
        for b in g['boxes']:
            ax.add_patch(Rectangle((b['x_min'],b['y_min']),b['x_max']-b['x_min'],
                                   b['y_max']-b['y_min'],color='.83'))
        for c in g['cylinders']:
            ax.add_patch(Circle((c['center_x'],c['center_y']),c['radius'],color='.83'))
        for variant in protocol['variants']:
            item=next(r for r in report if r['case']==case and r['variant']==variant)
            m=item['metrics']
            with np.load(directory/(case+'_'+variant)/'trace.npz') as a:
                p=a['pos']
                ax.plot(*p[:,:2].T,color=colors[variant],lw=1.3,
                        label=f'{variant}: {m["duration_s"]:.2f} s')
                ax.scatter(*p[-1,:2],c=colors[variant],marker='x' if m['collision'] else 'o',s=35)
                ax.scatter(*p[0,:2],c='black',marker='s',s=18)
            lines.append(f'| {case} | {variant} | {m["duration_s"]:.3f} | {m["collision"] or "none"} | '
                         f'{100*m["min_clearance_m"]:.2f} | {100*m["reverse_fraction_auto"]:.2f} |')
        ax.set(xlim=(-4,4),ylim=(-4,4),xlabel='X (m)',ylabel='Y (m)',title=case+' | sphere radius 0.20 m')
        ax.set_aspect('equal');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Fixed 45 s trials; X = collision, circle = end of horizon\n'
                 'Identical weights, starts, geometry and sensor seeds; no scheduled STOP')
    fig.savefig(directory/'comparison.png',dpi=160)
    lines.extend(['','Reverse fractions have different exposure durations when trials crash early.',
                  'These four selected scenes are diagnostics, not a flight qualification set.'])
    (directory/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    plot(parser.parse_args().directory)
