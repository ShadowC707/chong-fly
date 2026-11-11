"""Compare the one-shot Pitch repair with saved parent and baseline flights."""
import argparse
import json
from pathlib import Path
import numpy as np
from evaluate.report_practical_maneuver import window_metrics

ROOT = Path(__file__).resolve().parents[1]


def report(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle
    out = Path(out)
    prior = ROOT / 'output/practical_maneuver_v1'
    old = json.loads((prior / 'report.json').read_text())
    new = json.loads((out / 'flights/report.json').read_text())
    assert len(new) == 4
    fig, axes = plt.subplots(2, 2, figsize=(12, 11), layout='constrained')
    rows=[]; common={}
    for ax, item in zip(axes.flat, new):
        case=item['case']; common[case]={}
        parent=next(r for r in old if r['case']==case and r['driver']=='candidate')
        baseline=next(r for r in old if r['case']==case and r['driver']=='before')
        variants=[('baseline', baseline, prior/(case+'_before'), '#2764b5'),
                  ('trained parent', parent, prior/(case+'_candidate'), '#d45123'),
                  ('Pitch repair', item, out/'flights'/(case+'_candidate'), '#339258')]
        g=json.loads((variants[0][2]/'manifest.json').read_text())['geometry']
        for b in g['boxes']:
            ax.add_patch(Rectangle((b['x_min'],b['y_min']),b['x_max']-b['x_min'],b['y_max']-b['y_min'],color='.85'))
        for c in g['cylinders']:
            ax.add_patch(Circle((c['center_x'],c['center_y']),c['radius'],color='.85'))
        end=min(r['metrics']['duration_s'] for _,r,_,_ in variants)
        for label,r,path,color in variants:
            m=r['metrics']
            with np.load(path/'physical.npz') as a:
                common[case][label]=window_metrics(a,end)
                pos=a['pos']
                ax.plot(*pos[:,:2].T, color=color, lw=1.3,label=f'{label}: {m["duration_s"]:.2f} s')
                ax.scatter(*pos[-1,:2],c=color,marker='x' if m['collision'] else 'o',s=40)
                ax.scatter(*pos[0,:2],c='black',marker='s',s=15)
            rows.append(dict(case=case,variant=label,**m))
        split='used for fit' if case in ('068fe108','89d81a9f') else 'excluded from fit; previously inspected'
        ax.set(xlim=(-4,4),ylim=(-4,4),xlabel='X (m)',ylabel='Y (m)',title=case+' | '+split)
        ax.set_aspect('equal'); ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle('One four-weight Pitch repair | same runtime and geometry\n'
                 'Sphere radius 0.20 m; X = contact; circle = completed 45 s')
    fig.savefig(out/'comparison.png',dpi=150);plt.close(fig)
    (out/'flight_comparison.json').write_text(json.dumps(dict(rows=rows,common_window=common),indent=2),encoding='utf-8')
    lines=['# One-shot Pitch readout repair','',
           '| Room | Variant | End (s) | Contact | AUTO speed (m/s) | Reverse / AUTO (%) |',
           '|---|---|---:|---|---:|---:|']
    for r in rows:
        lines.append(f'| {r["case"]} | {r["variant"]} | {r["duration_s"]:.3f} | {r["collision"] or "none"} | '
                     f'{r["mean_auto_speed_mps"]:.3f} | {100*r["reverse_fraction_auto"]:.1f} |')
    lines += ['','No promotion: repair and parent both contact in all four rooms.',
              'Different speeds, AUTO admission times and paths confound time-to-contact comparisons.',
              'Only two rooms contributed fitting targets; the other two were previously inspected, not blind tests.']
    (out/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path)
    report(p.parse_args().out)
