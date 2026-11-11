"""Report encoding correctness separately from trained flight usefulness."""
import argparse
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def report(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle,Rectangle
    from edge.runtime import NumpyPolicy
    out=Path(out)
    flights=json.loads((out/'flights.json').read_text())
    assert len(flights)==8
    comparison=json.loads((out/'comparison.json').read_text())
    cases=list(dict.fromkeys(r['case'] for r in flights))
    fig,axes=plt.subplots(2,2,figsize=(12,11),layout='constrained')
    lines=['# Paired encoding experiment','',
           '| Room | Encoding | Distance (m) | Mean AUTO speed (m/s) | Moving AUTO (s) | Contact |',
           '|---|---|---:|---:|---:|---|']
    quiet={}
    for name,path in [('baseline',ROOT/'output/balanced_maneuver_v1/before/numpy'),
                      ('old_encoding',out/'old_encoding/numpy'),('release_encoding',out/'release_encoding/numpy')]:
        p=NumpyPolicy(path)
        for _ in range(100): y=p.step_np(np.zeros(2,np.float32),np.ones(64,np.float32),dt=.02)
        quiet[name]=y.tolist()
    for ax,case in zip(axes.flat,cases):
        g=json.loads((out/'release_encoding/flights'/case/'manifest.json').read_text())['geometry']
        for b in g['boxes']:
            ax.add_patch(Rectangle((b['x_min'],b['y_min']),b['x_max']-b['x_min'],b['y_max']-b['y_min'],color='.86'))
        for c in g['cylinders']:
            ax.add_patch(Circle((c['center_x'],c['center_y']),c['radius'],color='.86'))
        for label,path,color,style in [
            ('original network',ROOT/'output/practical_maneuver_v1'/(case+'_before')/'physical.npz','#2764b5','-'),
            ('teacher 1530',ROOT/'output/teacher_native1530_v1'/case/'physical.npz','#339258','-'),
            ('trained old encoding',out/'old_encoding/flights'/case/'physical.npz','#e59d20','-'),
            ('trained release encoding',out/'release_encoding/flights'/case/'physical.npz','#bd3d45','--')]:
            with np.load(path) as a:
                ax.plot(*a['pos'][:,:2].T,style,color=color,lw=1.4,label=label)
                ax.scatter(*a['pos'][-1,:2],c=color,s=22)
        ax.set(xlim=(-4,4),ylim=(-4,4),xlabel='X (m)',ylabel='Y (m)',title=case)
        ax.set_aspect('equal');ax.grid(alpha=.2);ax.legend(fontsize=7)
        for r in (v for v in flights if v['case']==case):
            m=r['metrics']
            lines.append(f'| {case} | {r["arm"]} | {m["distance_m"]:.3f} | {m["mean_auto_speed_mps"]:.5f} | '
                         f'{m["moving_auto_s"]:.3f} | {m["collision"] or "none"} |')
    fig.suptitle('Encoding experiment: both trained models remain near the start\n'
                 'No contacts is not successful navigation; original/teacher trajectories are saved references')
    fig.savefig(out/'comparison.png',dpi=150);plt.close(fig)
    assessment=dict(quiet_pwm=quiet,temporal=comparison,
        no_collision_but_no_navigation=True,all_trials_below_motion_threshold=all(r['metrics']['mean_auto_speed_mps']<.03 for r in flights),
        scope='Observed motion failure; 0.03 m/s is the existing moving/stationary metric, not a new flight admission gate',
        promotion=False,next_step='Preserve cruise behavior in the training objective before interpreting collision performance')
    (out/'assessment.json').write_text(json.dumps(assessment,indent=2,allow_nan=False),encoding='utf-8')
    lines+=['','Both trained arms fail the motion objective. Do not interpret 8/8 contact-free horizons as successful avoidance.',
            'New encoding separates the teacher release inputs; its learned output is still incorrect.',
            'Old packages and active weights remain unchanged; this is not a new flight release.']
    (out/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Quiet PWM:',quiet)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path)
    report(p.parse_args().out)
