"""Render the completed corrective-data comparison without rerunning training."""
import argparse
import json
from pathlib import Path
import numpy as np


def render(directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle,Circle
    d=Path(directory)
    validation=json.loads((d/'validation.json').read_text())
    assert validation['completed_physical_trials']==12
    scores=json.loads((d/'comparison.json').read_text())
    flights=json.loads((d/'physical_comparison.json').read_text())
    arms=('before','control','mixed'); colors=('#526477','#d18b24','#267cba')
    metrics=(
        ('Pitch MAE, new corrective episodes',lambda r:r['raw']['new_corrections']['all']['mae_roll_pitch_yaw'][1]),
        ('Pitch MAE, old external episodes',lambda r:r['raw']['old_external_seen']['all']['mae_roll_pitch_yaw'][1]),
        ('Cruise during post-threat teacher braking (198 frames)',lambda r:r['raw']['new_corrections']['teacher_braking_after_clear']['positive_pitch_frames']),
        ('Minimum narrow-threat yaw amplitude (PWM)',lambda r:r['temporal']['narrow_min_signed_yaw_pwm']))
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for ax,(label,fn) in zip(axes.flat,metrics):
        values=[fn(scores[a]) for a in arms]; bars=ax.bar(arms,values,color=colors)
        ax.bar_label(bars,fmt='%.2f');ax.set_title(label,fontsize=11)
        ax.set_ylim(0,max(values)*1.2);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Data intervention / 10 epochs / frozen core\nAmplitude is a regression check; other panels are imitation errors')
    fig.savefig(d/'metrics.png',dpi=150);plt.close(fig)
    cases=list(flights['before'])
    fig,axes=plt.subplots(4,3,figsize=(13,16),layout='constrained')
    for i,case in enumerate(cases):
        for j,arm in enumerate(arms):
            ax=axes[i,j];g=json.loads((d/arm/case/'manifest.json').read_text())['geometry']
            for b in g['boxes']:
                ax.add_patch(Rectangle((b['x_min'],b['y_min']),b['x_max']-b['x_min'],b['y_max']-b['y_min'],color='.8'))
            for c in g['cylinders']:
                ax.add_patch(Circle((c['center_x'],c['center_y']),c['radius'],color='.8'))
            with np.load(d/arm/case/'trace.npz') as trace:p=trace['pos']
            m=flights[arm][case]
            ax.plot(p[:,0],p[:,1],color=colors[j],lw=1.3)
            ax.scatter(*p[0,:2],marker='s',color='green',s=20)
            ax.scatter(*p[-1,:2],marker='x' if m['collision'] else 'o',color='red' if m['collision'] else colors[j],s=25)
            room=g['room'];ax.set(xlim=(room['x_min'],room['x_max']),ylim=(room['y_min'],room['y_max']),
                xlabel='X (m)',ylabel='Y (m)',title=f"{case} | {arm}\n{m['duration_s']:.3f}s / gap {100*m['min_clearance_m']:.2f}cm / AUTO {m['auto_s']:.2f}s")
            ax.set_aspect('equal');ax.grid(alpha=.2)
    fig.suptitle('Unchanged coordinator / starts / sensor seeds / 0.20m body / 45s horizon\nSquare: start; cross: collision; dot: horizon end')
    fig.savefig(d/'trajectories.png',dpi=140);plt.close(fig)
    lines=['# Парне донавчання на нових послідовностях','',
        '| Показник | До | Старі дані | Старі + коригувальні |','|---|---:|---:|---:|']
    for label,fn in metrics:
        lines.append('| '+label+' | '+' | '.join(f'{fn(scores[a]):.2f}' for a in arms)+' |')
    lines+=['','| Сцена | До | Старі дані | Старі + коригувальні |','|---|---|---|---|']
    for case in cases:
        cells=[]
        for a in arms:
            m=flights[a][case]
            cells.append(f"контакт {m['duration_s']:.3f} с" if m['collision'] else f"45 с; зазор {100*m['min_clearance_m']:.2f} см")
        lines.append('| '+case+' | '+' | '.join(cells)+' |')
    lines+=['','Повні метрики та час AUTO: physical_comparison.json. Один seed; відомі складні сцени. Робочий пакет не змінено.']
    (d/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
    render(p.parse_args().directory)
