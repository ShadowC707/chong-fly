"""Render the paired replay findings and the bounded physical intervention."""
import argparse
import json
from pathlib import Path
import numpy as np


def report(physical):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle,Circle
    physical=Path(physical)
    rows=json.loads((physical/'report.json').read_text())
    if len(rows)!=4:raise ValueError('Wait for all four fixed interventions')
    m=json.loads((physical/'control/manifest.json').read_text())['geometry']
    protocol=json.loads((physical/'protocol.json').read_text())
    references=[Path(p) for p in protocol['input_sha256'] if Path(p).name=='physical.npz']
    if len(references)!=1:raise ValueError('Expected one original physical trace')
    with np.load(references[0]) as a:
        dt=np.diff(a['time'],append=a['time'][-1])
        fwd=a['velocity'][:,0]*np.cos(a['yaw'])+a['velocity'][:,1]*np.sin(a['yaw'])
        prefix_reverse=float(dt[(a['time']<34.)&a['auto']&(fwd<-.03)].sum())
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for b in m['boxes']:
        axes[0].add_patch(Rectangle((b['x_min'],b['y_min']),b['x_max']-b['x_min'],b['y_max']-b['y_min'],color='.85'))
    for c in m['cylinders']:
        axes[0].add_patch(Circle((c['center_x'],c['center_y']),c['radius'],color='.85'))
    lines=['# Контроль завершення маневру: один випадок, чотири втручання','',
        'Усі варіанти мають точно однакову передісторію до t=34 с. Це діагностика, не нова прошивка.', '',
        '| Варіант | Час завершення, с | Контакт | Мін. зазор після 34 с, см | Рух назад після 34 с, с |',
        '|---|---:|---|---:|---:|']
    for row in rows:
        arm=row['arm']
        with np.load(physical/arm/'physical.npz') as a:
            mask=a['time']>=34.
            line,=axes[0].plot(*a['pos'][mask,:2].T,label=arm)
            axes[0].scatter(*a['pos'][-1,:2],color=line.get_color(),s=20)
            axes[1].plot(a['time'][mask],a['clearance'][mask]*100,label=arm)
            minimum=float(a['clearance'][mask].min())
            xy=a['pos'][mask,:2]
            distance=float(np.linalg.norm(np.diff(xy,axis=0),axis=1).sum())
            displacement=float(np.linalg.norm(xy[-1]-xy[0]))
        reverse=max(0.,row['metrics']['reverse_s']-prefix_reverse)
        row['post_switch']=dict(min_clearance_m=minimum,reverse_s=reverse,
            distance_m=distance,net_displacement_m=displacement)
        lines.append(f'| {arm} | {row["metrics"]["duration_s"]:.3f} | {row["metrics"]["collision"] or "немає"} | {minimum*100:.2f} | {reverse:.3f} |')
    axes[0].set(xlim=(.5,4),ylim=(-4,-1.5),xlabel='X (m)',ylabel='Y (m)',title='Trajectories after the common switch at 34 s')
    axes[0].set_aspect('equal')
    axes[1].set(xlabel='Time (s)',ylabel='Body clearance (cm)',title='Clearance after intervention')
    for ax in axes:ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.savefig(physical/'comparison.png',dpi=150);plt.close(fig)
    (physical/'assessment.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    lines+=['','Teacher_yaw змінює лише Yaw перед незмінним turn_latch. Raw_pitch вимикає лише anti-rollback.',
        'Після втручання траєкторії й сенсорні входи розходяться. Висновки обмежені цим стартовим станом і горизонтом.',
        'Інтервал обрано після перегляду аварії. Безконтактне завершення не доводить безпеки на інших сценах.',
        'Для teacher_yaw після втручання: '+str(next(r['post_switch'] for r in rows if r['arm']=='teacher_yaw')),
        'Мале чисте переміщення означає, що уникнення контакту ще не є виходом із кута.']
    (physical/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('physical',type=Path)
    a=p.parse_args();report(a.physical)
