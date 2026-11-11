"""Report fixed channel controls and the original contact box's visibility."""
import argparse
import json
from pathlib import Path
import numpy as np


def report(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle,Circle
    out=Path(out);rows=json.loads((out/'report.json').read_text())
    if len(rows)!=4:raise ValueError('Four completed variants required')
    protocol=json.loads((out/'protocol.json').read_text());switch=protocol['switch_s']
    geometry=json.loads((out/'control/manifest.json').read_text())['geometry']
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for b in geometry['boxes']:
        axes[0].add_patch(Rectangle((b['x_min'],b['y_min']),b['x_max']-b['x_min'],b['y_max']-b['y_min'],color='.86'))
    for c in geometry['cylinders']:
        axes[0].add_patch(Circle((c['center_x'],c['center_y']),c['radius'],color='.86'))
    lines=['# Завершення маневру: контроль Pitch та Yaw','',
        'Однакова передісторія до 21 с. Учитель спостерігає від початку; після втручання кожна гілка має власну фізику та сенсори.', '',
        '| Варіант | Час, с | Контакт | Зазор після 21 с, см | Шлях після 21 с, м | Чисте переміщення після 21 с, м |',
        '|---|---:|---|---:|---:|---:|']
    positions=[]
    for row in rows:
        arm=row['arm'];m=row['metrics'];a=row['post_switch']
        with np.load(out/arm/'physical.npz') as trace:
            mask=trace['time']>=switch
            positions.append(trace['pos'][mask,:2])
            line,=axes[0].plot(*trace['pos'][mask,:2].T,label=arm)
            axes[0].scatter(*trace['pos'][-1,:2],s=20,color=line.get_color())
            axes[1].plot(trace['time'][mask],trace['clearance'][mask]*100,label=arm)
        lines.append(f'| {arm} | {m["duration_s"]:.3f} | {m["collision"] or "немає"} | '
            f'{100*a["min_clearance_m"]:.2f} | {a["path_m"]:.3f} | {a["net_displacement_m"]:.3f} |')
    all_xy=np.concatenate(positions)
    lo=all_xy.min(axis=0)-.35;hi=all_xy.max(axis=0)+.35
    axes[0].set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),xlabel='X (m)',ylabel='Y (m)',title='Local trajectories after identical history through 21 s')
    axes[0].set_aspect('equal')
    axes[1].set(xlabel='Time (s)',ylabel='Body clearance (cm)',title='Minimum body clearance after intervention')
    for ax in axes:ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.savefig(out/'comparison.png',dpi=150);plt.close(fig)

    fig,axes=plt.subplots(3,1,figsize=(12,9),sharex=True,layout='constrained')
    samples=[]
    with np.load(out/'control/trace.npz') as a,np.load(out/'control/decisions.npz') as d:
        v=json.loads((out/'control/contact_box_visibility.json').read_text())
        vt=np.array([r['time_s'] for r in v]);rays=np.array([r['visible_rays'] for r in v])
        axes[0].plot(a['time'],a['min_tof'],label='held minimum ToF (controller input)')
        axes[0].plot(vt,[r['box_gap_m'] for r in v],label='true body gap to contact box (diagnostic only)')
        axes[0].axhline(.95,color='.4',ls=':',label='teacher release distance')
        axes[0].set(ylabel='Distance (m)',title='Recorded control: forward range is not body clearance')
        axes[1].plot(vt,rays,label='rays intersecting box in isolation (no occlusion)')
        axes[1].set(ylabel='Ray count /64')
        for channel,name,axis in ((2,'Pitch',axes[2]),(3,'Yaw',axes[2])):
            axis.plot(d['time'],d['network'][:,channel]-1500,label='network '+name)
            axis.plot(d['time'],d['teacher'][:,channel]-1500,ls='--',label='teacher '+name)
        axes[2].set(xlabel='Time (s)',ylabel='Command offset (PWM)',xlim=(16,24.584))
        for t in (18.,21.,22.,23.,24.,24.56):
            i=int(np.argmin(abs(a['time']-t)));j=int(np.argmin(abs(d['time']-t)));k=int(np.argmin(abs(vt-t)))
            samples.append(dict(time_s=float(a['time'][i]),held_tof_m=float(a['min_tof'][i]),
                tof_age_s=float(a['time'][i]-a['tof_timestamp'][i]),
                contact_box=v[k],network=d['network'][j].tolist(),teacher=d['teacher'][j].tolist()))
    for ax in axes:ax.axvline(switch,color='.7',ls=':');ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.savefig(out/'release_visibility.png',dpi=150);plt.close(fig)
    (out/'control_snapshots.json').write_text(json.dumps(samples,indent=2),encoding='utf-8')
    lines+=['','Підстановка Pitch змінює силу гальмування та час відновлення; Yaw — знак, амплітуду й тривалість. Це не ізоляція лише одного таймера.',
        'Безконтактне завершення з малим переміщенням не є виходом із пастки. Контактний бокс у геометричному аудиті обрано за відомою аварією.',
        'Ізольований raycast не враховує перекриття іншими об’єктами; нуль променів означає, що обраний бокс не спостерігається навіть без перекриття.',
        'Карта та істинний зазор не подаються контролеру. Жодної заміни активної моделі не виконано.']
    (out/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path)
    report(p.parse_args().out)
