"""Report both success and regression in the fixed magnitude-objective pair."""
import argparse
import json
from pathlib import Path
import numpy as np


def report(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle,Circle
    out=Path(out)
    rows=json.loads((out/'comparison.json').read_text())
    if set(rows)!={'control','magnitude'}:raise ValueError('Both arms must finish')
    initial=json.loads((out/'initial.json').read_text())
    protocol=json.loads((out/'protocol.json').read_text());parent=Path(protocol['parent'])
    lines=['# Навчання амплітуди двобічного повороту','',
        'Один початковий кандидат, два запуски по 10 додаткових епох; коефіцієнт додаткової втрати 0 / 1.', '',
        '| Варіант | Двобічні навчальні кадри зі слабким Yaw /428 | Напрямок /12 | Згасання /12 | Вузькі /8 | Широкі /4 | Рух уперед, м/с | Допуск до кімнат |',
        '|---|---:|---:|---:|---:|---:|---:|---|']
    for arm,r in rows.items():
        d=r['diagnostic']['training']['slow_bilateral'];t=r['temporal']
        speed=r['clear']['mean_forward_speed_mps']
        lines.append(f'| {arm} | {d["weak_network_frames"]} | {t["direction"]} | {t["recovery"]} | '
            f'{t["narrow_amplitude"]} | {t["broad_joint"]} | {speed:.3f} | {r["room_screen"]["passed"]} |')
    lines+=['','| Варіант | Повтор | Слабкий Yaw /119 | Знак проти вчителя /119 | Середня абсолютна амплітуда, PWM |',
        '|---|---|---:|---:|---:|']
    diagnostics={'parent':initial['diagnostic'],**{k:v['diagnostic'] for k,v in rows.items()}}
    for arm,diag in diagnostics.items():
        for side in ('original','mirrored'):
            d=diag['trace'][side]['terminal_slow_active']
            lines.append(f'| {arm} | {side} | {d["weak_network_frames"]} | '
                f'{d["opposite_network_frames"]} | {d["mean_abs_network_yaw"]:.2f} |')
    lines+=['','Знак проти вчителя означає розбіжність з його історичним вибором, не доведену небезпечність локального напряму.']
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for arm in rows:
        progress=json.loads((out/(arm+'_training.json')).read_text())
        axes[0,0].plot([p['epoch'] for p in progress],[p['quiet_final_pwm'][2] for p in progress],label=arm)
        with np.load(out/arm/'clear/trace.npz') as a:
            f=a['velocity'][:,0]*np.cos(a['yaw'])+a['velocity'][:,1]*np.sin(a['yaw'])
            axes[0,1].plot(a['time'],f,label=arm)
    axes[0,0].set(xlabel='Additional epoch',ylabel='Quiet Pitch (PWM)',title='Cruise preservation')
    axes[0,1].set(xlabel='Time (s)',ylabel='Forward speed (m/s)',title='Integrated start from rest')
    for col,side in enumerate(('original','mirrored')):
        ax=axes[1,col]
        with np.load(out/'initial_replay.npz') as a:
            ax.plot(a['time'],a[side+'_teacher'][:,3]-1500,label='teacher',color='.3')
            ax.plot(a['time'],a[side+'_network'][:,3]-1500,label='parent',ls=':')
        for arm in rows:
            with np.load(out/arm/'recorded_replay.npz') as a:
                ax.plot(a['time'],a[side+'_network'][:,3]-1500,label=arm)
        ax.set(xlim=(33,36.57),xlabel='Recorded time (s)',ylabel='Yaw offset (PWM)',title=side+' frozen-input replay')
    for ax in axes.flat:ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.savefig(out/'learning.png',dpi=150);plt.close(fig)
    flights=json.loads((out/'flights.json').read_text()) if (out/'flights.json').exists() else []
    lines+=['','| Кімната | Варіант | Час, с | Контакт | Мін. зазор, см | Шлях, м |',
        '|---|---|---:|---|---:|---:|']
    for row in flights:
        m=row['metrics']
        lines.append(f'| {row["case"]} | {row["arm"]} | {m["duration_s"]:.3f} | '
            f'{m["collision"] or "немає"} | {100*m["min_clearance_m"]:.2f} | {m["distance_m"]:.2f} |')
    if flights:
        cases=list(dict.fromkeys(r['case'] for r in flights))
        fig,axes=plt.subplots(2,2,figsize=(12,11),layout='constrained')
        for ax,case in zip(axes.flat,cases):
            geometry=json.loads((parent/'flights'/case/'manifest.json').read_text())['geometry']
            for b in geometry['boxes']:
                ax.add_patch(Rectangle((b['x_min'],b['y_min']),b['x_max']-b['x_min'],b['y_max']-b['y_min'],color='.86'))
            for c in geometry['cylinders']:
                ax.add_patch(Circle((c['center_x'],c['center_y']),c['radius'],color='.86'))
            paths=[('parent',parent/'flights'/case/'physical.npz')]
            paths += [(r['arm'],out/r['arm']/'flights'/case/'physical.npz') for r in flights if r['case']==case]
            for label,path in paths:
                with np.load(path) as a:
                    line,=ax.plot(*a['pos'][:,:2].T,label=label,lw=1.3)
                    ax.scatter(*a['pos'][-1,:2],s=18,color=line.get_color())
            ax.set(xlim=(-4,4),ylim=(-4,4),xlabel='X (m)',ylabel='Y (m)',title=case)
            ax.set_aspect('equal');ax.legend(fontsize=8);ax.grid(alpha=.2)
        fig.savefig(out/'trajectories.png',dpi=150);plt.close(fig)
    for arm,r in rows.items():
        if not r['room_screen']['passed']:
            lines+=['',f'{arm}: кімнати не запускалися — {r["room_screen"]["reasons"]}.']
    lines+=['','Навчальні кадри — оцінка засвоєння, не незалежна перевірка. Відомі аварійні входи не додавалися в навчання.',
        'Навіть проходження кімнат не означає виходу з пастки чи придатності до реального польоту. Активний пакет не змінено.']
    (out/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path)
    report(p.parse_args().out)
