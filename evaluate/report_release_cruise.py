"""Summarize the fixed cruise experiment, including skipped room trials."""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def report(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle
    out = Path(out)
    results = json.loads((out/'comparison.json').read_text())
    flights = json.loads((out/'flights.json').read_text()) if (out/'flights.json').exists() else []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
    lines = ['# Парне навчання зі збереженням крейсерського руху', '',
        'Два кодування, незмінний учитель v5/1530, заморожені W/tau/A. 10 епох, seed=42.', '',
        '| Кодування | Pitch у спокої | Вперед у clear, м/с | Допуск до кімнат | Напрямок /12 | Згасання /12 | Широкі /4 | Вузькі /8 |',
        '|---|---:|---:|---|---:|---:|---:|---:|']
    for name, result in results.items():
        progress = json.loads((out/(name+'_training.json')).read_text())
        initial = json.loads((out/(name+'_initial.json')).read_text())['cruise']['final_pwm'][2]
        axes[0].plot([0]+[p['epoch'] for p in progress],
            [initial]+[p['quiet_final_pwm'][2] for p in progress], marker='.', label=name)
        with np.load(out/name/'clear/trace.npz') as a:
            forward = a['velocity'][:,0]*np.cos(a['yaw'])+a['velocity'][:,1]*np.sin(a['yaw'])
            axes[1].plot(a['time'], forward, label=name)
        t = result['temporal']; speed = result['clear']['mean_forward_speed_mps']
        speed_text = 'N/A' if speed is None else f'{speed:.3f}'
        lines.append(f'| {name} | {result["cruise"]["final_pwm"][2]:.2f} | {speed_text} | '
            f'{result["motion_admission"]["passed"]} | {t["direction"]} | {t["recovery"]} | '
            f'{t["broad_joint"]} | {t["narrow_amplitude"]} |')
    if (out/'baseline_clear/trace.npz').is_file():
        with np.load(out/'baseline_clear/trace.npz') as a:
            forward = a['velocity'][:,0]*np.cos(a['yaw'])+a['velocity'][:,1]*np.sin(a['yaw'])
            axes[1].plot(a['time'],forward,ls=':',label='original baseline')
    axes[0].axhspan(1515,1535,color='green',alpha=.08)
    axes[0].axhline(1500,color='.5',ls=':')
    axes[0].set(xlabel='Epoch',ylabel='Quiet Pitch (PWM)',title='Cruise command during training')
    axes[1].axhline(.1,color='.5',ls=':',label='mean admission threshold')
    axes[1].set(xlabel='Time (s)',ylabel='Body forward velocity (m/s)',title='Integrated clear scene, start at rest')
    for ax in axes: ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.savefig(out/'cruise.png',dpi=150); plt.close(fig)
    lines += ['', '| Кімната | Кодування | Час, с | Контакт | Шлях, м | Середня швидкість AUTO, м/с | Мін. зазор, см |',
        '|---|---|---:|---|---:|---:|---:|']
    for row in flights:
        m = row['metrics']
        lines.append(f'| {row["case"]} | {row["arm"]} | {m["duration_s"]:.3f} | {m["collision"] or "немає"} | '
            f'{m["distance_m"]:.2f} | {m["mean_auto_speed_mps"]:.3f} | {100*m["min_clearance_m"]:.2f} |')
    if flights:
        cases = list(dict.fromkeys(r['case'] for r in flights))
        fig, axes = plt.subplots(2,2,figsize=(12,11),layout='constrained')
        for ax,case in zip(axes.flat,cases):
            rows = [r for r in flights if r['case']==case]
            geometry = json.loads((out/rows[0]['arm']/'flights'/case/'manifest.json').read_text())['geometry']
            for b in geometry['boxes']:
                ax.add_patch(Rectangle((b['x_min'],b['y_min']),b['x_max']-b['x_min'],b['y_max']-b['y_min'],color='.86'))
            for c in geometry['cylinders']:
                ax.add_patch(Circle((c['center_x'],c['center_y']),c['radius'],color='.86'))
            traces = [('original',ROOT/'output/practical_maneuver_v1'/(case+'_before')/'physical.npz','-'),
                ('teacher 1530',ROOT/'output/teacher_native1530_v1'/case/'physical.npz','-')]
            traces += [(r['arm'],out/r['arm']/'flights'/case/'physical.npz','--') for r in rows]
            for label,path,style in traces:
                with np.load(path) as a:
                    line, = ax.plot(*a['pos'][:,:2].T,style,lw=1.2,label=label)
                    ax.scatter(*a['pos'][-1,:2],s=16,color=line.get_color())
            ax.set(xlim=(-4,4),ylim=(-4,4),xlabel='X (m)',ylabel='Y (m)',title=case)
            ax.set_aspect('equal'); ax.grid(alpha=.2); ax.legend(fontsize=7)
        fig.suptitle('Fixed known rooms: recorded baseline/teacher and new trained models')
        fig.savefig(out/'trajectories.png',dpi=150); plt.close(fig)
    for name,result in results.items():
        if not result['motion_admission']['passed']:
            lines += ['', f'{name}: кімнати пропущено через {result["motion_admission"]["reasons"]}.']
    lines += ['', 'Це відомі діагностичні кімнати, один seed. Результат не є допуском до реального польоту.',
        'Активні ваги й конфігурація не замінені; автоматичного просування кандидата немає.']
    (out/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out',type=Path)
    report(parser.parse_args().out)
