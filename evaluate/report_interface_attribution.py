"""Summarize parameter-group attribution and the admitted hybrid's real rollouts."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Circle


def report(out):
    from optimizer.early_turn_training import PARENT,ROOT
    out=Path(out);rows=json.loads((out/'comparison.json').read_text())
    lines=['# Вхідні проєкції та декодер: локалізація регресії','',
        'Параметри переносяться без навчання. Єдиний новий гібрид, допущений до фізичних прогонів: старі входи + новий декодер (head_only).','',
        '| Варіант | Напрямок /12 | Згасання /12 | Макс. час, с | Вузькі /8 | Широкі /4 | Pitch MAE | Yaw MAE |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm,r in rows.items():
        t=r['temporal'];c=r['correction']
        lines.append(f'| {arm} | {t["direction"]} | {t["recovery"]} | {t["max_recovery_s"]} | '
            f'{t["narrow_amplitude"]} | {t["broad_joint"]} | {c["pitch_mae_pwm"]:.2f} | {c["yaw_mae_pwm"]:.2f} |')
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    names=list(rows)
    axes[0].bar(names,[r['correction']['yaw_mae_pwm'] for r in rows.values()]);axes[0].set(ylabel='Demonstration yaw MAE (PWM)')
    axes[1].bar(names,[r['temporal']['narrow_min_signed_yaw_pwm'] for r in rows.values()]);axes[1].axhline(80,ls='--',color='.4')
    axes[1].set(ylabel='Minimum signed narrow yaw (PWM)')
    for ax in axes:ax.tick_params(axis='x',rotation=15);ax.grid(axis='y',alpha=.2)
    fig.savefig(out/'group_comparison.png',dpi=150);plt.close(fig)
    dest=out/'head_only_flights';flights=json.loads((dest/'report.json').read_text())
    lines+=['','| Сцена | Гібрид: зазор, см | Порівняння | Зазор порівняння, см | Контакт гібрида | Шлях гібрида, м |',
        '|---|---:|---|---:|---|---:|']
    fig,axes=plt.subplots(2,3,figsize=(15,10),layout='constrained')
    for ax,row in zip(axes.flat,flights):
        name=row['case'];m=row['metrics'];run_dir=dest/name
        if (PARENT/'flights'/name/'physical.npz').exists():
            ref=PARENT/'flights'/name;label='parent'
        else:
            ref=ROOT/'output/early_turn_training_v1/control/flights'/name;label='historical control'
        with np.load(ref/'physical.npz') as a:ref_gap=float(a['clearance'].min());ref_xy=a['pos'][:,:2].copy()
        lines.append(f'| {name} | {100*m["min_clearance_m"]:.3f} | {label} | {100*ref_gap:.3f} | '
            f'{m["collision"] or "немає"} | {m["distance_m"]:.2f} |')
        geo=json.loads((run_dir/'manifest.json').read_text())['geometry']
        for b in geo['boxes']:ax.add_patch(Rectangle((b['x_min'],b['y_min']),b['x_max']-b['x_min'],b['y_max']-b['y_min'],color='.86'))
        for c in geo['cylinders']:ax.add_patch(Circle((c['center_x'],c['center_y']),c['radius'],color='.86'))
        ax.plot(*ref_xy.T,label=label,ls='--')
        with np.load(run_dir/'physical.npz') as a:
            line,=ax.plot(*a['pos'][:,:2].T,label='head hybrid');ax.scatter(*a['pos'][-1,:2],s=20,color=line.get_color())
        ax.set(title=name,xlim=(-4,4),ylim=(-4,4),xlabel='X (m)',ylabel='Y (m)');ax.set_aspect('equal');ax.legend(fontsize=8)
    fig.savefig(out/'physical_comparison.png',dpi=150);plt.close(fig)
    lines+=['','parent — архівний кандидат. Для ±3° показано попередній контроль із додатковими 10 епохами, '
        'оскільки окремий архівний parent на цих стартах не запускався. Це різні порівняння, не однаковий baseline.',
        'Фізичними командами керує нейронний гібрид. Учитель лише спостерігає. '
        'Інший, окремо навчений XY-head кандидат не пройшов допуск і тут не показаний як політ.']
    (out/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('out',type=Path)
    report(parser.parse_args().out)
