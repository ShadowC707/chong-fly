"""Report the fixed early-turn learning experiment and its physical regressions."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Circle


def report(out):
    from artifacts.research import load_research_run
    from optimizer.early_turn_training import PARENT,SNAPSHOT,correction_case
    torch.set_num_threads(1);out=Path(out)
    rows=json.loads((out/'comparison.json').read_text())
    initial=json.loads((out/'initial.json').read_text())
    lines=['# Навчання раннього Pitch/Yaw-маневру','','Рівний контроль і одна фіксована коригувальна ціль; ядро W/tau/A заморожене.','',
        '| Варіант | Pitch MAE на корекції | Yaw MAE на корекції | Напрямок /12 | Відновлення /12 | Макс. час, с | Вузькі /8 | Широкі /4 | Сцени /7 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|',
        f'| parent | {initial["correction"]["pitch_mae_pwm"]:.2f} | {initial["correction"]["yaw_mae_pwm"]:.2f} | 12 | 12 | 0.50 | 8 | 4 | 7 |']
    for arm,r in rows.items():
        t=r['temporal'];c=r['correction']
        lines.append(f'| {arm} | {c["pitch_mae_pwm"]:.2f} | {c["yaw_mae_pwm"]:.2f} | {t["direction"]} | {t["recovery"]} | '
            f'{t["max_recovery_s"]} | {t["narrow_amplitude"]} | {t["broad_joint"]} | {t["scenes"]} |')
    case=correction_case(SNAPSHOT/'evidence/early_turn/early_xy_+0/decisions.npz')
    fig,axes=plt.subplots(1,3,figsize=(16,4),layout='constrained')
    time=case['time'];target=case['Y'][0].numpy()
    for ax,ch in zip(axes[:2],(2,3)):ax.plot(time,target[:,ch],'k--',label='teacher target')
    curves={}
    for arm,path in [('parent',PARENT)]+[(a,out/a) for a in rows]:
        p,_=load_research_run(path)
        with torch.no_grad():pred=p(case['X'],dt=.02)[0][0].numpy()
        curves[arm]=pred
        for ax,ch in zip(axes[:2],(2,3)):ax.plot(time,pred[:,ch],label=arm)
    for ax,name in zip(axes[:2],('Pitch','Yaw')):
        ax.axvspan(6,10,color='.8',alpha=.3);ax.set(xlabel='Recorded time (s)',ylabel=name+' (PWM)');ax.legend(fontsize=8);ax.grid(alpha=.2)
    for arm in rows:
        progress_path=out/(arm+'_training.json')
        if not progress_path.exists() and arm=='control':
            reused=json.loads((out/'protocol.json').read_text()).get('reused_control')
            if reused: progress_path=Path(reused).parent/'control_training.json'
        progress=json.loads(progress_path.read_text())
        axes[2].plot([r['epoch'] for r in progress],[r['recorded_loss'] for r in progress],label=arm)
    axes[2].set(xlabel='Epoch',ylabel='Shared recorded loss');axes[2].legend();axes[2].grid(alpha=.2)
    fig.savefig(out/'learning.png',dpi=150);plt.close(fig)
    np.savez_compressed(out/'corrective_replay.npz',time=time,target=target,**curves)
    flights=json.loads((out/'flights.json').read_text()) if (out/'flights.json').exists() else []
    lines+=['','| Сцена | Варіант | Час, с | Контакт | Зазор, см | Шлях, м | Похибка висоти, см |','|---|---|---:|---|---:|---:|---:|']
    for row in flights:
        m=row['metrics'];lines.append(f'| {row["case"]} | {row["arm"]} | {m["duration_s"]:.3f} | {m["collision"] or "немає"} | '
            f'{100*m["min_clearance_m"]:.3f} | {m["distance_m"]:.2f} | {100*m["max_height_error_m"]:.2f} |')
    if flights:
        cases=list(dict.fromkeys(r['case'] for r in flights))
        fig,axes=plt.subplots(2,3,figsize=(15,10),layout='constrained')
        for ax,name in zip(axes.flat,cases):
            example=next(r for r in flights if r['case']==name)
            m=json.loads((out/example['arm']/'flights'/name/'manifest.json').read_text())
            for b in m['geometry']['boxes']:ax.add_patch(Rectangle((b['x_min'],b['y_min']),b['x_max']-b['x_min'],b['y_max']-b['y_min'],color='.86'))
            for c in m['geometry']['cylinders']:ax.add_patch(Circle((c['center_x'],c['center_y']),c['radius'],color='.86'))
            for row in [r for r in flights if r['case']==name]:
                with np.load(out/row['arm']/'flights'/name/'physical.npz') as a:ax.plot(*a['pos'][:,:2].T,label=row['arm'])
            ax.set(xlim=(-4,4),ylim=(-4,4),title=name);ax.set_aspect('equal');ax.legend(fontsize=8)
        fig.savefig(out/'trajectories.png',dpi=150);plt.close(fig)
    for arm,r in rows.items():
        if not r['screen']['passed']:lines+=['',f'{arm}: кімнати пропущені — {r["screen"]["reasons"]}.']
    lines+=['','Коригувальний replay є навчальним прикладом; його MAE не є незалежною оцінкою. '
        'У фізичних прогонах керує лише мережа, без часу та без учителя на вході. '
        '±3° — нові сусідні старти у відомій кімнаті. Архівний та активний пакети збережено.']
    (out/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('out',type=Path)
    report(parser.parse_args().out)
