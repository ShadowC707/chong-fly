"""Render the completed paired flow-reflex experiment; no training or rollouts."""
import argparse
import json
from pathlib import Path

import numpy as np

ARMS=('before','control','active_fast')
COLORS=('#526477','#d18b24','#267cba')


def render(directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle,Circle
    d=Path(directory)
    validation=json.loads((d/'validation.json').read_text())
    assert validation['completed_arms']==3 and validation['completed_physical_trials']==12
    scores=json.loads((d/'comparison.json').read_text())
    physical=json.loads((d/'physical_comparison.json').read_text())
    cases=list(physical['before'])
    assert len(cases)==4 and all(set(physical[a])==set(cases) for a in ARMS)
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    for ax,(split,metric,title) in zip(axes.flat,(
        ('internal_validation','strong_yaw_frames','Internal: excessive yaw on high flow / close ToF'),
        ('external_seen_evaluation','strong_yaw_frames','Known external: excessive yaw on high flow / close ToF'),
        ('external_seen_evaluation','yaw_mae','Known external: yaw MAE in active-fast subset'),
        ('external_seen_evaluation','pitch_mae','Known external: pitch MAE across ALL valid frames'))):
        vals=[]
        for arm in ARMS:
            raw=scores[arm]['raw'][split]
            if metric=='strong_yaw_frames':value=raw['neutral_yaw_active_fast'][metric]
            elif metric=='yaw_mae':value=raw['neutral_yaw_active_fast']['mae_roll_pitch_yaw'][2]
            else:value=raw['all']['mae_roll_pitch_yaw'][1]
            vals.append(value)
        bars=ax.bar(ARMS,vals,color=COLORS)
        ax.bar_label(bars,fmt='%.2f' if metric.endswith('mae') else '%.0f')
        ax.set(title=title,ylabel='PWM' if metric.endswith('mae') else 'Frames (correlated within episodes)')
        ax.set_ylim(0,max(vals)*1.2);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Same parent / 10 epochs per training arm / frozen recurrent core\nConditional yaw and global pitch metrics; lower is better')
    fig.savefig(d/'raw_metrics.png',dpi=150);plt.close(fig)

    fig,axes=plt.subplots(4,3,figsize=(13,16),layout='constrained')
    for i,case in enumerate(cases):
        for j,arm in enumerate(ARMS):
            ax=axes[i,j];g=json.loads((d/arm/case/'manifest.json').read_text())['geometry']
            for box in g['boxes']:
                ax.add_patch(Rectangle((box['x_min'],box['y_min']),box['x_max']-box['x_min'],
                    box['y_max']-box['y_min'],color='.82'))
            for cylinder in g['cylinders']:
                ax.add_patch(Circle((cylinder['center_x'],cylinder['center_y']),cylinder['radius'],color='.82'))
            with np.load(d/arm/case/'trace.npz') as trace:p=trace['pos']
            m=physical[arm][case]
            ax.plot(p[:,0],p[:,1],color=COLORS[j],lw=1.2)
            ax.scatter(*p[0,:2],marker='s',color='green',s=20)
            ax.scatter(*p[-1,:2],marker='x' if m['collision'] else 'o',color='red' if m['collision'] else COLORS[j],s=30)
            room=g['room']
            ax.set(xlim=(room['x_min'],room['x_max']),ylim=(room['y_min'],room['y_max']),
                xlabel='X (m)',ylabel='Y (m)',title=f"{case} | {arm}\n{m['duration_s']:.3f}s, gap {100*m['min_clearance_m']:.2f}cm, AUTO {m['auto_s']:.2f}s")
            ax.set_aspect('equal');ax.grid(alpha=.2)
    fig.suptitle('Fixed coordinator / starts / sensor seeds / radius 0.20m / horizon 45s\nSquare = start; X = collision; dot = horizon end. No survey memory or scheduled STOP.')
    fig.savefig(d/'trajectories.png',dpi=150);plt.close(fig)

    lines=['# Парне донавчання: підсумок','',
        '| Показник | До | Контроль +10 епох | Active-fast +10 епох |',
        '|---|---:|---:|---:|']
    metrics=[('Сильний yaw, зовнішній зріз /180',lambda r:r['raw']['external_seen_evaluation']['neutral_yaw_active_fast']['strong_yaw_frames']),
        ('Yaw MAE, зовнішній зріз',lambda r:r['raw']['external_seen_evaluation']['neutral_yaw_active_fast']['mae_roll_pitch_yaw'][2]),
        ('Pitch MAE, всі зовнішні кадри',lambda r:r['raw']['external_seen_evaluation']['all']['mae_roll_pitch_yaw'][1]),
        ('Макс. згасання, с',lambda r:r['temporal']['max_recovery_s']),
        ('Мінімальна вузька амплітуда PWM',lambda r:r['temporal']['narrow_min_signed_yaw_pwm'])]
    for label,fn in metrics:lines.append('| '+label+' | '+' | '.join(f'{fn(scores[a]):.2f}' for a in ARMS)+' |')
    lines+=['','| Сцена | До | Контроль | Active-fast |','|---|---|---|---|']
    for case in cases:
        cells=[]
        for arm in ARMS:
            m=physical[arm][case]
            cells.append(f"контакт {m['duration_s']:.3f} с" if m['collision'] else f"45 с; зазор {100*m['min_clearance_m']:.2f} см")
        lines.append('| '+case+' | '+' | '.join(cells)+' |')
    lines+=['','Завершені складні прогони (окремо перевіряйте фактичний час AUTO): '+', '.join(
        f"{a}: {sum(m['completed'] for m in physical[a].values())}/{len(cases)}" for a in ARMS)+'.',
        'Прості сцени: '+', '.join(f"{a}: {scores[a]['temporal']['scenes']}/7" for a in ARMS)+'.',
        'Автоматична заміна робочого пакета не виконується.',
        'Набори й сцени вже відомі; один seed; кількість кадрів не дорівнює кількості незалежних випробувань.']
    (d/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
    render(p.parse_args().directory)
