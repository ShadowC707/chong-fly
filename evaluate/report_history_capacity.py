"""Render the completed, bounded history-capacity experiment."""
import argparse
import json
from pathlib import Path

import numpy as np


def render(directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from artifacts.research import load_research_run
    from optimizer.probe_history_capacity import make_cases
    import torch

    d=Path(directory)
    validation=json.loads((d/'validation.json').read_text())
    if validation['completed_arms']!=3:
        raise ValueError('Expected a completed experiment')
    comparison=json.loads((d/'comparison.json').read_text())
    arms=('before','direct','curriculum'); colors=('#526477','#d18b24','#267cba')
    suffixes=(5,12,25,50,150);times=np.array(suffixes)*.02
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for arm,color in zip(arms,colors):
        r=comparison[arm]; train=r['capacity']['training']
        axes[0,0].plot(times,[train[str(n)]['max_abs_pitch_error_pwm'] for n in suffixes],'-o',color=color,label=arm)
        axes[0,1].plot(times,[train[str(n)]['loss_over_memoryless_floor'] for n in suffixes],'-o',color=color)
        # Actual zeros remain zero rather than being invented for a logarithmic axis.
        axes[1,0].plot(times,[max(g['pitch_span_pwm'] for g in train[str(n)]['groups']) for n in suffixes],'-o',color=color)
    axes[0,0].axhline(20,color='.5',ls='--',label='20 PWM fit criterion')
    axes[0,1].axhline(1,color='.5',ls='--')
    axes[0,0].set_title('Worst terminal Pitch error on training cases')
    axes[0,1].set_title('MSE / best history-blind common prediction MSE')
    axes[1,0].set_title('Maximum predicted difference between three histories')
    vals=[comparison[a]['cruise']['final_pwm'][2] for a in arms]
    bars=axes[1,1].bar(arms,np.array(vals)-1500,color=colors)
    axes[1,1].bar_label(bars,labels=[f'{v:.2f}' for v in vals])
    axes[1,1].axhline(15,color='.5',ls='--')
    axes[1,1].set(title='Clear-space Pitch offset (labels: absolute PWM)',ylabel='PWM relative to 1500')
    for ax in (axes[0,0],axes[0,1],axes[1,0]):
        ax.set_xlabel('Identical input suffix (seconds)');ax.grid(alpha=.2)
    axes[0,0].legend();axes[0,0].set_ylabel('PWM');axes[1,0].set_ylabel('PWM')
    fig.suptitle('Frozen core: 300 updates per training arm; imposed histories, not flights')
    fig.savefig(d/'capacity_summary.png',dpi=150);plt.close(fig)

    # All contexts at one training flow, one side; no selection of best case.
    torch.set_num_threads(1)
    cases=make_cases((.12,),150);names=('always clear','unsettled','settled')
    fig,axes=plt.subplots(1,3,figsize=(14,4),sharey=True,layout='constrained')
    for ax,arm in zip(axes,arms):
        policy,_=load_research_run(d/arm);policy.eval()
        with torch.no_grad():pred=policy(cases['X'][:3],dt=.02)[0].numpy()
        t=np.arange(200)*.02
        for i,(name,color) in enumerate(zip(names,('#267cba','#bc3a32','#33804a'))):
            ax.plot(t,pred[i,:,2],color=color,label=name)
            ax.plot(t,cases['Y'][i,:,2].numpy(),color=color,ls='--',alpha=.55)
        ax.axvspan(1.,4.,color='.92',zorder=-1)
        ax.set(title=arm,xlabel='Time (s)',ylabel='Pitch PWM');ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('Same final 3 seconds of inputs | solid: model; dashed: teacher | flow=0.12, left threat')
    fig.savefig(d/'history_traces.png',dpi=150);plt.close(fig)

    rows=['# Перевірка пам’яті маневру','',
        '| Показник | До | Пряме навчання | Поступова затримка |',
        '|---|---:|---:|---:|']
    metrics=(('Максимальна тренувальна помилка Pitch, 3 с, PWM',lambda r:r['capacity']['training']['150']['max_abs_pitch_error_pwm']),
        ('Максимальна перевірочна помилка Pitch, 3 с, PWM',lambda r:r['capacity']['evaluation']['150']['max_abs_pitch_error_pwm']),
        ('MSE / мінімум для спільного прогнозу без історії',lambda r:r['capacity']['training']['150']['loss_over_memoryless_floor']),
        ('Найбільша різниця команд між історіями, 3 с, PWM',lambda r:max(g['pitch_span_pwm'] for g in r['capacity']['training']['150']['groups'])),
        ('Pitch у вільному просторі за нульового flow',lambda r:r['cruise']['final_pwm'][2]),
        ('Найменша вузька амплітуда Yaw, PWM',lambda r:r['temporal']['narrow_min_signed_yaw_pwm']))
    for label,fn in metrics:
        rows.append('| '+label+' | '+' | '.join(f'{fn(comparison[a]):.5f}' for a in arms)+' |')
    rows+=['','300 оновлень у кожній навчальній гілці; це діагностична ціль, не курс навчання для польоту.',
        'Невдача в цьому бюджеті не доводить неможливість за будь-яких ваг. Пакет не просунуто в робочий.']
    (d/'summary.md').write_text('\n'.join(rows)+'\n',encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
    render(p.parse_args().directory)
