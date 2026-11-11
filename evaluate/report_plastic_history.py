"""Plot the completed protected plastic-core diagnostic; no retraining."""
import argparse
import json
from pathlib import Path
import numpy as np


def render(directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import torch
    from artifacts.research import load_research_run
    from optimizer.probe_history_capacity import make_cases
    d=Path(directory)
    check=json.loads((d/'validation.json').read_text())
    if check['completed_arms']!=3:raise ValueError('Experiment is not complete')
    results=json.loads((d/'comparison.json').read_text())
    arms=('before','frozen','plastic');colors=('#526477','#d18b24','#267cba')
    ticks=(5,12,25,50,150); times=np.asarray(ticks)*.02
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for arm,color in zip(arms,colors):
        r=results[arm]
        for ax,split in zip(axes[0],('training','evaluation')):
            ax.plot(times,[r['capacity'][split][str(n)]['max_abs_pitch_error_pwm'] for n in ticks],'-o',color=color,label=arm)
            ax.axhline(20,color='.6',ls='--')
            ax.set(xlabel='Identical suffix (s)',ylabel='Worst terminal error (PWM)',title=split)
        axes[1,0].plot(times,[max(g['pitch_span_pwm'] for g in r['capacity']['training'][str(n)]['groups']) for n in ticks],'-o',color=color)
    axes[1,0].set(xlabel='Identical suffix (s)',ylabel='Pitch PWM',title='Maximum difference between histories')
    vals=[results[a]['cruise']['final_pwm'][2]-1500 for a in arms]
    bars=axes[1,1].bar(arms,vals,color=colors)
    axes[1,1].bar_label(bars,labels=[f'{v+1500:.2f}' for v in vals]);axes[1,1].axhline(15,color='.6',ls='--')
    axes[1,1].set(title='Clear-space cruise (labels: absolute PWM)',ylabel='Pitch offset from 1500')
    axes[0,0].legend()
    for ax in axes.flat:ax.grid(alpha=.2)
    fig.suptitle('Same history + anchor loss / same 300 updates\nFrozen interface-only vs existing-edge W/tau learning')
    fig.savefig(d/'comparison.png',dpi=150);plt.close(fig)
    torch.set_num_threads(1);cases=make_cases((.12,),150)
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained',sharey=True)
    traces={}
    for ax,arm in zip(axes,arms):
        policy,_=load_research_run(d/arm);policy.eval()
        with torch.no_grad():pred=policy(cases['X'][:3],dt=.02)[0].numpy()
        traces[arm]=pred
        for i,(label,color) in enumerate(zip(('always clear','unsettled','settled'),('#267cba','#bc3a32','#33804a'))):
            ax.plot(np.arange(200)*.02,pred[i,:,2],color=color,label=label)
            ax.plot(np.arange(200)*.02,cases['Y'][i,:,2],ls='--',color=color,alpha=.5)
        ax.axvspan(1,4,color='.93',zorder=-1);ax.grid(alpha=.2)
        ax.set(title=arm,xlabel='Time (s)',ylabel='Pitch PWM')
    axes[0].legend(fontsize=8);fig.suptitle('Identical last 3s inputs | solid: model; dashed: teacher | flow=.12, left threat')
    fig.savefig(d/'history_traces.png',dpi=150);plt.close(fig)
    np.savez_compressed(d/'plot_traces.npz',**traces,target=cases['Y'][:3].numpy())
    # Endpoint fitting can hide phase-shifted oscillations: report the entire
    # common suffix as well, on exactly the three histories shown in the plot.
    target=cases['Y'][:3,50:,2].numpy()
    suffix_metrics={}
    for arm,pred in traces.items():
        pitch=pred[:,50:,2]
        suffix_metrics[arm]={
            'mean_abs_pitch_error_pwm':float(np.abs(pitch-target).mean()),
            'max_abs_pitch_error_pwm':float(np.abs(pitch-target).max()),
            'pitch_peak_to_peak_pwm_by_history':np.ptp(pitch,axis=1).tolist(),
        }
    (d/'full_suffix_diagnostic.json').write_text(json.dumps({
        'scope':'Plotted flow=0.12, left threat, 3 histories, t=1.00..3.98 s; no retraining',
        'arms':suffix_metrics,
    },indent=2)+'\n',encoding='utf-8')
    lines=['# Навчання часових сталих та наявних зв’язків','',
        '| Метрика | До | Ядро заморожене | W/tau навчаються |','|---|---:|---:|---:|']
    fields=(('Макс. навчальна помилка після 3 с, PWM',lambda r:r['capacity']['training']['150']['max_abs_pitch_error_pwm']),
        ('Макс. перевірочна помилка після 3 с, PWM',lambda r:r['capacity']['evaluation']['150']['max_abs_pitch_error_pwm']),
        ('Різниця між історіями після 3 с, PWM',lambda r:max(g['pitch_span_pwm'] for g in r['capacity']['training']['150']['groups'])),
        ('Pitch у вільному просторі',lambda r:r['cruise']['final_pwm'][2]),
        ('Напрямок Yaw /12',lambda r:r['temporal']['direction']),
        ('Відновлення Yaw /12',lambda r:r['temporal']['recovery']),
        ('Вузька амплітуда /8',lambda r:r['temporal']['narrow_amplitude']))
    for label,fn in fields:lines.append('| '+label+' | '+' | '.join(f'{fn(results[a]):.3f}' for a in arms)+' |')
    lines+=['','300 оновлень на гілку; один seed; це діагностичні сенсорні історії, не складні польотні сцени.',
        'Увага: plastic має сильні коливання Pitch протягом спільних 3 с. Кінцеве розрізнення історій не означає стабільного виконання маневру; див. history_traces.png та full_suffix_diagnostic.json.',
        'Пакет не замінено. training_audit описує фактичне навчання; constraint_audit збереженого пакета — заморожування після нього.']
    (d/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
    render(p.parse_args().directory)
