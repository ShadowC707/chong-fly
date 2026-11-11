"""Summarize the fixed full-episode trial without changing weights or targets."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from artifacts.research import load_research_run
from optimizer.continue_flow_reflex import write
from optimizer.probe_history_capacity import make_cases
from generator.generate_reflex_dataset import make_teacher
from edge.runtime import NumpyPolicy


def run(directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    torch.set_num_threads(1);d=Path(directory)
    if json.loads((d/'validation.json').read_text())['completed_arms']!=3:
        raise ValueError('Incomplete experiment')
    results=json.loads((d/'comparison.json').read_text())
    balanced=json.loads((d/'protocol.json').read_text())['version']=='balanced-full-maneuvers-v1'
    lines=['# Повні маневри: один парний запуск','',
        '| Метрика | Початкова | Заморожене ядро | Навчання W/tau |','|---|---:|---:|---:|']
    arms=('before','frozen','plastic')
    fields=[('Pitch за нульового flow',lambda r:r['cruise']['final_pwm'][2]),
        ('Напрямок /12',lambda r:r['temporal']['direction']),
        ('Відновлення /12',lambda r:r['temporal']['recovery']),
        ('Прості сцени /7',lambda r:r['temporal']['scenes'])]
    for dataset in ('old_external','corrective','expert'):
        fields.append((dataset+' MAE Pitch (нові цілі)',lambda r,k=dataset:r['raw'][k]['revised']['all']['mae_roll_pitch_yaw'][1]))
    for label,fn in fields:lines.append('| '+label+' | '+' | '.join(f'{fn(results[a]):.2f}' for a in arms)+' |')
    clear_speeds={}
    for arm in arms:
        report=json.loads((d/arm/'report.json').read_text())
        clear_speeds[arm]=next(s for s in report['scenes'] if s['scenario']=='clear')['mean_fwd_speed']
    lines.append('| Середня швидкість уперед у clear, м/с | '+' | '.join(f'{clear_speeds[a]:.4f}' for a in arms)+' |')
    lines+=['','64 записані зовнішні епізоди; оцінка на входах записів, не нові польоти.',
        'Крейсерська складова вчителя змінена на 1530 в обох гілках. Старі мітки також оцінені у comparison.json.',
        'Старий mvp_passed не включає цю умову. 7/7 сцен не є допуском нового кандидата.',
        'Автоматична заміна робочого пакета заборонена протоколом.']
    if balanced:
        for arm in arms:
            admission=results[arm]['behavioral_admission']
            lines.append(f"{arm}: behavioral_admission={admission['passed']}; reasons={admission['reasons']}")
    (d/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    case=make_cases((.10,.16),400);target=case['Y'].clone()
    for i,x in enumerate(case['X']):
        teacher=make_teacher('geometry-reflex-v5',noise_std_pwm=0,pwm_cruise_pitch=1530.)
        for t,row in enumerate(x):target[i,t]=torch.from_numpy(teacher.step(row.numpy(),dt=.02))
    fig,axes=plt.subplots(3,2,figsize=(13,10),sharey=True,layout='constrained')
    traces={};equivalence={};neutral_readout={}
    for row,arm in enumerate(arms):
        p,_=load_research_run(d/arm);p.eval()
        with torch.no_grad():
            pred=p(case['X'],dt=.02)[0].numpy()
            zero_input=torch.ones(1,100,74);zero_input[:,:,:2]=0
            neutral_pwm,neutral_h=p(zero_input,dt=.02)
            bias_pwm=p.pwm_layer(p.dn_head.proj.bias[None,:])
            neutral_readout[arm]=dict(hidden_max_abs=float(neutral_h.abs().max()),
                actual_pitch_pwm=float(neutral_pwm[0,-1,2]),bias_only_pitch_pwm=float(bias_pwm[0,2]),
                max_abs_difference_pwm=float((neutral_pwm[0,-1]-bias_pwm[0]).abs().max()),
                scope='Reset state and zero external drive only, not recovery from an arbitrary previous state')
        traces[arm]=pred;runtime=NumpyPolicy(d/arm/'numpy');worst=0.
        for x,expected in zip(case['X'].numpy(),pred):
            runtime.reset_state()
            actual=np.stack([runtime.step_np(v[:2],v[2:66],v[66:],dt=.02) for v in x])
            worst=max(worst,float(np.abs(actual-expected).max()))
        equivalence[arm]=dict(max_abs_pwm_error=worst,within_002_pwm=worst<=.02,episodes=12,steps=450)
        for ax,start,flow in zip(axes[row],(0,6),(.10,.16)):
            for i,(label,color) in enumerate(zip(('clear','unsettled','settled'),('#267cba','#bc3a32','#33804a'))):
                ax.plot(np.arange(450)*.02,pred[start+i,:,2],color=color,label=label)
                ax.plot(np.arange(450)*.02,target[start+i,:,2],color=color,ls='--',alpha=.5)
            ax.axvspan(1,9,color='.93',zorder=-1);ax.grid(alpha=.2)
            ax.set(title=f'{arm}, flow={flow:.2f}, left threat',xlabel='Time (s)',ylabel='Pitch PWM')
    axes[0,0].legend(fontsize=8)
    fig.suptitle(('Long-history diagnostics (flow .10 trained in auxiliary; .16 held out)' if balanced else 'Diagnostic histories NOT used for training')+
        '\nSolid: network; dashed: v5 with cruise=1530')
    fig.savefig(d/'history_comparison.png',dpi=140);plt.close(fig)
    np.savez_compressed(d/'history_traces.npz',**traces,target=target.numpy())
    write(d/'long_numpy_equivalence.json',equivalence)
    write(d/'neutral_readout_audit.json',neutral_readout)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path)
    run(parser.parse_args().directory)
