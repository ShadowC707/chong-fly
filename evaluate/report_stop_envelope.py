"""Summarize measured STOP displacement; never turns samples into a safety bound."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from simulation.xy_coordinator import XYCoordinator,XYConfig,BehaviorConfig


def hover_diagnostics(folder,report):
    manifest=json.loads((folder/'manifest.json').read_text())
    contract=manifest['system']['resolved']['coordinator']
    c=XYCoordinator(anti_rollback=True,turn_latch=True,stop_hover=True,
        config=XYConfig(**contract['config']),behavior_config=BehaviorConfig(**contract['behavior_config']))
    warm=manifest['experiment']['warm_state']
    c.filtered_flow,c.filtered_flow_y=warm['filtered_flow']
    c.previous_pr=np.array(warm['previous_pr'])
    times=[];false_ticks=0;first=None
    with np.load(folder/'trace.npz') as a:
        for i in range(0,len(a['command_time']),5):
            t=float(a['command_time'][i])
            stop=t+1e-8>=report['delay_s']
            # In a clear scene HOVER depends only on filtered flow + dwell;
            # network PWM does not enter this predicate. Verify count below.
            c.apply([1213,1500,1526,1500],flow_xy=a['flow'][i],tof_m=np.full(64,3.5),dt=.02,stop_requested=stop)
            if c.last_decision['state']=='HOVER':
                times.append(t)
                if first is None:first=t
                if np.linalg.norm(a['velocity'][i,:2])>=.03:false_ticks+=1
    assert len(times)*5==report['behavior_counts']['HOVER'], 'HOVER replay differs'
    return dict(first_hover_s=first,hover_control_ticks=len(times),
                hover_ticks_true_speed_ge_003=false_ticks,
                meaning='HOVER is an observable filtered-flow label, not the ground-truth settling criterion.')


def summarize(directory):
    directory=Path(directory)
    protocol=json.loads((directory/'protocol.json').read_text())
    reports=json.loads((directory/'report.json').read_text())
    if len(reports)!=protocol['trials']:raise ValueError('Incomplete grid')
    groups=defaultdict(list)
    hover=[]
    for row in reports:
        groups[(row['case'],row['delay_s'])].append(row)
        hover.append(dict(id=row['id'],**hover_diagnostics(directory/row['id'],row)))
    summary=[]
    lines=['# STOP: observed displacement over 8 seconds','',
        '| State | STOP delay (ms) | Max excursion min..max (cm) | Stable-low-speed onset min..max (s) | Settled trials |',
        '|---|---:|---:|---:|---:|']
    for (case,delay),rows in sorted(groups.items()):
        excursion=[r['metrics']['max_excursion_m'] for r in rows]
        times=[r['metrics']['settled_from_s'] for r in rows if r['metrics']['settled']]
        result=dict(case=case,delay_s=delay,trials=len(rows),settled=len(times),
            excursion_min_m=min(excursion),excursion_max_m=max(excursion),
            settled_from_min_s=min(times) if times else None,settled_from_max_s=max(times) if times else None,
            observed_swept_radius_m=protocol['radius_m']+max(excursion))
        summary.append(result)
        timing=f'{min(times):.3f}–{max(times):.3f}' if times else 'censored'
        lines.append(f'| {case} | {delay*1000:.0f} | {100*min(excursion):.2f}–{100*max(excursion):.2f} | {timing} | {len(times)}/{len(rows)} |')
    (directory/'summary.json').write_text(json.dumps(dict(groups=summary,hover_diagnostics=hover,
        all_settled=all(r['metrics']['settled'] for r in reports),
        collisions=sum(r['collision'] is not None for r in reports),
        max_height_error_m=max(r['metrics']['max_height_error_m'] for r in reports)),indent=2),encoding='utf-8')
    lines += ['','Displacement is from the initial centre; body radius must be added only when starting from a centre-to-surface distance.',
              'This is a finite sampled envelope in free space, not an operational safety bound.']
    (directory/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    plot(directory,reports,summary)


def plot(directory,reports,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,9),layout='constrained')
    speeds=[.1,.2,.4,.6]
    for delay,label in [(0.,'Immediate STOP'),(.1,'STOP request delayed 100 ms')]:
        x=[];lo=[];hi=[]
        for speed in speeds:
            key=f'forward_{round(speed*10):02d}'
            r=next((r for r in summary if r['case']==key and r['delay_s']==delay),None)
            if r:
                x.append(speed);lo.append(r['excursion_min_m']);hi.append(r['excursion_max_m'])
        axes[0,0].plot(x,np.array(hi)*100,'o-',label=label)
        axes[0,0].fill_between(x,np.array(lo)*100,np.array(hi)*100,alpha=.15)
    axes[0,0].set(xlabel='Initial forward speed (m/s)',ylabel='Centre excursion (cm)',title='Level body: maximum over three sensor seeds')
    axes[0,0].legend(fontsize=8)
    names=['pitch_backward_5','forward_02','pitch_forward_5']
    maxima=[next(r['excursion_max_m'] for r in summary if r['case']==name and r['delay_s']==0) for name in names]
    axes[0,1].bar(['Pitch -5°','Level','Pitch +5°'],np.array(maxima)*100,color=['#418f68','#387cb8','#d17b35'])
    axes[0,1].set(ylabel='Centre excursion (cm)',title='Same initial 0.20 m/s; different initial tilt')
    for name,label in [('forward_02','Forward 0.2'),('forward_06','Forward 0.6'),('pitch_forward_5','Forward 0.2 + pitch 5°')]:
        row=next(r for r in reports if r['case']==name and r['delay_s']==0 and r['seed']==20261021)
        with np.load(directory/row['id']/'trace.npz') as a:
            axes[1,0].plot(a['time'],np.linalg.norm(a['velocity'][:,:2],axis=1),label=label)
    axes[1,0].axhline(.03,color='black',ls=':',label='0.03 m/s')
    axes[1,0].set(xlabel='Time from initial state (s)',ylabel='True horizontal speed (m/s)',xlim=(0,3.5),title='Settling is not instantaneous')
    axes[1,0].legend(fontsize=8)
    for name in ['left_02','right_02','diagonal_left','diagonal_right','roll_left_5','roll_right_5']:
        row=next(r for r in reports if r['case']==name and r['delay_s']==0 and r['seed']==20261021)
        with np.load(directory/row['id']/'trace.npz') as a:axes[1,1].plot(*a['pos'][:,:2].T,label=name)
    axes[1,1].set(xlabel='X displacement (m)',ylabel='Y displacement (m)',title='Sideways excursion matters too')
    axes[1,1].set_aspect('equal');axes[1,1].legend(fontsize=7)
    for ax in axes.flat:ax.grid(alpha=.2)
    fig.suptitle('Unchanged STOP | 54 simulations | 1 m height | 50/30/15 Hz sensors\n'
                 'Synthetic initial physical states after normal AUTO warmup; not hardware calibration')
    fig.savefig(directory/'envelope.png',dpi=160)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path)
    summarize(parser.parse_args().directory)
