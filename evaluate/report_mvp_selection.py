"""Read-only analysis of the frozen MVP selection series; no flights or fitting."""
import argparse
import json
from pathlib import Path
import numpy as np
from evaluate.mvp_selection import ROOT, MODES, write


def report(out):
    from edge.runtime import sha256
    from demo.contacts import contact_and_clearance
    from simulation.drone_env import RoomBoundaries,BoxObstacle,CylinderObstacle
    protocol=json.loads((out/'protocol.json').read_text())
    rows=json.loads((out/'report.json').read_text())
    if len(rows)!=12 or not all(sha256(ROOT/p)==h for p,h in protocol['input_sha256'].items()):
        raise ValueError('Incomplete series or modified experiment inputs')
    counts={k:dict(correct=sum(r['same_input'][k][v]['correct'] for r in rows for v in ('native','mirrored')),
        total=24,missing=sum(r['same_input'][k][v]['time_s'] is None for r in rows for v in ('native','mirrored')))
        for k in ('original','candidate')}
    # Diagnose the sole smooth clearance failure without changing its verdict.
    failed=[r for r in rows if r['mode']=='candidate_smooth' and not r['checks']['clearance_10cm']]
    failures=[]
    for r in failed:
        folder=out/r['id'];m=json.loads((folder/'manifest.json').read_text());g=m['geometry']
        with np.load(folder/'physical.npz') as a:
            i=int(a['clearance'].argmin());t=float(a['time'][i]);pos=a['pos'][i];yaw=float(a['yaw'][i])
            speed=float(np.linalg.norm(a['velocity'][i,:2]));gap=float(a['clearance'][i])
        distances=[]
        for kind,cls in (('boxes',BoxObstacle),('cylinders',CylinderObstacle)):
            for j,obj in enumerate(g[kind]):
                b=[cls(**obj)] if kind=='boxes' else [];c=[cls(**obj)] if kind=='cylinders' else []
                _,distance=contact_and_clearance(pos,pos,RoomBoundaries(**g['room']),b,c,.2)
                distances.append(dict(kind=kind,index=j,clearance_m=distance,geometry=obj))
        nearest=min(distances,key=lambda d:d['clearance_m'])
        decisions=json.loads((folder/'decisions.json').read_text())
        # Use last actual control before physical minimum, not the next tick.
        d=max((d for d in decisions if d['time_s']<=t),key=lambda d:d['time_s'])
        with np.load(folder/'trace.npz') as a:
            j=int(np.flatnonzero(a['time']<=t)[-1])
            control={k:a[k][j].tolist() for k in a.files}
        failures.append(dict(id=r['id'],time_s=t,position_m=pos.tolist(),yaw_rad=yaw,speed_mps=speed,
            clearance_m=gap,nearest=nearest,preceding_decision=d,preceding_control=control,
            first_phase_start_s=next((d['time_s'] for d in decisions if d.get('phased_event')=='started'),None)))
    write(out/'posthoc_audit.json',dict(scope='Descriptive only; protocol/decision unchanged',
        same_input=counts,smooth_clearance_failures=failures,script_sha256=sha256(__file__),source_unchanged=True))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for ax,name in zip(axes.flat,protocol['scenes']):
        folder=out/(name+'_original_xy')
        with np.load(folder/'same_input.npz') as a:
            onset=next(r for r in rows if r['id']==folder.name)['same_input']['candidate']['native']['threat_onset_s']
            for model in ('original','candidate'):
                ax.plot(a['time'],a[model][:,3]-1500,label=model)
            ax.axvline(onset,color='.5',ls=':',label='first ToF < 1m')
            ax.set(xlim=(max(0,onset-.5),onset+2),ylim=(-300,300),title=name+' | identical original-driver inputs',
                xlabel='time (s)',ylabel='network Yaw - 1500 (PWM)')
            ax.axhline(50,color='.7',ls='--');ax.axhline(-50,color='.7',ls='--');ax.legend(fontsize=8)
    fig.savefig(out/'same_input.png',dpi=150);plt.close(fig)

    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    # Show the remaining neural asymmetry explicitly, including the failed replay.
    folder=out/'b0a9f212_native_candidate_xy'
    with np.load(folder/'same_input.npz') as a,np.load(folder/'mirrored_replay.npz') as b:
        for model in ('original','candidate'):
            axes[0,0].plot(a['time'],a[model][:,3]-1500,label=model)
            axes[0,1].plot(a['time'],b[model][:,3]-1500,label=model)
        for ax,title in zip(axes[0],('Native replay: expected initial negative Yaw','Mirrored replay: expected initial positive Yaw')):
            ax.set(xlim=(4.5,8),ylim=(-200,200),title=title,xlabel='time (s)',ylabel='network Yaw - 1500')
            ax.axhline(50,color='.7',ls='--');ax.axhline(-50,color='.7',ls='--');ax.legend()
    name='9d5431c1_mirror'
    for mode in MODES[1:]:
        with np.load(out/(name+'_'+mode)/'physical.npz') as a:
            axes[1,0].plot(a['time'],a['clearance']*100,label=mode)
    axes[1,0].axhline(10,color='red',ls='--');axes[1,0].set(xlim=(8,16),ylim=(0,45),
        title='Remaining clearance failure',xlabel='time (s)',ylabel='body clearance (cm)');axes[1,0].legend()
    folder=out/(name+'_candidate_smooth')
    with np.load(folder/'trace.npz') as a:
        axes[1,1].plot(a['time'],a['min_tof'],label='held frontal ToF')
    for value,label in ((.85,'BRAKE trigger'),(.1,'body criterion (different quantity)')):
        if value==.85: axes[1,1].axhline(value,color='red',ls='--',label=label)
    axes[1,1].axvline(failures[0]['time_s'],color='black',ls=':',label='minimum body clearance')
    axes[1,1].axvline(failures[0]['first_phase_start_s'],color='orange',ls=':',label='first BRAKE')
    axes[1,1].set(xlim=(8,16),ylim=(0,2),title='Frontal distance does not measure side clearance',xlabel='time (s)',ylabel='ToF (m)');axes[1,1].legend(fontsize=8)
    fig.savefig(out/'remaining_limits.png',dpi=150);plt.close(fig)
    print(json.dumps(dict(same_input=counts,failed=[f['id'] for f in failures]),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out',type=Path)
    report(parser.parse_args().out)
