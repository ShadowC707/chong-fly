"""Summarize the fixed teacher check and new XY sequences; never train a model."""
import argparse
import json
from pathlib import Path
import numpy as np


def render(teacher_dir, dataset_dir, output):
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle,Circle
    from artifacts.io import write_json
    teacher_dir, dataset_dir, output = map(Path,(teacher_dir,dataset_dir,output))
    output.mkdir(parents=True,exist_ok=False)
    reports=json.loads((teacher_dir/'report.json').read_text())
    data=torch.load(dataset_dir/'train.pt',weights_only=True,map_location='cpu')
    audit=json.loads((dataset_dir/'train_report.json').read_text())['audit']
    fig,axes=plt.subplots(4,2,figsize=(10,15),layout='constrained')
    for ax,ep in zip(axes.flat,(0,1,10,11,20,21,30,31)):
        record=data['metadata']['episodes'][ep]; geometry=record['scene_parameters']
        for b in geometry['boxes']:
            ax.add_patch(Rectangle((b['x_min'],b['y_min']),b['x_max']-b['x_min'],b['y_max']-b['y_min'],color='.8'))
        for c in geometry['cylinders']:
            ax.add_patch(Circle((c['center_x'],c['center_y']),c['radius'],color='.8'))
        v=data['valid'][ep].numpy()
        p=data['diagnostics']['position'][ep].numpy()[v]
        ax.plot(p[:,0],p[:,1],lw=1.5)
        ax.scatter(*p[0,:2],marker='s',color='green',s=25)
        ax.scatter(*record['final_position_m'][:2],marker='x' if record['crashed'] else 'o',color='red' if record['crashed'] else 'blue',s=25)
        ax.set(xlim=(-6,6),ylim=(-6,6),xlabel='X (m)',ylabel='Y (m)',
            title=f"{record['family']} / {record['motion']} / side {record['side']}\n12s; body radius 0.20m; teacher-driven")
        ax.set_aspect('equal'); ax.grid(alpha=.2)
    fig.suptitle('New XY sequences: actual physics, v5 teacher, no XY coordinator\nMirrored geometry; labels recalculated, not manually mirrored')
    fig.savefig(output/'sequence_examples.png',dpi=140);plt.close(fig)

    with np.load(teacher_dir/'068fe108_before/decisions.npz') as a:
        t=a['time']; x=a['X']; ys={k:a[k].copy() for k in ('before','control','active_fast','teacher')}
    fig,axes=plt.subplots(3,1,figsize=(11,9),sharex=True,layout='constrained')
    for k,y in ys.items():
        axes[0].plot(t,y[:,2],label=k)
        axes[1].plot(t,y[:,3],label=k)
    axes[0].set(ylabel='Pitch PWM',title='Same inputs / recorded baseline history; observer commands do not drive this path')
    axes[1].set(ylabel='Yaw PWM')
    axes[0].legend(ncol=4)
    axes[2].plot(t,x[:,2:66].min(1)*3.5,color='black',label='min ToF (m)')
    axes[2].axhline(.8,color='.6',ls='--',label='threat 0.8m')
    axes[2].set(ylabel='ToF (m)',xlabel='Time (s)')
    flowax=axes[2].twinx(); flowax.plot(t,np.linalg.norm(x[:,:2],axis=1),color='purple',label='flow norm')
    flowax.set_ylabel('Normalized flow',color='purple')
    axes[2].legend(loc='upper right');flowax.legend(loc='upper left')
    for ax in axes: ax.grid(alpha=.2)
    fig.savefig(output/'common_input_decisions.png',dpi=150);plt.close(fig)

    common={}
    for case in reports:
        with np.load(teacher_dir/(case+'_before')/'decisions.npz') as a:
            distance=a['X'][:,2:66].min(1)*3.5
            # Current teacher state already incorporates the actual reset history.
            mask=(a['teacher'][:,2]<1495)&(distance>=.8)
            common[case]=dict(teacher_brake_with_clear_tof_frames=int(mask.sum()),
                network_cruise_during_teacher_brake={k:int((mask&(a[k][:,2]>1510)).sum())
                    for k in ('before','control','active_fast')})
    groups={}
    for family in sorted({r['family'] for r in audit['episodes']}):
        rows=[r for r in audit['episodes'] if r['family']==family]
        groups[family]=dict(episodes=len(rows),completed=sum(r['completed'] for r in rows),
            turn_bursts=sum(r['turn_bursts'] for r in rows),
            left_frames=sum(r['left_frames'] for r in rows),right_frames=sum(r['right_frames'] for r in rows),
            brake_after_clear_frames=sum(r['brake_after_clear_frames'] for r in rows))
    write_json(output/'summary.json',dict(teacher_check={case:{k:r['metrics'] for k,r in arms.items()}
        for case,arms in reports.items()},common_input=common,training_groups=groups,
        limitation='Whole-teacher replacement is not causal isolation of one channel. Known scenes, one seed each.'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--teacher',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();render(a.teacher,a.data,a.out)
