"""Read-only paired replay and input interventions for the residual turn failure."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def mirror_observations(x):
    x=np.asarray(x)
    if x.shape[-1]!=74: raise ValueError('Expected 74 input channels')
    y=x.copy();y[...,1]=-x[...,1]
    y[...,2:66]=x[...,2:66].reshape(*x.shape[:-1],8,8)[...,::-1].reshape(*x.shape[:-1],64)
    y[...,66:]=x[...,66:][...,[0,7,6,5,4,3,2,1]]
    return y


def replay(policy,x,segment,*,teacher=False):
    if len(x)!=len(segment): raise ValueError('Unaligned segments')
    result=[]
    for i,row in enumerate(x):
        if i==0 or segment[i]!=segment[i-1]:
            (policy.reset if teacher else policy.reset_state)()
        result.append(policy.step(row,dt=.02) if teacher else policy.step_np(row[:2],row[2:66],row[66:],dt=.02))
    return np.asarray(result)


def threat_groups(x,valid):
    grid=x[...,2:66].reshape(*x.shape[:-1],8,8)*3.5
    left=grid[...,:,:4].min(axis=(-2,-1))<.8
    right=grid[...,4:].min(axis=(-2,-1))<.8
    slow=np.linalg.norm(x[...,:2],axis=-1)<=.05
    return dict(active=valid&(left|right),slow_active=valid&slow&(left|right),
        slow_unilateral=valid&slow&(left^right),slow_bilateral=valid&slow&left&right,
        fast_active=valid&~slow&(left|right))


def metrics(y,target,mask):
    delta=y[...,3]-1500;td=target[...,3]-1500
    strong=mask&(np.abs(td)>=80)
    return dict(frames=int(mask.sum()),strong_target_frames=int(strong.sum()),
        weak_network_frames=int((strong&(np.abs(delta)<50)).sum()),
        opposite_network_frames=int((strong&(delta*td<0)).sum()),
        signed_amplitude_ge80_frames=int((strong&(delta*np.sign(td)>=80)).sum()),
        yaw_mae_pwm=float(np.abs(delta[mask]-td[mask]).mean()) if mask.any() else None,
        mean_abs_network_yaw=float(np.abs(delta[strong]).mean()) if strong.any() else None,
        mean_abs_teacher_yaw=float(np.abs(td[strong]).mean()) if strong.any() else None)


def run(out):
    import torch
    from artifacts.research import load_research_run
    from edge.runtime import NumpyPolicy,sha256
    from generator.generate_reflex_dataset import make_teacher
    from optimizer.continue_flow_reflex import write
    from optimizer.navigation_loss import navigation_weights
    torch.set_num_threads(1)
    source=ROOT/'output/release_cruise_v1/release_encoding'
    flight=source/'flights/9d5431c1'
    dataset=ROOT/'output/balanced_maneuver_v1/train.pt'
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    files=[source/'checkpoint.pt',source/'numpy/metadata.json',flight/'decisions.npz',flight/'trace.npz',dataset,
        ROOT/'configs/mvp_runtime_v3.json',*[p for p in (ROOT/'data/edge/k306_tau025_integrated_v1').iterdir() if p.is_file()]]
    hashes={str(p):sha256(p) for p in files}
    write(out/'protocol.json',dict(version='active-turn-diagnosis-v1',input_sha256=hashes,source_sha256=sha256(__file__),
        replay='Full recorded decision history, original segment resets; original and reflected sensor inputs',
        mirror='Reverse ToF columns, negate flow_y, reverse ring sectors around front; recompute teacher labels',
        masks='Active ToF <0.8m; slow flow norm <=0.05; strong teacher yaw >=80 PWM; weak network yaw <50 PWM',
        static=dict(times_s=[34.,35.,36.],duration_s=2.,variants=['original','zero_flow','left_only','right_only']),
        limits='Offline input interventions do not generate physical counterfactual trajectories; one known failure; no training or promotion'))
    with np.load(flight/'decisions.npz') as a: recorded={k:a[k].copy() for k in a.files}
    x=recorded['X'];segment=recorded['segment'];time=recorded['time'];auto=recorded['auto']
    policy=NumpyPolicy(source/'numpy')
    def teacher(): return make_teacher('geometry-reflex-v5',noise_std_pwm=0,pwm_cruise_pitch=1530.)
    predictions={};targets={};stats={}
    for name,inputs in [('original',x),('mirrored',mirror_observations(x))]:
        y=replay(policy,inputs,segment);target=replay(teacher(),inputs,segment,teacher=True)
        if name=='original':
            np.testing.assert_array_equal(y,recorded['candidate'])
            np.testing.assert_array_equal(target,recorded['teacher'])
        predictions[name]=y;targets[name]=target
        groups=threat_groups(inputs,auto)
        groups['terminal_slow_active']=groups['slow_active']&(time>=34.)
        stats[name]={key:metrics(y,target,mask) for key,mask in groups.items()}
        np.savez_compressed(out/(name+'.npz'),X=inputs,time=time,segment=segment,auto=auto,network=y,teacher=target)
    np.savez_compressed(out/'applied_reference.npz',**{k:v for k,v in np.load(flight/'trace.npz').items()})
    symmetry=dict(network_yaw_mirror_residual_max_pwm=float(np.max(np.abs(predictions['original'][:,3]+predictions['mirrored'][:,3]-3000))),
        teacher_yaw_mirror_residual_max_pwm=float(np.max(np.abs(targets['original'][:,3]+targets['mirrored'][:,3]-3000))),
        note='Symmetric ties may legitimately select the same teacher turn; mirror equality is measured, not assumed.')
    static=[]
    for t in (34.,35.,36.):
        i=int(np.argmin(abs(time-t)))
        for side in ('original','mirrored'):
            base=x[i].copy() if side=='original' else mirror_observations(x[i])
            for variant in ('original','zero_flow','left_only','right_only'):
                obs=base.copy()
                if variant=='zero_flow':obs[:2]=0
                if variant=='left_only':obs[2:66].reshape(8,8)[:,4:]=1
                if variant=='right_only':obs[2:66].reshape(8,8)[:,:4]=1
                inputs=np.repeat(obs[None],100,axis=0);seg=np.zeros(100,int)
                y=replay(policy,inputs,seg);target=replay(teacher(),inputs,seg,teacher=True)
                static.append(dict(time_s=float(time[i]),side=side,variant=variant,
                    network_final_pwm=y[-1].tolist(),fresh_teacher_final_pwm=target[-1].tolist(),
                    recorded_network_pwm=predictions[side][i].tolist(),recorded_teacher_pwm=targets[side][i].tolist()))
    write(out/'static_interventions.json',static)
    # Dataset inspection uses original, unmodified labels, including the exact training weights.
    train=torch.load(dataset,weights_only=True,map_location='cpu')
    model,_=load_research_run(source)
    with torch.no_grad():
        pred=np.concatenate([model(train['X'][start:start+8],dt=.02)[0].numpy() for start in range(0,len(train['X']),8)])
    valid=train['valid'].numpy();target=train['Y'].numpy();inputs=train['X'].numpy()
    weights=navigation_weights(train['Y'],train['valid'],train['behavior'],len(train['metadata']['behavior_names']),
        yaw_balance=.7,amplitude_bins=True,recovery_steps=50).numpy()[...,2]
    coverage={}
    for name,mask in threat_groups(inputs,valid).items():
        row=metrics(pred,target,mask)
        row.update(episodes_with_frames=int(mask.any(axis=1).sum()),
            valid_frame_fraction=float(mask.sum()/valid.sum()),
            yaw_weight_mass_fraction=float(weights[mask].sum()/weights.sum()))
        coverage[name]=row
    write(out/'report.json',dict(replay_exact=True,trace=stats,symmetry=symmetry,training_coverage=coverage,
        no_parameter_changes=True,limitations='No causal collision attribution; static inputs are diagnostic, not training additions'))
    assert all(sha256(Path(p))==h for p,h in hashes.items())
    write(out/'validation.json',dict(replay_exact=True,input_and_active_bundle_unchanged=True,promotion=False))
    plot(out)
    print(json.dumps(dict(trace=stats,symmetry=symmetry,training_coverage=coverage),indent=2))


def plot(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(12,9),sharex=True,layout='constrained')
    for name,style in [('original','-'),('mirrored','--')]:
        with np.load(out/(name+'.npz')) as a:
            sign=1 if name=='original' else -1
            axes[0].plot(a['time'],sign*(a['network'][:,3]-1500),style,label=name+' network')
            axes[0].plot(a['time'],sign*(a['teacher'][:,3]-1500),style,alpha=.65,label=name+' teacher')
            axes[2].plot(a['time'],a['X'][:,2:66].min(axis=1)*3.5,style,label=name+' min ToF')
    with np.load(out/'applied_reference.npz') as a:
        axes[1].plot(a['time'],a['network'][:,3]-1500,label='network yaw')
        axes[1].plot(a['time'],a['coordinated'][:,3]-1500,label='coordinated yaw')
        axes[1].plot(a['time'],a['applied'][:,3]-1500,ls=':',label='applied yaw')
        axes[2].plot(a['time'],a['clearance'],label='physical body clearance (original only)')
    axes[0].set(ylabel='Yaw offset (PWM)',title='Original and mirrored input replay; mirrored yaw sign normalized for comparison')
    axes[1].set(ylabel='Yaw offset (PWM)',title='Original recorded control stages; mirrored replay has no physical trajectory')
    axes[2].axhline(.8,color='.4',ls=':',label='active threat threshold')
    axes[2].set(ylabel='Distance (m)',xlabel='Recorded time (s)',xlim=(30,36.57))
    for ax in axes:ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.savefig(out/'terminal_replay.png',dpi=150);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    run(parser.parse_args().out)
