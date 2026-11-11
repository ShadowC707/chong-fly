"""Fixed three-part objective: recorded maneuvers, history, and free cruise."""
import math
import numpy as np
import torch
from generator.generate_reflex_dataset import make_teacher
from optimizer.probe_history_capacity import make_cases
from optimizer.history_objective import sequence_pitch_loss, sequence_metrics
from optimizer.navigation_loss import navigation_weights, navigation_loss

STAGES=(12,25,50,100,200)
TRAIN_FLOWS=(.06,.10,.14,.18)
EVAL_FLOWS=(.08,.12,.16,.20)
SETTINGS=dict(updates=300,updates_per_stage=60,seed=42,batch_size=8,
    lr_interface=.001,lr_W=.001,lr_tau=.01,gradient_clip=1.,
    loss='channel-v4 recorded + full-history Pitch MSE + 10 * cruise navigation MSE; PWM errors /500',
    history_suffix_ticks=list(STAGES),history_train_flows=list(TRAIN_FLOWS),
    history_eval_flows=list(EVAL_FLOWS),history_eval_suffix_ticks=[150,400],cruise_pitch=1530.)


def teacher_targets(x):
    y=torch.empty((*x.shape[:2],4))
    for i,episode in enumerate(x):
        teacher=make_teacher('geometry-reflex-v5',noise_std_pwm=0,pwm_cruise_pitch=1530.)
        for t,row in enumerate(episode):y[i,t]=torch.from_numpy(teacher.step(row.numpy(),dt=.02))
    return y


def auxiliary_cases(suffix, flows=TRAIN_FLOWS):
    history=make_cases(flows,suffix)
    history['Y']=teacher_targets(history['X'])
    flow=torch.tensor([(0,0),(.01,0),(-.01,0),(.03,0),(.06,0),(.12,0),(-.06,0),(0,.03),(0,-.03)])
    x=torch.ones(len(flow),200,74);x[:,:,:2]=flow[:,None,:]
    return history,dict(X=x,Y=teacher_targets(x))


def behavioral_admission(result, *, clear_speed, history_error):
    t=result['temporal'];pitch=result['cruise']['final_pwm'][2]
    checks=dict(scenes=t['scenes']==7,direction=t['direction']==12,
        recovery=t['recovery']==12 and t['max_recovery_s'] is not None and t['max_recovery_s']<=.8,
        narrow_amplitude=t['narrow_amplitude']==8,handover=result['handover']['passed'],
        cruise_pitch=1515<=pitch<=1535,
        clear_forward_speed=math.isfinite(clear_speed) and clear_speed>=.10,
        history_suffix=math.isfinite(history_error) and history_error<=20.)
    return dict(passed=all(checks.values()),checks=checks,reasons=[k for k,v in checks.items() if not v],
        scope='Research screen only; no automatic deployment or full handover claim')


def fit(policy,data,guard,auxiliary,*,progress_callback):
    device=next(policy.parameters()).device
    x,y,valid,behavior=(data[k] for k in ('X','Y','valid','behavior'))
    weights=navigation_weights(y,valid,behavior,len(data['metadata']['behavior_names']),
        yaw_balance=.7,amplitude_bins=True,recovery_steps=50).to(device)
    normalizer_per_episode=float(valid.sum())/len(x)*3
    x,y=x.to(device),y.to(device)
    interface=[p for n,p in policy.named_parameters() if p.requires_grad and not n.endswith(('tau_raw','_w_macro_param'))]
    groups=[dict(params=interface,lr=.001)]
    for name,lr in (('_w_macro_param',.001),('tau_raw',.01)):
        p=getattr(policy.cfc_network.cell,name)
        if p.requires_grad:groups.append(dict(params=[p],lr=lr))
    optimizer=torch.optim.Adam(groups);rng=np.random.default_rng(42)
    order=rng.permutation(len(x));cursor=0;progress=[]
    cruise=auxiliary['cruise'];cx,cy=cruise['X'].to(device),cruise['Y'].to(device)
    for suffix in STAGES:
        h=auxiliary['training'][suffix];hx,hy=h['X'].to(device),h['Y'].to(device)
        sums=np.zeros(3)
        for _ in range(60):
            if cursor==len(order):order=rng.permutation(len(x));cursor=0
            ids=order[cursor:cursor+8];cursor+=len(ids)
            policy.train();optimizer.zero_grad(set_to_none=True)
            pred=policy(x[ids],dt=.02)[0]
            recorded=navigation_loss(pred,y[ids],weights[ids],normalizer=normalizer_per_episode*len(ids))
            hist=sequence_pitch_loss(policy(hx,dt=.02)[0],hy)
            quiet=((policy(cx,dt=.02)[0][...,1:]-cy[...,1:])/500.).square().mean()
            loss=recorded+hist+10*quiet
            if not torch.isfinite(loss):raise ValueError('Nonfinite mixed objective')
            loss.backward();torch.nn.utils.clip_grad_norm_(policy.parameters(),1.,error_if_nonfinite=True)
            optimizer.step();policy.post_step();guard(policy)
            sums += [float(v.detach()) for v in (recorded,hist,quiet)]
        policy.eval()
        with torch.no_grad():
            row=dict(updates=guard.steps,suffix=suffix,mean_losses=(sums/60).tolist(),
                sequence=sequence_metrics(policy(hx,dt=.02)[0],hy),
                cruise_final_pwm=policy(cx[:1],dt=.02)[0][0,-1].cpu().tolist())
        progress.append(row);progress_callback(progress)
    return dict(settings=SETTINGS,progress=progress,validation_indices=[],
        source_episode_count=len(data['X']),note='Fixed all-episode permutation stream; one batch + all auxiliary cases per update')
