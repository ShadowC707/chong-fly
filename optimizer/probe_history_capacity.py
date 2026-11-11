"""Bounded overfit test of maneuver-history capacity with a frozen recurrent core.

Diagnostic objective only, not a flight-training recipe: terminal Pitch loss,
direct long-delay training vs a fixed short-to-long curriculum. No state labels
or teacher memory are supplied as network inputs. No deployment promotion.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')

import argparse
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import torch

from artifacts.research import load_research_run
from edge.runtime import sha256
from generator.generate_reflex_dataset import make_teacher
from optimizer.continue_flow_reflex import ROOT,PARENT,DATA,write,export_and_verify
from optimizer.mvp_constraints import RecurrentConstraints
from optimizer.finalize_mvp_candidate import evaluate_phase,save_phase,comparison_row

SUFFIXES=(5,12,25,50,150)


def make_cases(flows, suffix):
    if (type(suffix) is not int or suffix<5 or not flows or
        any(not np.isfinite(f) or not .05<f<=1 for f in flows)):
        raise ValueError('Require flow above turn_full_flow and a positive suffix of at least five ticks')
    xs=[];ys=[];records=[]
    for flow in flows:
        for side in ('left','right'):
            for context in ('always_clear','unsettled','settled'):
                x=np.ones((50+suffix,74),np.float32);x[:,:2]=[flow,0.]
                if context!='always_clear':
                    x[:25,:2]=[.01,0.]
                    grid=x[:25,2:66].reshape(25,8,8)
                    grid[:,:,:4] = .5/3.5 if side=='left' else 1.
                    grid[:,:,4:] = .5/3.5 if side=='right' else 1.
                if context=='settled': x[25:50,:2]=0.
                teacher=make_teacher('geometry-reflex-v5',noise_std_pwm=0)
                xs.append(x);ys.append(np.array([teacher.step(row,dt=.02) for row in x]))
                records.append(dict(flow=float(flow),side=side,context=context))
    return dict(X=torch.from_numpy(np.stack(xs)),Y=torch.from_numpy(np.stack(ys)),records=records)


def endpoint_loss(prediction,target):
    return ((prediction[:,-5:,2]-target[:,-5:,2])/500.).square().mean()


def memoryless_floor(target):
    y=target[:,-5:,2].reshape(-1,3,5)
    return float(((y-y.mean(1,keepdim=True))/500.).square().mean())


@torch.no_grad()
def capacity_metrics(policy,data):
    device=next(policy.parameters()).device
    pred,h=policy(data['X'].to(device),dt=.02); pred=pred.cpu();h=h.cpu()
    error=(pred[:,-5:,2]-data['Y'][:,-5:,2]).abs()
    groups=[]
    for i in range(0,len(h),3):
        groups.append(dict(flow=data['records'][i]['flow'],side=data['records'][i]['side'],
            predicted_pitch=pred[i:i+3,-1,2].tolist(),target_pitch=data['Y'][i:i+3,-1,2].tolist(),
            final_hidden_span_max=float((h[i:i+3].amax(0)-h[i:i+3].amin(0)).max()),
            pitch_span_pwm=float(pred[i:i+3,-1,2].max()-pred[i:i+3,-1,2].min())))
    floor=memoryless_floor(data['Y'])
    loss=float(endpoint_loss(pred,data['Y']))
    return dict(loss=loss,memoryless_mse_floor=floor,loss_over_memoryless_floor=loss/floor,
        max_abs_pitch_error_pwm=float(error.max()),mean_abs_pitch_error_pwm=float(error.mean()),
        within_20_pwm=bool((error<=20).all()),groups=groups)


def prefix_gradient(policy,data):
    """Sensitivity of terminal Pitch to early ToF; no optimizer step."""
    x=data['X'].to(next(policy.parameters()).device).clone().requires_grad_(True)
    pred,_=policy(x,dt=.02)
    # Only unsettled histories, to avoid cancellation between independent cases.
    gradient=torch.autograd.grad(pred[1::3,-1,2].sum(),x)[0]
    return dict(early_tof_max_abs=float(gradient[1::3,:25,2:66].abs().max()),
        latest_flow_max_abs=float(gradient[1::3,-5:,:2].abs().max()))


@torch.no_grad()
def cruise(policy):
    x=torch.ones(1,100,74,device=next(policy.parameters()).device);x[:,:,:2]=0
    y,_=policy(x,dt=.02)
    return dict(final_pwm=y[0,-1].cpu().tolist(),pitch_at_least_1515=bool(y[0,-1,2]>=1515))


def run(output,device):
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    training={n:make_cases((.08,.12),n) for n in SUFFIXES}
    evaluation={n:make_cases((.10,.16),n) for n in SUFFIXES}
    torch.save(dict(training=training,evaluation=evaluation),output/'cases.pt')
    fixed=json.loads((ROOT/'output/neural_refocus_v1/protocol.json').read_text())['fixed_files_sha256']
    assert all(sha256(ROOT/p)==v for p,v in fixed.items())
    sources={str(p.relative_to(ROOT)):sha256(p) for folder in ('core','optimizer','generator','simulation','edge','configs','artifacts') for p in (ROOT/folder).glob('*.py')}
    old=torch.load(DATA/'train.pt',weights_only=True,map_location='cpu')
    _,parent=load_research_run(PARENT);old_info=parent['pretrain_info']
    config=json.loads((PARENT/'report.json').read_text())['config']
    protocol=dict(version='frozen-core-history-capacity-v1',parent_sha256=sha256(PARENT/'checkpoint.pt'),
        cases_sha256=sha256(output/'cases.pt'),source_sha256=sources,fixed_baseline_sha256=fixed,
        train_flows=[.08,.12],evaluation_flows=[.10,.16],suffix_ticks=SUFFIXES,
        arms=dict(direct=[150]*5,curriculum=list(SUFFIXES)),
        settings=dict(updates=300,updates_per_stage=60,full_batch=12,lr=.003,seed=42,device=device,
            objective='Unweighted terminal five-frame Pitch MSE / 500^2; no navigation/anchor loss',
            frozen=['tau_raw','A','_w_macro_param'],fresh_adam=True,tf32=False),
        criteria=dict(capacity='All training terminal frames within 20 PWM at suffix=3s; evaluation reported separately',
            regression='Cruise >=1515 at clear zero-flow; 12 directions/recoveries and <=.8s; narrow amplitude >=80'),
        limits=['Open-loop imposed inputs, not physically driven trajectories.',
            'Teacher latent state is not provided as an input.',
            'Failure is bounded optimization evidence, not a proof for all possible interface weights.',
            'No deployment, no physical trials unless a later explicit experiment admits a candidate.'])
    write(output/'protocol.json',protocol)
    reports={}
    for arm in ('before','direct','curriculum'):
        policy,payload=load_research_run(PARENT)
        constraint=RecurrentConstraints(policy,metadata=payload['training_constraints'])
        policy.to(device);torch.manual_seed(42)
        if device=='cuda':torch.cuda.manual_seed_all(42)
        progress=[]
        if arm!='before':
            params=[p for p in policy.parameters() if p.requires_grad]
            optimizer=torch.optim.Adam(params,lr=.003)
            for stage,suffix in enumerate(protocol['arms'][arm]):
                data=training[suffix];x=data['X'].to(device);y=data['Y'].to(device)
                policy.train()
                for _ in range(60):
                    optimizer.zero_grad(set_to_none=True)
                    pred,_=policy(x,dt=.02);loss=endpoint_loss(pred,y)
                    if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                    loss.backward()
                    if not all(p.grad is None or torch.isfinite(p.grad).all() for p in params):
                        raise ValueError('Nonfinite gradient')
                    optimizer.step();policy.post_step();constraint(policy)
                policy.eval()
                row=dict(update=constraint.steps,suffix_ticks=suffix,
                    stage_training=capacity_metrics(policy,data),long_training=capacity_metrics(policy,training[150]))
                progress.append(row);write(output/(arm+'_progress.json'),progress)
                print(arm,'step',constraint.steps,'suffix',suffix,'stage error',row['stage_training']['max_abs_pitch_error_pwm'],
                      '3s error',row['long_training']['max_abs_pitch_error_pwm'],flush=True)
            assert constraint.steps==300
        policy.eval()
        capacity={label:{str(n):capacity_metrics(policy,d) for n,d in collection.items()}
            for label,collection in (('training',training),('evaluation',evaluation))}
        gradient={str(n):prefix_gradient(policy,training[n]) for n in (5,150)}
        quiet=cruise(policy)
        policy.cpu().eval()
        metrics=evaluate_phase(policy,old,old_info,config)
        if arm!='before':
            payload['initial_pretrain_info']=deepcopy(payload['pretrain_info'])
            payload['pretrain_info']=dict(experiment='frozen-core-history-capacity-v1',
                dataset_sha256=protocol['cases_sha256'],subset_indices=list(range(12)),validation_indices=[],
                loss_mode='terminal-pitch-mse-v1',updates=300,research_only=True)
        payload['continuation']=dict(experiment='frozen-core-history-capacity-v1',arm=arm,
            parent_sha256=protocol['parent_sha256'],updates=constraint.steps,diagnostic_only=True)
        report=dict(candidate=payload['candidate'],config=config,
            continuation_epochs=0 if arm=='before' else None,continuation_updates=constraint.steps,
            capacity=capacity,gradient=gradient,cruise=quiet,progress=progress,**metrics,
            research_only=True,training_ready=False)
        sample=old['X'][old_info['validation_indices'][:2]]
        report=save_phase(output/arm,policy=policy,checkpoint=payload,parent_directory=PARENT,
            report=report,constraints=constraint,sample=sample)
        export_and_verify(output/arm,sample)
        reports[arm]=dict(capacity=capacity,gradient=gradient,cruise=quiet,temporal=comparison_row(report),
            updates=constraint.steps,diagnostic_fit=capacity['training']['150']['within_20_pwm'])
        write(output/'comparison.json',reports)
        print(arm,'finished',dict(fit=reports[arm]['diagnostic_fit'],cruise=quiet,
            temporal=reports[arm]['temporal']),flush=True)
        del policy
        if device=='cuda':torch.cuda.empty_cache()
    assert all(sha256(ROOT/p)==v for m in (fixed,sources) for p,v in m.items())
    assert sha256(output/'cases.pt')==protocol['cases_sha256']
    write(output/'validation.json',dict(completed_arms=3,source_and_input_hashes_unchanged=True,
        active_baseline_unchanged=True,frozen_core_verified=True,roundtrip_and_numpy_verified=True,
        automatic_promotion=False,physical_trials=0))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',choices=('cuda','cpu'),required=True)
    args=p.parse_args();run(args.out,args.device)
