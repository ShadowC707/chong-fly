"""Read-only local gradient audit; no optimizer or parameter update."""
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
from optimizer.continue_flow_reflex import ROOT,write
from optimizer.probe_plastic_history import PlasticConstraints
from optimizer.navigation_loss import navigation_weights,navigation_loss
from optimizer.history_objective import sequence_pitch_loss
from optimizer.functional_probe import make_stimuli,WARMUP,ACTIVE


def alignment(a,b):
    dot=float(torch.dot(a,b));na=float(a.norm());nb=float(b.norm())
    return dict(dot=dot,norm_a=na,norm_b=nb,cosine=dot/(na*nb) if na>1e-14 and nb>1e-14 else None)


def descent_effect(target,direction):
    """First derivative of target loss along theta - epsilon*direction."""
    return -float(torch.dot(target,direction))


def capture_state(policy):
    return {n:v.detach().cpu().clone() if isinstance(v,torch.Tensor) else deepcopy(v)
            for n,v in policy.state_dict().items()}


def unchanged_state(policy,snapshot):
    state=policy.state_dict()
    return state.keys()==snapshot.keys() and all(
        torch.equal(v.detach().cpu(),snapshot[n]) if isinstance(v,torch.Tensor) else v==snapshot[n]
        for n,v in state.items())


def pack_gradient(loss,params):
    grads=torch.autograd.grad(loss,params,allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).detach().cpu().double().flatten()
        for p,g in zip(params,grads)])


def group_indices(named):
    groups={'all':[],'W':[],'tau':[],'sensor_input':[],'decoder':[]};offset=0
    for name,p in named:
        group='W' if name.endswith('_w_macro_param') else 'tau' if name.endswith('tau_raw') else 'decoder' if name.startswith('dn_head.') else 'sensor_input'
        ids=list(range(offset,offset+p.numel()));groups[group]+=ids;groups['all']+=ids;offset+=p.numel()
    return {k:torch.tensor(v,dtype=torch.long) for k,v in groups.items()}


def run(source,out,device):
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    source,out=Path(source),Path(out);out.mkdir(parents=True,exist_ok=False)
    paths=[source/'train.pt',source/'auxiliary.pt',source/'protocol.json']
    paths += [source/a/'checkpoint.pt' for a in ('before','plastic')]
    hashes={str(p):sha256(p) for p in paths}
    baseline=json.loads((source/'protocol.json').read_text())['fixed_baseline_sha256']
    assert all(sha256(ROOT/p)==v for p,v in baseline.items())
    train=torch.load(source/'train.pt',weights_only=True,map_location='cpu')
    aux=torch.load(source/'auxiliary.pt',weights_only=True,map_location='cpu')
    weights=navigation_weights(train['Y'],train['valid'],train['behavior'],len(train['metadata']['behavior_names']),
        yaw_balance=.7,amplitude_bins=True,recovery_steps=50)
    write(out/'protocol.json',dict(version='maneuver-gradient-audit-v1',inputs_sha256=hashes,
        source_sha256=sha256(Path(__file__)),device=device,
        methods='Exact full training objective gradient (104 episodes), last-stage 24 histories, cruise weighted by10; no updates',
        hypothesis='Opposing objectives vs insufficient history fitting; gradient alignment is local evidence only',
        limits='Raw Euclidean gradients, not Adam-preconditioned, clipped, or constraint-projected steps. No causal proof of training history.'))
    results={}
    for arm in ('before','plastic'):
        p,_=load_research_run(source/arm);p.to(device);p.eval()
        PlasticConstraints(p) # enable exactly the plastic training parameters, no call/project/update
        snapshot=capture_state(p)
        named=[(n,v) for n,v in p.named_parameters() if v.requires_grad];params=[v for _,v in named]
        groups=group_indices(named);grads={};losses={}
        whole=torch.zeros(sum(v.numel() for v in params),dtype=torch.float64);value=0.
        for start in range(0,len(train['X']),8):
            ids=slice(start,start+8)
            pred=p(train['X'][ids].to(device),dt=.02)[0]
            loss=navigation_loss(pred,train['Y'][ids].to(device),weights[ids].to(device),
                normalizer=float(train['valid'].sum())*3)
            value+=float(loss.detach());whole+=pack_gradient(loss,params)
        grads['recorded']=whole;losses['recorded']=value
        history=aux['training'][200]
        for label,ids in [('history',slice(None)),('clear_history',slice(0,None,3)),
                          ('unsettled_history',slice(1,None,3)),('settled_history',slice(2,None,3))]:
            pred=p(history['X'][ids].to(device),dt=.02)[0]
            loss=sequence_pitch_loss(pred,history['Y'][ids].to(device))
            losses[label]=float(loss.detach());grads[label]=pack_gradient(loss,params)
        x,y=aux['cruise']['X'].to(device),aux['cruise']['Y'].to(device)
        loss=10*((p(x,dt=.02)[0][...,1:]-y[...,1:])/500.).square().mean()
        losses['cruise_x10']=float(loss.detach());grads['cruise_x10']=pack_gradient(loss,params)
        cases,x=make_stimuli();idx=next(i for i,c in enumerate(cases) if c['side']=='left' and c['normalized_distance']==.2 and c['width']==1)
        yaw=p(x[idx:idx+1].to(device),dt=.02)[0][0,WARMUP+ACTIVE-1,3]
        narrow=float((yaw-1500).detach())
        loss=(torch.relu(80-(yaw-1500))/500.).square()
        losses['narrow_repair']=float(loss.detach());grads['narrow_repair']=pack_gradient(loss,params)
        grads['total']=grads['recorded']+grads['history']+grads['cruise_x10']
        decomposition=(grads['history']-(grads['clear_history']+grads['unsettled_history']+grads['settled_history'])/3).abs().max()
        pair_names=[('recorded','history'),('history','cruise_x10'),('settled_history','unsettled_history'),
            ('settled_history','recorded'),('settled_history','cruise_x10'),('narrow_repair','total'),
            ('narrow_repair','recorded'),('narrow_repair','history'),('narrow_repair','cruise_x10')]
        group_report={}
        for name,ids in groups.items():
            group_report[name]=dict(norms={k:float(g[ids].norm()) for k,g in grads.items()},
                alignments={a+'__'+b:alignment(grads[a][ids],grads[b][ids]) for a,b in pair_names},
                along_total_descent={k:descent_effect(g[ids],grads['total'][ids]) for k,g in grads.items() if k!='total'})
        assert unchanged_state(p,snapshot)
        results[arm]=dict(losses=losses,narrow_signed_pwm=narrow,groups=group_report,
            history_gradient_decomposition_max_error=float(decomposition),parameters_unchanged=True,
            parameters=[dict(name=n,shape=list(v.shape)) for n,v in named])
        np.savez_compressed(out/(arm+'_gradients.npz'),**{k:v.numpy() for k,v in grads.items()})
        write(out/'report.json',results)
        print(arm,'losses',losses,'cosines',{k:v['cosine'] for k,v in group_report['all']['alignments'].items()},flush=True)
    assert all(sha256(Path(p))==v for p,v in hashes.items())
    assert all(sha256(ROOT/p)==v for p,v in baseline.items())
    write(out/'validation.json',dict(checkpoints_and_inputs_unchanged=True,active_baseline_unchanged=True,
        no_optimizer_steps=True,no_parameter_updates=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'output/balanced_maneuver_v1')
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--device',choices=('cpu','cuda'),required=True)
    args=parser.parse_args();run(args.source,args.out,args.device)
