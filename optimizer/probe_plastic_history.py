"""Bounded existing-edge W/tau adaptation with reference-action anchors.

Research only. No added synapses, sensor channels or external maneuver state.
Both arms use identical curriculum and loss; only one may adapt core W/tau.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')

import argparse
from copy import deepcopy
import json
from pathlib import Path
import torch

from core.contracts import TAU_FLOOR_S
from core.invariants import verify_recurrent_contract
from optimizer.mvp_constraints import RecurrentConstraints,verify_clear_drive
from optimizer.probe_history_capacity import make_cases,capacity_metrics,endpoint_loss,cruise,prefix_gradient,SUFFIXES
from optimizer.continue_flow_reflex import ROOT,PARENT,DATA,write,export_and_verify
from optimizer.finalize_mvp_candidate import evaluate_phase,save_phase,comparison_row
from artifacts.research import load_research_run
from edge.runtime import sha256
from optimizer.history_objective import sequence_pitch_loss, sequence_metrics


class PlasticConstraints:
    def __init__(self,policy):
        self.cell=policy.cfc_network.cell
        if not policy.neutral_origin or not self.cell.preserve_signs or self.cell.mode!='masked':
            raise ValueError('Requires zero-origin masked signed graph')
        verify_recurrent_contract(self.cell);verify_clear_drive(policy)
        self.amplitude=self.cell.A.detach().clone()
        self.weight=self.cell._w_macro_param.detach().clone()
        self.tau=self.cell.tau.detach().clone()
        self.w_max_norm=float(self.cell._effective_W().detach().norm())
        self.cell.A.requires_grad_(False);self.cell.A.grad=None
        self.cell.tau_raw.requires_grad_(True);self.cell._w_macro_param.requires_grad_(True)
        self.steps=0;self.projections=0

    @torch.no_grad()
    def __call__(self,policy):
        c=policy.cfc_network.cell
        if c is not self.cell or c.A.requires_grad or not torch.equal(c.A,self.amplitude.to(c.A)):
            raise ValueError('Frozen amplitude changed')
        verify_recurrent_contract(c)
        if not torch.isfinite(c.tau_raw).all():raise ValueError('Nonfinite tau')
        bounds=torch.tensor([.005,2.],device=c.tau_raw.device,dtype=c.tau_raw.dtype)-TAU_FLOOR_S
        raw=bounds+torch.log(-torch.expm1(-bounds))
        c.tau_raw.clamp_(min=raw[0],max=raw[1])
        norm=float(c._effective_W().norm())
        if norm>self.w_max_norm:
            c._w_macro_param.mul_((self.w_max_norm/norm)*(1-4*torch.finfo(c.A.dtype).eps))
            self.projections+=1
        verify_clear_drive(policy)
        self.steps+=1

    def audit(self):
        c=self.cell
        return dict(version='plastic-existing-core-v1',optimizer_steps=self.steps,
            frozen_parameters=['A'],tau_bounds_s=[.005,2.],w_max_norm=self.w_max_norm,
            final_norm=float(c._effective_W().detach().norm()),norm_projections=self.projections,
            A_unchanged=torch.equal(c.A.detach(),self.amplitude.to(c.A)),
            W_delta_norm=float((c._w_macro_param.detach()-self.weight.to(c.A)).norm()),
            final_tau_min_s=float(c.tau.detach().min()),final_tau_max_s=float(c.tau.detach().max()),
            max_tau_change_s=float((c.tau.detach()-self.tau.to(c.A)).abs().max()),
            topology_and_signs='Verified after every native post_step; weights may become zero')


def anchor_inputs():
    x=torch.ones(12,200,74);x[:,:,:2]=0
    for i in range(8):
        width=(1,4,1,4)[i%4];distance=(.5,.5,.7,.7)[i%4]
        grid=x[i,25:75,2:66].reshape(50,8,8)
        if i<4:grid[:,:,:width]=distance/3.5
        else:grid[:,:,8-width:]=distance/3.5
    x[9,:,0]=.12;x[10,:,0]=-.12;x[11,:,1]=.08
    return x


def protected_loss(prediction,target):
    diff=(prediction-target)/500.
    weights=diff.new_tensor([1.,1.,4.])
    imitation=(diff[...,1:].square()*weights).mean()
    # Extra late recovery loss, same for both arms; avoids forcing state reset.
    recovery=diff[:8,125:,3].square().mean()
    return imitation+recovery


def run(output,device,objective='endpoint'):
    if objective not in ('endpoint','sequence'):raise ValueError('Unknown history objective')
    history_loss=endpoint_loss if objective=='endpoint' else sequence_pitch_loss
    experiment='protected-plastic-history-v1' if objective=='endpoint' else 'sequence-plastic-history-v1'
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    original,payload=load_research_run(PARENT);original.eval()
    anchors=anchor_inputs()
    with torch.no_grad():anchor_target=original(anchors,dt=.02)[0]
    training={n:make_cases((.08,.12),n) for n in SUFFIXES}
    evaluation={n:make_cases((.10,.16),n) for n in SUFFIXES}
    duration_holdout={n:make_cases((.10,.16),n) for n in (75,100,225,400)} if objective=='sequence' else {}
    torch.save(dict(training=training,evaluation=evaluation,duration_holdout=duration_holdout,
        anchors=anchors,anchor_target=anchor_target),out/'cases.pt')
    fixed=json.loads((ROOT/'output/neural_refocus_v1/protocol.json').read_text())['fixed_files_sha256']
    assert all(sha256(ROOT/p)==v for p,v in fixed.items())
    sources={str(p.relative_to(ROOT)):sha256(p) for folder in ('optimizer','core','generator','simulation','edge','configs','artifacts') for p in (ROOT/folder).glob('*.py')}
    protocol=dict(version=experiment,parent_sha256=sha256(PARENT/'checkpoint.pt'),
        cases_sha256=sha256(out/'cases.pt'),source_sha256=sources,fixed_baseline_sha256=fixed,
        settings=dict(arms=['frozen','plastic'],seed=42,updates=300,updates_per_stage=60,
            suffix_ticks=list(SUFFIXES),lr_interface=.003,lr_W=.003,lr_tau=.01,
            tau_bounds_s=[.005,2.],W_frobenius_max=float(original.cfc_network.cell._effective_W().detach().norm()),
            A_frozen=True,signs_preserved=True,new_edges=False,device=device,tf32=False,
            objective=objective+'_pitch_mse + protected_anchor_loss; exact implementation hashed',
            anchors='Frozen parent actions, 8 threat pulses and 4 clear/flow histories; not teacher labels'),
        criteria=dict(capacity='All 3s terminal training errors <=20 PWM, evaluation separately',
            regression='Cruise >=1515, direction/recovery=12, narrow=8, recovery<=.8s',
            promotion='No automatic promotion or physical flights from this diagnostic'),
        duration_holdout_ticks=list(duration_holdout),
        sequence_criteria='At 3s: all suffix errors <=20 PWM; report all-frame errors, unseen durations and physical regressions',
        caveat='Joint W/tau intervention; no causal attribution to one parameter. Anchors are partial, not a safety guarantee.')
    write(out/'protocol.json',protocol)
    old=torch.load(DATA/'train.pt',weights_only=True,map_location='cpu');info=payload['pretrain_info']
    config=json.loads((PARENT/'report.json').read_text())['config']
    results={}
    for arm in ('before','frozen','plastic'):
        p,checkpoint=load_research_run(PARENT)
        guard=PlasticConstraints(p) if arm=='plastic' else RecurrentConstraints(p,metadata=checkpoint['training_constraints'])
        p.to(device);torch.manual_seed(42)
        if device=='cuda':torch.cuda.manual_seed_all(42)
        progress=[]
        if arm!='before':
            interface=[v for n,v in p.named_parameters() if v.requires_grad and not n.endswith(('tau_raw','_w_macro_param'))]
            groups=[dict(params=interface,lr=.003)]
            if arm=='plastic':
                groups += [dict(params=[p.cfc_network.cell._w_macro_param],lr=.003),
                           dict(params=[p.cfc_network.cell.tau_raw],lr=.01)]
            opt=torch.optim.Adam(groups)
            for suffix in SUFFIXES:
                x=training[suffix]['X'].to(device);y=training[suffix]['Y'].to(device)
                for _ in range(60):
                    p.train();opt.zero_grad(set_to_none=True)
                    pred,_=p(x,dt=.02);anchor_pred,_=p(anchors.to(device),dt=.02)
                    loss=history_loss(pred,y)+protected_loss(anchor_pred,anchor_target.to(device))
                    if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                    loss.backward()
                    if not all(v.grad is None or torch.isfinite(v.grad).all() for v in p.parameters()):raise ValueError('Nonfinite gradient')
                    opt.step();p.post_step();guard(p)
                p.eval()
                metrics=capacity_metrics(p,training[suffix])
                with torch.no_grad():whole=sequence_metrics(p(x,dt=.02)[0],y)
                row=dict(update=guard.steps,suffix_ticks=suffix,training=metrics,sequence=whole,cruise=cruise(p))
                progress.append(row);write(out/(arm+'_progress.json'),progress)
                print(arm,guard.steps,'suffix',suffix,'error',metrics['max_abs_pitch_error_pwm'],flush=True)
        p.eval()
        capacity={label:{str(n):capacity_metrics(p,d) for n,d in sets.items()}
            for label,sets in (('training',training),('evaluation',evaluation))}
        gradients={str(n):prefix_gradient(p,training[n]) for n in (5,150)}
        sequence={}
        with torch.no_grad():
            for label,sets in (('training',training),('evaluation',evaluation),('duration_holdout',duration_holdout)):
                sequence[label]={str(n):sequence_metrics(p(d['X'].to(device),dt=.02)[0],d['Y'].to(device)) for n,d in sets.items()}
        quiet=cruise(p);training_audit=guard.audit()
        with torch.no_grad():anchor_error=float((p(anchors.to(device),dt=.02)[0]-anchor_target.to(device))[...,1:].abs().max())
        p.cpu().eval()
        temporal=evaluate_phase(p,old,info,config)
        # Export final parameters as frozen inference weights. This is not a
        # claim that W/tau were frozen during the plastic arm's training.
        frozen_export=RecurrentConstraints(p,freeze_core=True)
        checkpoint['initial_pretrain_info']=deepcopy(checkpoint['pretrain_info'])
        checkpoint['pretrain_info']=dict(experiment=experiment,arm=arm,
            updates=guard.steps,dataset_sha256=protocol['cases_sha256'],loss_mode=objective+'-history-plus-anchors-v1')
        checkpoint['continuation']=dict(arm=arm,parent_sha256=protocol['parent_sha256'],
            training_constraint_audit=training_audit,export_freezing='After fit; see training audit')
        report=dict(candidate=checkpoint['candidate'],config=config,continuation_epochs=None,
            continuation_updates=guard.steps,capacity=capacity,sequence=sequence,gradient=gradients,cruise=quiet,
            anchor_max_error_pwm=anchor_error,training_constraint_audit=training_audit,
            progress=progress,**temporal,research_only=True,training_ready=False)
        sample=old['X'][info['validation_indices'][:2]]
        report=save_phase(out/arm,policy=p,checkpoint=checkpoint,parent_directory=PARENT,
            report=report,constraints=frozen_export,sample=sample)
        export_and_verify(out/arm,sample)
        results[arm]=dict(capacity=capacity,sequence=sequence,gradient=gradients,cruise=quiet,
            anchor_max_error_pwm=anchor_error,training_audit=training_audit,temporal=comparison_row(report))
        write(out/'comparison.json',results)
        print(arm,'done',dict(cruise=quiet,temporal=results[arm]['temporal']),flush=True)
        del p
        if device=='cuda':torch.cuda.empty_cache()
    assert all(sha256(ROOT/p)==v for m in (fixed,sources) for p,v in m.items())
    assert sha256(out/'cases.pt')==protocol['cases_sha256']
    write(out/'validation.json',dict(completed_arms=3,source_and_cases_unchanged=True,
        active_baseline_unchanged=True,checkpoint_and_numpy_roundtrips_verified=True,
        no_new_recurrent_edges=True,automatic_promotion=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--device',choices=('cuda','cpu'),required=True)
    parser.add_argument('--objective',choices=('endpoint','sequence'),default='endpoint')
    a=parser.parse_args();run(a.out,a.device,a.objective)
