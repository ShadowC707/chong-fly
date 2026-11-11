"""Fixed data intervention: old episodes vs old + corrective XY sequences.

Same initial checkpoint, 104 episodes, 130 updates, frozen tau/A/W. Explicit
external validation only; mirror-related new episodes never cross the split.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import torch

from optimizer.continue_flow_reflex import (ROOT,PARENT,DATA,CASES,write,score_raw,
    export_and_verify,physical_trial)
from artifacts.research import load_research_run
from edge.runtime import sha256
from optimizer.pretrain import pretrain_policy
from optimizer.mvp_constraints import RecurrentConstraints
from optimizer.finalize_mvp_candidate import evaluate_phase,save_phase,comparison_row
from optimizer.diagnose_candidate import heldout_predictions


def combine(parts, *, steps):
    keys=('X','Y','applied_actions','valid','behavior')
    arrays={k:[] for k in keys}; records=[]; sources=[]
    meta=deepcopy(parts[0][0]['metadata'])
    for data,ids,name in parts:
        for key in ('teacher_version','teacher_aggregation_version','teacher_parameters',
                    'dt','control_contract','required_sensor_motor_paths','behavior_names'):
            if data['metadata'][key]!=meta[key]: raise ValueError('Incompatible '+key)
        for i in ids:
            if data['X'].shape[1]>steps: raise ValueError('Do not truncate histories')
            for k in keys:
                row=data[k][i];fill=-1 if k=='behavior' else 1500 if k in ('Y','applied_actions') else 0
                padded=torch.full((steps,*row.shape[1:]),fill,dtype=row.dtype)
                padded[:len(row)]=row;arrays[k].append(padded)
            records.append(deepcopy(data['metadata']['episodes'][i])); sources.append([name,int(i)])
    seeds=[r['seed'] for r in records]
    if len(set(seeds))!=len(seeds):raise ValueError('Duplicate episode seed')
    result={k:torch.stack(v) for k,v in arrays.items()}
    scenarios=[r['scenario'] for r in records]
    meta.update(episodes=records,episode_scenarios=scenarios,scenario_counts=dict(Counter(scenarios)),
        source_episodes=sources,seq_len=steps,num_episodes=len(records),scene_profile='explicit-maneuver-mixture-v1',
        behavior_counts=dict(Counter(meta['behavior_names'][int(b)] for b in result['behavior'][result['valid']])),
        collection_contract='Per-episode provenance in source_episodes; targets unchanged; no new internal split')
    result['metadata']=meta
    return result


def run(output,device):
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    old=torch.load(DATA/'train.pt',weights_only=True,map_location='cpu')
    external=torch.load(DATA/'evaluation.pt',weights_only=True,map_location='cpu')
    newdir=ROOT/'data/maneuver_corrections_v1'
    new=torch.load(newdir/'train.pt',weights_only=True,map_location='cpu')
    new_eval=torch.load(newdir/'evaluation.pt',weights_only=True,map_location='cpu')
    expert_eval=torch.load(ROOT/'data/maneuver_sequences_v1/evaluation.pt',weights_only=True,map_location='cpu')
    _,checkpoint=load_research_run(PARENT); info=checkpoint['pretrain_info']
    config=json.loads((PARENT/'report.json').read_text())['config']
    rng=np.random.default_rng(42); kept=[]
    for name in sorted(set(old['metadata']['episode_scenarios'])):
        ids=[i for i in info['subset_indices'] if old['metadata']['episode_scenarios'][i]==name]
        kept.extend(rng.choice(ids,size=9,replace=False).tolist())
    assert len(kept)==72 and not set(kept)&set(info['validation_indices'])
    assert len(new['X'])==32
    assert not ({r['pair_id'] for r in new['metadata']['episodes']} &
                {r['pair_id'] for r in new_eval['metadata']['episodes']})
    datasets={'control':combine([(old,info['subset_indices'],'old')],steps=600),
        'mixed':combine([(old,kept,'old'),(new,list(range(32)),'correction')],steps=600)}
    assert all(len(d['X'])==104 for d in datasets.values())
    fixed=json.loads((ROOT/'output/neural_refocus_v1/protocol.json').read_text())['fixed_files_sha256']
    assert all(sha256(ROOT/p)==v for p,v in fixed.items())
    sources={str(p.relative_to(ROOT)):sha256(p) for folder in ('core','optimizer','generator','simulation','edge','demo','configs','artifacts') for p in (ROOT/folder).glob('*.py')}
    inputs={str(p.relative_to(ROOT)):sha256(p) for p in (DATA/'train.pt',DATA/'evaluation.pt',newdir/'train.pt',newdir/'evaluation.pt',ROOT/'data/maneuver_sequences_v1/evaluation.pt')}
    paths={case:next((ROOT/'flight simulations/changed contract').glob('*'+case+'*.json')) for case in CASES}
    write(output/'protocol.json',dict(version='paired-corrective-data-v1',parent_sha256=sha256(PARENT/'checkpoint.pt'),
        settings=dict(epochs=10,batch_size=8,updates=130,lr=.001,seed=42,device=device,
            loss='channel-v4',yaw_balance=.7,active_fast_balance=0.,frozen=['tau_raw','A','_w_macro_param']),
        datasets={k:dict(source_episodes=v['metadata']['source_episodes'],valid_frames=int(v['valid'].sum())) for k,v in datasets.items()},
        old_validation=info['validation_indices'],new_validation='Separate geometry draws; both teacher- and learner-driven; no training use',
        source_sha256=sources,input_sha256=inputs,fixed_baseline_sha256=fixed,
        physical_cases={k:dict(path=str(p.relative_to(ROOT)),sha256=sha256(p)) for k,p in paths.items()},
        comparison='Equal optimizer updates, not equal valid-frame exposure; new sequences have 12s instead of 5s',
        promotion=False))
    reports={}; sample=old['X'][info['validation_indices'][:2]]
    for arm in ('before','control','mixed'):
        policy,payload=load_research_run(PARENT)
        constraint=RecurrentConstraints(policy,metadata=payload['training_constraints'])
        if arm!='before':
            torch.manual_seed(42)
            if device=='cuda':torch.cuda.manual_seed_all(42)
            policy.to(device)
            def post_step(p):
                constraint(p)
                if constraint.steps%13==0:print(arm,'epoch',constraint.steps//13,'/10',flush=True)
            pretrain_policy(policy,datasets[arm],epochs=10,subset_ratio=1.,batch_size=8,seed=42,
                validation_fraction=0.,lr=.001,device=device,research_only=True,
                loss_mode='channel-v4',yaw_balance=.7,post_step_callback=post_step)
            assert constraint.steps==130 and policy._pretrain_info['validation_indices']==[]
            payload['pretrain_info']=deepcopy(policy._pretrain_info)
        policy.cpu().eval()
        raw={}
        with torch.no_grad():
            for name,data,ids in (('old_internal',old,info['validation_indices']),
                ('old_external_seen',external,list(range(len(external['X'])))),
                ('new_corrections',new_eval,list(range(len(new_eval['X'])))),
                ('new_expert',expert_eval,list(range(len(expert_eval['X']))))):
                raw[name]=score_raw(heldout_predictions(policy,data,ids),data,ids)
        metrics=evaluate_phase(policy,old,info,config)
        payload['continuation']=dict(experiment='paired-corrective-data-v1',parent_sha256=sha256(PARENT/'checkpoint.pt'),
            arm=arm,epochs=0 if arm=='before' else 10,optimizer_reset=True,frozen_core=True)
        report=dict(candidate=payload['candidate'],config=config,pretrain=payload['pretrain_info'],
            continuation_epochs=0 if arm=='before' else 10,raw=raw,**metrics,research_only=True,training_ready=False)
        report=save_phase(output/arm,policy=policy,checkpoint=payload,parent_directory=PARENT,
            report=report,constraints=constraint,sample=sample)
        export_and_verify(output/arm,sample)
        reports[arm]=dict(raw=raw,temporal=comparison_row(report));write(output/'comparison.json',reports)
        print(arm,'temporal',reports[arm]['temporal'],flush=True)
        del policy
        if device=='cuda':torch.cuda.empty_cache()
    flights={}
    for arm in ('before','control','mixed'):
        flights[arm]={}
        for case,path in paths.items():
            report=physical_trial(output/arm,path,output/arm/case)
            flights[arm][case]=report['metrics'];write(output/'physical_comparison.json',flights)
            print(arm,case,report['metrics'],flush=True)
    assert all(sha256(ROOT/p)==v for mapping in (fixed,sources,inputs) for p,v in mapping.items())
    write(output/'validation.json',dict(completed_arms=3,completed_physical_trials=12,
        source_and_input_hashes_unchanged=True,frozen_baseline_unchanged=True,
        frozen_core_verified=True,checkpoint_roundtrips_exact=True,numpy_equivalence=True,
        automatic_promotion=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',choices=('cpu','cuda'),required=True)
    a=p.parse_args();run(a.out,a.device)
