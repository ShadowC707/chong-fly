"""Paired frozen-core continuation: only the versioned sensory encoding differs."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]


def migrate_release_encoding(policy):
    """Explicit research transfer, never reinterpret a checkpoint during load."""
    if policy.sensor_layer.encoding!='proximity-v1' or not policy.neutral_origin:
        raise ValueError('Transfer requires a neutral-origin proximity-v1 source')
    result=deepcopy(policy)
    cell=result.cfc_network.cell
    if cell._architecture_contract['sensor_encoding_parameters']!={'threshold_m':.8,'max_range_m':3.5}:
        raise ValueError('Unexpected source encoding parameters')
    result.sensor_layer.encoding='proximity-release-v1'
    cell._architecture_contract['sensor_encoding']='proximity-release-v1'
    cell._architecture_contract['sensor_encoding_parameters']={'threshold_m':1.,'max_range_m':3.5}
    result.reset_state()
    for name,value in policy.state_dict().items():
        if torch.is_tensor(value): assert torch.equal(value,result.state_dict()[name])
    return result


def run(out, resume=False):
    from artifacts.research import load_research_run
    from edge.runtime import sha256
    from optimizer.continue_flow_reflex import PARENT,DATA,write,export_and_verify
    from optimizer.finalize_mvp_candidate import evaluate_phase,save_phase,comparison_row
    from optimizer.mvp_constraints import RecurrentConstraints
    from optimizer.pretrain import pretrain_policy
    from evaluate.practical_maneuver import trial,CASES
    from evaluate.teacher_encoding_contract import histories,teacher_targets
    from edge.validate_numpy import validate

    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    out=Path(out)
    if resume:
        if not (out/'protocol.json').is_file(): raise ValueError('No saved protocol to resume')
    else: out.mkdir(parents=True,exist_ok=False)
    dataset=ROOT/'output/balanced_maneuver_v1/train.pt'
    train=torch.load(dataset,weights_only=True,map_location='cpu')
    assert len(train['X'])==104 and train['metadata']['teacher_parameters']['pwm_cruise_pitch']==1530.
    parent,payload=load_research_run(PARENT);info=payload['pretrain_info']
    data=torch.load(DATA/'train.pt',weights_only=True,map_location='cpu')
    config=json.loads((PARENT/'report.json').read_text())['config']
    sample=data['X'][info['validation_indices'][:2]]
    active=ROOT/'data/edge/k306_tau025_integrated_v1'
    files=[dataset,PARENT/'checkpoint.pt',ROOT/'configs/mvp_runtime_v3.json',*[p for p in active.iterdir() if p.is_file()]]
    hashes={str(p):sha256(p) for p in files}
    protocol=dict(version='release-encoding-paired-v1',input_sha256=hashes,
        source_sha256={str(p.relative_to(ROOT)):sha256(p) for folder in ('edge','core','simulation','optimizer','demo') for p in (ROOT/folder).glob('*.py')},
        arms=['old_encoding','release_encoding'],parent=str(PARENT),
        encoding=dict(old_encoding='proximity-v1; support 0.8m',release_encoding='proximity-release-v1; support 1.0m'),
        training=dict(seed=42,epochs=10,updates=130,batch_size=8,lr=.001,device='cpu',loss='channel-v4',yaw_balance=.7,
            frozen=['tau_raw','A','W'],history_auxiliary=False,teacher_unchanged=True,dataset_regenerated=False),
        criteria='Report alias pair, temporal yaw, seven scenes, four known physical rooms; no automatic promotion',
        limitations='One seed, known evaluation rooms; changing encoding changes below-0.8m drive too; this is not isolation of the margin alone',
        no_parameter_search=True)
    if resume:
        prior=json.loads((out/'protocol.json').read_text())
        assert prior['input_sha256']==hashes and prior['training']==protocol['training']
        assert prior['version']==protocol['version'] and prior['encoding']==protocol['encoding']
        prior_sources={Path(path).as_posix():digest for path,digest in prior['source_sha256'].items()}
        for path,digest in prior_sources.items():
            if path not in ('optimizer/release_encoding.py','edge/pinned_system.py'): assert sha256(ROOT/path)==digest,path
        write(out/'resume.json',dict(reason='Resume saved exports after temporary-directory access failure; no retraining completed arms',
            runner_sha256=sha256(__file__),original_runner_sha256=prior_sources['optimizer/release_encoding.py'],
            pinned_loader_sha256=sha256(ROOT/'edge/pinned_system.py'),
            loader_change='Source-only import: no bytecode written into immutable bundles; numerical source unchanged'))
    else: write(out/'protocol.json',protocol)
    results={}
    for arm in ('old_encoding','release_encoding'):
        if resume and (out/arm/'checkpoint.pt').exists():
            restored,saved=load_research_run(out/arm)
            assert saved['continuation']['arm']==arm and saved['pretrain_info']['epochs']==10
            report=json.loads((out/arm/'report.json').read_text())
            parity=validate(out/arm,out/arm/'numpy',steps=400,execution='bundled')
            write(out/arm/'bundled_parity.json',parity)
            results[arm]=dict(temporal=comparison_row(report),witness=report['witness'])
            write(out/'comparison.json',results)
            print(arm,'resumed saved checkpoint; no optimizer updates',results[arm],flush=True)
            continue
        p,ck=load_research_run(PARENT)
        if arm=='release_encoding':
            p=migrate_release_encoding(p);ck['sensor_encoding']='proximity-release-v1'
        guard=RecurrentConstraints(p,freeze_core=True)
        print(arm,'zero-epoch evaluation',flush=True)
        before=evaluate_phase(p,data,info,config)
        write(out/(arm+'_initial.json'),before)
        torch.manual_seed(42)
        def post_step(policy):
            guard(policy)
            if guard.steps%26==0: print(arm,'updates',guard.steps,'/130',flush=True)
        pretrain_policy(p,train,epochs=10,subset_ratio=1.,validation_fraction=0.,lr=.001,batch_size=8,seed=42,
            device='cpu',research_only=True,loss_mode='channel-v4',yaw_balance=.7,post_step_callback=post_step)
        assert guard.steps==130
        p.eval()
        metrics=evaluate_phase(p,data,info,config)
        witness=torch.from_numpy(histories())
        with torch.no_grad(): predictions=p(witness,dt=.02)[0].numpy()
        wreport=dict(final_pwm=predictions[:,-1].tolist(),teacher_final_pwm=teacher_targets(histories())[:,-1].tolist(),
            max_pair_difference_pwm=float(np.abs(predictions[0]-predictions[1]).max()))
        ck['continuation']=dict(experiment='release-encoding-paired-v1',arm=arm,parent_sha256=hashes[str(PARENT/'checkpoint.pt')],
            explicit_encoding_transfer=arm=='release_encoding',initial_numeric_parameters_unchanged=True)
        ck['initial_pretrain_info']=deepcopy(ck['pretrain_info']);ck['pretrain_info']=deepcopy(p._pretrain_info)
        report=dict(candidate=ck['candidate'],config=config,continuation_epochs=10,**metrics,witness=wreport,
            pretrain=p._pretrain_info,research_only=True,training_ready=False)
        save_phase(out/arm,policy=p,checkpoint=ck,parent_directory=PARENT,report=report,constraints=guard,sample=sample)
        export_and_verify(out/arm,sample)
        parity=validate(out/arm,out/arm/'numpy',steps=400,execution='bundled')
        write(out/arm/'bundled_parity.json',parity)
        results[arm]=dict(temporal=comparison_row(report),witness=wreport)
        write(out/'comparison.json',results)
        print(arm,results[arm],flush=True)
    # Exact historical baseline regression with bundled old runtime, before new flights.
    runs=dict(before=ROOT/'output/balanced_maneuver_v1/before',candidate=out/'old_encoding')
    case=CASES[0];path=next((ROOT/'flight simulations/changed contract').glob('*'+case+'*.json'))
    baseline=trial(case,path,'before',runs,out/'baseline_regression')
    assert baseline['baseline_exact']
    flights=[]
    for arm in results:
        dest=out/arm/'flights';dest.mkdir()
        runs=dict(before=ROOT/'output/balanced_maneuver_v1/before',candidate=out/arm)
        for case in CASES:
            path=next((ROOT/'flight simulations/changed contract').glob('*'+case+'*.json'))
            r=trial(case,path,'candidate',runs,dest/case);r['arm']=arm;flights.append(r)
            write(out/'flights.json',flights);print(arm,case,r['metrics'],flush=True)
    assert all(sha256(Path(p))==h for p,h in hashes.items())
    write(out/'validation.json',dict(completed_arms=2,completed_flights=8,baseline_trajectory_exact=True,
        input_and_active_bundle_unchanged=True,automatic_promotion=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args();run(args.out,args.resume)
