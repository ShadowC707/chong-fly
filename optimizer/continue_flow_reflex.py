"""Fixed paired continuation of k306: channel-v4 vs active-fast loss mixture.

Same parent, episode split/order, fresh Adam and frozen tau/A/W. Evaluates raw
commands, temporal probes, seven elementary scenes and four recorded layouts
through the unchanged integrated simulator. Never promotes a model implicitly.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')

import argparse
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch

from artifacts.research import load_research_run
from edge.export_numpy import export_run
from edge.runtime import NumpyPolicy, sha256
from edge.system import RunConfig
from optimizer.diagnose_candidate import heldout_predictions
from optimizer.finalize_mvp_candidate import evaluate_phase, save_phase, comparison_row
from optimizer.mvp_constraints import RecurrentConstraints
from optimizer.pretrain import pretrain_policy

ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT/'data/selection/mvp_frozen_core_v1/trial_00003/after'
DATA=ROOT/'data/reflex_training_v5_pilot'
ARMS={'control':0.,'active_fast':.25}
CASES=('068fe108','89d81a9f','9d5431c1','b0a9f212')


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def score_raw(prediction,data,ids):
    x=data['X'][ids];y=data['Y'][ids];valid=data['valid'][ids]
    distance=x[...,2:66].amin(-1)*3.5;flow=x[...,:2].norm(dim=-1)
    recent=torch.zeros_like(valid);age=torch.full((len(ids),),10000,dtype=torch.long)
    for t in range(x.shape[1]):
        age=torch.where(valid[:,t] & (distance[:,t]<.8),0,age+1)
        recent[:,t]=(age>0)&(age<=50)
    masks=dict(all=valid,active_fast=valid & (distance<.8)&(flow>=.12),
        neutral_yaw_active_fast=valid & (distance<.8)&(flow>=.12)&((y[...,3]-1500).abs()<=20),
        clear_fast_forward=valid &(distance>=.95)&(x[...,0]>=.12)&(x[...,1].abs()<.03),
        recently_cleared=valid & recent &(distance>=.8),
        teacher_braking_after_clear=valid & recent &(distance>=.8)&(y[...,2]<1495),
        first_half_second=valid & (torch.arange(x.shape[1])[None,:]<25))
    if prediction.shape!=y.shape or not torch.isfinite(prediction).all():raise ValueError('Invalid raw predictions')
    result={}
    for name,mask in masks.items():
        def metrics(m):
            count=int(m.sum())
            return dict(frames=count,mae_roll_pitch_yaw=(prediction-y)[m][:,1:].abs().mean(0).tolist() if count else None,
                strong_yaw_frames=int((((prediction[...,3]-1500).abs()>50)&m).sum()),
                positive_pitch_frames=int(((prediction[...,2]>1510)&m).sum()))
        result[name]=dict(**metrics(mask),episodes=int(mask.any(1).sum()),per_episode=[
            dict(episode_id=int(ids[i]),**metrics(mask & (torch.arange(len(ids))[:,None]==i)))
            for i in range(len(ids)) if mask[i].any()])
    return result


def evaluate_raw(policy,data,external,info):
    return {name:score_raw(heldout_predictions(policy,d,ids),d,ids) for name,d,ids in (
        ('training',data,info['subset_indices']),('internal_validation',data,info['validation_indices']),
        ('external_seen_evaluation',external,list(range(len(external['X'])))))}


def export_and_verify(run,sample):
    bundle=run/'numpy';export_run(run,bundle)
    policy,_=load_research_run(run)
    with torch.no_grad():expected=policy(sample,dt=.02)[0].numpy()
    numpy_policy=NumpyPolicy(bundle);worst=0.
    for episode,target in zip(sample.numpy(),expected):
        numpy_policy.reset_state()
        actual=np.stack([numpy_policy.step_np(x[:2],x[2:66],x[66:74],dt=.02) for x in episode])
        worst=max(worst,float(np.max(abs(actual-target))))
        np.testing.assert_allclose(actual,target,atol=.02,rtol=0)
    write(run/'numpy_equivalence.json',dict(max_abs_pwm_error=worst,atol_pwm=.02,steps=sample.shape[1]))
    base=RunConfig.from_json(ROOT/'configs/mvp_runtime_v3.json')
    config=replace(base,bundle_dir=str(bundle.resolve()),expected_metadata_sha256=sha256(bundle/'metadata.json'))
    write(run/'runtime_config.json',config.to_dict())


def physical_trial(run,path,destination):
    from demo.runner import DemoRunner
    from demo.recording import plain
    from evaluate.compare_maneuver_repair import summarize
    m=json.loads(path.read_text(encoding='utf-8'))['manifest']
    runner=DemoRunner(config=run/'runtime_config.json',scene=m['scene'],seed=m['seed'],
        start=m['start']['position_m'],yaw=m['start']['yaw_rad'],scene_options=m.get('scene_options'),
        collision_radius_m=.2,max_duration_s=45.,contact_model='sphere-sweep-v1',run_id=run.name+'-'+path.stem)
    try:
        for key in ('room','boxes','cylinders'):
            assert runner.manifest()['geometry'][key]==m['geometry'][key]
        destination.mkdir()
        write(destination/'manifest.json',runner.manifest())
        runner.resume();runner.event(dict(type='mode',mode='AUTO'))
        for step in range(11250):
            runner.advance()
            if runner.terminal:break
        report=dict(metrics=summarize(runner.rows,runner.time_s),events=runner.events,
                    source_sha256=sha256(path),coordinator=runner.system.runtime.coordinator.contract())
        controls=[row for i,row in enumerate(runner.rows) if i%5==0]
        np.savez_compressed(destination/'trace.npz',time=[r['time_s'] for r in controls],
            pos=[r['position_m'] for r in controls],velocity=[r['velocity_mps'] for r in controls],
            yaw=[r['attitude_rad'][2] for r in controls],
            network=[r['trace'].get('network_pwm') or [np.nan]*4 for r in controls],
            coordinated=[r['trace'].get('coordinated_pwm') or [np.nan]*4 for r in controls],
            applied=[r.get('plant_input_pwm') or [np.nan]*4 for r in controls],
            min_tof=[r['min_tof_m'] for r in controls],clearance=[r['clearance_m'] for r in controls])
        write(destination/'report.json',plain(report))
        return report
    finally:runner.env.close()


def run_experiment(output,device):
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    parent_report=json.loads((PARENT/'report.json').read_text(encoding='utf-8'))
    config=parent_report['config']
    original,checkpoint=load_research_run(PARENT)
    info=checkpoint['pretrain_info']
    data=torch.load(DATA/'train.pt',weights_only=True,map_location='cpu')
    external=torch.load(DATA/'evaluation.pt',weights_only=True,map_location='cpu')
    sources={str(p.relative_to(ROOT)):sha256(p) for folder in ('optimizer','core','simulation','edge','demo','configs','artifacts','generator')
             for p in (ROOT/folder).glob('*.py')}
    fixed=json.loads((ROOT/'output/neural_refocus_v1/protocol.json').read_text())['fixed_files_sha256']
    assert all(sha256(ROOT/p)==v for p,v in fixed.items()),'Frozen baseline changed'
    paths={case:next((ROOT/'flight simulations/changed contract').glob('*'+case+'*.json')) for case in CASES}
    protocol=dict(version='paired-active-fast-v1',parent=str(PARENT.relative_to(ROOT)),
        parent_sha256=sha256(PARENT/'checkpoint.pt'),dataset_sha256=sha256(DATA/'train.pt'),
        evaluation_sha256=sha256(DATA/'evaluation.pt'),source_sha256=sources,frozen_baseline_sha256=fixed,
        settings=dict(arms=ARMS,epochs=10,learning_rate=.001,batch_size=8,seed=42,
            yaw_balance=.7,loss_mode='channel-v4',frozen=['tau_raw','A','_w_macro_param'],
            device=device,torch=str(torch.__version__),gpu=torch.cuda.get_device_name(0) if device=='cuda' else None,
            deterministic_algorithms=True,tf32=False,fresh_adam=True),
        training_indices=info['subset_indices'],validation_indices=info['validation_indices'],
        physical=dict(cases={k:dict(path=str(v.relative_to(ROOT)),sha256=sha256(v)) for k,v in paths.items()},
            horizon_s=45.,radius_m=.2,coordinator='both/original_v1',levy=False,scheduled_stop=False),
        interpretation=dict(primary='Raw active-fast error plus paired physical collisions/clearance; no automatic promotion.',
            regression_checks='All 12 temporal cases, seven elementary scenes, global/channel MAE, reverse motion.',
            known_evaluation_data=True,one_seed=True,geometry_memory_disabled=True))
    write(output/'protocol.json',protocol)
    sample=data['X'][info['validation_indices'][:2]]
    reports={}
    for arm in ('before',*ARMS):
        policy,payload=load_research_run(PARENT)
        constraint=RecurrentConstraints(policy,metadata=payload['training_constraints'])
        epochs=0 if arm=='before' else 10
        if epochs:
            policy.to(device);torch.manual_seed(42)
            if device=='cuda':torch.cuda.manual_seed_all(42)
            print(arm,'training',device,'mix',ARMS[arm],flush=True)
            def post_step(p):
                constraint(p)
                if constraint.steps%13==0:print(arm,'epoch',constraint.steps//13,'/10',flush=True)
            pretrain_policy(policy,data,epochs=10,subset_ratio=1.,batch_size=8,seed=42,
                lr=.001,device=device,research_only=True,loss_mode='channel-v4',yaw_balance=.7,
                post_step_callback=post_step,active_fast_balance=ARMS[arm])
            for key in ('subset_indices','validation_indices','dataset_sha256'):
                assert policy._pretrain_info[key]==info[key],key+' changed'
            payload['pretrain_info']=deepcopy(policy._pretrain_info)
        policy.cpu().eval()
        print(arm,'raw evaluation / temporal probes / seven elementary scenes',flush=True)
        raw=evaluate_raw(policy,data,external,info)
        metrics=evaluate_phase(policy,data,info,config)
        payload['continuation']=dict(parent_checkpoint_sha256=protocol['parent_sha256'],
            epochs=epochs,learning_rate=.001,loss_mode='channel-v4',active_fast_balance=ARMS.get(arm,0.),
            frozen_core=True,optimizer_reset=True,experiment='paired-active-fast-v1')
        report=dict(candidate=payload['candidate'],config={**config,'device':device,'active_fast_balance':ARMS.get(arm,0.)},
            pretrain=payload['pretrain_info'],continuation_epochs=epochs,raw=raw,**metrics,
            research_only=True,training_ready=False)
        report=save_phase(output/arm,policy=policy,checkpoint=payload,parent_directory=PARENT,
            report=report,constraints=constraint,sample=sample)
        export_and_verify(output/arm,sample)
        if arm=='before':
            with np.load(output/arm/'numpy/model.npz') as a,np.load(ROOT/'data/edge/k306_tau025_integrated_v1/model.npz') as b:
                assert a.files==b.files and all(np.array_equal(a[k],b[k]) for k in a.files),'Initial exported model changed'
        reports[arm]=dict(raw=raw,temporal=comparison_row(report))
        write(output/'comparison.json',reports)
        print(arm,'temporal',reports[arm]['temporal'],flush=True)
        del policy
        if device=='cuda':torch.cuda.empty_cache()
    flights={}
    for arm in ('before',*ARMS):
        flights[arm]={}
        for case,path in paths.items():
            result=physical_trial(output/arm,path,output/arm/case)
            flights[arm][case]=result['metrics']
            write(output/'physical_comparison.json',flights)
            print(arm,case,result['metrics'],flush=True)
    assert all(sha256(ROOT/p)==v for p,v in fixed.items()),'Fixed baseline changed'
    assert all(sha256(ROOT/p)==v for p,v in sources.items()),'Executed sources changed'
    assert sha256(DATA/'train.pt')==protocol['dataset_sha256']
    assert sha256(DATA/'evaluation.pt')==protocol['evaluation_sha256']
    write(output/'validation.json',dict(completed_arms=3,completed_physical_trials=12,
        fixed_baseline_unchanged=True,source_hashes_unchanged=True,training_splits_unchanged=True,
        checkpoint_roundtrips_exact=True,numpy_multistep_equivalence_passed=True,
        automatic_promotion=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--device',choices=('cpu','cuda'),required=True)
    args=parser.parse_args();run_experiment(args.out,args.device)


if __name__=='__main__':main()
