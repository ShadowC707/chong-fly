"""Frozen-core paired continuation on one demonstrated early Pitch/Yaw maneuver."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from copy import deepcopy
import argparse
import json
import shutil
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
SNAPSHOT=ROOT/'archive/candidates/k306_before_early_turn_v1'
PARENT=SNAPSHOT/'model'


def correction_case(path):
    with np.load(path) as a:
        time=a['time'].copy(); segment=a['segment'].copy(); x=a['X'].copy(); y=a['teacher'].copy()
    window=(time>=6)&(time<10)
    if not window.any() or len(np.unique(segment[window]))!=1:
        raise ValueError('Correction window must exist within one reset segment')
    selected=(segment==segment[window][0])&(time<10)
    if x.shape!=(len(time),74) or y.shape!=(len(time),4) or not np.isfinite(x[selected]).all() or not np.isfinite(y[selected]).all():
        raise ValueError('Invalid corrective trace')
    return dict(X=torch.from_numpy(x[selected]).unsqueeze(0).float(),
        Y=torch.from_numpy(y[selected]).unsqueeze(0).float(),
        mask=torch.from_numpy(window[selected]).unsqueeze(0),time=torch.from_numpy(time[selected]))


def correction_loss(prediction,target,mask):
    if (prediction.shape!=target.shape or prediction.shape[-1]!=4 or mask.shape!=prediction.shape[:2]
        or mask.dtype!=torch.bool or not mask.any()):
        raise ValueError('Invalid corrective loss contract')
    return (((prediction[...,2:4]-target[...,2:4])/500.).square()*mask[...,None]).sum()/(2*mask.sum())


def mirrored_case(case):
    from evaluate.diagnose_active_turn import mirror_observations
    from generator.generate_reflex_dataset import make_teacher
    x=torch.from_numpy(mirror_observations(case['X'][0].numpy())).unsqueeze(0)
    teacher=make_teacher('geometry-reflex-v5',noise_std_pwm=0,pwm_cruise_pitch=1530.)
    y=torch.from_numpy(np.stack([teacher.step(row,dt=.02) for row in x[0].numpy()])).unsqueeze(0)
    return dict(X=torch.cat([case['X'],x]),Y=torch.cat([case['Y'],y]),
        mask=case['mask'].expand(2,-1).clone(),time=case['time'])


@torch.no_grad()
def correction_metrics(policy,case):
    pred=policy(case['X'],dt=.02)[0]
    error=(pred-case['Y'])[case['mask']][:,2:4].abs()
    return dict(pitch_mae_pwm=float(error[:,0].mean()),yaw_mae_pwm=float(error[:,1].mean()),
        joint_loss=float(correction_loss(pred,case['Y'],case['mask'])))


def run(out, *, balanced=False, head_only=False):
    from artifacts.research import load_research_run
    from edge.runtime import sha256
    from edge.validate_numpy import validate
    from optimizer.continue_flow_reflex import DATA,write,export_and_verify,PARENT as ORIGINAL
    from optimizer.release_cruise import fit,cruise_cases,clear_trial,motion_admission
    from optimizer.bilateral_turn import bilateral_mask,magnitude_loss,admission
    from optimizer.finalize_mvp_candidate import evaluate_phase,comparison_row,save_phase
    from optimizer.mvp_constraints import RecurrentConstraints
    from optimizer.probe_history_capacity import cruise
    from evaluate.practical_maneuver import trial,CASES
    if head_only:balanced=True
    version='early-turn-xy-head-only-v1' if head_only else ('early-turn-balanced-v1' if balanced else 'early-turn-frozen-core-paired-v1')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    archive_hashes=json.loads((SNAPSHOT/'SHA256_MANIFEST.json').read_text())
    assert all(sha256(SNAPSHOT/p)==h for p,h in archive_hashes.items())
    dataset=ROOT/'output/balanced_maneuver_v1/train.pt'
    data=torch.load(dataset,weights_only=True,map_location='cpu')
    correction_path=SNAPSHOT/'evidence/early_turn/early_xy_+0/decisions.npz'
    case=correction_case(correction_path); assert int(case['mask'].sum())==200
    original_case=case
    if balanced: case=mirrored_case(case)
    torch.save(case,out/'correction.pt')
    mask=bilateral_mask(data['X'],data['Y'],data['valid']);mass=float(mask.sum())/len(data['X'])
    _,original=load_research_run(ORIGINAL);info=original['pretrain_info']
    old=torch.load(DATA/'train.pt',weights_only=True,map_location='cpu')
    sample=old['X'][info['validation_indices'][:2]]
    config=json.loads((PARENT/'report.json').read_text())['config']
    active=ROOT/'data/edge/k306_tau025_integrated_v1'
    paths={k:next((ROOT/'flight simulations/changed contract').glob('*'+k+'*.json')) for k in CASES}
    files=[dataset,correction_path,PARENT/'checkpoint.pt',SNAPSHOT/'SHA256_MANIFEST.json',DATA/'train.pt',
        ORIGINAL/'checkpoint.pt',ROOT/'configs/mvp_runtime_v3.json',*paths.values(),*[p for p in active.iterdir() if p.is_file()]]
    hashes={str(p):sha256(p) for p in files}
    reuse=ROOT/'output/early_turn_training_v1'
    if balanced:
        hashes[str(reuse/'comparison.json')]=sha256(reuse/'comparison.json')
        hashes[str(reuse/'control/checkpoint.pt')]=sha256(reuse/'control/checkpoint.pt')
    source_manifest=json.loads((PARENT/'flights/9d5431c1/manifest.json').read_text())
    for offset in (-3,3):
        m=deepcopy(source_manifest);m['start']['yaw_rad']+=float(np.deg2rad(offset))
        path=out/f'start_{offset:+d}.json';write(path,dict(manifest=m));paths[f'9d_heading_{offset:+d}']=path
    write(out/'protocol.json',dict(version=version,snapshot=str(SNAPSHOT),
        head_only=head_only,trainable='DN Pitch/Yaw rows only' if head_only else 'sensor_layer, W_in, dn_head',
        comparison_scope='Historical full-interface control, not a matched head-only control' if head_only else 'Matched full-interface control',
        mirrored_correction=balanced,reused_control=str(reuse/'control') if balanced else None,
        input_sha256=hashes,sources={str(p.relative_to(ROOT)):sha256(p) for folder in ('optimizer','simulation','core','edge','evaluate','generator','demo') for p in (ROOT/folder).glob('*.py')},
        arms=['control','corrected'],epochs=10,updates=130,seed=42,lr=.001,batch_size=8,
        frozen=['W','tau_raw','A'],base_loss='channel-v4 + 10 cruise MSE + bilateral magnitude MSE',
            correction=dict(coefficient=1.,loss='Pitch/Yaw MSE /500, mean over all selected frames and 2 channels',
            source='0deg early_xy demonstration'+(' plus mirrored observations with recomputed teacher' if balanced else '')+', full reset prefix, loss [6,10)s',
            frames=int(case['mask'].sum()),
            reason='Unit-weight normalized corrective MSE, one fixed recipe, no coefficient search',
            no_time_input=True,no_teacher_state_input=True,no_runtime_teacher=True),
        evaluation='Temporal probes + clear-start; six physical trials per arm only if existing screen passed. Four known rooms and previously untested initial headings +/-3deg.',
        limits='Correction observations are new, from one known room. +/-3deg are nearby validation starts, not unseen geometry or statistical safety proof.',
        no_promotion=True))
    auxiliary=cruise_cases();results={};flights=[]
    if balanced:
        shutil.copytree(reuse/'control',out/'control')
        results['control']=json.loads((reuse/'comparison.json').read_text())['control']
        flights=json.loads((reuse/'flights.json').read_text())
        for row in flights: row['reused_control']=True
        write(out/'flights.json',flights)
    p,_=load_research_run(PARENT);write(out/'initial.json',dict(correction=correction_metrics(p,original_case),cruise=cruise(p)))
    for arm in (('corrected',) if balanced else ('control','corrected')):
        p,ck=load_research_run(PARENT)
        handles=[]
        if head_only:
            from optimizer.interface_attribution import configure_xy_head,fit_xy_head
            handles=configure_xy_head(p)
        guard=RecurrentConstraints(p,freeze_core=True);torch.manual_seed(42)
        def extra(pred,target,ids):
            value=magnitude_loss(pred,target,mask[ids],normalizer=mass*len(ids))
            if arm=='corrected': value=value+correction_loss(p(case['X'],dt=.02)[0],case['Y'],case['mask'])
            return value
        def progress(rows):
            write(out/(arm+'_training.json'),rows);print(arm,rows[-1],flush=True)
        training=(fit_xy_head(p,data,guard,auxiliary,case,progress) if head_only else
            fit(p,data,guard,auxiliary,progress,extra_objective=extra))
        for handle in handles:handle.remove()
        p.eval()
        training.update(loss='channel-v4 + 10 cruise + bilateral + optional early joint correction',
            correction_coefficient=1. if arm=='corrected' else 0.)
        ck['initial_pretrain_info']=deepcopy(ck['pretrain_info'])
        ck['pretrain_info']=dict(training,dt=.02,dataset_sha256=hashes[str(dataset)],
            teacher_version=data['metadata']['teacher_version'],teacher_parameters=data['metadata']['teacher_parameters'],
            correction_sha256=hashes[str(correction_path)],subset_size=104,subset_indices=list(range(104)))
        ck['continuation']=dict(experiment=version,arm=arm,parent_sha256=hashes[str(PARENT/'checkpoint.pt')],
            training_frozen_parameters=[n for n,v in p.named_parameters() if not v.requires_grad],
            trained_output_rows=[2,3] if head_only else None)
        metrics=evaluate_phase(p,old,info,config);quiet=cruise(p);fitting=correction_metrics(p,original_case)
        report=dict(candidate=ck['candidate'],config=config,continuation_epochs=10,**metrics,cruise=quiet,
            correction=fitting,research_only=True,training_ready=False,pretrain=ck['pretrain_info'])
        save_phase(out/arm,policy=p,checkpoint=ck,parent_directory=PARENT,report=report,constraints=guard,sample=sample)
        export_and_verify(out/arm,sample)
        write(out/arm/'bundled_parity.json',validate(out/arm,out/arm/'numpy',steps=400,execution='bundled'))
        clear=clear_trial(out/arm,out/arm/'clear'); temporal=comparison_row(report)
        screen=admission(motion_admission(quiet['final_pwm'][2],clear),temporal)
        results[arm]=dict(temporal=temporal,cruise=quiet,correction=fitting,clear=clear,screen=screen)
        write(out/'comparison.json',results);print(arm,'screen',screen,flush=True)
        if screen['passed']:
            dest=out/arm/'flights';dest.mkdir()
            for name,path in paths.items():
                row=trial(name,path,'candidate',dict(before=ROOT/'output/balanced_maneuver_v1/before',candidate=out/arm),dest/name)
                row['arm']=arm;flights.append(row);write(out/'flights.json',flights);print(arm,name,row['metrics'],flush=True)
    assert all(sha256(Path(p))==h for p,h in hashes.items())
    assert all(sha256(SNAPSHOT/p)==h for p,h in archive_hashes.items())
    write(out/'validation.json',dict(arms_completed=2,new_training_arms=1 if balanced else 2,
        new_room_trials=sum(not row.get('reused_control',False) for row in flights),
        reused_room_trials=sum(row.get('reused_control',False) for row in flights),
        archive_and_inputs_unchanged=True,no_promotion=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--mirrored-correction',action='store_true')
    parser.add_argument('--xy-head-only',action='store_true');args=parser.parse_args()
    run(args.out,balanced=args.mirrored_correction,head_only=args.xy_head_only)
