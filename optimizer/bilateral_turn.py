"""Fixed paired interface continuation: existing objective vs bilateral yaw magnitude."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT/'output/release_cruise_v1/release_encoding'


def bilateral_mask(x,target,valid):
    if (x.shape[:-1]!=target.shape[:-1] or x.shape[-1]!=74 or target.shape[-1]!=4
        or valid.shape!=x.shape[:-1] or valid.dtype!=torch.bool):
        raise ValueError('Invalid bilateral training shapes')
    grid=x[...,2:66].reshape(*x.shape[:-1],8,8)*3.5
    left=grid[...,:,:4].amin(dim=(-2,-1))<.8
    right=grid[...,4:].amin(dim=(-2,-1))<.8
    return valid&left&right&(x[...,:2].norm(dim=-1)<=.05)&((target[...,3]-1500).abs()>=80)


def magnitude_loss(prediction,target,mask,*,normalizer):
    if (not math.isfinite(normalizer) or normalizer<=0 or prediction.shape!=target.shape
        or prediction.shape[-1]!=4 or mask.shape!=prediction.shape[:-1] or mask.dtype!=torch.bool):
        raise ValueError('Invalid magnitude loss contract')
    delta=prediction[...,3]-1500.
    # A detached branch gives the absolute-magnitude objective away from zero.
    # At exactly neutral use teacher direction rather than abs()'s zero gradient.
    sign=torch.where(delta.detach()!=0,delta.detach().sign(),(target[...,3]-1500.).sign())
    error=(delta-sign*(target[...,3]-1500.).abs())/500.
    return (error.square()*mask).sum()/normalizer


def diagnostic(policy,dataset,recorded):
    from evaluate.diagnose_active_turn import mirror_observations,metrics,threat_groups
    from generator.generate_reflex_dataset import make_teacher
    def sequence(x,segments):
        # Preserve original segment resets, but run vectorized complete segments.
        cuts=np.r_[0,np.flatnonzero(segments[1:]!=segments[:-1])+1,len(x)]
        with torch.no_grad():
            return np.concatenate([policy(torch.from_numpy(x[a:b]).unsqueeze(0),dt=.02)[0][0].numpy()
                for a,b in zip(cuts[:-1],cuts[1:])])
    with torch.no_grad():
        pred=np.concatenate([policy(dataset['X'][i:i+8],dt=.02)[0].numpy() for i in range(0,len(dataset['X']),8)])
    train={k:metrics(pred,dataset['Y'].numpy(),m) for k,m in
        threat_groups(dataset['X'].numpy(),dataset['valid'].numpy()).items()}
    trace={};arrays={}
    for side in ('original','mirrored'):
        x=recorded['X'] if side=='original' else mirror_observations(recorded['X'])
        y=sequence(x,recorded['segment'])
        teacher=make_teacher('geometry-reflex-v5',noise_std_pwm=0,pwm_cruise_pitch=1530.)
        targets=[]
        for i,row in enumerate(x):
            if i==0 or recorded['segment'][i]!=recorded['segment'][i-1]:teacher.reset()
            targets.append(teacher.step(row,dt=.02))
        targets=np.asarray(targets)
        groups=threat_groups(x,recorded['auto'])
        groups['terminal_slow_active']=groups['slow_active']&(recorded['time']>=34.)
        trace[side]={k:metrics(y,targets,m) for k,m in groups.items()}
        arrays[side+'_network']=y;arrays[side+'_teacher']=targets
    return dict(training=train,trace=trace),arrays


def admission(motion,temporal):
    checks=dict(motion=motion['passed'],direction=temporal['direction']==12,
        recovery=temporal['recovery']==12 and temporal['max_recovery_s'] is not None and temporal['max_recovery_s']<=.8,
        narrow=temporal['narrow_amplitude']==8,elementary_scenes=temporal['scenes']==7,
        broad_no_regression=temporal['broad_joint']>=3)
    return dict(passed=all(checks.values()),checks=checks,reasons=[k for k,v in checks.items() if not v],
        scope='Diagnostic room-run screen only, not flight admission; parent broad score 3/4 retained explicitly')


def run(out):
    from artifacts.research import load_research_run
    from edge.runtime import sha256
    from edge.validate_numpy import validate
    from optimizer.continue_flow_reflex import DATA,write,export_and_verify,PARENT as ORIGINAL
    from optimizer.release_cruise import fit,cruise_cases,clear_trial,motion_admission,SETTINGS
    from optimizer.finalize_mvp_candidate import evaluate_phase,comparison_row,save_phase
    from optimizer.mvp_constraints import RecurrentConstraints
    from optimizer.probe_history_capacity import cruise
    from evaluate.practical_maneuver import trial,CASES
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    dataset=ROOT/'output/balanced_maneuver_v1/train.pt'
    train=torch.load(dataset,weights_only=True,map_location='cpu')
    assert len(train['X'])==104 and train['metadata']['teacher_parameters']['pwm_cruise_pitch']==1530.
    mask=bilateral_mask(train['X'],train['Y'],train['valid'])
    assert int(mask.sum())==428
    expected_mass=float(mask.sum())/len(train['X'])
    _,original=load_research_run(ORIGINAL);info=original['pretrain_info']
    data=torch.load(DATA/'train.pt',weights_only=True,map_location='cpu')
    config=json.loads((PARENT/'report.json').read_text())['config']
    sample=data['X'][info['validation_indices'][:2]]
    record_path=PARENT/'flights/9d5431c1/decisions.npz'
    with np.load(record_path) as a:recorded={k:a[k].copy() for k in a.files}
    active=ROOT/'data/edge/k306_tau025_integrated_v1'
    paths={case:next((ROOT/'flight simulations/changed contract').glob('*'+case+'*.json')) for case in CASES}
    files=[dataset,PARENT/'checkpoint.pt',ORIGINAL/'checkpoint.pt',DATA/'train.pt',record_path,
        ROOT/'configs/mvp_runtime_v3.json',*paths.values(),*[p for p in active.iterdir() if p.is_file()]]
    hashes={str(p):sha256(p) for p in files}
    auxiliary=cruise_cases()
    write(out/'protocol.json',dict(version='bilateral-magnitude-paired-v1',parent=str(PARENT),training=SETTINGS,
        arms=dict(control=0.,magnitude=1.),input_sha256=hashes,
        sources={str(p.relative_to(ROOT)):sha256(p) for folder in ('optimizer','simulation','core','edge','evaluate','generator','demo') for p in (ROOT/folder).glob('*.py')},
        objective=dict(selected_frames=int(mask.sum()),episodes=int(mask.any(dim=1).sum()),
            selection='valid, each ToF half min <0.8m, flow norm <=0.05, teacher yaw magnitude >=80',
            formula='mean_selected ((abs(yaw-1500)-abs(teacher_yaw-1500))/500)^2; exact zero takes teacher-sign subgradient',
            normalizer='428 / 104 * current batch episode count; fixed expected mass, no batch-local renormalization',
            coefficient=1.,reason='One unit-weight conditional MSE, same PWM normalization as existing objectives; no coefficient search'),
        evaluation='Initial/final temporal probes, seven scenes, clear start, frozen recorded original/mirror; four known rooms only if screen passes',
        screen=dict(direction=12,recovery=12,max_recovery_s=.8,narrow=8,broad_min=3,scenes=7,motion='existing cruise screen'),
        limitations='Control receives same extra 10 epochs and fresh Adam. Magnitude objective can strengthen wrong signs; signed navigation loss retained. Known diagnosis is not an independent holdout.',
        no_new_training_data=True,no_promotion=True))
    p,_=load_research_run(PARENT)
    before=evaluate_phase(p,data,info,config)
    with torch.no_grad():before['cruise']=cruise(p)
    before_diag,arrays=diagnostic(p,train,recorded)
    write(out/'initial.json',dict(evaluation=before,diagnostic=before_diag))
    np.savez_compressed(out/'initial_replay.npz',time=recorded['time'],**arrays)
    results={};flights=[]
    for arm,coefficient in (('control',0.),('magnitude',1.)):
        p,ck=load_research_run(PARENT);guard=RecurrentConstraints(p,freeze_core=True)
        torch.manual_seed(42)
        def progress(rows):
            write(out/(arm+'_training.json'),rows);print(arm,rows[-1],flush=True)
        def extra(pred,target,ids):
            return magnitude_loss(pred,target,mask[ids],normalizer=expected_mass*len(ids))
        training=fit(p,train,guard,auxiliary,progress,extra_objective=extra if coefficient else None)
        p.eval()
        ck['initial_pretrain_info']=deepcopy(ck['pretrain_info'])
        ck['pretrain_info']=dict(training,dt=.02,dataset_sha256=hashes[str(dataset)],
            teacher_version=train['metadata']['teacher_version'],teacher_parameters=train['metadata']['teacher_parameters'],
            subset_size=104,subset_indices=list(range(104)),learned_channels=['roll','pitch','yaw'],
            loss_contract=dict(version='channel-v4-plus-cruise-bilateral-v1',magnitude_coefficient=coefficient,
                selected_frames=428,cruise_weight=10.,yaw_balance=.7,recovery_window_s=1.))
        ck['continuation']=dict(experiment='bilateral-magnitude-paired-v1',arm=arm,parent_sha256=hashes[str(PARENT/'checkpoint.pt')])
        metrics=evaluate_phase(p,data,info,config)
        with torch.no_grad():quiet=cruise(p)
        diag,arrays=diagnostic(p,train,recorded)
        report=dict(candidate=ck['candidate'],config=config,continuation_epochs=10,**metrics,
            cruise=quiet,diagnostic=diag,research_only=True,training_ready=False,pretrain=ck['pretrain_info'])
        save_phase(out/arm,policy=p,checkpoint=ck,parent_directory=PARENT,report=report,constraints=guard,sample=sample)
        np.savez_compressed(out/arm/'recorded_replay.npz',time=recorded['time'],**arrays)
        export_and_verify(out/arm,sample)
        write(out/arm/'bundled_parity.json',validate(out/arm,out/arm/'numpy',steps=400,execution='bundled'))
        clear=clear_trial(out/arm,out/arm/'clear')
        motion=motion_admission(quiet['final_pwm'][2],clear);temporal=comparison_row(report)
        screen=admission(motion,temporal)
        results[arm]=dict(temporal=temporal,cruise=quiet,clear=clear,motion_admission=motion,
            room_screen=screen,diagnostic=diag)
        write(out/'comparison.json',results);print(arm,'screen',screen,flush=True)
        if screen['passed']:
            dest=out/arm/'flights';dest.mkdir()
            for case in CASES:
                row=trial(case,paths[case],'candidate',dict(before=ROOT/'output/balanced_maneuver_v1/before',candidate=out/arm),dest/case)
                row['arm']=arm;flights.append(row);write(out/'flights.json',flights)
                print(arm,case,row['metrics'],flush=True)
    assert all(sha256(Path(p))==h for p,h in hashes.items())
    write(out/'validation.json',dict(completed_arms=2,completed_room_flights=len(flights),
        input_and_active_bundle_unchanged=True,no_promotion=True))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    run(p.parse_args().out)
