"""Four fixed interface hybrids, isolating changes in input projection and readout."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from copy import deepcopy
import argparse
import json
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
INPUT_PREFIXES=('sensor_layer.','cfc_network.cell.W_in.')
HEAD_PREFIX='dn_head.'
ARMS={'parent':(False,False),'inputs_only':(True,False),'head_only':(False,True),'corrected':(True,True)}


def configure_xy_head(policy):
    for p in policy.parameters():p.requires_grad_(False);p.grad=None
    head=policy.dn_head.proj
    head.weight.requires_grad_(True);head.bias.requires_grad_(True)
    mask=torch.zeros_like(head.weight);mask[2:4]=1
    bias_mask=torch.zeros_like(head.bias);bias_mask[2:4]=1
    return [head.weight.register_hook(lambda grad:grad*mask),head.bias.register_hook(lambda grad:grad*bias_mask)]


def head_prediction(policy,hidden):
    b,t,k=hidden.shape
    return policy.pwm_layer(policy.dn_head(hidden.reshape(b*t,k))).reshape(b,t,4)


def fit_xy_head(policy,data,guard,auxiliary,case,progress):
    import numpy as np
    from optimizer.navigation_loss import navigation_weights,navigation_loss
    from optimizer.release_cruise import cruise_loss,SETTINGS
    from optimizer.bilateral_turn import bilateral_mask,magnitude_loss
    from optimizer.early_turn_training import correction_loss
    from optimizer.maneuver_state import forward_hidden
    x,y,valid,behavior=(data[k] for k in ('X','Y','valid','behavior'))
    weights=navigation_weights(y,valid,behavior,len(data['metadata']['behavior_names']),yaw_balance=.7,amplitude_bins=True,recovery_steps=50)
    bilateral=bilateral_mask(x,y,valid);mass=float(bilateral.sum())/len(x)
    normalizer=float(valid.sum())/len(x)*3
    # Frozen sensors/core make hidden states independent of all head updates.
    # Cache complete reset episodes once; gradients only traverse DN and PWM.
    with torch.no_grad():
        h=torch.cat([forward_hidden(policy,x[i:i+8])[1] for i in range(0,len(x),8)])
        quiet_h=forward_hidden(policy,auxiliary['X'])[1]
        correction_h=forward_hidden(policy,case['X'])[1]
        torch.testing.assert_close(head_prediction(policy,correction_h),policy(case['X'],dt=.02)[0],rtol=0,atol=0)
    fixed={n:p.detach().clone() for n,p in policy.named_parameters() if not n.startswith('dn_head.')}
    old_weight=policy.dn_head.proj.weight.detach().clone();old_bias=policy.dn_head.proj.bias.detach().clone()
    optimizer=torch.optim.Adam([p for p in policy.parameters() if p.requires_grad],lr=.001)
    rng=np.random.default_rng(42);rows=[]
    for epoch in range(10):
        sums=np.zeros(3);batches=0
        for ids in np.array_split(rng.permutation(len(x)),13):
            optimizer.zero_grad(set_to_none=True)
            pred=head_prediction(policy,h[ids])
            recorded=navigation_loss(pred,y[ids],weights[ids],normalizer=normalizer*len(ids))
            quiet=cruise_loss(head_prediction(policy,quiet_h),auxiliary['Y'])
            extra=magnitude_loss(pred,y[ids],bilateral[ids],normalizer=mass*len(ids))+correction_loss(head_prediction(policy,correction_h),case['Y'],case['mask'])
            loss=recorded+10*quiet+extra
            if not torch.isfinite(loss):raise ValueError('Nonfinite head loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(policy.parameters(),1.,error_if_nonfinite=True)
            optimizer.step();policy.post_step();guard(policy)
            assert all(torch.equal(p.detach(),fixed[n]) for n,p in policy.named_parameters() if n in fixed)
            assert torch.equal(policy.dn_head.proj.weight[:2],old_weight[:2]) and torch.equal(policy.dn_head.proj.bias[:2],old_bias[:2])
            sums+=[float(v.detach()) for v in (recorded,quiet,extra)];batches+=1
        with torch.no_grad():final=head_prediction(policy,quiet_h[:1])[0,-1].tolist()
        rows.append(dict(epoch=epoch+1,updates=guard.steps,recorded_loss=float(sums[0]/batches),
            cruise_loss=float(sums[1]/batches),extra_loss=float(sums[2]/batches),quiet_final_pwm=final))
        progress(rows)
    with torch.no_grad():
        torch.testing.assert_close(head_prediction(policy,correction_h),policy(case['X'],dt=.02)[0],rtol=0,atol=0)
    return dict(**{k:v for k,v in SETTINGS.items() if k!='frozen'},progress=rows,
        frozen=['all parameters except DN Pitch/Yaw rows'],cached_hidden_exact=True,validation_indices=[],research_only=True)


def hybrid_policy(parent,corrected,*,new_inputs,new_head):
    old=parent.state_dict();new=corrected.state_dict()
    if old.keys()!=new.keys():raise ValueError('Incompatible policy state keys')
    old_buffers=dict(parent.named_buffers());new_buffers=dict(corrected.named_buffers())
    if old_buffers.keys()!=new_buffers.keys():raise ValueError('Outside interface buffer schema changed')
    for name,value in old_buffers.items():
        if not torch.equal(value,new_buffers[name]):raise ValueError('Outside interface buffer changed: '+name)
    allowed={name for name,p in parent.named_parameters()
        if p.requires_grad and name.startswith(INPUT_PREFIXES+(HEAD_PREFIX,))}
    for name in old:
        equal=torch.equal(old[name],new[name]) if isinstance(old[name],torch.Tensor) else old[name]==new[name]
        if not equal and name not in allowed:raise ValueError('Outside interface changed: '+name)
    mixed={}
    for name,value in old.items():
        take_new=(new_inputs and name.startswith(INPUT_PREFIXES)) or (new_head and name.startswith(HEAD_PREFIX))
        mixed[name]=deepcopy(new[name] if take_new else value)
    policy=deepcopy(parent);policy.load_state_dict(mixed,strict=True);policy.reset_state()
    return policy.eval()


def run(out):
    from artifacts.research import load_research_run
    from edge.runtime import sha256
    from edge.validate_numpy import validate
    from optimizer.early_turn_training import PARENT,SNAPSHOT,correction_case,mirrored_case,correction_metrics
    from optimizer.continue_flow_reflex import DATA,PARENT as ORIGINAL,write,export_and_verify
    from optimizer.finalize_mvp_candidate import evaluate_phase,comparison_row,save_phase
    from optimizer.mvp_constraints import RecurrentConstraints
    from optimizer.probe_history_capacity import cruise
    from optimizer.release_cruise import clear_trial,motion_admission
    from optimizer.bilateral_turn import admission
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    changed=ROOT/'output/early_turn_balanced_v1/corrected'
    parent,ck=load_research_run(PARENT);new,_=load_research_run(changed)
    archive_hashes=json.loads((SNAPSHOT/'SHA256_MANIFEST.json').read_text())
    assert all(sha256(SNAPSHOT/p)==h for p,h in archive_hashes.items())
    case_path=SNAPSHOT/'evidence/early_turn/early_xy_+0/decisions.npz'
    case=correction_case(case_path);pair=mirrored_case(case)
    _,original=load_research_run(ORIGINAL);info=original['pretrain_info']
    data=torch.load(DATA/'train.pt',weights_only=True,map_location='cpu');sample=data['X'][info['validation_indices'][:2]]
    config=json.loads((PARENT/'report.json').read_text())['config']
    active=ROOT/'data/edge/k306_tau025_integrated_v1'
    files=[PARENT/'checkpoint.pt',changed/'checkpoint.pt',case_path,DATA/'train.pt',ORIGINAL/'checkpoint.pt',
        ROOT/'configs/mvp_runtime_v3.json',*[p for p in active.iterdir() if p.is_file()]]
    hashes={str(p):sha256(p) for p in files}
    write(out/'protocol.json',dict(version='interface-factorial-attribution-v1',parent=str(PARENT),changed=str(changed),
        input_sha256=hashes,script_sha256=sha256(__file__),arms=ARMS,
        groups=dict(inputs=list(INPUT_PREFIXES),head=HEAD_PREFIX),
        method='Exact source parameters, no optimizer, no interpolation. Reject any changed non-interface state, including route buffers.',
        evaluation='Same corrected demonstration and mirror, full temporal/elementary evaluation, clear-start for temporally eligible hybrids.',
        limitations='Hybrids change learned coadaptation; results identify sufficient parameter-group interventions, not biological functional localization.',
        no_training=True,no_promotion=True))
    results={}
    for arm,(inp,head) in ARMS.items():
        p=hybrid_policy(parent,new,new_inputs=inp,new_head=head)
        guard=RecurrentConstraints(p,freeze_core=True)
        metrics=evaluate_phase(p,data,info,config);quiet=cruise(p)
        fit=correction_metrics(p,case);mirror=correction_metrics(p,pair)
        payload=deepcopy(ck)
        payload['continuation']=dict(experiment='interface-factorial-attribution-v1',training_performed=False,
            parameter_sources=dict(inputs=str(changed if inp else PARENT),head=str(changed if head else PARENT)),
            parent_checkpoint_sha256=hashes[str(PARENT/'checkpoint.pt')],corrected_checkpoint_sha256=hashes[str(changed/'checkpoint.pt')])
        report=dict(candidate=payload['candidate'],config=config,continuation_epochs=0,**metrics,cruise=quiet,
            correction=fit,paired_correction=mirror,research_only=True,training_ready=False,
            note='Parameter-group transplant, not new training; pretrain_info belongs to parent lineage.')
        save_phase(out/arm,policy=p,checkpoint=payload,parent_directory=PARENT,report=report,constraints=guard,sample=sample)
        export_and_verify(out/arm,sample)
        write(out/arm/'bundled_parity.json',validate(out/arm,out/arm/'numpy',steps=400,execution='bundled'))
        temporal=comparison_row(report)
        tcheck=admission(dict(passed=True),temporal)
        clear=clear_trial(out/arm,out/arm/'clear') if tcheck['passed'] else None
        motion=motion_admission(quiet['final_pwm'][2],clear) if clear else dict(passed=False,reasons=['temporal screen failed; not run'])
        screen=admission(motion,temporal)
        results[arm]=dict(temporal=temporal,cruise=quiet,correction=fit,paired_correction=mirror,clear=clear,screen=screen)
        write(out/'comparison.json',results);print(arm,results[arm],flush=True)
    assert all(sha256(Path(p))==h for p,h in hashes.items())
    assert all(sha256(SNAPSHOT/p)==h for p,h in archive_hashes.items())
    write(out/'validation.json',dict(hybrids_completed=4,archive_and_inputs_unchanged=True,no_training=True,no_promotion=True))


def fly_head_hybrid(out):
    import numpy as np
    from edge.runtime import sha256
    from optimizer.continue_flow_reflex import write
    from optimizer.early_turn_training import PARENT,SNAPSHOT
    from evaluate.practical_maneuver import trial,CASES
    out=Path(out);results=json.loads((out/'comparison.json').read_text())
    if not results['head_only']['screen']['passed']:raise ValueError('Head hybrid not admitted')
    run_dir=out/'head_only';dest=out/'head_only_flights';dest.mkdir(exist_ok=False)
    paths={k:next((ROOT/'flight simulations/changed contract').glob('*'+k+'*.json')) for k in CASES}
    original=json.loads((PARENT/'flights/9d5431c1/manifest.json').read_text())
    for offset in (-3,3):
        m=deepcopy(original);m['start']['yaw_rad']+=float(np.deg2rad(offset))
        path=dest/f'start_{offset:+d}.json';write(path,dict(manifest=m));paths[f'9d_heading_{offset:+d}']=path
    files=[run_dir/'checkpoint.pt',run_dir/'runtime_config.json',out/'comparison.json',*paths.values(),
        *[p for p in (run_dir/'numpy').iterdir() if p.is_file()],
        ROOT/'configs/mvp_runtime_v3.json',*[p for p in (ROOT/'data/edge/k306_tau025_integrated_v1').iterdir() if p.is_file()]]
    hashes={str(p):sha256(p) for p in files}
    write(dest/'protocol.json',dict(version='admitted-head-hybrid-physical-v1',candidate=str(run_dir),cases=list(paths),
        input_sha256=hashes,horizon_s=45.,radius_m=.2,driver='Neural hybrid only; teacher passive observer',
        selection='Only non-parent hybrid preserving all existing admission checks. XY-head training was rejected, not used here.',
        limitation='Known geometries and previously inspected nearby starts; diagnostic validation, not a blind test.',no_promotion=True))
    rows=[]
    for name,path in paths.items():
        row=trial(name,path,'candidate',dict(before=ROOT/'output/balanced_maneuver_v1/before',candidate=run_dir),dest/name)
        rows.append(row);write(dest/'report.json',rows);print(name,row['metrics'],flush=True)
    assert all(sha256(Path(p))==h for p,h in hashes.items())
    archive_hashes=json.loads((SNAPSHOT/'SHA256_MANIFEST.json').read_text())
    assert all(sha256(SNAPSHOT/p)==h for p,h in archive_hashes.items())
    write(dest/'validation.json',dict(trials_completed=6,inputs_active_and_archive_unchanged=True,no_promotion=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--fly-head-hybrid',action='store_true');args=parser.parse_args()
    fly_head_hybrid(args.out) if args.fly_head_hybrid else run(args.out)
