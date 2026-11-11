"""One bounded paired full-episode experiment; immutable baseline, no promotion."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
from copy import deepcopy
import json
from pathlib import Path
import torch

from artifacts.research import load_research_run
from edge.runtime import sha256
from edge.system import RunConfig, build_system
from optimizer.continue_flow_reflex import ROOT,PARENT,DATA,write,score_raw,export_and_verify
from optimizer.continue_maneuver_data import combine
from optimizer.maneuver_relabel import replay_cruise_targets
from optimizer.probe_plastic_history import PlasticConstraints
from optimizer.probe_history_capacity import make_cases,cruise
from optimizer.history_objective import sequence_metrics
from optimizer.pretrain import pretrain_policy
from optimizer.mvp_constraints import RecurrentConstraints
from optimizer.diagnose_candidate import heldout_predictions
from optimizer.finalize_mvp_candidate import evaluate_phase,save_phase,comparison_row


def run(output,device,balanced=False):
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    files=dict(old=DATA/'train.pt',correction=ROOT/'data/maneuver_corrections_v1/train.pt',
        old_external=DATA/'evaluation.pt',corrective=ROOT/'data/maneuver_corrections_v1/evaluation.pt',
        expert=ROOT/'data/maneuver_sequences_v1/evaluation.pt')
    inputs={str(p):sha256(p) for p in files.values()}
    raw={k:torch.load(p,weights_only=True,map_location='cpu') for k,p in files.items()}
    selection_path=ROOT/'output/neural_maneuver_data_v1/protocol.json'
    inputs[str(selection_path)]=sha256(selection_path)
    selection=json.loads(selection_path.read_text())['datasets']['mixed']['source_episodes']
    old_ids=[i for name,i in selection if name=='old']
    new_ids=[i for name,i in selection if name=='correction']
    _,checkpoint=load_research_run(PARENT);info=checkpoint['pretrain_info']
    assert len(old_ids)==72 and len(new_ids)==32
    assert not set(old_ids)&set(info['validation_indices'])
    assert not ({r['pair_id'] for r in raw['correction']['metadata']['episodes']} &
                {r['pair_id'] for r in raw['corrective']['metadata']['episodes']})
    mixed=combine([(raw['old'],old_ids,'old'),(raw['correction'],new_ids,'correction')],steps=600)
    train=replay_cruise_targets(mixed,cruise_pwm=1530.)
    evaluation={k:replay_cruise_targets(raw[k],cruise_pwm=1530.) for k in ('old_external','corrective','expert')}
    torch.save(train,out/'train.pt')
    torch.save(evaluation,out/'evaluation.pt')
    auxiliary=None
    if balanced:
        from optimizer.balanced_maneuvers import auxiliary_cases,STAGES,EVAL_FLOWS,SETTINGS
        auxiliary=dict(training={n:auxiliary_cases(n)[0] for n in STAGES},
            evaluation={n:auxiliary_cases(n,EVAL_FLOWS)[0] for n in (150,400)},cruise=auxiliary_cases(12)[1])
        torch.save(auxiliary,out/'auxiliary.pt')
    fixed=json.loads((ROOT/'output/neural_refocus_v1/protocol.json').read_text())['fixed_files_sha256']
    assert all(sha256(ROOT/p)==v for p,v in fixed.items())
    sources={str(p.relative_to(ROOT)):sha256(p) for folder in ('optimizer','generator','core','simulation','edge','artifacts','configs') for p in (ROOT/folder).glob('*.py')}
    protocol=dict(version='full-maneuver-cruise1530-v1',parent_sha256=sha256(PARENT/'checkpoint.pt'),
        input_sha256=inputs,source_sha256=sources,fixed_baseline_sha256=fixed,
        prepared_sha256={p.name:sha256(p) for p in (out/'train.pt',out/'evaluation.pt')},
        source_episodes=selection,teacher_revision=train['metadata']['target_revision'],
        settings=dict(seed=42,epochs=10,updates=130,batch_size=8,lr=.001,device=device,
            loss_mode='channel-v4',yaw_balance=.7,validation_fraction=0.,
            arms=['before','frozen','plastic'],plastic='W/tau/interface trainable; A frozen; same topology/signs and norm cap',
            frozen='Only interface trainable',extra_parent_anchors=False,history_probes_training=False),
        criteria=dict(regression='7/7 scenes, 12/12 direction/recovery, recovery<=0.8s, narrow=8; neutral handover snapshot<=35 PWM',
            improvement='Report all external episodes vs revised AND original labels, especially braking after clearance; no best epoch selection',
            promotion=False),
        limits='Revised teacher commands on recorded inputs; no new geometry or closed-loop collection. One seed. Existing external sets previously examined.')
    write(out/'protocol.json',protocol)
    if balanced:
        protocol['version']='balanced-full-maneuvers-v1'
        protocol['settings']={**protocol['settings'],**SETTINGS,'epochs':None,'history_probes_training':True}
        protocol['prepared_sha256']['auxiliary.pt']=sha256(out/'auxiliary.pt')
        protocol['criteria']['behavioral_screen']='Cruise Pitch1515..1535, clear mean forward speed>=0.10m/s, all heldout history suffix errors<=20 PWM; plus scene/yaw/handover conditions'
        write(out/'protocol.json',protocol)
    config=json.loads((PARENT/'report.json').read_text())['config']
    runtime_config=RunConfig.from_json(ROOT/'configs/mvp_runtime_v3.json');build_system(runtime_config)
    results={}
    for arm in ('before','frozen','plastic'):
        policy,payload=load_research_run(PARENT)
        guard=PlasticConstraints(policy) if arm=='plastic' else RecurrentConstraints(policy,metadata=payload['training_constraints'])
        policy.to(device)
        if arm!='before':
            torch.manual_seed(42)
            if device=='cuda':torch.cuda.manual_seed_all(42)
            def post_step(p):
                guard(p)
                if guard.steps%13==0:print(arm,'epoch',guard.steps//13,'/10',flush=True)
            if balanced:
                from optimizer.balanced_maneuvers import fit
                def progress(rows):
                    write(out/(arm+'_progress.json'),rows)
                    print(arm,rows[-1]['updates'],'suffix',rows[-1]['suffix'],
                        'history max',rows[-1]['sequence']['suffix_max_error_pwm'],
                        'cruise',rows[-1]['cruise_final_pwm'][2],flush=True)
                policy._pretrain_info=fit(policy,train,guard,auxiliary,progress_callback=progress)
                assert guard.steps==300
            else:
                pretrain_policy(policy,train,epochs=10,subset_ratio=1.,batch_size=8,seed=42,
                    validation_fraction=0.,lr=.001,device=device,research_only=True,
                    loss_mode='channel-v4',yaw_balance=.7,post_step_callback=post_step)
                assert guard.steps==130
            assert policy._pretrain_info['validation_indices']==[]
            payload['initial_pretrain_info']=deepcopy(payload['pretrain_info'])
            payload['pretrain_info']=deepcopy(policy._pretrain_info)
        policy.eval();training_audit=guard.audit();aux_metrics={}
        quiet=cruise(policy);scores={};history={}
        with torch.no_grad():
            if balanced:
                aux_metrics={split:{str(n):sequence_metrics(policy(d['X'].to(device),dt=.02)[0],d['Y'].to(device))
                    for n,d in ds.items()} for split,ds in auxiliary.items() if split!='cruise'}
            for name,d in evaluation.items():
                ids=list(range(len(d['X'])))
                pred=heldout_predictions(policy,d,ids)
                scores[name]=dict(revised=score_raw(pred,d,ids),original=score_raw(pred,raw[name],ids))
            for suffix in (150,400):
                d=make_cases((.08,.10,.12,.16),suffix)
                # Keep historical v5-1600 targets for comparability and add v5-1530.
                pred=policy(d['X'].to(device),dt=.02)[0].cpu()
                from generator.generate_reflex_dataset import make_teacher
                target=d['Y'].clone()
                for i,x in enumerate(d['X']):
                    teacher=make_teacher('geometry-reflex-v5',noise_std_pwm=0,pwm_cruise_pitch=1530.)
                    for t,row in enumerate(x):target[i,t]=torch.from_numpy(teacher.step(row.numpy(),dt=.02))
                history[str(suffix)]=dict(original=sequence_metrics(pred,d['Y']),revised=sequence_metrics(pred,target))
        policy.cpu().eval()
        metrics=evaluate_phase(policy,raw['old'],info,config)
        deviation=max(abs(v-1500) for v in quiet['final_pwm'][1:3])
        handover=dict(deviation_pwm=deviation,threshold_pwm=runtime_config.bridge.max_attitude_difference_pwm,
            passed=deviation<=runtime_config.bridge.max_attitude_difference_pwm,scope='Snapshot condition only, not full handover')
        payload['continuation']=dict(experiment=protocol['version'],arm=arm,
            parent_sha256=protocol['parent_sha256'],teacher_revision=train['metadata']['target_revision'],
            training_constraint_audit=training_audit,export_freezing='After training')
        report=dict(candidate=payload['candidate'],config=config,continuation_epochs=0 if arm=='before' else None if balanced else 10,
            continuation_updates=guard.steps,cruise=quiet,raw=scores,history=history,handover=handover,
            auxiliary=aux_metrics,training_constraint_audit=training_audit,**metrics,research_only=True,training_ready=False)
        if balanced:
            from optimizer.balanced_maneuvers import behavioral_admission
            report['behavioral_admission']=behavioral_admission(dict(temporal=comparison_row(report),cruise=quiet,handover=handover),
                clear_speed=next(s for s in metrics['scenes'] if s['scenario']=='clear')['mean_fwd_speed'],
                history_error=max(v['suffix_max_error_pwm'] for v in aux_metrics['evaluation'].values()))
        frozen_export=RecurrentConstraints(policy,freeze_core=True)
        sample=raw['old']['X'][info['validation_indices'][:2]]
        report=save_phase(out/arm,policy=policy,checkpoint=payload,parent_directory=PARENT,
            report=report,constraints=frozen_export,sample=sample)
        export_and_verify(out/arm,sample)
        results[arm]=dict(temporal=comparison_row(report),cruise=quiet,handover=handover,
            raw=scores,history=history,auxiliary=aux_metrics,behavioral_admission=report.get('behavioral_admission'),training_audit=training_audit)
        write(out/'comparison.json',results)
        print(arm,'done',dict(temporal=results[arm]['temporal'],handover=handover),flush=True)
        del policy
        if device=='cuda':torch.cuda.empty_cache()
    assert all(sha256(ROOT/p)==v for m in (fixed,sources) for p,v in m.items())
    assert all(sha256(Path(p))==v for p,v in inputs.items())
    assert all(sha256(out/p)==v for p,v in protocol['prepared_sha256'].items())
    write(out/'validation.json',dict(completed_arms=3,all_input_source_and_baseline_hashes_unchanged=True,
        checkpoint_and_numpy_roundtrips_verified=True,automatic_promotion=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--device',choices=('cpu','cuda'),required=True)
    parser.add_argument('--balanced',action='store_true')
    args=parser.parse_args();run(args.out,args.device,args.balanced)
