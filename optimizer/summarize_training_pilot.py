"""Compare exact saved pilot weights, with a separate external evaluation set.

Run only after the pilot finishes; do not tune on this report's external scores.
"""
import argparse
import json
from pathlib import Path

import torch

from artifacts.io import publication, write_json
from generator.prepare_reflex_training import audit_labels
from generator.source_contract import file_sha256
from generator.continuous_reflex_teacher import CorrectedExpertReflexPolicy
from generator.reflex_contract import TEACHER_VERSION
from artifacts.research import load_research_run
from optimizer.diagnose_candidate import heldout_predictions, prediction_metrics
from optimizer.rollout import simulate_policy_rollout
from scripts.audit_tof_threats import TeacherAdapter
from simulation.altitude_control import AltitudeHold
from simulation.benchmark_scenes import NavigationScene, SCENES
from simulation.policy import ChongFlyMSPPolicy


def summarize(run, bundle, output):
    run, bundle = Path(run), Path(bundle)
    manifest=json.loads((bundle/'manifest.json').read_text(encoding='utf-8'))
    preflight=json.loads((run/'preflight.json').read_text(encoding='utf-8'))
    if manifest['teacher_version'] != TEACHER_VERSION:
        raise ValueError('Unexpected teacher bundle')
    for name in ('train','evaluation'):
        if file_sha256(bundle/f'{name}.pt') != manifest['datasets'][name]['file_sha256']:
            raise ValueError('Dataset hash mismatch')
    if preflight['experiment']['dataset_sha256'] != manifest['datasets']['train']['file_sha256']:
        raise ValueError('Pilot trained on a different dataset')
    data=torch.load(bundle/'evaluation.pt',weights_only=True,map_location='cpu')
    audit=audit_labels(data)
    rows=[]
    for trial in sorted(run.glob('trial_*')):
        report=json.loads((trial/'report.json').read_text(encoding='utf-8'))
        if report['experiment_fingerprint'] != preflight['experiment_fingerprint']:
            raise ValueError('Mixed experiment trials')
        learned, checkpoint=load_research_run(trial)
        if checkpoint['pretrain_info']['teacher_version'] != TEACHER_VERSION:
            raise ValueError('Unexpected checkpoint teacher')
        # Same initialization seed, graph and options as candidate_objective.
        torch.manual_seed(report['config']['seed'])
        meta=next((trial/'graph').glob('meta_*.json'))
        initial=ChongFlyMSPPolicy.from_meta(str(meta),dt=checkpoint['dt'],**report['policy_options'])
        ids=list(range(len(data['X'])))
        external={}
        for name,policy in [('initial',initial),('trained',learned)]:
            prediction=heldout_predictions(policy,data,ids)
            external[name]=prediction_metrics(prediction,data['Y'],data['valid'],data['metadata']['episode_scenarios'])
        info=report['pretrain']
        row=dict(candidate=report['candidate'],trial=trial.name,
            checkpoint_sha256=report['checkpoint_sha256'],checkpoint_roundtrip_exact=report['checkpoint_roundtrip_exact'],
            epochs=info['epochs'],train_episodes=info['subset_size'],validation_episodes=len(info['validation_indices']),
            training_loss_history=info['loss_history'],validation_loss_history=info['validation_loss_history'],
            validation=report['heldout'],external=external,
            functional_admission=report['functional_admission'],
            scenes=report['scenes'],spatial_response=report['spatial_response'])
        rows.append(row)
        print(report['candidate'], 'external MAE before/after:',
              external['initial']['channel_mae_pwm'],external['trained']['channel_mae_pwm'],flush=True)
    if {r['candidate'] for r in rows} != set(preflight['eligible_candidates']):
        raise ValueError('Pilot does not cover every eligible candidate')
    teachers=[]
    config=preflight['experiment']['config']
    for scenario in SCENES:
        for seed in config['scene_seeds']:
            env=NavigationScene(scenario)
            try:
                _,metrics=simulate_policy_rollout(TeacherAdapter(CorrectedExpertReflexPolicy),env=env,
                    eval_steps=config['scene_steps'],seed=seed,altitude_hold=AltitudeHold())
                teachers.append(dict(scenario=scenario,seed=seed,**metrics))
            finally:
                env.close()
    root=Path(__file__).resolve().parents[1]
    with publication(output) as stage:
        result=dict(version='teacher-v5-pilot-comparison-v1',research_only=True,training_ready=False,
            teacher_version=TEACHER_VERSION,experiment_fingerprint=preflight['experiment_fingerprint'],
            evaluation_dataset_sha256=manifest['datasets']['evaluation']['file_sha256'],
            evaluation_audit=audit,teacher_scenes=teachers,candidates=rows,
            code_sha256={p:file_sha256(root/p) for p in ['optimizer/summarize_training_pilot.py',
                'optimizer/benchmark_candidate.py','optimizer/diagnose_candidate.py','optimizer/rollout.py',
                'simulation/policy.py','simulation/policy_diagnostics.py','core/models.py',
                'generator/generate_reflex_dataset.py','generator/continuous_reflex_teacher.py', 'artifacts/research.py', 'artifacts/io.py', 'core/invariants.py']},
            limitations=['Two graphs, one initialization seed, fixed first-trial hyperparameters; not an optimized champion.',
                'External evaluation uses different initial scenes in familiar families, not new scenario families.',
                'External errors are on teacher trajectories; closed-loop outcomes are reported separately.',
                'Initial baseline reconstructs initialization with the recorded seed and current model code.',
                'Failures are retained. No thresholds adjusted to admit a candidate.'])
        write_json(stage/'report.json',result)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
        for row in rows:
            axes[0].plot(range(1,len(row['training_loss_history'])+1),row['training_loss_history'],label=row['candidate'])
            axes[1].plot(range(1,len(row['validation_loss_history'])+1),row['validation_loss_history'],label=row['candidate'])
        for ax,title in zip(axes,('Training loss (weighted)','Validation loss (unweighted)')):
            ax.set(title=title,xlabel='Epoch',ylabel='Loss');ax.grid(alpha=.2);ax.legend()
        fig.suptitle('V5 pilot | compare curves within each panel, not train/validation scales')
        fig.savefig(stage/'learning_curves.png',dpi=140);plt.close(fig)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','bundle','out-dir'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(1)
    summarize(args.run,args.bundle,args.out_dir)


if __name__ == '__main__':
    main()
