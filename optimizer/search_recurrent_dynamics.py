"""Four-trial Optuna grid: equal-budget continuation of both pilot candidates.

Scales initialize weights and tau; both remain trainable afterward. This is a
bounded continuation screen, not a new model architecture or a full HPO study.
"""
import argparse
import json
from pathlib import Path
import shutil

import optuna
import torch

from artifacts.io import publication, write_json
from generator.source_contract import file_sha256
from artifacts.research import load_research_run
from optimizer.candidate_selection import contract_sha256
from optimizer.optuna_tuner import feasible_pareto_trials
from optimizer.pretrain import pretrain_policy
from optimizer.probe_recurrent_dynamics import scaled_policy, summarize_probe
from optimizer.probe_yaw_recovery import evaluate
from core.invariants import verify_recurrent_contract
from optimizer.rollout import BENCHMARK_VERSION
from optimizer.search_pipeline import GATE
from simulation.control_contract import control_contract


def trial_settings():
    return dict(tau_scale=[.25, .5], recurrent_scale=[.75], epochs=10,
                learning_rate=.001, loss_mode='channel-v3', device='cpu')


def run_search(run, dataset, sensitivity, output):
    run, dataset, sensitivity, output = map(Path, (run, dataset, sensitivity, output))
    if output.exists():
        raise FileExistsError('Use a new output directory; prior trials are immutable')
    parent = json.loads((run/'preflight.json').read_text(encoding='utf-8'))
    grid = json.loads(sensitivity.read_text(encoding='utf-8'))
    if (grid['parent_experiment_fingerprint'] != parent['experiment_fingerprint']
            or grid['gate'] != GATE or not any(c['shortlisted'] for c in grid['candidates'])):
        raise ValueError('Expected a matching sensitivity grid with a promising region')
    digest = file_sha256(dataset)
    if digest != parent['experiment']['dataset_sha256']:
        raise ValueError('Must use original training dataset')
    data = torch.load(dataset, weights_only=True, map_location='cpu')
    if file_sha256(dataset) != digest:
        raise ValueError('Dataset changed during loading')
    parents = {}
    for directory in sorted(run.glob('trial_*')):
        policy, checkpoint = load_research_run(directory)
        report = json.loads((directory/'report.json').read_text(encoding='utf-8'))
        if report['experiment_fingerprint'] != parent['experiment_fingerprint']:
            raise ValueError('Mixed parent trials')
        parents[checkpoint['candidate']] = (directory, report, file_sha256(directory/'checkpoint.pt'))
    if set(parents) != set(parent['eligible_candidates']):
        raise ValueError('Incomplete parent candidate coverage')
    settings = trial_settings()
    root = Path(__file__).resolve().parents[1]
    files = set(parent['experiment']['code_sha256']) | {
        'optimizer/search_recurrent_dynamics.py', 'optimizer/probe_recurrent_dynamics.py',
        'optimizer/probe_yaw_recovery.py', 'optimizer/benchmark_candidate.py', 'core/invariants.py', 'artifacts/research.py', 'artifacts/io.py'}
    hashes = {p:file_sha256(root/p) for p in sorted(files)}
    contract = dict(version='dynamics-continuation-grid-v1', settings=settings,
        parent_experiment_fingerprint=parent['experiment_fingerprint'],
        parent_checkpoints={name:p[2] for name,p in parents.items()},
        dataset_sha256=digest, sensitivity_sha256=file_sha256(sensitivity),
        code_sha256=hashes, gate=GATE,
        objectives=['max_recovery_s_censored_at_2s', 'mean_final_yaw_amplitude_error_pwm', 'heldout_yaw_mae_pwm'],
        recovery_censoring='A failure to settle within the 2s probe is scored as 2s and always fails admission.',
        limitations=['One seed, 10 extra epochs, fresh Adam for every trial; initialization scales can drift during training.',
            'Known validation/probes used for development, not independent confirmation.',
            'k306 did not pass the untrained shortlist; included for equal-budget transfer comparison.',
            'No external evaluation dataset read. No thresholds relaxed.'])
    fingerprint = contract_sha256(contract)
    with publication(output) as stage:
        write_json(stage/'preflight.json', dict(experiment=contract, experiment_fingerprint=fingerprint,
                                              training_ready=False, research_only=True))
    study = optuna.create_study(study_name='dynamics_continuation_v1',
        storage='sqlite:///'+(output.resolve()/'study.db').as_posix(),
        sampler=optuna.samplers.GridSampler(dict(candidate=sorted(parents),
            tau_scale=settings['tau_scale'], recurrent_scale=settings['recurrent_scale']), seed=42),
        directions=['minimize']*3)
    study.set_user_attr('experiment_fingerprint', fingerprint)
    study.set_user_attr('experiment_contract', contract)

    def objective(trial):
        name = trial.suggest_categorical('candidate', sorted(parents))
        tau = trial.suggest_categorical('tau_scale', settings['tau_scale'])
        gain = trial.suggest_categorical('recurrent_scale', settings['recurrent_scale'])
        directory, parent_report, checkpoint_hash = parents[name]
        original, checkpoint = load_research_run(directory)
        if file_sha256(directory/'checkpoint.pt') != checkpoint_hash:
            raise ValueError('Parent checkpoint changed')
        policy = scaled_policy(original, tau_scale=tau, recurrent_scale=gain)
        info = checkpoint['pretrain_info']
        config = parent_report['config']
        torch.manual_seed(config['seed'])
        print(name, 'tau scale', tau, 'recurrent scale', gain, 'training', flush=True)
        pretrain_policy(policy, data, epochs=settings['epochs'], subset_ratio=1.,
            batch_size=config['batch_size'], lr=settings['learning_rate'], seed=config['seed'],
            device='cpu', research_only=True, loss_mode=settings['loss_mode'], yaw_balance=config['yaw_balance'])
        for key in ('subset_indices', 'validation_indices'):
            if policy._pretrain_info[key] != info[key]:
                raise ValueError('Training/validation split changed')
        verify_recurrent_contract(policy.cfc_network.cell)
        metrics = evaluate(policy, data, info['validation_indices'], config)
        summary = summarize_probe(metrics['spatial_response'], GATE)
        cell = policy.cfc_network.cell
        learned_dynamics = dict(tau_min_s=float(cell.tau.min()), tau_max_s=float(cell.tau.max()),
            recurrent_frobenius_norm=float(cell._effective_W().detach().norm()))
        with publication(output/f'trial_{trial.number:05d}') as stage:
            shutil.copytree(directory/'graph', stage/'graph')
            checkpoint.update(state_dict=policy.state_dict(), pretrain_info=policy._pretrain_info,
                parent_checkpoint_sha256=checkpoint_hash,
                continuation=dict(**settings, selected_tau_scale=tau, selected_recurrent_scale=gain,
                                  optimizer_reset=True, scales_are_initialization_only=True))
            torch.save(checkpoint, stage/'checkpoint.pt')
            report = dict(candidate=name, trial_number=trial.number,
                checkpoint_sha256=file_sha256(stage/'checkpoint.pt'), parent_checkpoint_sha256=checkpoint_hash,
                experiment_fingerprint=fingerprint, parameters=dict(trial.params),
                pretrain=policy._pretrain_info, learned_dynamics=learned_dynamics,
                summary=summary, **metrics, research_only=True, training_ready=False)
            write_json(stage/'report.json', report)
            restored, _ = load_research_run(stage)
            with torch.no_grad():
                sample = data['X'][info['validation_indices'][:2]]
                torch.testing.assert_close(policy(sample)[0], restored(sample)[0], rtol=0, atol=0)
            report['checkpoint_roundtrip_exact'] = True
            write_json(stage/'report.json', report)
        admission = metrics['functional_admission']['passed']
        for key, value in dict(benchmark_version=BENCHMARK_VERSION, control_contract=control_contract(),
                              feasible=admission, constraints=[0. if admission else 1.],
                              experiment_fingerprint=fingerprint, functional_admission=metrics['functional_admission']).items():
            trial.set_user_attr(key, value)
        print(name, tau, summary, 'full admission', admission, flush=True)
        return (max(r['recovery_settle_s'] if r['recovery_settle_s'] is not None else 2.
                    for r in metrics['spatial_response']['cases']),
                summary['mean_final_amplitude_error_pwm'], metrics['heldout']['channel_mae_pwm']['yaw'])

    study.optimize(objective, n_trials=len(parents)*len(settings['tau_scale']))
    if hashes != {p:file_sha256(root/p) for p in hashes} or file_sha256(dataset) != digest:
        raise ValueError('Experiment code/data changed')
    summary = dict(experiment_fingerprint=fingerprint, training_ready=False,
        completed_trials=len(study.trials), feasible_pareto_trials=[t.number for t in feasible_pareto_trials(study)],
        trials=[dict(number=t.number, parameters=t.params, values=t.values,
                     passed=t.user_attrs['feasible']) for t in study.trials])
    write_json(output/'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'dataset', 'sensitivity', 'out-dir'):
        parser.add_argument('--'+name, required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run_search(args.run, args.dataset, args.sensitivity, args.out_dir)


if __name__ == '__main__':
    main()
