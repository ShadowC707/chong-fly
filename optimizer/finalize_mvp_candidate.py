"""Four fixed-budget MVP continuations with frozen tau/A and bounded W.

Uses the original v5 training bundle. Preserves the old gate as a diagnostic;
MVP admission is the explicitly agreed, separately versioned criterion.
"""
import argparse
import json
import math
from pathlib import Path
import shutil

import optuna
import torch

from configs.flight_config import CONTROL_DT
from artifacts.io import publication, write_json
from generator.reflex_contract import TEACHER_VERSION
from generator.source_contract import file_sha256
from artifacts.research import load_research_run
from optimizer.candidate_selection import contract_sha256
from optimizer.mvp_constraints import RecurrentConstraints
from optimizer.optuna_tuner import feasible_pareto_trials
from optimizer.pretrain import pretrain_policy
from optimizer.probe_recurrent_dynamics import scaled_policy, summarize_probe
from optimizer.probe_yaw_recovery import evaluate
from core.invariants import verify_recurrent_contract
from optimizer.rollout import BENCHMARK_VERSION
from optimizer.search_pipeline import GATE
from simulation.benchmark_scenes import SCENES
from simulation.control_contract import control_contract


MVP_GATE = dict(version='course-mvp-navigation-v1', direction_required=12,
    recovery_required=12, recovery_max_s=.8, tolerance_pwm=20., broad_joint_required=4,
    narrow_required=8, narrow_min_signed_yaw_pwm=80., scenes_required=7)
CANDIDATES = {'bridge_role_k301', 'bridge_role_k306'}


def trial_settings(*, freeze_core=False):
    result = dict(tau_scale=[.25, .5], recurrent_scale=[.75], epochs=10,
                learning_rate=.001, loss_mode='channel-v4', device='cpu',
                frozen_parameters=['tau_raw', 'A'], recovery_window_s=1.)
    if freeze_core:
        result.update(frozen_parameters=['tau_raw', 'A', '_w_macro_param'],
                      decoder_norm_policy='monitor_only')
    return result


def assess_mvp(metrics, *, scene_seed, scene_steps):
    """Fail closed on missing cases, weak turns, crashes or incomplete scenes.

    Broad 'joint' deliberately retains all existing broad-case conditions,
    including onset and clear yaw. Narrow amplitude is signed, not absolute.
    Old held-out MAE/burst/clearance gates remain reported, not added here.
    """
    probe = metrics['spatial_response']
    cases = probe.get('cases', [])
    expected = {(side, distance, width) for side in ('left', 'right')
                for distance in (.1, .2) for width in (1, 2, 4)}
    identities = {(r.get('obstacle_side'), r.get('normalized_distance'), r.get('active_columns'))
                  for r in cases}
    reasons = []
    if (len(cases) != 12 or identities != expected or probe.get('dt') != CONTROL_DT
            or probe.get('teacher_version') != TEACHER_VERSION
            or probe.get('tolerance_pwm') != MVP_GATE['tolerance_pwm']
            or probe.get('recovery_s') != 2.):
        return dict(passed=False, reasons=['invalid_temporal_contract'], thresholds=dict(MVP_GATE))
    summary = summarize_probe(probe, GATE)
    narrow = []
    direction_count = 0
    for row in cases:
        sign = 1 if row['obstacle_side'] == 'left' else -1
        command = row['threat_yaw_pwm']
        finite = isinstance(command, (int, float)) and math.isfinite(command)
        signed = sign*(command-1500) if finite else None
        direction_count += int(row['direction_correct'] is True and finite and signed > 20)
        if row['active_columns'] in (1, 2):
            narrow.append(dict(obstacle_side=row['obstacle_side'],
                normalized_distance=row['normalized_distance'], active_columns=row['active_columns'],
                signed_yaw_pwm=signed, passed=bool(finite and signed >= 80)))
    narrow_count = sum(r['passed'] for r in narrow)
    if direction_count != 12: reasons.append('direction')
    if summary['recovery_passed'] != 12: reasons.append('recovery')
    if summary['broad_joint_passed'] != 4: reasons.append('broad_joint')
    if narrow_count != 8: reasons.append('narrow_amplitude')
    scenes = metrics.get('scenes', [])
    if len(scenes) != 7 or {(r.get('scenario'), r.get('seed')) for r in scenes} != {
            (name, scene_seed) for name in SCENES}:
        reasons.append('missing_or_duplicate_scenes')
    scene_results = []
    for row in scenes:
        passed = (row.get('benchmark_version') == BENCHMARK_VERSION
            and row.get('control_contract') == control_contract()
            and row.get('feasible') is True and row.get('crashed') is False
            and row.get('fatal_failure') is False and row.get('terminated') is False
            and row.get('horizon_completed') is True and row.get('control_steps') == scene_steps)
        scene_results.append(dict(scenario=row.get('scenario'), passed=passed))
        if not passed: reasons.append(f"scene:{row.get('scenario')}")
    return dict(passed=not reasons, reasons=reasons, thresholds=dict(MVP_GATE),
        direction_passed=direction_count, recovery_passed=summary['recovery_passed'],
        broad_joint_passed=summary['broad_joint_passed'], narrow_amplitude_passed=narrow_count,
        narrow_cases=narrow, scenes_passed=sum(r['passed'] for r in scene_results),
        scene_results=scene_results, scope='course_simulation_experiment_not_flight_admission')


def evaluate_phase(policy, data, info, config):
    metrics = evaluate(policy, data, info['validation_indices'], config)
    return dict(**metrics, summary=summarize_probe(metrics['spatial_response'], GATE),
                mvp_admission=assess_mvp(metrics, scene_seed=config['scene_seeds'][0],
                                        scene_steps=config['scene_steps']))


def save_phase(destination, *, policy, checkpoint, parent_directory, report, constraints, sample):
    """Publish an exact checkpoint and its graph; verify reload before publication."""
    with publication(destination) as stage:
        shutil.copytree(parent_directory/'graph', stage/'graph')
        payload = dict(checkpoint, state_dict=policy.state_dict(),
                       training_constraints=constraints.metadata(),
                       frozen_parameters=constraints.frozen_parameters, w_max_norm=constraints.w_max_norm)
        torch.save(payload, stage/'checkpoint.pt')
        report = dict(report, checkpoint_sha256=file_sha256(stage/'checkpoint.pt'),
                      constraint_audit=constraints.audit())
        write_json(stage/'report.json', report)
        restored, saved = load_research_run(stage)
        restored_constraints = RecurrentConstraints(restored, metadata=saved['training_constraints'])
        assert restored_constraints.w_max_norm == constraints.w_max_norm
        policy.eval()
        with torch.no_grad():
            torch.testing.assert_close(policy(sample)[0], restored(sample)[0], rtol=0, atol=0)
        report['checkpoint_roundtrip_exact'] = True
        report['freeze_metadata_roundtrip_verified'] = True
        write_json(stage/'report.json', report)
    return report


def comparison_row(report):
    admission, summary = report['mvp_admission'], report['summary']
    narrow = admission['narrow_cases']
    return dict(epochs=report['continuation_epochs'],
        direction=admission['direction_passed'], recovery=admission['recovery_passed'],
        broad_joint=admission['broad_joint_passed'], narrow_amplitude=admission['narrow_amplitude_passed'],
        narrow_min_signed_yaw_pwm=min(r['signed_yaw_pwm'] for r in narrow),
        max_recovery_s=summary['max_recovery_s'], scenes=admission['scenes_passed'],
        yaw_mae_pwm=report['heldout']['channel_mae_pwm']['yaw'],
        old_gate_passed=report['functional_admission']['passed'], mvp_passed=admission['passed'])


def write_comparison(output, rows, *, freeze_core=False):
    lines = ['# Порівняння MVP: до навчання та після 10 епох', '',
        ('Ті самі початкові checkpoint і дані v5; tau/A/W повністю заморожені. Норма декодера лише вимірюється.'
         if freeze_core else 'Ті самі початкові checkpoint і дані v5; tau/A заморожені, W обмежена нормою після масштабування.'),
        'Loss: channel-v4. CPU, один seed, свіжий Adam. Це дослідницький допуск у симуляції.', '',
        '| Кандидат | tau × | Епохи | Напрямок /12 | Згасання /12 | Широкі /4 | Вузькі ≥80 /8 | Мін. вузька команда | Макс. згасання, с | Сцени /7 | MAE yaw | MVP | Старий GATE |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|']
    for row in sorted(rows, key=lambda x: (x['candidate'], x['tau_scale'])):
        for phase in ('before', 'after'):
            r = row[phase]
            settle = '>2 / не всі згасли' if r['max_recovery_s'] is None else f"{r['max_recovery_s']:.2f}"
            lines.append(f"| {row['candidate']} | {row['tau_scale']} | {r['epochs']} | {r['direction']} | "
                f"{r['recovery']} | {r['broad_joint']} | {r['narrow_amplitude']} | {r['narrow_min_signed_yaw_pwm']:.2f} | "
                f"{settle} | {r['scenes']} | {r['yaw_mae_pwm']:.2f} | {'так' if r['mvp_passed'] else 'ні'} | "
                f"{'так' if r['old_gate_passed'] else 'ні'} |")
    lines += ['', 'Мінімальна вузька команда — відхилення від 1500 у потрібному напрямку, не абсолютна RC-команда.',
        'Широкі /4 — повний broad_joint: амплітуда, напрямок, onset, згасання та нейтральний фон.',
        'Старий GATE збережений без змін і не визначає новий MVP-допуск.',
        'Повні 12 стимулів, фізичні сцени та довша діагностика згасання містяться у report.json кожної фази.']
    (output/'comparison.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def run_search(run, dataset, output, *, freeze_core=False):
    run, dataset, output = map(Path, (run, dataset, output))
    if output.exists():
        raise FileExistsError('Use a new output directory; prior results are immutable')
    if type(freeze_core) is not bool:
        raise ValueError('freeze_core must be boolean')
    parent = json.loads((run/'preflight.json').read_text(encoding='utf-8'))
    digest = file_sha256(dataset)
    if digest != parent['experiment']['dataset_sha256']:
        raise ValueError('Must use the original v5 training dataset')
    data = torch.load(dataset, weights_only=True, map_location='cpu')
    if file_sha256(dataset) != digest:
        raise ValueError('Dataset changed during loading')
    parents = {}
    for directory in sorted(run.glob('trial_*')):
        _, checkpoint = load_research_run(directory)
        report = json.loads((directory/'report.json').read_text(encoding='utf-8'))
        name = checkpoint['candidate']
        if report['experiment_fingerprint'] != parent['experiment_fingerprint'] or name in parents:
            raise ValueError('Mixed or duplicate parent trials')
        config = report['config']
        if len(config['scene_seeds']) != 1 or config['scene_steps'] != 500:
            raise ValueError('Expected seven original ten-second scenes at one seed')
        if checkpoint['pretrain_info']['teacher_version'] != TEACHER_VERSION:
            raise ValueError('Expected original v5-trained checkpoints')
        parents[name] = (directory, report, file_sha256(directory/'checkpoint.pt'))
    if set(parents) != CANDIDATES:
        raise ValueError('Expected exactly k301 and k306 pilot checkpoints')
    settings = trial_settings(freeze_core=freeze_core)
    root = Path(__file__).resolve().parents[1]
    files = set(parent['experiment']['code_sha256']) | {
        'optimizer/finalize_mvp_candidate.py', 'optimizer/mvp_constraints.py',
        'optimizer/pretrain.py', 'optimizer/navigation_loss.py',
        'optimizer/probe_recurrent_dynamics.py', 'optimizer/probe_yaw_recovery.py',
        'optimizer/benchmark_candidate.py', 'simulation/policy_diagnostics.py', 'core/invariants.py', 'artifacts/research.py', 'artifacts/io.py'}
    hashes = {p: file_sha256(root/p) for p in sorted(files)}
    contract = dict(version='mvp-frozen-core-grid-v1' if freeze_core else 'mvp-frozen-dynamics-grid-v1', settings=settings,
        mvp_gate=MVP_GATE, old_gate=GATE, parent_experiment_fingerprint=parent['experiment_fingerprint'],
        parent_checkpoints={name: p[2] for name, p in parents.items()},
        dataset_sha256=digest, code_sha256=hashes,
        objectives=['max_recovery_s_censored_at_2s', 'mean_final_yaw_amplitude_error_pwm', 'heldout_yaw_mae_pwm'],
        constraints=('Frozen scaled tau/A/W, bitwise verification after each native post_step; decoder norms monitored only.'
            if freeze_core else 'Frozen scaled tau/A; W Frobenius upper bound captured after scaling; native post_step first.'),
        limitations=['One seed, 10 extra epochs, fresh Adam; continuation, not training from scratch.',
            'Known validation/probes are development data, not independent confirmation.',
            'Frobenius projection does not guarantee settling or biological equivalence.',
            'External evaluation dataset is not read. No flight admission.'])
    fingerprint = contract_sha256(contract)
    with publication(output) as stage:
        write_json(stage/'preflight.json', dict(experiment=contract, experiment_fingerprint=fingerprint,
                                              training_ready=False, research_only=True))
    study = optuna.create_study(study_name='mvp_frozen_core_v1' if freeze_core else 'mvp_frozen_dynamics_v1',
        storage='sqlite:///'+(output.resolve()/'study.db').as_posix(),
        sampler=optuna.samplers.GridSampler(dict(candidate=sorted(parents),
            tau_scale=settings['tau_scale'], recurrent_scale=settings['recurrent_scale']), seed=42),
        directions=['minimize']*3)
    study.set_user_attr('experiment_fingerprint', fingerprint)
    study.set_user_attr('experiment_contract', contract)
    rows = []

    def objective(trial):
        name = trial.suggest_categorical('candidate', sorted(parents))
        tau = trial.suggest_categorical('tau_scale', settings['tau_scale'])
        gain = trial.suggest_categorical('recurrent_scale', settings['recurrent_scale'])
        directory, parent_report, checkpoint_hash = parents[name]
        original, checkpoint = load_research_run(directory)
        if file_sha256(directory/'checkpoint.pt') != checkpoint_hash:
            raise ValueError('Parent checkpoint changed')
        policy = scaled_policy(original, tau_scale=tau, recurrent_scale=gain)
        constraint = RecurrentConstraints(policy, freeze_core=freeze_core)
        info, config = checkpoint['pretrain_info'], parent_report['config']
        checkpoint.update(parent_checkpoint_sha256=checkpoint_hash,
            continuation=dict(**settings, selected_tau_scale=tau, selected_recurrent_scale=gain,
                optimizer_reset=True, scales_are_initialization_only=False))
        common = dict(candidate=name, trial_number=trial.number, parameters=dict(trial.params),
            experiment_fingerprint=fingerprint, parent_checkpoint_sha256=checkpoint_hash,
            config={**config, **settings}, research_only=True, training_ready=False)
        sample = data['X'][info['validation_indices'][:2]]
        print(name, 'tau', tau, 'evaluating and saving 0 epochs', flush=True)
        before = evaluate_phase(policy, data, info, config)
        baseline_checkpoint = dict(checkpoint, continuation={**checkpoint['continuation'], 'epochs': 0})
        before_report = save_phase(output/f'trial_{trial.number:05d}'/'before', policy=policy,
            checkpoint=baseline_checkpoint, parent_directory=directory, constraints=constraint, sample=sample,
            report=dict(**common, **before, continuation_epochs=0, pretrain=info))
        torch.manual_seed(config['seed'])
        print(name, 'tau', tau, 'training 10 epochs; frozen', constraint.frozen_parameters, flush=True)
        pretrain_policy(policy, data, epochs=settings['epochs'], subset_ratio=1.,
            batch_size=config['batch_size'], lr=settings['learning_rate'], seed=config['seed'],
            device=settings['device'], research_only=True, loss_mode=settings['loss_mode'],
            yaw_balance=config['yaw_balance'], post_step_callback=constraint)
        for key in ('subset_indices', 'validation_indices'):
            if policy._pretrain_info[key] != info[key]:
                raise ValueError('Training/validation split changed')
        if policy._pretrain_info['loss_contract']['yaw_recovery_window_s'] != 1.:
            raise ValueError('Recovery loss window changed')
        verify_recurrent_contract(policy.cfc_network.cell)
        metrics = evaluate_phase(policy, data, info, config)
        checkpoint['pretrain_info'] = policy._pretrain_info
        after_report = save_phase(output/f'trial_{trial.number:05d}'/'after', policy=policy,
            checkpoint=checkpoint, parent_directory=directory, constraints=constraint, sample=sample,
            report=dict(**common, **metrics, continuation_epochs=10, pretrain=policy._pretrain_info))
        admission = metrics['mvp_admission']
        for key, value in dict(benchmark_version=BENCHMARK_VERSION, control_contract=control_contract(),
            feasible=admission['passed'], constraints=[0. if admission['passed'] else 1.],
            experiment_fingerprint=fingerprint, mvp_admission=admission,
            functional_admission=metrics['functional_admission']).items():
            trial.set_user_attr(key, value)
        rows.append(dict(trial=trial.number, candidate=name, tau_scale=tau,
            before=comparison_row(before_report), after=comparison_row(after_report)))
        write_comparison(output, rows, freeze_core=freeze_core)
        write_json(output/'comparison.json', dict(experiment_fingerprint=fingerprint, rows=rows))
        print(name, tau, 'MVP', admission['passed'], 'reasons', admission['reasons'], flush=True)
        return (max(r['recovery_settle_s'] if r['recovery_settle_s'] is not None else 2.
                    for r in metrics['spatial_response']['cases']),
                metrics['summary']['mean_final_amplitude_error_pwm'], metrics['heldout']['channel_mae_pwm']['yaw'])

    study.optimize(objective, n_trials=4)
    if hashes != {p: file_sha256(root/p) for p in hashes} or file_sha256(dataset) != digest:
        raise ValueError('Experiment code/data changed')
    for directory, _, digest_parent in parents.values():
        if file_sha256(directory/'checkpoint.pt') != digest_parent:
            raise ValueError('Parent checkpoint changed')
    summary = dict(experiment_fingerprint=fingerprint, training_ready=False,
        completed_trials=len(study.trials), feasible_pareto_trials=[t.number for t in feasible_pareto_trials(study)],
        trials=[dict(number=t.number, parameters=t.params, values=t.values,
                     passed=t.user_attrs['feasible']) for t in study.trials])
    write_json(output/'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'dataset', 'out-dir'):
        parser.add_argument('--'+name, required=True, type=Path)
    parser.add_argument('--freeze-core', action='store_true',
                        help='Freeze W as well as tau/A; monitor decoder norms without limiting them')
    args = parser.parse_args()
    torch.set_num_threads(1)
    run_search(args.run, args.dataset, args.out_dir, freeze_core=args.freeze_core)


if __name__ == '__main__':
    main()
