"""Paired continuation experiment: ordinary loss versus post-turn yaw balance.

Uses the SAME saved weights, training episodes, batch order, learning rate and
epoch budget for both arms. Does not read the external evaluation dataset.
State reset is a diagnostic counterfactual, never a deployed controller change.
"""
import argparse
from core.invariants import verify_recurrent_contract
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from configs.flight_config import CONTROL_DT
from generator.generate_reflex_dataset import make_teacher
from artifacts.io import publication, write_json
from generator.reflex_contract import TEACHER_VERSION
from generator.source_contract import file_sha256
from artifacts.research import load_research_run
from optimizer.diagnose_candidate import heldout_predictions, prediction_metrics
from optimizer.navigation_loss import yaw_recovery_mask
from optimizer.pretrain import pretrain_policy
from optimizer.rollout import simulate_policy_rollout
from optimizer.search_pipeline import assess_navigation
from simulation.altitude_control import AltitudeHold
from simulation.benchmark_scenes import NavigationScene, SCENES
from simulation.policy_diagnostics import temporal_response_probe


@torch.no_grad()
def recovery_probe(policy):
    parameter = next(policy.parameters())
    warmup, threat, recovery = 50, 25, 250
    cases = [(side, distance, width) for width in (1, 4) for distance in (.1, .2)
             for side in ('left', 'right')]
    obs = torch.zeros(len(cases), warmup+threat+recovery, 74,
                      dtype=parameter.dtype, device=parameter.device)
    obs[..., 2:] = 1
    for i, (side, distance, width) in enumerate(cases):
        grid = obs[i, warmup:warmup+threat, 2:66].reshape(threat, 8, 8)
        grid[..., :width] = distance if side == 'left' else 1
        if side == 'right':
            grid[..., -width:] = distance
    policy.eval()
    pwm, _ = policy(obs, dt=CONTROL_DT)
    reset_pwm, _ = policy(obs[:, warmup+threat:], dt=CONTROL_DT)
    if not torch.isfinite(pwm).all() or not torch.isfinite(reset_pwm).all():
        raise ValueError('Nonfinite recovery probe')
    rows = []
    for i, (side, distance, width) in enumerate(cases):
        teacher = make_teacher(TEACHER_VERSION, noise_std_pwm=0)
        target = np.stack([teacher.step(x, dt=CONTROL_DT) for x in obs[i].cpu().numpy()])
        tail = pwm[i, warmup+threat:, 3].cpu()-1500
        outside = (tail.abs() > 20).nonzero().flatten()
        last = int(outside[-1])+1 if len(outside) else 0
        teacher_tail = target[warmup+threat:, 3]-1500
        teacher_outside = np.flatnonzero(np.abs(teacher_tail) > 1)
        rows.append(dict(side=side, normalized_distance=distance, width=width,
            threat_yaw_pwm=float(pwm[i, warmup+threat-1, 3]),
            teacher_threat_yaw_pwm=float(target[warmup+threat-1, 3]),
            recovery_abs_yaw_at_0_8s_pwm=float(tail[39].abs()),
            recovery_abs_yaw_at_2s_pwm=float(tail[99].abs()),
            settle_s=last*CONTROL_DT if last < recovery else None,
            teacher_release_s=(int(teacher_outside[-1])+1)*CONTROL_DT if len(teacher_outside) else 0.,
            yaw_pwm=pwm[i, :, 3].cpu().tolist(), teacher_yaw_pwm=target[:, 3].tolist()))
    return dict(dt=CONTROL_DT, teacher_version=TEACHER_VERSION, cases=rows,
        warmup_steps=warmup, threat_steps=threat, recovery_steps=recovery,
        clear_from_reset_max_abs_yaw_pwm=float((reset_pwm[..., 3]-1500).abs().max()))


def evaluate(policy, data, ids, config):
    heldout = prediction_metrics(heldout_predictions(policy, data, ids), data['Y'][ids],
        data['valid'][ids], [data['metadata']['episode_scenarios'][i] for i in ids])
    temporal = temporal_response_probe(policy)
    spatial = temporal_response_probe(policy, coverage_columns=(1, 2, 4))
    scenes = []
    for scenario in SCENES:
        for seed in config['scene_seeds']:
            env = NavigationScene(scenario)
            try:
                _, metrics = simulate_policy_rollout(policy, env=env, eval_steps=config['scene_steps'],
                    seed=seed, dt=CONTROL_DT, altitude_hold=AltitudeHold())
                scenes.append(dict(scenario=scenario, seed=seed, **metrics))
            finally:
                env.close()
    from core.routing import sensor_motor_paths
    from generator.reflex_contract import REQUIRED_PATHS
    cell = policy.cfc_network.cell
    paths = sensor_motor_paths(cell._effective_W().detach() != 0,
                               cell.sensor_indices, policy.dn_head._proj_mask)
    admission = assess_navigation(temporal, heldout, scenes, spatial=spatial)
    for group, channels in REQUIRED_PATHS.items():
        for channel in channels:
            if paths[group]['minimum_hops'][channel] is None:
                admission['reasons'].append(f'lost_route:{group}->{channel}')
                admission['passed'] = False
    return dict(heldout=heldout, temporal_response=temporal, spatial_response=spatial, scenes=scenes,
        functional_admission=admission, learned_sensor_motor_paths=paths,
        recovery=recovery_probe(policy))


def run_experiment(run, dataset, output, *, epochs=10, learning_rate=.001):
    if type(epochs) is not int or not 1 <= epochs <= 30:
        raise ValueError('Use 1..30 continuation epochs')
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('Learning rate must be positive and finite')
    run, dataset = Path(run), Path(dataset)
    preflight = json.loads((run/'preflight.json').read_text(encoding='utf-8'))
    dataset_hash = file_sha256(dataset)
    if dataset_hash != preflight['experiment']['dataset_sha256']:
        raise ValueError('Continuation requires the original training dataset')
    data = torch.load(dataset, map_location='cpu', weights_only=True)
    if file_sha256(dataset) != dataset_hash:
        raise ValueError('Dataset changed during loading')
    root = Path(__file__).resolve().parents[1]
    source_files = list(preflight['experiment']['code_sha256']) + ['optimizer/probe_yaw_recovery.py',
        'optimizer/benchmark_candidate.py', 'core/invariants.py', 'artifacts/research.py', 'artifacts/io.py']
    hashes = {p:file_sha256(root/p) for p in source_files}
    results = []
    with publication(output) as stage:
        for trial in sorted(run.glob('trial_*')):
            parent_report = json.loads((trial/'report.json').read_text(encoding='utf-8'))
            if parent_report['experiment_fingerprint'] != preflight['experiment_fingerprint']:
                raise ValueError('Mixed parent experiments')
            initial, checkpoint = load_research_run(trial)
            parent_hash = file_sha256(trial/'checkpoint.pt')
            info = checkpoint['pretrain_info']
            train_ids, val_ids = info['subset_indices'], info['validation_indices']
            if info['teacher_version'] != TEACHER_VERSION or not val_ids:
                raise ValueError('Expected a v5 checkpoint with held-out episodes')
            recovery = yaw_recovery_mask(data['Y'][train_ids], data['valid'][train_ids], steps=50)
            row = dict(candidate=checkpoint['candidate'], parent_checkpoint_sha256=parent_hash,
                training_recovery_frames=int(recovery.sum()),
                training_recovery_episodes=int(recovery.any(dim=1).sum()),
                training_valid_frames=int(data['valid'][train_ids].sum()),
                initial=evaluate(initial, data, val_ids, parent_report['config']), arms={})
            for arm, loss_mode in [('control', 'channel-v3'), ('recovery_balanced', 'channel-v4')]:
                policy, current = load_research_run(trial)
                if file_sha256(trial/'checkpoint.pt') != parent_hash:
                    raise ValueError('Parent changed during experiment')
                torch.manual_seed(parent_report['config']['seed'])
                print(checkpoint['candidate'], arm, 'training', flush=True)
                pretrain_policy(policy, data, epochs=epochs, subset_ratio=1., batch_size=parent_report['config']['batch_size'],
                    lr=learning_rate, seed=parent_report['config']['seed'], device='cpu',
                    research_only=True, loss_mode=loss_mode, yaw_balance=parent_report['config']['yaw_balance'])
                if (policy._pretrain_info['subset_indices'] != train_ids
                        or policy._pretrain_info['validation_indices'] != val_ids):
                    raise ValueError('Continuation changed the training/validation split')
                cell = policy.cfc_network.cell
                verify_recurrent_contract(cell)
                metrics = evaluate(policy, data, val_ids, parent_report['config'])
                destination = stage/trial.name/arm
                destination.mkdir(parents=True)
                shutil.copytree(trial/'graph', destination/'graph')
                current.update(state_dict=policy.state_dict(), pretrain_info=policy._pretrain_info,
                    parent_checkpoint_sha256=parent_hash,
                    continuation=dict(epochs=epochs, learning_rate=learning_rate, optimizer_reset=True, arm=arm))
                torch.save(current, destination/'checkpoint.pt')
                write_json(destination/'report.json', dict(candidate=checkpoint['candidate'],
                    checkpoint_sha256=file_sha256(destination/'checkpoint.pt'), research_only=True,
                    training_ready=False, parent_checkpoint_sha256=parent_hash,
                    pretrain=policy._pretrain_info, **metrics))
                restored, _ = load_research_run(destination)
                with torch.no_grad():
                    sample = data['X'][val_ids[:2]]
                    torch.testing.assert_close(policy(sample)[0], restored(sample)[0], rtol=0, atol=0)
                row['arms'][arm] = dict(checkpoint=str((Path(output)/trial.name/arm/'checkpoint.pt').resolve()),
                    checkpoint_roundtrip_exact=True, **metrics)
                print(checkpoint['candidate'], arm, 'gate', metrics['functional_admission']['passed'],
                    'heldout', metrics['heldout']['channel_mae_pwm'], flush=True)
            results.append(row)
        if hashes != {p:file_sha256(root/p) for p in hashes} or file_sha256(dataset) != dataset_hash:
            raise ValueError('Experiment inputs changed')
        result = dict(version='yaw-recovery-paired-v1', research_only=True, training_ready=False,
            dataset_sha256=dataset_hash, code_sha256=hashes, epochs_per_arm=epochs,
            learning_rate=learning_rate, candidates=results,
            limitations=['One seed; continuation of existing trained models, not training from scratch.',
                'Fresh Adam in both arms; optimizer state was not saved by the original pilot.',
                'Original validation and diagnostic probes are development data, not new independent confirmation.',
                'External evaluation dataset is not read. Numerical admission thresholds unchanged.',
                'Synthetic reset comparison isolates state dependence, not a proposed flight policy.'])
        write_json(stage/'report.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=10)
    args = p.parse_args()
    torch.set_num_threads(1)
    run_experiment(args.run, args.dataset, args.out_dir, epochs=args.epochs)


if __name__ == '__main__':
    main()
