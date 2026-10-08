"""Bounded, explicitly research-only pretrain diagnostic; never grants flight admission."""
import argparse
import json
import math
from pathlib import Path
import shutil

import torch

from configs.flight_config import CONTROL_DT, PWM_MID, PWM_HALF
from generator.prepare_flywire_candidates import publication, write_json
from generator.source_contract import file_sha256
from generator.validate_flywire_candidates import validate
from optimizer.pretrain import pretrain_policy
from simulation.control_contract import NAVIGATION_INDICES, NAVIGATION_CHANNELS, control_contract
from simulation.policy import ChongFlyMSPPolicy
from simulation.policy_diagnostics import response_probe, temporal_response_probe


def prediction_metrics(prediction, target, valid, scenarios):
    """Unweighted held-out scores; padding and throttle cannot hide turn failures."""
    if (prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[-1] != 4
        or valid.shape != target.shape[:2] or valid.dtype != torch.bool
        or len(scenarios) != len(target) or not valid.any()
        or not torch.isfinite(prediction).all() or not torch.isfinite(target).all()):
        raise ValueError('Invalid prediction/target/mask contract')
    def summarize(pred, truth, mask):
        error = (pred-truth)[..., list(NAVIGATION_INDICES)][mask]
        direction = {}
        for label, sign in [('right', 1), ('left', -1)]:
            turns = mask & ((truth[..., 3]-PWM_MID)*sign > 1)
            direction[label] = (float(((pred[..., 3]-PWM_MID)*sign > 0)[turns].float().mean())
                                if turns.any() else None)
        yaw_groups = {}
        delta = truth[..., 3]-PWM_MID
        for label, selected in [('neutral', delta.abs() <= 1), ('right', delta > 1), ('left', delta < -1)]:
            selected = selected & mask
            yaw_groups[label] = {'frames':int(selected.sum()),
                'mae_pwm':float((pred[..., 3]-truth[..., 3])[selected].abs().mean()) if selected.any() else None,
                'mean_command_pwm':float(pred[..., 3][selected].mean()) if selected.any() else None,
                'mean_target_pwm':float(truth[..., 3][selected].mean()) if selected.any() else None}
        return {'frames':int(mask.sum()), 'normalized_mse':float((error/PWM_HALF).square().mean()),
                'channel_mae_pwm':dict(zip(NAVIGATION_CHANNELS, error.abs().mean(0).tolist())),
                'turn_direction_accuracy':direction, 'yaw_groups':yaw_groups}
    result = summarize(prediction, target, valid)
    result['scenarios'] = {}
    for name in sorted(set(scenarios)):
        ids = [i for i,label in enumerate(scenarios) if label == name]
        result['scenarios'][name] = summarize(prediction[ids], target[ids], valid[ids])
    return result


@torch.no_grad()
def heldout_predictions(policy, data, ids, *, batch_size=8):
    parameter = next(policy.parameters())
    policy.eval()
    return torch.cat([policy(data['X'][ids[start:start+batch_size]].to(parameter.device),
                             dt=CONTROL_DT)[0].cpu()
                      for start in range(0, len(ids), batch_size)])


def short_rollouts(policy, *, seed, steps):
    from optimizer.rollout import simulate_policy_rollout
    from simulation.altitude_control import AltitudeHold
    results = []
    for offset in range(3):
        _, metrics = simulate_policy_rollout(policy, eval_steps=steps, dt=CONTROL_DT,
                                             seed=seed+10000+offset, altitude_hold=AltitudeHold())
        results.append({'seed':seed+10000+offset, **metrics})
    return results


def run_diagnostic(source, candidates, candidate, dataset, output, *, research_only=False,
                   epochs=3, lr=.005, batch_size=8, seed=42, device='cpu', rollout_steps=250,
                   preserve_signs=True, mode='masked', sensor_encoding='distance-v1',
                   loss_mode='behavior-v1', yaw_balance=.5, evaluation_dataset=None, neutral_origin=False):
    if not research_only:
        raise ValueError('Explicit research-only opt-in required for unapproved candidates')
    if (type(epochs) is not int or not 1 <= epochs <= 10 or type(batch_size) is not int or batch_size < 1
        or not math.isfinite(lr) or lr <= 0 or type(seed) is not int
        or type(rollout_steps) is not int or not 1 <= rollout_steps <= 2000):
        raise ValueError('Use a bounded diagnostic: 1..10 epochs, 1..2000 rollout steps, positive batch size/lr')
    if device not in {'cpu', 'cuda'}:
        raise ValueError('Choose cpu or cuda explicitly')
    if type(preserve_signs) is not bool or mode not in {'masked', 'fixed'}:
        raise ValueError('Use fixed/masked topology and an explicit sign policy')
    if sensor_encoding not in {'distance-v1', 'threat-v1', 'proximity-v1', 'proximity-mean-v1'} or loss_mode not in {'behavior-v1', 'channel-v2', 'channel-v3'}:
        raise ValueError('Unknown sensor encoding or loss mode')
    if not math.isfinite(yaw_balance) or not 0 <= yaw_balance <= 1:
        raise ValueError('yaw_balance must be in [0, 1]')
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; no CPU fallback')
    source, candidates, dataset, output = map(Path, (source, candidates, dataset, output))
    if output.exists():
        raise FileExistsError('Use a new diagnostic output directory')
    verification = validate(source, candidates)
    registry = json.loads((candidates/'registry.json').read_text(encoding='utf-8'))
    rows = [row for row in registry['candidates'] if row['candidate'] == candidate]
    admissions = [row for row in verification['candidates'] if row['candidate'] == candidate]
    if len(rows) != 1 or len(admissions) != 1 or admissions[0]['structural_admission'] != 'passed':
        raise ValueError('Research diagnostic requires one structurally admitted candidate')
    row = rows[0]
    meta_path = (candidates/row['meta_file']).resolve()
    if not meta_path.is_relative_to(candidates.resolve()) or row['meta_file'] not in registry['files']:
        raise ValueError('Candidate metadata is outside the verified registry')
    data = torch.load(dataset, weights_only=True, map_location='cpu')
    if data.get('metadata', {}).get('dt') != CONTROL_DT:
        raise ValueError('Diagnostic dataset must use the current control timestep')
    if any(record['seed'] in {seed+10000+i for i in range(3)}
           for record in data['metadata'].get('episodes', [])):
        raise ValueError('Rollout seeds overlap recorded demonstration seeds')
    torch.manual_seed(seed)
    policy = ChongFlyMSPPolicy.from_meta(str(meta_path), dt=CONTROL_DT,
        mode=mode, preserve_signs=preserve_signs, sensor_encoding=sensor_encoding,
        neutral_origin=neutral_origin).to(device)
    # Reuse the real pretrain validator/splitter without changing any parameters.
    pretrain_policy(policy, data, epochs=0, subset_ratio=1., seed=seed, device=device,
                    batch_size=batch_size, research_only=True, loss_mode=loss_mode, yaw_balance=yaw_balance)
    split = policy._pretrain_info
    external = None
    if evaluation_dataset is not None:
        evaluation_dataset = Path(evaluation_dataset)
        external = torch.load(evaluation_dataset, weights_only=True, map_location='cpu')
        train_records = data['metadata'].get('episodes', [])
        eval_records = external.get('metadata', {}).get('episodes', [])
        if (len(train_records) != len(data['X']) or len(eval_records) != len(external['X'])
            or {r['seed'] for r in train_records} & {r['seed'] for r in eval_records}
            or external['metadata'].get('dt') != CONTROL_DT):
            raise ValueError('External evaluation requires recorded disjoint episode seeds and control dt')
        # Reuse the dataset validator without training or using these labels to
        # calculate the actual training weights/split.
        pretrain_policy(policy, external, epochs=0, subset_ratio=1., seed=seed, device=device,
                        research_only=True, loss_mode=loss_mode, yaw_balance=yaw_balance)
        policy._pretrain_info = split
    ids = split['validation_indices']
    if not ids or set(ids) & set(split['subset_indices']):
        raise ValueError('Disjoint held-out episodes are required')
    scenarios = [data['metadata']['episode_scenarios'][i] for i in ids]
    target, valid_mask = data['Y'][ids], data['valid'][ids]
    before = prediction_metrics(heldout_predictions(policy, data, ids), target, valid_mask, scenarios)
    train_ids = split['subset_indices']
    constant = data['Y'][train_ids][data['valid'][train_ids]].mean(0)
    baseline = prediction_metrics(constant.expand_as(target), target, valid_mask, scenarios)
    initial_w = policy.cfc_network.cell._effective_W().detach().cpu().clone()
    responses_before = response_probe(policy)
    temporal_before = temporal_response_probe(policy)
    spatial_before = temporal_response_probe(policy, coverage_columns=(1,2,4))
    rollouts_before = short_rollouts(policy, seed=seed, steps=rollout_steps)
    print(f'Research-only {candidate}: {len(train_ids)} training / {len(ids)} held-out episodes', flush=True)
    with publication(output) as stage:
        pretrain_policy(policy, data, epochs=epochs, subset_ratio=1., lr=lr, batch_size=batch_size,
                        seed=seed, device=device, research_only=True, loss_mode=loss_mode, yaw_balance=yaw_balance)
        if policy._pretrain_info['validation_indices'] != ids:
            raise ValueError('Pretrain changed the held-out split')
        after = prediction_metrics(heldout_predictions(policy, data, ids), target, valid_mask, scenarios)
        responses_after = response_probe(policy)
        temporal_after = temporal_response_probe(policy)
        spatial_after = temporal_response_probe(policy, coverage_columns=(1,2,4))
        rollouts_after = short_rollouts(policy, seed=seed, steps=rollout_steps)
        learned_w = policy.cfc_network.cell._effective_W().detach().cpu()
        sign_changes = int(((initial_w != 0) & (initial_w * learned_w < 0)).sum())
        forbidden = int(torch.count_nonzero(learned_w[initial_w == 0]))
        if forbidden:
            raise ValueError('Training introduced forbidden recurrent edges')
        if preserve_signs and sign_changes:
            raise ValueError('Training violated declared synaptic signs')
        from core.routing import sensor_motor_paths
        learned_paths = sensor_motor_paths(learned_w != 0, policy.cfc_network.cell.sensor_indices,
                                          policy.dn_head._proj_mask)
        # Bundle the exact graph required to reconstruct this checkpoint.
        graph = stage/'graph'; graph.mkdir()
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        for name in [meta_path.name, meta['w_file'], meta['cmap_file']]:
            original = (meta_path.parent/name).resolve()
            relative = original.relative_to(candidates.resolve()).as_posix()
            if relative not in registry['files']:
                raise ValueError('Checkpoint graph references an unverified file')
            shutil.copyfile(original, graph/original.name)
        checkpoint = {'format_version':'candidate-research-checkpoint-v1', 'training_ready':False,
                      'candidate':candidate, 'registry_sha256':file_sha256(candidates/'registry.json'),
                      'meta_sha256':file_sha256(meta_path), 'dt':CONTROL_DT,
                      'mode':mode, 'preserve_signs':preserve_signs,
                      'sensor_encoding':sensor_encoding,
                      'neutral_origin':neutral_origin,
                      'control_contract':control_contract(), 'pretrain_info':policy._pretrain_info,
                      'state_dict':{k:v.detach().cpu() if isinstance(v, torch.Tensor) else v
                                    for k,v in policy.state_dict().items()}}
        torch.save(checkpoint, stage/'checkpoint.pt')
        reloaded = torch.load(stage/'checkpoint.pt', weights_only=True, map_location='cpu')
        restored = ChongFlyMSPPolicy.from_meta(str(graph/meta_path.name), dt=CONTROL_DT,
            mode=mode, preserve_signs=preserve_signs, sensor_encoding=sensor_encoding,
            neutral_origin=neutral_origin).to(device)
        restored.load_state_dict(reloaded['state_dict'], strict=True)
        torch.testing.assert_close(heldout_predictions(restored, data, ids),
                                   heldout_predictions(policy, data, ids), rtol=0, atol=0)
        write_json(stage/'source_validation.json', verification)
        report = {'format_version':'candidate-diagnostic-v3', 'research_only':True, 'training_ready':False,
            'candidate':candidate, 'source_quality_review_required':verification['quality']['quality_review_required'],
            'config':{'epochs':epochs, 'lr':lr, 'batch_size':batch_size, 'seed':seed, 'device':device,
                      'dtype':'float32', 'control_dt':CONTROL_DT, 'rollout_steps':rollout_steps,
                      'mode':mode, 'preserve_signs':preserve_signs, 'sensor_encoding':sensor_encoding,
                      'loss_mode':loss_mode, 'yaw_balance':yaw_balance, 'neutral_origin':neutral_origin},
            'source_manifest_sha256':file_sha256(source/'manifest.json'),
            'registry_sha256':file_sha256(candidates/'registry.json'), 'dataset_file_sha256':file_sha256(dataset),
            'checkpoint_sha256':file_sha256(stage/'checkpoint.pt'), 'checkpoint_roundtrip_exact':True,
            'weight_scaling':meta['provenance']['weight_scaling'], 'before':before, 'after':after,
            'training_mean_baseline':baseline, 'response_before':responses_before, 'response_after':responses_after,
            'temporal_response_before':temporal_before, 'temporal_response_after':temporal_after,
            'spatial_response_before':spatial_before, 'spatial_response_after':spatial_after,
            'pretrain':policy._pretrain_info, 'rollouts_before':rollouts_before, 'rollouts_after':rollouts_after,
            'recurrent_sign_changes':sign_changes, 'forbidden_recurrent_edges':forbidden,
            'zeroed_initial_edges':int(((initial_w != 0) & (learned_w == 0)).sum()),
            'learned_sensor_motor_paths':learned_paths,
            'limitations':['Short exploratory run; held-out episodes are validation, not a final test set.',
                           'Sign preservation enforces model assumptions, not biological proof.',
                           'Official-view quality and sensory/RC assumptions remain unresolved.'],
            'code_sha256':{**verification['code_sha256'],
                          'optimizer/diagnose_candidate.py':file_sha256(Path(__file__)),
                          **{name:file_sha256(Path(__file__).resolve().parents[1]/name) for name in
                             ['optimizer/pretrain.py', 'optimizer/navigation_loss.py', 'optimizer/rollout.py',
                              'simulation/policy_diagnostics.py', 'simulation/drone_env.py',
                              'simulation/altitude_control.py', 'simulation/memory.py',
                              'generator/generate_reflex_dataset.py', 'generator/reflex_contract.py']}},
            'torch_version':str(torch.__version__)}
        if external is not None:
            ext_ids = list(range(len(external['X'])))
            ext_target, ext_valid = external['Y'], external['valid']
            ext_scenarios = external['metadata']['episode_scenarios']
            report['external_evaluation'] = {
                'dataset_file_sha256':file_sha256(evaluation_dataset), 'episode_seeds_disjoint':True,
                'metadata':external['metadata'],
                'after':prediction_metrics(heldout_predictions(policy, external, ext_ids),
                                          ext_target, ext_valid, ext_scenarios),
                'training_mean_baseline':prediction_metrics(constant.expand_as(ext_target),
                                                            ext_target, ext_valid, ext_scenarios)}
        write_json(stage/'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'candidates', 'dataset', 'out-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--research-only', action='store_true')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=.005)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--rollout-steps', type=int, default=250)
    parser.add_argument('--sensor-encoding', choices=['distance-v1', 'threat-v1', 'proximity-v1', 'proximity-mean-v1'], default='distance-v1')
    parser.add_argument('--loss-mode', choices=['behavior-v1', 'channel-v2', 'channel-v3'], default='behavior-v1')
    parser.add_argument('--yaw-balance', type=float, default=.5)
    parser.add_argument('--evaluation-dataset', type=Path)
    parser.add_argument('--neutral-origin', action='store_true',
                        help='Zero constant neural/input drive and yaw readout bias; requires threat-v1 or proximity-v1')
    parser.add_argument('--mode', choices=['masked', 'fixed'], default='masked')
    parser.add_argument('--allow-sign-changes', action='store_true',
                        help='Explicit research control only; otherwise initial signs are preserved')
    args = parser.parse_args()
    # Small sequential graphs are cheaper without oversubscribing CPU threads.
    torch.set_num_threads(1)
    result = run_diagnostic(args.source, args.candidates, args.candidate, args.dataset, args.out_dir,
                            research_only=args.research_only, epochs=args.epochs, lr=args.lr,
                            batch_size=args.batch_size, seed=args.seed, device=args.device,
                            rollout_steps=args.rollout_steps, mode=args.mode,
                            preserve_signs=not args.allow_sign_changes, sensor_encoding=args.sensor_encoding,
                            loss_mode=args.loss_mode, yaw_balance=args.yaw_balance,
                            evaluation_dataset=args.evaluation_dataset, neutral_origin=args.neutral_origin)
    print(json.dumps({'report':str(args.out_dir/'report.json'), 'training_ready':False,
                      'before':result['before']['channel_mae_pwm'], 'after':result['after']['channel_mae_pwm'],
                      'sign_changes':result['recurrent_sign_changes']}, indent=2))


if __name__ == '__main__':
    main()
