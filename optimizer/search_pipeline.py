"""Versioned registry-based research selection with exact-weight exports and gates."""
import argparse
import json
import math
from pathlib import Path
import shutil

import numpy as np
import torch

from configs.flight_config import CONTROL_DT
from generator.prepare_flywire_candidates import publication, write_json
from generator.source_contract import file_sha256
from optimizer.candidate_selection import CandidatePool, contract_sha256
from optimizer.diagnose_candidate import heldout_predictions, prediction_metrics
from optimizer.optuna_tuner import create_study, feasible_pareto_trials
from optimizer.pretrain import pretrain_policy
from optimizer.rollout import BENCHMARK_VERSION, simulate_policy_rollout
from simulation.altitude_control import AltitudeHold
from simulation.benchmark_scenes import NavigationScene, SCENES
from simulation.control_contract import control_contract
from simulation.policy_diagnostics import temporal_response_probe


DEFAULT_CONFIG = {'epochs':30, 'batch_size':8, 'seed':42, 'device':'cpu',
    'sensor_encoding':'proximity-v1', 'neutral_origin':True, 'preserve_signs':True,
    'loss_mode':'channel-v3', 'yaw_balance':.7, 'learning_rate_min':.001,
    'learning_rate_max':.005, 'tau_choices':[.05, .1, .2],
    'scene_steps':500, 'scene_seeds':[20261009]}

GATE = {'version':'navigation-research-gate-v2', 'clear_yaw_max_pwm':20.,
    'turn_final_error_max_pwm':100., 'onset_max_s':.2, 'recovery_max_s':.8,
    'roll_mae_max_pwm':25., 'pitch_mae_max_pwm':35., 'turn_mae_max_pwm':100.,
    'mean_forward_speed_min_mps':.1, 'minimum_clearance_m':.1}


def validate_search_config(value):
    if not isinstance(value, dict) or set(value)-set(DEFAULT_CONFIG):
        raise ValueError('Unknown search configuration fields')
    config = {**DEFAULT_CONFIG, **value}
    for key, limit in [('epochs', 50), ('batch_size', 64), ('scene_steps', 2000)]:
        if type(config[key]) is not int or not 1 <= config[key] <= limit:
            raise ValueError(f'{key} must be an integer in [1,{limit}]')
    if type(config['seed']) is not int or config['device'] not in {'cpu', 'cuda'}:
        raise ValueError('Use an integer seed and explicit cpu/cuda device')
    if config['sensor_encoding'] not in {'distance-v1', 'threat-v1', 'proximity-v1', 'proximity-mean-v1'}:
        raise ValueError('Unknown sensor encoding')
    for key in ('neutral_origin', 'preserve_signs'):
        if type(config[key]) is not bool:
            raise ValueError(f'{key} must be an explicit boolean')
    if config['neutral_origin'] and config['sensor_encoding'] == 'distance-v1':
        raise ValueError('Neutral origin requires threat/proximity encoding')
    if config['loss_mode'] not in {'behavior-v1', 'channel-v2', 'channel-v3'}:
        raise ValueError('Unknown loss mode')
    if not math.isfinite(config['yaw_balance']) or not 0 <= config['yaw_balance'] <= 1:
        raise ValueError('Invalid yaw balance')
    a, b = config['learning_rate_min'], config['learning_rate_max']
    if not all(math.isfinite(x) for x in (a, b)) or not 0 < a <= b:
        raise ValueError('Learning-rate bounds must be positive finite and ordered')
    taus = config['tau_choices']
    if not isinstance(taus, list) or not taus or len(set(taus)) != len(taus) or any(
            not isinstance(t, (int, float)) or isinstance(t, bool) or not math.isfinite(t) or t <= .001 for t in taus):
        raise ValueError('tau_choices must be distinct positive time constants above the floor')
    seeds = config['scene_seeds']
    if not isinstance(seeds, list) or not 1 <= len(seeds) <= 3 or len(set(seeds)) != len(seeds) or any(type(s) is not int for s in seeds):
        raise ValueError('Use one to three distinct integer scene seeds')
    return config


def assess_navigation(temporal, heldout, scenes, *, spatial=None):
    """Provisional engineering gate; cannot grant scientific or flight admission."""
    reasons = []
    def finite(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    cases = temporal.get('cases', [])
    expected = {(side, distance) for side in ('left','right') for distance in (.1,.2)}
    if (temporal.get('dt') != CONTROL_DT or len(cases) != 4 or
        {(row.get('obstacle_side'), row.get('normalized_distance')) for row in cases} != expected):
        reasons.append('missing_temporal_cases')
    spatial = spatial or {}
    spatial_cases = spatial.get('cases', [])
    expected_spatial = {(side, distance, width) for side, distance in expected for width in (1,2,4)}
    if (spatial.get('dt') != CONTROL_DT or len(spatial_cases) != 12 or
        {(r.get('obstacle_side'), r.get('normalized_distance'), r.get('active_columns'))
         for r in spatial_cases} != expected_spatial):
        reasons.append('missing_spatial_cases')
    combined = [(f'temporal_{i}', row) for i,row in enumerate(cases)]
    combined += [(f'spatial_{i}', row) for i,row in enumerate(spatial_cases)]
    for label_prefix, row in combined:
        for key, threshold in [('clear_abs_yaw_pwm', GATE['clear_yaw_max_pwm']),
                               ('onset_s', GATE['onset_max_s']), ('recovery_settle_s', GATE['recovery_max_s'])]:
            if not finite(row.get(key)) or not 0 <= row[key] <= threshold:
                label = 'clear_yaw' if key == 'clear_abs_yaw_pwm' else key
                reasons.append(f'{label_prefix}:{label}')
        if row.get('direction_correct') is not True:
            reasons.append(f'{label_prefix}:direction')
        command, target = row.get('threat_yaw_pwm'), row.get('threat_target_yaw_pwm')
        if not finite(command) or not finite(target) or abs(command-target) > GATE['turn_final_error_max_pwm']:
            reasons.append(f'{label_prefix}:amplitude')
    for channel in ('roll', 'pitch'):
        value = heldout.get('channel_mae_pwm', {}).get(channel)
        if not finite(value) or value > GATE[f'{channel}_mae_max_pwm']:
            reasons.append(f'heldout:{channel}')
    for side in ('right', 'left'):
        value = heldout.get('yaw_groups', {}).get(side, {}).get('mae_pwm')
        if not finite(value) or value > GATE['turn_mae_max_pwm']:
            reasons.append(f'heldout:{side}_amplitude')
    if set(r.get('scenario') for r in scenes) != set(SCENES):
        reasons.append('missing_challenge_scenes')
    if not any(finite(r.get('saccades_yaw_count')) and r['saccades_yaw_count'] > 0 for r in scenes):
        reasons.append('no_physical_yaw_bursts')
    for row in scenes:
        name = row.get('scenario', 'unknown')
        if (row.get('benchmark_version') != BENCHMARK_VERSION or
            row.get('control_contract') != control_contract() or row.get('feasible') is not True):
            reasons.append(f'{name}:flight_failure_or_contract')
        speed, clearance = row.get('mean_fwd_speed'), row.get('min_clearance_m')
        if not finite(speed) or speed < GATE['mean_forward_speed_min_mps']:
            reasons.append(f'{name}:progress')
        if not finite(clearance) or clearance < GATE['minimum_clearance_m']:
            reasons.append(f'{name}:clearance')
    return {'passed':not reasons, 'reasons':reasons, 'thresholds':dict(GATE),
            'scope':'research_comparison_only', 'training_ready':False}


def validate_demonstration_context(data, scene_seeds):
    """Require traceable whole episodes and the same timebase as evaluation."""
    if not isinstance(data, dict) or not isinstance(data.get('metadata'), dict):
        raise ValueError('Search requires demonstration metadata')
    metadata = data['metadata']
    if metadata.get('dt') != CONTROL_DT:
        raise ValueError('Demonstration dt differs from the control/evaluation timebase')
    episodes = metadata.get('episodes')
    x = data.get('X')
    if (not isinstance(x, torch.Tensor) or x.ndim != 3 or
        not isinstance(episodes, list) or len(episodes) != len(x)):
        raise ValueError('Search requires a recorded seed for every demonstration episode')
    seeds = [row.get('seed') if isinstance(row, dict) else None for row in episodes]
    if any(type(s) is not int for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError('Demonstration seeds must be distinct recorded integers')
    if set(seeds) & set(scene_seeds):
        raise ValueError('Challenge seeds overlap demonstration seeds')
    scenarios = metadata.get('episode_scenarios')
    if not isinstance(scenarios, list) or [row.get('scenario') for row in episodes] != scenarios:
        raise ValueError('Recorded demonstration scenarios are inconsistent')


def export_trial(policy, pool, name, output, report, options):
    """Publish exact learned weights and their graph, including failed admissions."""
    pool.assert_unchanged()
    meta_path = pool.directory/pool.rows[name]['meta_file']
    metadata = json.loads(meta_path.read_text(encoding='utf-8'))
    with publication(Path(output)) as stage:
        graph = stage/'graph'; graph.mkdir()
        for filename in [meta_path.name, metadata['w_file'], metadata['cmap_file']]:
            original = (meta_path.parent/filename).resolve()
            relative = original.relative_to(pool.directory).as_posix()
            if relative not in pool.registry['files']:
                raise ValueError('Checkpoint references an unverified graph file')
            shutil.copyfile(original, graph/original.name)
        graph_hashes = {p.name:file_sha256(p) for p in graph.iterdir()}
        checkpoint = {'format_version':'candidate-research-checkpoint-v1', 'training_ready':False,
            'candidate':name, 'registry_sha256':pool.identity['registry_sha256'],
            'meta_sha256':file_sha256(meta_path), 'dt':CONTROL_DT, 'mode':'masked',
            **{k:options[k] for k in ('preserve_signs', 'sensor_encoding', 'neutral_origin')},
            'tau_init':options['tau_init'], 'graph_file_sha256':graph_hashes,
            'control_contract':control_contract(), 'pretrain_info':policy._pretrain_info,
            'state_dict':{k:v.detach().cpu() if isinstance(v, torch.Tensor) else v for k,v in policy.state_dict().items()}}
        torch.save(checkpoint, stage/'checkpoint.pt')
        from simulation.policy import ChongFlyMSPPolicy
        restored = ChongFlyMSPPolicy.from_meta(str(graph/meta_path.name), dt=CONTROL_DT, **options)
        restored.load_state_dict(torch.load(stage/'checkpoint.pt', weights_only=True)['state_dict'], strict=True)
        device = next(policy.parameters()).device
        x = torch.ones(2, 10, 74, device=device); x[..., :2] = 0
        x[1, :, 2:34] = .1
        policy.eval(); restored.to(device).eval()
        with torch.no_grad():
            torch.testing.assert_close(policy(x)[0], restored(x)[0], rtol=0, atol=0)
        write_json(stage/'report.json', {**report, 'checkpoint_sha256':file_sha256(stage/'checkpoint.pt'),
            'checkpoint_roundtrip_exact':True, 'graph_file_sha256':graph_hashes,
            'training_ready':False, 'research_only':pool.research_only})


def candidate_objective(trial, *, pool, data, config, output, experiment):
    torch.manual_seed(config['seed'])
    name = trial.suggest_categorical('candidate', pool.names)
    lr = trial.suggest_float('learning_rate', config['learning_rate_min'], config['learning_rate_max'], log=True)
    tau = trial.suggest_categorical('tau_init', config['tau_choices'])
    options = {k:config[k] for k in ('sensor_encoding', 'neutral_origin', 'preserve_signs')}
    options.update(tau_init=tau, mode='masked')
    policy = pool.create_policy(name, options).to(config['device'])
    initial = policy.cfc_network.cell._effective_W().detach().clone()
    pretrain_policy(policy, data, epochs=config['epochs'], subset_ratio=1., batch_size=config['batch_size'],
        lr=lr, seed=config['seed'], device=config['device'], research_only=pool.research_only,
        loss_mode=config['loss_mode'], yaw_balance=config['yaw_balance'])
    ids = policy._pretrain_info['validation_indices']
    if not ids:
        raise ValueError('Search requires nonempty held-out episodes')
    heldout = prediction_metrics(heldout_predictions(policy, data, ids), data['Y'][ids], data['valid'][ids],
                                [data['metadata']['episode_scenarios'][i] for i in ids])
    temporal = temporal_response_probe(policy)
    spatial = temporal_response_probe(policy, coverage_columns=(1,2,4))
    learned = policy.cfc_network.cell._effective_W().detach()
    flips = int(((initial*learned) < 0).sum())
    if (learned[initial == 0] != 0).any() or (config['preserve_signs'] and flips):
        raise ValueError('Training violated the declared recurrent graph/signs')
    scenes, scores = [], []
    for scenario in SCENES:
        for seed in config['scene_seeds']:
            env = NavigationScene(scenario)
            try:
                score, metrics = simulate_policy_rollout(policy, env=env, eval_steps=config['scene_steps'],
                    seed=seed, dt=CONTROL_DT, altitude_hold=AltitudeHold())
                scenes.append({'scenario':scenario, 'seed':seed, **metrics}); scores.append(score)
            finally:
                env.close()
    admission = assess_navigation(temporal, heldout, scenes, spatial=spatial)
    # Connectivity must also survive zeroing allowed edges during learning.
    from core.routing import sensor_motor_paths
    from generator.reflex_contract import REQUIRED_PATHS
    paths = sensor_motor_paths(learned != 0, policy.cfc_network.cell.sensor_indices, policy.dn_head._proj_mask)
    for group, channels in REQUIRED_PATHS.items():
        for channel in channels:
            if paths[group]['minimum_hops'][channel] is None:
                admission['reasons'].append(f'lost_route:{group}->{channel}')
                admission['passed'] = False
    report = {'format_version':'registry-search-trial-v1', 'candidate':name, 'trial_number':trial.number,
        'experiment_fingerprint':contract_sha256(experiment), 'config':config,
        'policy_options':options, 'learning_rate':lr, 'pretrain':policy._pretrain_info,
        'heldout':heldout, 'temporal_response':temporal, 'spatial_response':spatial, 'scenes':scenes,
        'functional_admission':admission, 'recurrent_sign_changes':flips,
        'learned_sensor_motor_paths':paths, 'source_quality_review_required':pool.verification['quality']['quality_review_required']}
    trial_dir = Path(output)/f'trial_{trial.number:05d}'
    export_trial(policy, pool, name, trial_dir, report, options)
    trial.set_user_attr('benchmark_version', BENCHMARK_VERSION)
    trial.set_user_attr('control_contract', control_contract())
    trial.set_user_attr('feasible', admission['passed'])
    trial.set_user_attr('constraints', [0. if admission['passed'] else 1.])
    trial.set_user_attr('functional_admission', admission)
    trial.set_user_attr('experiment_fingerprint', contract_sha256(experiment))
    trial.set_user_attr('research_only', pool.research_only)
    trial.set_user_attr('artifact_directory', str(trial_dir.resolve()))
    if hasattr(trial, 'set_constraint'):
        trial.set_constraint('flight_safety', 0. if admission['passed'] else 1.)
    return tuple(float(v) for v in np.mean(scores, axis=0))


def run_search(source, candidates, dataset, output, *, config=None, research_only=False,
               trials=2, storage=None, study_name='registry_navigation_v8', preflight_only=False):
    config = validate_search_config(config or {})
    if type(trials) is not int or not 1 <= trials <= 20:
        raise ValueError('Use 1..20 trials per bounded research run')
    if config['device'] == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; no silent CPU fallback')
    output = Path(output)
    dataset = Path(dataset)
    pool = CandidatePool.load(source, candidates, research_only=research_only)
    dataset_hash = file_sha256(dataset)
    data = torch.load(dataset, map_location='cpu', weights_only=True)
    if file_sha256(dataset) != dataset_hash:
        raise ValueError('Dataset changed during loading')
    validate_demonstration_context(data, config['scene_seeds'])
    # Full pretrain contract validation before creating a study or updating weights.
    preflight_policy = pool.create_policy(pool.names[0], {k:config[k] for k in
                                            ('sensor_encoding', 'neutral_origin', 'preserve_signs')})
    pretrain_policy(preflight_policy, data, epochs=0, subset_ratio=1., seed=config['seed'],
                    research_only=research_only, loss_mode=config['loss_mode'], yaw_balance=config['yaw_balance'])
    if not preflight_policy._pretrain_info['validation_indices']:
        raise ValueError('Search requires nonempty held-out demonstration episodes')
    root = Path(__file__).resolve().parents[1]
    files = ['optimizer/search_pipeline.py', 'optimizer/candidate_selection.py', 'optimizer/optuna_tuner.py',
        'optimizer/pretrain.py', 'optimizer/navigation_loss.py', 'optimizer/diagnose_candidate.py',
        'optimizer/rollout.py', 'core/models.py', 'core/routing.py', 'core/contracts.py',
        'simulation/policy.py', 'simulation/policy_diagnostics.py', 'simulation/metrics.py',
        'simulation/benchmark_scenes.py', 'simulation/drone_env.py', 'simulation/drone_interface.py',
        'simulation/altitude_control.py',
        'simulation/avionics_filter.py', 'simulation/control_contract.py', 'simulation/memory.py',
        'simulation/pmw3901_emulator.py', 'configs/flight_config.py', 'generator/reflex_contract.py',
        'generator/generate_reflex_dataset.py']
    experiment = {'format_version':'registry-search-contract-v1', **pool.identity,
        'dataset_sha256':dataset_hash, 'study_name':study_name, 'config':config, 'functional_gate':GATE,
        'benchmark_version':BENCHMARK_VERSION, 'control_contract':control_contract(),
        'code_sha256':{**pool.verification.get('code_sha256', {}),
                       **{name:file_sha256(root/name) for name in files}}, 'torch_version':str(torch.__version__)}
    if output.exists():
        prior = json.loads((output/'preflight.json').read_text(encoding='utf-8'))
        if prior.get('experiment_fingerprint') != contract_sha256(experiment):
            raise ValueError('Output belongs to a different experiment fingerprint; use a new directory')
    else:
        with publication(output) as stage:
            write_json(stage/'preflight.json', {'experiment':experiment,
                'experiment_fingerprint':contract_sha256(experiment), 'eligible_candidates':pool.names,
                'source_verification':pool.verification, 'training_ready':False})
    if preflight_only:
        return {'eligible_candidates':pool.names, 'experiment_fingerprint':contract_sha256(experiment)}
    if storage is None:
        storage = 'sqlite:///' + (output.resolve()/'study.db').as_posix()
    study = create_study(study_name=study_name, storage=storage, seed=config['seed'], experiment_contract=experiment)
    if not study.trials:
        # Ensure the first comparison actually covers every eligible graph.
        for candidate in pool.names:
            study.enqueue_trial({'candidate':candidate,
                'learning_rate':math.sqrt(config['learning_rate_min']*config['learning_rate_max']),
                'tau_init':min(config['tau_choices'], key=lambda t:abs(t-.1))})
    study.optimize(lambda trial:candidate_objective(trial, pool=pool, data=data, config=config,
                    output=output, experiment=experiment), n_trials=trials)
    frontier = feasible_pareto_trials(study)
    write_json(output/'summary.json', {'study_name':study.study_name, 'research_only':research_only,
        'experiment_fingerprint':contract_sha256(experiment), 'functional_pareto_trials':[t.number for t in frontier],
        'training_ready':False, 'complete_trials':len([t for t in study.trials if t.values is not None])})
    return study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'candidates', 'dataset', 'out-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--device', choices=['cpu', 'cuda'], help='Explicit override of the configuration device')
    parser.add_argument('--research-only', action='store_true')
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--trials', type=int, default=2)
    parser.add_argument('--storage', default=None)
    parser.add_argument('--study-name', default='registry_navigation_v8')
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = json.loads(args.config.read_text(encoding='utf-8')) if args.config else {}
    if args.device is not None:
        config['device'] = args.device
    result = run_search(args.source, args.candidates, args.dataset, args.out_dir, config=config,
        research_only=args.research_only, trials=args.trials, storage=args.storage,
        study_name=args.study_name, preflight_only=args.preflight_only)
    if isinstance(result, dict):
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
