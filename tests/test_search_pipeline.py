import json
import pytest
import torch
from tests.test_candidate_selection import registry_fixture


@pytest.mark.parametrize('fault', ['dt', 'missing_seeds', 'overlap', 'duplicate_seed', 'scenario_mismatch'])
def test_search_rejects_unverifiable_demonstration_timebase_and_seeds(fault):
    from generator.generate_reflex_dataset import generate_reflex_dataset
    from optimizer.search_pipeline import validate_demonstration_context
    data = generate_reflex_dataset(num_episodes=16, seq_len=4, seed=71)
    if fault == 'dt':
        data['metadata']['dt'] = .04
    elif fault == 'missing_seeds':
        data['metadata']['episodes'] = []
    elif fault == 'overlap':
        data['metadata']['episodes'][0]['seed'] = 999
    elif fault == 'duplicate_seed':
        data['metadata']['episodes'][1]['seed'] = data['metadata']['episodes'][0]['seed']
    else:
        data['metadata']['episodes'][0]['scenario'] = 'different'
    with pytest.raises(ValueError):
        validate_demonstration_context(data, [999])


def test_functional_gate_accepts_complete_current_engineering_evidence():
    from optimizer.search_pipeline import assess_navigation
    from optimizer.rollout import BENCHMARK_VERSION
    from configs.flight_config import CONTROL_DT
    from simulation.benchmark_scenes import SCENES
    from simulation.control_contract import control_contract
    cases = [{'obstacle_side':side, 'normalized_distance':distance, 'clear_abs_yaw_pwm':0.,
        'direction_correct':True, 'threat_yaw_pwm':1620. if side == 'left' else 1380.,
        'threat_target_yaw_pwm':1620. if side == 'left' else 1380., 'onset_s':.1,
        'recovery_settle_s':.2} for side in ('left','right') for distance in (.1,.2)]
    temporal = {'dt':CONTROL_DT, 'cases':cases}
    heldout = {'channel_mae_pwm':{'roll':5., 'pitch':20.},
               'yaw_groups':{'right':{'mae_pwm':50.}, 'left':{'mae_pwm':50.}}}
    scenes = [{'scenario':name, 'benchmark_version':BENCHMARK_VERSION,
        'control_contract':control_contract(), 'feasible':True, 'mean_fwd_speed':.3,
        'min_clearance_m':.2, 'saccades_yaw_count':1} for name in SCENES]
    spatial = {'dt':CONTROL_DT, 'cases':[{**row, 'active_columns':width}
               for row in cases for width in (1,2,4)]}
    assert assess_navigation(temporal, heldout, scenes, spatial=spatial)['passed'] is True
    spatial['cases'][0]['threat_yaw_pwm'] = 1900
    failed = assess_navigation(temporal, heldout, scenes, spatial=spatial)
    assert failed['passed'] is False and any('spatial' in r and 'amplitude' in r for r in failed['reasons'])
    spatial['cases'][0]['threat_yaw_pwm'] = spatial['cases'][0]['threat_target_yaw_pwm']
    temporal['cases'][0] = dict(temporal['cases'][1])
    assert assess_navigation(temporal, heldout, scenes, spatial=spatial)['passed'] is False


def test_functional_admission_rejects_constant_turn_and_hover_even_without_crash():
    from optimizer.search_pipeline import assess_navigation
    from optimizer.rollout import BENCHMARK_VERSION
    from simulation.control_contract import control_contract
    temporal = {'cases':[{'clear_abs_yaw_pwm':35., 'direction_correct':True,
        'threat_yaw_pwm':1650., 'threat_target_yaw_pwm':1850., 'onset_s':.1,
        'recovery_settle_s':None}]}
    heldout = {'channel_mae_pwm':{'roll':5., 'pitch':20.},
               'yaw_groups':{'right':{'mae_pwm':50.}, 'left':{'mae_pwm':50.}}}
    scenes = [{'scenario':'clear', 'benchmark_version':BENCHMARK_VERSION,
               'control_contract':control_contract(), 'feasible':True,
               'mean_fwd_speed':0., 'min_clearance_m':.5}]
    admission = assess_navigation(temporal, heldout, scenes)
    assert admission['passed'] is False
    assert any('clear_yaw' in r for r in admission['reasons'])
    assert any('progress' in r for r in admission['reasons'])
    assert any('amplitude' in r for r in admission['reasons'])


def test_functional_admission_requires_all_scene_evidence_and_current_benchmark():
    from optimizer.search_pipeline import assess_navigation
    admission = assess_navigation({'cases':[]}, {'channel_mae_pwm':{}, 'yaw_groups':{}}, [])
    assert admission['passed'] is False and 'missing_challenge_scenes' in admission['reasons']


def test_search_configuration_rejects_unknown_fields_and_impossible_device():
    from optimizer.search_pipeline import validate_search_config
    with pytest.raises(ValueError, match='Unknown'):
        validate_search_config({'made_up_option':True})
    with pytest.raises(ValueError, match='epochs'):
        validate_search_config({'epochs':0})


def test_registry_search_exports_reloadable_weights_and_resumes_local_database(registry_fixture, tmp_path):
    from generator.generate_reflex_dataset import generate_reflex_dataset
    from optimizer.search_pipeline import run_search
    from optimizer.benchmark_candidate import load_research_run
    directory, _ = registry_fixture
    data = generate_reflex_dataset(num_episodes=16, seq_len=6, seed=71)
    dataset = tmp_path/'demonstrations.pt'
    torch.save(data, dataset)
    output = tmp_path/'search'
    config = {'epochs':1, 'scene_steps':2, 'scene_seeds':[999], 'tau_choices':[.2]}
    study = run_search(directory, directory, dataset, output, config=config, research_only=True, trials=1)
    assert len(study.trials) == 1 and (output/'study.db').is_file()
    policy, checkpoint = load_research_run(output/'trial_00000')
    assert checkpoint['tau_init'] == .2 and checkpoint['training_ready'] is False
    assert checkpoint['graph_file_sha256']
    assert policy.neutral_origin is True
    report = json.loads((output/'trial_00000'/'report.json').read_text())
    assert report['checkpoint_roundtrip_exact'] is True
    assert report['functional_admission']['passed'] is False
    resumed = run_search(directory, directory, dataset, output, config=config, research_only=True, trials=1)
    assert len(resumed.trials) == 2
    assert (output/'trial_00001'/'checkpoint.pt').is_file()
    torch.save({**data, 'metadata':{**data['metadata'], 'seed':72}}, dataset)
    with pytest.raises(ValueError, match='fingerprint'):
        run_search(directory, directory, dataset, output, config=config, research_only=True, trials=1)
    assert len(resumed.trials) == 2
    graph_file = output/'trial_00000'/'graph'/next(iter(checkpoint['graph_file_sha256']))
    graph_file.write_bytes(graph_file.read_bytes()+b' ')
    with pytest.raises(ValueError, match='hash'):
        load_research_run(output/'trial_00000')


def test_checkpoint_rejects_changed_altitude_controller_even_with_matching_file_hash(registry_fixture, tmp_path):
    from generator.source_contract import file_sha256
    from optimizer.candidate_selection import CandidatePool
    from optimizer.search_pipeline import export_trial
    from optimizer.benchmark_candidate import load_research_run
    directory, _ = registry_fixture
    pool = CandidatePool.load(directory, directory, research_only=True)
    options = {'sensor_encoding':'proximity-v1', 'neutral_origin':True,
               'preserve_signs':True, 'tau_init':.1, 'mode':'masked'}
    policy = pool.create_policy(pool.names[0], options)
    policy._pretrain_info = {}
    output = tmp_path/'export'
    export_trial(policy, pool, pool.names[0], output, {}, options)
    checkpoint = torch.load(output/'checkpoint.pt', weights_only=True)
    checkpoint['control_contract']['altitude']['config']['hover_pwm'] += 1
    torch.save(checkpoint, output/'checkpoint.pt')
    report = json.loads((output/'report.json').read_text())
    report['checkpoint_sha256'] = file_sha256(output/'checkpoint.pt')
    (output/'report.json').write_text(json.dumps(report))
    with pytest.raises(ValueError, match='control contract'):
        load_research_run(output)
