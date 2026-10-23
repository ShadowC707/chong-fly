import json

import optuna
import pytest


@pytest.fixture
def registry_fixture(tiny_model_meta, monkeypatch):
    from generator.source_contract import file_sha256
    import generator.validate_flywire_candidates as validator
    directory = tiny_model_meta.parent
    metadata = json.loads(tiny_model_meta.read_text())
    metadata['provenance']['training_ready'] = False
    tiny_model_meta.write_text(json.dumps(metadata))
    paths = [tiny_model_meta, directory/metadata['w_file'], directory/metadata['cmap_file']]
    registry = {'format_version':'filtered783-candidate-registry-v1',
        'files':{p.name:file_sha256(p) for p in paths},
        'candidates':[{'candidate':'tiny_k8', 'meta_file':tiny_model_meta.name,
                       'k':8, 'structural_admission':'passed', 'training_ready':False}]}
    (directory/'registry.json').write_text(json.dumps(registry))
    verification = {'registry_sha256':file_sha256(directory/'registry.json'),
        'quality':{'quality_review_required':True},
        'candidates':[{'candidate':'tiny_k8', 'structural_admission':'passed'}]}
    monkeypatch.setattr(validator, 'validate', lambda *a: verification)
    return directory, verification


def test_pool_requires_explicit_research_mode_for_unapproved_registry(registry_fixture):
    from optimizer.candidate_selection import CandidatePool
    directory, _ = registry_fixture
    with pytest.raises(ValueError, match='eligible|admission'):
        CandidatePool.load(directory, directory, research_only=False)
    pool = CandidatePool.load(directory, directory, research_only=True)
    assert pool.names == ['tiny_k8']


def test_pool_uses_recomputed_admission_not_registry_claim(registry_fixture):
    from optimizer.candidate_selection import CandidatePool
    directory, verification = registry_fixture
    verification['candidates'][0]['structural_admission'] = 'rejected'
    with pytest.raises(ValueError, match='eligible'):
        CandidatePool.load(directory, directory, research_only=True)


def test_factory_selects_real_registry_names_and_propagates_policy_contract(registry_fixture):
    from optimizer.candidate_selection import CandidatePool
    from optimizer.evaluate import create_model
    directory, _ = registry_fixture
    pool = CandidatePool.load(directory, directory, research_only=True)
    class Trial:
        def suggest_categorical(self, key, choices):
            assert key == 'candidate' and choices == ['tiny_k8']
            return choices[0]
    policy = create_model(Trial(), candidate_pool=pool, policy_options={
        'sensor_encoding':'proximity-v1', 'neutral_origin':True, 'preserve_signs':True})
    assert policy.cfc_network.cell.hidden_size == 8
    assert policy.neutral_origin is True
    assert policy.training_admission is False
    assert policy.reduction_diagnostics['candidate'] == 'tiny_k8'


def test_search_contract_cannot_be_changed_when_resuming(tmp_path):
    from optimizer.optuna_tuner import create_study
    storage = 'sqlite:///' + (tmp_path/'fingerprint.db').as_posix()
    expected = {'dataset_sha256':'a', 'registry_sha256':'b', 'policy':{'neutral_origin':True}}
    create_study(storage=storage, experiment_contract=expected)
    changed = {**expected, 'dataset_sha256':'changed'}
    with pytest.raises(ValueError, match='experiment|fingerprint'):
        create_study(storage=storage, experiment_contract=changed)
    with pytest.raises(ValueError, match='experiment|fingerprint'):
        create_study(storage=storage)


def test_old_optuna_grid_requires_a_registry(tmp_path):
    from optimizer.evaluate import create_model
    class Trial:
        def suggest_categorical(self, *args):
            raise AssertionError('Do not select the retired 128/256 grid')
    with pytest.raises(ValueError, match='registry'):
        create_model(Trial(), base_dir=str(tmp_path))


def test_registry_artifact_changes_are_detected_before_constructing_policy(registry_fixture):
    from optimizer.candidate_selection import CandidatePool
    directory, _ = registry_fixture
    pool = CandidatePool.load(directory, directory, research_only=True)
    meta = next(directory.glob('meta_*.json'))
    meta.write_text(meta.read_text()+' ')
    with pytest.raises(ValueError, match='SHA256|hash|changed|integrity'):
        pool.create_policy('tiny_k8', {})
