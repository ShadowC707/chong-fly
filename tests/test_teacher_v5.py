"""The repaired teacher must reach collection, training, and evaluation."""
import numpy as np
import pytest
import torch


def test_factory_preserves_v4_and_selects_repaired_v5():
    from generator.reflex_contract import TEACHER_VERSION, BRAKE_FIRST_TEACHER_VERSION
    from generator.generate_reflex_dataset import make_teacher, ExpertReflexPolicy
    from generator.continuous_reflex_teacher import CorrectedExpertReflexPolicy
    assert TEACHER_VERSION == 'geometry-reflex-v5'
    old = make_teacher(BRAKE_FIRST_TEACHER_VERSION, noise_std_pwm=0)
    new = make_teacher(TEACHER_VERSION, noise_std_pwm=0)
    assert type(old) is ExpertReflexPolicy
    assert isinstance(new, CorrectedExpertReflexPolicy)
    force = np.zeros(64); force[:2] = [100., 1.]
    assert old._summarize_repulsion(force, force>0)[0] == 50.5
    assert new._summarize_repulsion(force, force>0)[0] == 100.


def test_default_collection_labels_replay_with_v5_and_differ_from_v4():
    from generator.generate_reflex_dataset import generate_reflex_dataset, make_teacher
    data = generate_reflex_dataset(num_episodes=8, seq_len=12, seed=23, noise_std_pwm=0)
    meta = data['metadata']
    assert meta['teacher_version'] == 'geometry-reflex-v5'
    assert meta['teacher_aggregation_version'] == 'max-brake-force-weighted-direction-v1'
    different = False
    for ep in range(8):
        current = make_teacher(meta['teacher_version'], **meta['teacher_parameters'])
        old = make_teacher('geometry-reflex-v4', **meta['teacher_parameters'])
        for t in range(int(data['valid'][ep].sum())):
            x = data['X'][ep,t].numpy()
            np.testing.assert_array_equal(current.step(x),data['Y'][ep,t].numpy())
            different |= not np.array_equal(old.step(x),data['Y'][ep,t].numpy())
    assert different, 'The dataset must exercise the repaired aggregation'


def test_explicit_historical_collection_retains_its_version():
    from generator.generate_reflex_dataset import generate_reflex_dataset
    data = generate_reflex_dataset(num_episodes=2, seq_len=2, teacher_version='geometry-reflex-v4')
    assert data['metadata']['teacher_version'] == 'geometry-reflex-v4'
    assert data['metadata']['teacher_aggregation_version'] == 'active-ray-mean-v1'


def test_pretrain_rejects_old_teacher_and_wrong_aggregation_before_updates():
    from tests.test_pretrain_contract import dataset, TimedPolicy
    from optimizer.pretrain import pretrain_policy
    for key, value in [('teacher_version','geometry-reflex-v4'),
                       ('teacher_aggregation_version','active-ray-mean-v1')]:
        data = dataset(); data['metadata'][key] = value
        model = TimedPolicy()
        with pytest.raises(ValueError):
            pretrain_policy(model,data,epochs=1)
        assert model.seen_dt == []


def test_temporal_probe_records_and_uses_v5_reference():
    from tests.test_pretrain_contract import TimedPolicy
    from simulation.policy_diagnostics import temporal_response_probe
    class ProbePolicy(TimedPolicy):
        def forward(self,x,dt=None):
            return super().forward(x,dt), None
    report = temporal_response_probe(ProbePolicy(),coverage_columns=(1,4))
    assert report['teacher_version'] == 'geometry-reflex-v5'
