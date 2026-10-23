import pytest


def test_bundle_audit_checks_every_label_and_rejects_corruption():
    from generator.generate_reflex_dataset import generate_reflex_dataset
    from generator.prepare_reflex_training import audit_labels
    data=generate_reflex_dataset(num_episodes=8,seq_len=2,seed=91,noise_std_pwm=0)
    result=audit_labels(data)
    assert result['mismatch_frames'] == 0 and result['valid_frames'] == 16
    data['Y'][7,1,3] += 1
    with pytest.raises(ValueError,match='replay'):
        audit_labels(data)


def test_current_gate_rejects_historical_teacher_reference():
    from optimizer.search_pipeline import assess_navigation
    result=assess_navigation({'teacher_version':'geometry-reflex-v4'}, {}, [],
                             spatial={'teacher_version':'geometry-reflex-v4'})
    assert 'teacher_reference_mismatch' in result['reasons']
