import pytest


@pytest.fixture
def interrupted_demonstrations(monkeypatch):
    from generator import generate_reflex_dataset as generator
    from simulation.altitude_control import AltitudeTelemetryError
    original = generator.AltitudeHold
    class FaultAfterOne(original):
        def apply(self, action, sample, now_s):
            if now_s > 0:
                raise AltitudeTelemetryError('test discontinuity')
            return super().apply(action, sample, now_s=now_s)
    monkeypatch.setattr(generator, 'AltitudeHold', FaultAfterOne)
    return generator.generate_reflex_dataset(num_episodes=2, seq_len=4, seed=71, scene_profile='varied-v1')


def test_failure_replay_checks_recorded_prefix_and_retains_fault(interrupted_demonstrations):
    from optimizer.replay_demonstrations import audit_failures
    result = audit_failures(interrupted_demonstrations)
    assert len(result['failures']) == 2
    for row in result['failures']:
        assert row['prefix_exact'] is True and row['valid_steps'] == 1
        assert row['reproduced_failure_reason'] == 'test discontinuity'
        assert row['trace'][-1]['tof_min_m'] > 0


def test_failure_replay_refuses_changed_observations(interrupted_demonstrations):
    from optimizer.replay_demonstrations import audit_failures
    interrupted_demonstrations['X'][0,0,2] = 0
    with pytest.raises(ValueError, match='prefix'):
        audit_failures(interrupted_demonstrations)


def test_failure_replay_refuses_unrecorded_execution_noise(interrupted_demonstrations):
    from optimizer.replay_demonstrations import audit_failures
    interrupted_demonstrations['metadata']['execution_noise_std_pwm'] = 1
    with pytest.raises(ValueError, match='noise'):
        audit_failures(interrupted_demonstrations)
