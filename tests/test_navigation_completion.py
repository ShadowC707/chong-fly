import numpy as np
import pytest
import torch


def test_proximity_encoding_resolves_distance_and_suppresses_far_background():
    from simulation.policy import SensorInputLayer
    layer = SensorInputLayer(74, learnable_scale=False, encoding='proximity-v1')
    raw = torch.ones(3, 74)
    raw[:, :2] = .2
    raw[:, 2:66] = torch.tensor([.35, .7, 1.5]).unsqueeze(1)/3.5
    encoded = layer(raw)
    assert encoded[0, 2] == pytest.approx(.5625)
    assert encoded[1, 2] == pytest.approx(.125)
    assert encoded[2, 2:66].count_nonzero() == 0
    torch.testing.assert_close(encoded[:, :2], raw[:, :2])


def test_proximity_can_use_neutral_origin_and_has_distinct_checkpoint(tiny_model_meta):
    from simulation.policy import ChongFlyMSPPolicy
    new = ChongFlyMSPPolicy.from_meta(str(tiny_model_meta), sensor_encoding='proximity-v1', neutral_origin=True)
    old = ChongFlyMSPPolicy.from_meta(str(tiny_model_meta), sensor_encoding='threat-v1', neutral_origin=True)
    with pytest.raises(RuntimeError, match='contract'):
        new.load_state_dict(old.state_dict(), strict=False)
    clear = torch.ones(1, 5, 74); clear[..., :2] = 0
    assert torch.all(new(clear)[0][..., 3] == 1500)


def test_amplitude_balancing_does_not_ignore_rare_strong_turns():
    from optimizer.navigation_loss import navigation_weights
    target = torch.full((1, 10, 4), 1500.)
    target[0, :6, 3] = 1620
    target[0, 6, 3] = 1850
    valid = torch.ones(1, 10, dtype=torch.bool)
    behavior = torch.zeros(1, 10, dtype=torch.long)
    weights = navigation_weights(target, valid, behavior, 1, yaw_balance=1., amplitude_bins=True)
    assert weights[0, :6, 2].sum() == pytest.approx(weights[0, 6, 2])
    torch.testing.assert_close(weights.sum((0, 1)), torch.full((3,), 10.))


def test_varied_physical_demonstrations_record_scene_profile_and_remain_reproducible():
    from generator.generate_reflex_dataset import generate_reflex_dataset
    first = generate_reflex_dataset(num_episodes=8, seq_len=4, seed=71, scene_profile='varied-v1')
    second = generate_reflex_dataset(num_episodes=8, seq_len=4, seed=71, scene_profile='varied-v1')
    assert first['metadata']['scene_profile'] == 'varied-v1'
    torch.testing.assert_close(first['X'], second['X'], rtol=0, atol=0)
    assert first['metadata']['episodes'][1]['scene_parameters']['initial_speed_mps'] != .3


def test_physical_yaw_bursts_ignore_command_jitter_and_sustained_rotation():
    from simulation.metrics import measure_yaw_bursts
    pulse = np.r_[np.zeros(10), np.ones(20), np.zeros(10)]
    report = measure_yaw_bursts(pulse, dt=.02)
    assert report['count'] == 1
    assert report['events'][0]['angle_rad'] == pytest.approx(.4)
    assert measure_yaw_bursts(np.ones(200), dt=.02)['count'] == 0
    assert measure_yaw_bursts(np.zeros(100), dt=.02)['count'] == 0
    # Opposite pulses are different maneuvers, not one long turn.
    assert measure_yaw_bursts(np.r_[pulse, -pulse], dt=.02)['count'] == 2


def test_physical_yaw_burst_contract_rejects_bad_timebase():
    from simulation.metrics import measure_yaw_bursts
    with pytest.raises(ValueError, match='dt'):
        measure_yaw_bursts([0., 1.], dt=0)


def test_varied_collection_records_telemetry_failure_as_prefix_not_new_labels(monkeypatch):
    from generator import generate_reflex_dataset as generator
    from simulation.altitude_control import AltitudeTelemetryError
    original = generator.AltitudeHold
    class FaultAfterOne(original):
        def apply(self, action, sample, now_s):
            if now_s > 0:
                raise AltitudeTelemetryError('test discontinuity')
            return super().apply(action, sample, now_s=now_s)
    monkeypatch.setattr(generator, 'AltitudeHold', FaultAfterOne)
    data = generator.generate_reflex_dataset(num_episodes=1, seq_len=5, scene_profile='varied-v1')
    assert data['valid'].tolist() == [[True, False, False, False, False]]
    assert data['metadata']['episodes'][0]['failure_reason'] == 'test discontinuity'
    assert data['metadata']['episodes'][0]['terminated'] is True
    assert data['behavior'][0, 1:].tolist() == [-1]*4
