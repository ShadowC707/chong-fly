import pytest
import torch

from configs.flight_config import CONTROL_DT


def test_threat_encoding_has_zero_clear_origin_and_preserves_flow_and_locations():
    from simulation.policy import SensorInputLayer
    layer = SensorInputLayer(74, learnable_scale=False, encoding='threat-v1')
    raw = torch.ones(2, 74)
    raw[:, :2] = torch.tensor([.2, -.3])
    raw[1, 5] = .2
    encoded = layer(raw)
    torch.testing.assert_close(encoded[:, :2], raw[:, :2])
    assert encoded[0, 2:].count_nonzero() == 0
    assert encoded[1, 5] == pytest.approx(.8)
    assert raw[1, 5] == pytest.approx(.2)  # input is not mutated


def test_sensor_encoding_is_guarded_by_checkpoint_contract(tiny_model_meta):
    from simulation.policy import ChongFlyMSPPolicy
    old = ChongFlyMSPPolicy.from_meta(str(tiny_model_meta))
    changed = ChongFlyMSPPolicy.from_meta(str(tiny_model_meta), sensor_encoding='threat-v1')
    with pytest.raises(RuntimeError, match='contract'):
        changed.load_state_dict(old.state_dict(), strict=False)
    restored = ChongFlyMSPPolicy.from_meta(str(tiny_model_meta), sensor_encoding='threat-v1')
    restored.load_state_dict(changed.state_dict())
    obs = torch.ones(1, 4, 74)
    obs[..., :2] = 0
    torch.testing.assert_close(restored(obs)[0], changed(obs)[0], rtol=0, atol=0)


def test_yaw_balancing_uses_its_targets_not_other_channels_behavior_labels():
    from optimizer.navigation_loss import navigation_weights
    y = torch.full((1, 10, 4), 1500.)
    y[0, 8:, 3] = torch.tensor([1700., 1300.])
    valid = torch.ones(1, 10, dtype=torch.bool)
    labels = torch.zeros(1, 10, dtype=torch.long)
    first = navigation_weights(y, valid, labels, 2, yaw_balance=.5)
    labels[0, :7] = 1  # classify neutral yaw as braking, without changing yaw targets
    second = navigation_weights(y, valid, labels, 2, yaw_balance=.5)
    torch.testing.assert_close(first[..., 2], second[..., 2])
    torch.testing.assert_close(first.sum((0, 1)), torch.full((3,), 10.))
    assert first[0, :8, 2].sum() == pytest.approx(10*(.5*.8+.5/3))


def test_navigation_loss_ignores_padding_and_throttle_and_keeps_global_normalizer():
    from optimizer.navigation_loss import navigation_weights, navigation_loss
    target = torch.full((1, 3, 4), 1500.)
    target[0, 1, 3] = 1700
    valid = torch.tensor([[True, True, False]])
    labels = torch.tensor([[0, 1, -1]])
    weights = navigation_weights(target, valid, labels, 2, yaw_balance=.5)
    prediction = target.clone().requires_grad_()
    with torch.no_grad():
        prediction[0, 0, 3] += 100
        prediction[..., 0] = 2000
        prediction[0, 2] = 1000
    full = navigation_loss(prediction, target, weights, normalizer=6)
    chunks = sum(navigation_loss(prediction[:, i:i+1], target[:, i:i+1],
                                 weights[:, i:i+1], normalizer=6) for i in range(3))
    torch.testing.assert_close(full, chunks)
    full.backward()
    assert prediction.grad[..., 0].count_nonzero() == 0
    assert prediction.grad[0, 2].count_nonzero() == 0
    assert weights[0, 2].count_nonzero() == 0


@pytest.mark.parametrize('balance', [-.1, 1.1, float('nan')])
def test_invalid_balance_is_rejected(balance):
    from optimizer.navigation_loss import navigation_weights
    with pytest.raises(ValueError, match='balance'):
        navigation_weights(torch.full((1, 2, 4), 1500.), torch.ones(1, 2, dtype=torch.bool),
                           torch.zeros(1, 2, dtype=torch.long), 1, yaw_balance=balance)


def test_temporal_probe_detects_a_turn_that_never_releases():
    from simulation.policy_diagnostics import temporal_response_probe
    class Latched(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
        def forward(self, obs, dt=None):
            assert dt == CONTROL_DT
            tof = obs[..., 2:66].reshape(*obs.shape[:2], 8, 8)
            delta = 200*(tof[..., 4:].mean((-1, -2))-tof[..., :4].mean((-1, -2)))
            # Infinite latching: danger goes away, but the command persists.
            yaw = delta.clone()
            for t in range(1, obs.shape[1]):
                yaw[:, t] = torch.where(delta[:, t] == 0, yaw[:, t-1], delta[:, t])
            pwm = torch.full((*obs.shape[:2], 4), 1500.)
            pwm[..., 3] += yaw
            return pwm, None
    result = temporal_response_probe(Latched())
    assert result['dt'] == CONTROL_DT
    for row in result['cases']:
        assert row['direction_correct'] is True
        assert row['recovery_settle_s'] is None
        assert row['recovery_final_abs_yaw_pwm'] > 20


def test_pretrain_records_explicit_channel_objective_and_train_only_group_counts():
    from optimizer.pretrain import pretrain_policy
    from tests.test_pretrain_contract import dataset, TimedPolicy
    data = dataset()
    data['Y'][0, :, 3] = 1700
    model = TimedPolicy()
    pretrain_policy(model, data, epochs=1, subset_ratio=1., seed=42,
                    loss_mode='channel-v2', yaw_balance=.5)
    info = model._pretrain_info
    ids = info['subset_indices']
    target = data['Y'][ids, :, 3]
    assert info['loss_contract']['version'] == 'channel-v2'
    assert info['loss_contract']['yaw_balance'] == .5
    assert info['training_yaw_counts']['right'] == int((target > 1501).sum())
    assert sum(info['training_yaw_counts'].values()) == len(ids)*5


def test_invalid_loss_configuration_fails_before_updating_parameters():
    from optimizer.pretrain import pretrain_policy
    from tests.test_pretrain_contract import dataset, TimedPolicy
    model = TimedPolicy()
    with pytest.raises(ValueError, match='loss_mode'):
        pretrain_policy(model, dataset(), loss_mode='typo')
    assert model.seen_dt == []


def test_neutral_origin_stays_neutral_after_updates_without_gating_recurrent_state(tiny_model_meta):
    from simulation.policy import ChongFlyMSPPolicy
    policy = ChongFlyMSPPolicy.from_meta(str(tiny_model_meta), sensor_encoding='threat-v1',
                                        neutral_origin=True, preserve_signs=True)
    optimizer = torch.optim.Adam(policy.parameters(), lr=.01)
    for _ in range(3):
        optimizer.zero_grad()
        prediction, _ = policy(torch.rand(2, 10, 74))
        (prediction[..., 3]-1700).square().mean().backward()
        optimizer.step()
        policy.post_step()
    clear = torch.ones(1, 10, 74)
    clear[..., :2] = 0
    pwm, state = policy(clear)
    torch.testing.assert_close(pwm[..., 3], torch.full((1, 10), 1500.), rtol=0, atol=0)
    assert state.count_nonzero() == 0
    # Clear sensors must NOT erase a previously activated motor neuron.
    cell = policy.cfc_network.cell
    previous = torch.zeros(1, cell.hidden_size)
    previous[0, cell.motor_indices['yaw'][0]] = 1
    _, remaining = policy(clear[:, 0], hx=previous)
    assert remaining.count_nonzero() > 0
    assert not cell.b.requires_grad and not policy.sensor_layer.bias.requires_grad


def test_neutral_origin_requires_explicit_threat_encoding_and_guards_checkpoints(tiny_model_meta):
    from simulation.policy import ChongFlyMSPPolicy
    with pytest.raises(ValueError, match='threat'):
        ChongFlyMSPPolicy.from_meta(str(tiny_model_meta), neutral_origin=True)
    active = ChongFlyMSPPolicy.from_meta(str(tiny_model_meta), sensor_encoding='threat-v1', neutral_origin=True)
    old = ChongFlyMSPPolicy.from_meta(str(tiny_model_meta), sensor_encoding='threat-v1')
    with pytest.raises(RuntimeError, match='contract'):
        old.load_state_dict(active.state_dict(), strict=False)


def test_legacy_sequence_padding_matches_explicit_clear_memory(tiny_model_meta):
    from simulation.policy import ChongFlyMSPPolicy
    policy = ChongFlyMSPPolicy.from_meta(str(tiny_model_meta), sensor_encoding='threat-v1')
    raw = torch.rand(2, 5, 66)
    explicit = torch.cat([raw, torch.ones(2, 5, 8)], dim=-1)
    torch.testing.assert_close(policy(raw)[0], policy(explicit)[0], rtol=0, atol=0)
