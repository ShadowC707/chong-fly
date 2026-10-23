import pytest
import torch


def test_recovery_labels_are_causal_episode_local_and_ignore_padding():
    from optimizer.navigation_loss import yaw_recovery_mask
    target = torch.full((2, 7, 4), 1500.)
    target[0, 1, 3] = 1700
    target[0, 5, 3] = 1300
    valid = torch.ones(2, 7, dtype=torch.bool)
    valid[0, 6] = False
    actual = yaw_recovery_mask(target, valid, steps=2)
    assert actual[0].tolist() == [False, False, True, True, False, False, False]
    assert not actual[1].any()
    # Future turns cannot change past weights; padded labels cannot create recovery.
    changed = target.clone()
    changed[:, 6, 3] = 2000
    assert torch.equal(yaw_recovery_mask(changed, valid, steps=2)[:, :6], actual[:, :6])


def test_recovery_weighting_preserves_mass_and_only_changes_yaw():
    from optimizer.navigation_loss import navigation_weights
    target = torch.full((1, 10, 4), 1500.)
    target[0, 1, 3] = 1750
    valid = torch.ones(1, 10, dtype=torch.bool)
    behavior = torch.zeros(1, 10, dtype=torch.long)
    old = navigation_weights(target, valid, behavior, 1, yaw_balance=.7, amplitude_bins=True)
    new = navigation_weights(target, valid, behavior, 1, yaw_balance=.7,
                             amplitude_bins=True, recovery_steps=2)
    torch.testing.assert_close(new[..., :2], old[..., :2], rtol=0, atol=0)
    torch.testing.assert_close(new.sum((0, 1)), torch.full((3,), 10.))
    assert new[0, 2, 2] > new[0, 0, 2]
    with pytest.raises(ValueError, match='recovery'):
        navigation_weights(target, valid, behavior, 1, recovery_steps=-1)


def test_probe_separates_carried_state_from_clear_input_and_teacher_release():
    from optimizer.probe_yaw_recovery import recovery_probe
    class SlowPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
        def forward(self, obs, hx=None, dt=None):
            h = torch.zeros(len(obs), 1) if hx is None else hx
            rows = []
            for x in obs.unbind(1):
                grid = x[:, 2:66].reshape(-1, 8, 8)
                drive = grid[:, :, 4:].mean((1, 2))-grid[:, :, :4].mean((1, 2))
                h = .98*h + .02*drive[:, None]
                row = torch.full((len(obs), 4), 1500.)
                row[:, 3] += 400*h[:, 0]
                rows.append(row)
            return torch.stack(rows, 1), h
    report = recovery_probe(SlowPolicy())
    assert report['clear_from_reset_max_abs_yaw_pwm'] == 0
    broad = next(r for r in report['cases'] if r['width'] == 4)
    assert broad['recovery_abs_yaw_at_0_8s_pwm'] > 20
    assert broad['teacher_release_s'] < .8
    assert broad['settle_s'] > .8


def test_recovery_mode_preserves_split_and_records_time_contract(tiny_model_meta):
    from simulation.policy import ChongFlyMSPPolicy
    from optimizer.pretrain import pretrain_policy
    from tests.test_pretrain_contract import dataset
    data = dataset()
    policy = ChongFlyMSPPolicy.from_meta(str(tiny_model_meta))
    pretrain_policy(policy, data, epochs=0, seed=42, loss_mode='channel-v3')
    before = policy._pretrain_info
    pretrain_policy(policy, data, epochs=1, seed=42, loss_mode='channel-v4')
    after = policy._pretrain_info
    assert after['subset_indices'] == before['subset_indices']
    assert after['validation_indices'] == before['validation_indices']
    assert after['loss_contract']['yaw_recovery_window_s'] == pytest.approx(1.)
    assert after['loss_contract']['yaw_amplitude_threshold_pwm'] == 200


def test_continuation_checks_anatomical_mask_not_previous_zero_weights():
    from types import SimpleNamespace
    from optimizer.probe_yaw_recovery import verify_recurrent_contract
    weights = torch.tensor([[0., .2], [-.1, 0.]])
    cell = SimpleNamespace(_effective_W=lambda: weights,
        _synapse_mask=torch.tensor([[False, True], [True, False]]),
        _synapse_sign=torch.tensor([[0., 1.], [-1., 0.]]), preserve_signs=True)
    verify_recurrent_contract(cell)
    weights[0, 0] = .1
    with pytest.raises(ValueError, match='support/signs'):
        verify_recurrent_contract(cell)
    weights[0, 0] = 0
    weights[0, 1] = -.2
    with pytest.raises(ValueError, match='support/signs'):
        verify_recurrent_contract(cell)


@pytest.mark.parametrize('yaw', [1500., 1750.])
def test_absent_recovery_group_does_not_change_existing_weights(yaw):
    from optimizer.navigation_loss import navigation_weights
    from optimizer.search_pipeline import validate_search_config
    target = torch.full((2, 8, 4), 1500.)
    target[..., 3] = yaw
    valid = torch.ones(2, 8, dtype=torch.bool)
    behavior = torch.zeros(2, 8, dtype=torch.long)
    old = navigation_weights(target, valid, behavior, 1, amplitude_bins=True)
    new = navigation_weights(target, valid, behavior, 1, amplitude_bins=True, recovery_steps=5)
    torch.testing.assert_close(old, new, rtol=0, atol=0)
    assert validate_search_config({'loss_mode': 'channel-v4'})['loss_mode'] == 'channel-v4'
