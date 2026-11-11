import pytest
import torch

from optimizer.maneuver_state import teacher_labels, forward_hidden, StateReadout, state_loss


def test_pending_state_is_not_current_yaw_and_resets_between_episodes():
    x = torch.ones(2, 60, 74); x[..., :2] = 0
    x[0, :10, 2:66].reshape(10, 8, 8)[:, :, :4] = .5/3.5
    x[0, 10:35, 0] = .16  # yaw gated to neutral, maneuver still unfinished
    valid = torch.ones(2, 60, dtype=torch.bool)
    result = teacher_labels(x, valid)
    assert torch.all(result['state'][0, 10:35] == 2)
    assert torch.all(result['Y'][0, 10:35, 3] == 1500)
    assert torch.all(result['state'][0, 44:] == 0)  # 0.20s quiet settles
    assert torch.all(result['state'][1] == 0)  # no episode-state leakage
    mirrored = x.clone()
    mirrored[..., 2:66] = x[..., 2:66].reshape(2,60,8,8).flip(-1).reshape(2,60,64)
    assert torch.all(teacher_labels(mirrored, valid)['state'][0,10:35] == 1)


def test_padding_is_not_a_teacher_observation_and_holes_are_rejected():
    x = torch.ones(1, 5, 74); x[..., :2] = 0; x[:,3:] = float('nan')
    valid = torch.tensor([[True,True,True,False,False]])
    r = teacher_labels(x, valid)
    assert r['state'].tolist() == [[0,0,0,-100,-100]]
    assert torch.isnan(x[:,3:]).all()
    with pytest.raises(ValueError, match='prefix'):
        teacher_labels(x, torch.tensor([[True,False,True,False,False]]))


def test_hidden_capture_preserves_commands_gradients_and_removes_hook(tiny_model_meta):
    from simulation.policy import ChongFlyMSPPolicy
    p = ChongFlyMSPPolicy.from_meta(str(tiny_model_meta), preserve_signs=True,
        neutral_origin=True, sensor_encoding='proximity-release-v1', dt=.02)
    x = torch.rand(2,7,74)
    expected, last = p(x,dt=.02)
    actual, h = forward_hidden(p,x)
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    torch.testing.assert_close(h[:,-1],last,atol=0,rtol=0)
    h.sum().backward()
    assert p.cfc_network.cell.W_in.weight.grad is not None
    assert not p.dn_head._forward_pre_hooks
    with pytest.raises(Exception): forward_hidden(p,torch.ones(1,2,3))
    assert not p.dn_head._forward_pre_hooks


def test_detached_control_does_not_train_hidden_state_and_padding_has_no_gradient():
    h = torch.randn(2,4,5, requires_grad=True)
    readout = StateReadout(torch.ones(5))
    labels = torch.tensor([[0,1,2,-100],[0,1,2,-100]])
    valid = labels != -100
    loss = state_loss(readout(h.detach()), labels, valid, normalizer=6., class_weights=torch.ones(3))
    loss.backward()
    assert h.grad is None and readout.linear.weight.grad is not None
    readout.zero_grad()
    state_loss(readout(h), labels, valid, normalizer=6., class_weights=torch.ones(3)).backward()
    assert torch.any(h.grad[:,:3] != 0) and torch.all(h.grad[:,3] == 0)
    with torch.no_grad(): readout.linear.weight.mul_(1000)
    readout.project()
    assert readout.linear.weight.norm() <= 4.


def test_class_loss_is_partition_invariant_and_validates_labels():
    logits = torch.randn(2,4,3)
    labels = torch.tensor([[0,1,2,-100],[2,0,1,-100]])
    valid = labels != -100
    kwargs = dict(normalizer=6.,class_weights=torch.tensor([1.,2.,3.]))
    full = state_loss(logits, labels, valid, **kwargs)
    parts = sum(state_loss(logits[i:i+1], labels[i:i+1], valid[i:i+1], **kwargs) for i in range(2))
    torch.testing.assert_close(full, parts)
    with pytest.raises(ValueError): state_loss(logits, labels, valid, normalizer=0.,class_weights=torch.ones(3))
