import pytest
import torch

from optimizer.tau_continuation import TauConstraints, make_optimizer


def policy_from(meta):
    from simulation.policy import ChongFlyMSPPolicy
    return ChongFlyMSPPolicy.from_meta(str(meta), preserve_signs=True,
        neutral_origin=True, sensor_encoding='proximity-release-v1', dt=.02)


def test_tau_training_keeps_graph_and_amplitude_exact(tiny_model_meta):
    p = policy_from(tiny_model_meta)
    c = p.cfc_network.cell
    old_w, old_a, old_tau = (v.detach().clone() for v in (c._w_macro_param, c.A, c.tau_raw))
    guard = TauConstraints(p)
    opt = make_optimizer(p, train_tau=True)
    opt.zero_grad()
    (c.tau.sum() + p.dn_head.proj.bias.square().sum()).backward()
    opt.step(); p.post_step(); guard(p)
    assert torch.equal(c._w_macro_param, old_w) and torch.equal(c.A, old_a)
    assert not torch.equal(c.tau_raw, old_tau)
    assert c._w_macro_param.grad is None and c.A.grad is None
    assert guard.audit()['W_unchanged'] and guard.steps == 1
    params = [v for group in opt.param_groups for v in group['params']]
    assert len(set(map(id, params))) == len(params)
    assert set(map(id, params)) == {id(v) for v in p.parameters() if v.requires_grad}


def test_tau_bounds_and_corruption_rejection(tiny_model_meta):
    p = policy_from(tiny_model_meta); c = p.cfc_network.cell
    guard = TauConstraints(p)
    with torch.no_grad(): c.tau_raw[0] = -100; c.tau_raw[1] = 100
    p.post_step(); guard(p)
    assert c.tau.min().item() == pytest.approx(.005, abs=1e-7)
    assert c.tau.max().item() == pytest.approx(2., abs=1e-6)
    assert guard.audit()['clipped_elements'] == 2
    with torch.no_grad(): c._w_macro_param.add_(.001)
    with pytest.raises(ValueError, match='Frozen W'): guard(p)


def test_nonfinite_tau_is_not_silently_projected(tiny_model_meta):
    p = policy_from(tiny_model_meta); guard = TauConstraints(p)
    with torch.no_grad(): p.cfc_network.cell.tau_raw[0] = float('nan')
    with pytest.raises(ValueError, match='Nonfinite tau'): guard(p)


def test_control_optimizer_matches_existing_adam(tiny_model_meta):
    from optimizer.mvp_constraints import RecurrentConstraints
    p = policy_from(tiny_model_meta); RecurrentConstraints(p, freeze_core=True)
    opt = make_optimizer(p, train_tau=False)
    assert len(opt.param_groups) == 1 and opt.param_groups[0]['lr'] == .001
    assert set(map(id, opt.param_groups[0]['params'])) == {id(v) for v in p.parameters() if v.requires_grad}
    with pytest.raises(ValueError, match='tau trainability'): make_optimizer(p, train_tau=True)
