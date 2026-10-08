import numpy as np
import pytest
import torch

from core.models import BiologicalCfCCell
from simulation.policy import ChongFlyMSPPolicy


def cell(preserve=True):
    result = BiologicalCfCCell(2, 1, mode='masked', input_routes={0:[0]}, preserve_signs=preserve)
    result.set_w_macro(np.array([[0., .1], [-.1, 0.]], dtype=np.float32))
    return result


def test_optimizer_cannot_reverse_initial_signs_and_zero_edges_can_regrow():
    model = cell()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.)
    w = model._effective_W()
    (w[0,1]-w[1,0]).backward()  # deliberately overshoot both signs
    optimizer.step()
    assert model._effective_W()[0,1] >= 0
    assert model._effective_W()[1,0] <= 0
    model.apply_topology_mask()
    assert torch.count_nonzero(model._w_macro_param) == 0
    optimizer.zero_grad()
    w = model._effective_W()
    (-w[0,1]+w[1,0]).backward()
    optimizer.step()
    model.apply_topology_mask()
    assert model._effective_W()[0,1] > 0
    assert model._effective_W()[1,0] < 0
    assert torch.count_nonzero(model._effective_W().diag()) == 0


def test_sign_policy_is_an_explicit_checkpoint_contract():
    signed = cell(True)
    unconstrained_signs = cell(False)
    with pytest.raises(RuntimeError, match='contract'):
        unconstrained_signs.load_state_dict(signed.state_dict())
    restored = cell(True)
    restored.load_state_dict(signed.state_dict())
    torch.testing.assert_close(restored._effective_W(), signed._effective_W())


def test_policy_factory_preserves_declared_signs(tiny_model_meta):
    policy = ChongFlyMSPPolicy.from_meta(str(tiny_model_meta), preserve_signs=True)
    model = policy.cfc_network.cell
    initial = model._effective_W().detach().clone()
    with torch.no_grad():
        model._w_macro_param.copy_(-initial)
    policy.post_step()
    assert torch.all(initial*model._effective_W() >= 0)


def test_free_connectivity_cannot_claim_preserved_signs():
    with pytest.raises(ValueError, match='(?i)sign'):
        BiologicalCfCCell(2, 1, mode='free', connectivity='unconstrained', preserve_signs=True)


def test_signed_checkpoint_cannot_change_the_initial_sign_reference():
    source, other = cell(True), cell(True)
    other.set_w_macro(np.array([[0., -.1], [.1, 0.]], dtype=np.float32))
    with pytest.raises(RuntimeError, match='contract'):
        other.load_state_dict(source.state_dict())
