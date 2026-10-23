"""Analytical oracles and directed micrographs, independent of training luck."""
import math

import numpy as np
import pytest
import torch

from core.models import BiologicalCfCCell


def isolated_cell(weights, solver="CfC", tau=.1, mode="fixed"):
    cell = BiologicalCfCCell(len(weights), 1, mode=mode, solver_type=solver,
                             tau_init=tau, backbone_units=4, backbone_layers=1,
                             input_routes={0: list(range(len(weights)))},
                             connectivity="unconstrained" if mode == "free" else "structured").double()
    cell.set_w_macro(np.array(weights, dtype=np.float64))
    # Isolate recurrence from the explicit input adapter; free mode is the dense baseline.
    with torch.no_grad():
        for p in cell.backbone.parameters(): p.zero_()
        cell.W_in.weight.zero_()
        cell.W_in.bias.zero_()
        cell.b.zero_()
    return cell


@pytest.mark.parametrize("solver", ["CfC", "Euler_dt_0.02"])
@pytest.mark.parametrize("mode", ["fixed", "masked", "free"])
def test_one_way_edge_carries_activity_from_source_to_target(solver, mode):
    cell = isolated_cell([[0, 2], [0, 0]], solver, mode=mode)
    u = torch.zeros(1, 1, dtype=torch.float64)
    forward = cell(u, torch.tensor([[1., 0.]], dtype=torch.float64), dt=.01)
    reverse = cell(u, torch.tensor([[0., 1.]], dtype=torch.float64), dt=.01)
    assert forward[0, 1] > 0
    assert reverse[0, 0] == 0


def test_chain_propagates_one_edge_per_explicit_step():
    cell = isolated_cell([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    u = torch.zeros(1, 1, dtype=torch.float64)
    first = cell(u, torch.tensor([[1., 0., 0.]], dtype=torch.float64), dt=.01)
    second = cell(u, first, dt=.01)
    assert first[0, 1] > 0 and first[0, 2] == 0
    assert second[0, 2] > 0


def test_inhibitory_edge_retains_sign():
    cell = isolated_cell([[0, -2], [0, 0]])
    out = cell(torch.zeros(1, 1, dtype=torch.float64), torch.tensor([[1., 0.]], dtype=torch.float64), dt=.01)
    assert out[0, 1] < 0


@pytest.mark.parametrize("solver", ["CfC", "Euler_dt_0.02"])
def test_zero_elapsed_time_is_identity_including_state_gradient(solver):
    cell = isolated_cell([[0, 1], [0, 0]], solver)
    h = torch.tensor([[.2, -.3]], dtype=torch.float64, requires_grad=True)
    out = cell(torch.ones(1, 1, dtype=torch.float64), h, dt=0.)
    torch.testing.assert_close(out, h, rtol=0, atol=0)
    out.sum().backward()
    torch.testing.assert_close(h.grad, torch.ones_like(h), rtol=0, atol=0)


@pytest.mark.parametrize("solver", ["CfC", "Euler_dt_0.02"])
def test_solvers_implement_same_leaky_rate_equation(solver):
    cell = isolated_cell([[0]], solver, tau=.2)
    with torch.no_grad(): cell.b.fill_(.4)
    h = torch.tensor([[.7]], dtype=torch.float64)
    actual = cell(torch.zeros(1, 1, dtype=torch.float64), h, dt=.01)
    target = math.tanh(.4)
    expected = (.7 + .01/.2*(target-.7) if solver.startswith("Euler")
                else target + (.7-target)*math.exp(-.01/.2))
    assert actual.item() == pytest.approx(expected, abs=1e-8)


def test_exponential_solver_is_exact_for_constant_drive_across_substeps():
    cell = isolated_cell([[0]], tau=.1)
    with torch.no_grad(): cell.b.fill_(.4)
    u, initial = torch.zeros(1, 1, dtype=torch.float64), torch.tensor([[.8]], dtype=torch.float64)
    whole = cell(u, initial, dt=.2)
    split = initial
    for _ in range(20): split = cell(u, split, dt=.01)
    torch.testing.assert_close(whole, split, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("solver", ["CfC", "Euler_dt_0.02"])
def test_nonlinear_recurrent_solution_converges_as_timestep_shrinks(solver):
    cell = isolated_cell([[.3]], solver, tau=.1)
    initial, u = torch.tensor([[.6]], dtype=torch.float64), torch.zeros(1, 1, dtype=torch.float64)
    def integrate(n):
        h = initial
        for _ in range(n): h = cell(u, h, dt=.1/n)
        return h.detach().item()
    # High-resolution RK4 reference to dh/dt=(tanh(.3h)-h)/.1.
    ref, step = .6, .0001
    def rhs(h): return (math.tanh(.3*h)-h)/.1
    for _ in range(1000):
        a = rhs(ref); b = rhs(ref+step*a/2); c = rhs(ref+step*b/2); d = rhs(ref+step*c)
        ref += step*(a+2*b+2*c+d)/6
    assert abs(integrate(20)-ref) < abs(integrate(10)-ref) * .65
    assert abs(integrate(100)-ref) < .001


@pytest.mark.parametrize("dt", [-.01, float("nan"), float("inf")])
def test_invalid_time_is_rejected(dt):
    cell = isolated_cell([[0]])
    with pytest.raises(ValueError, match="dt"):
        cell(torch.zeros(1, 1, dtype=torch.float64), torch.zeros(1, 1, dtype=torch.float64), dt=dt)


def test_unknown_solver_is_not_silently_treated_as_cfc():
    with pytest.raises(ValueError, match="solver"):
        BiologicalCfCCell(2, 1, solver_type="typo")


def test_pruning_does_not_create_trainable_edges_at_existing_zeros():
    from generator.graph_reducer import ReducedModel
    model = ReducedModel(np.array([[0., 2.], [0., 0.]], dtype=np.float32), 2, "test",
                         np.arange(2), {}, {})
    cell = BiologicalCfCCell.from_reduced_model(model, input_size=1, pruning_sparsity=.5, input_routes={})
    assert cell._synapse_mask.sum().item() == 1


def test_checkpoint_without_dynamics_contract_is_rejected_even_non_strict():
    cell = isolated_cell([[0, 1], [0, 0]])
    old = {k: v for k, v in cell.state_dict().items() if not k.endswith("_extra_state")}
    with pytest.raises(RuntimeError, match="(?i)contract|version|legacy"):
        cell.load_state_dict(old, strict=False)


def test_checkpoint_restores_masked_topology_and_identical_outputs():
    original = isolated_cell([[0, -2], [0, 0]], mode="masked")
    restored = isolated_cell([[1, 1], [1, 1]], mode="masked")
    restored.load_state_dict(original.state_dict())
    u = torch.zeros(1, 1, dtype=torch.float64)
    h = torch.tensor([[1., .2]], dtype=torch.float64)
    torch.testing.assert_close(original(u, h), restored(u, h), rtol=0, atol=0)
    restored(u, h).sum().backward()
    assert torch.count_nonzero(restored._w_macro_param.grad[~restored._synapse_mask]) == 0


def test_exponential_step_stays_in_state_target_envelope_for_large_dt():
    cell = isolated_cell([[100, -100], [-100, 100]])
    h = torch.tensor([[.3, -.4]], dtype=torch.float64)
    for _ in range(10):
        h = cell(torch.zeros(1, 1, dtype=torch.float64), h, dt=10.)
        assert torch.isfinite(h).all() and torch.all(h.abs() <= 1.)


def test_state_and_input_gradients_match_numerical_derivatives():
    cell = isolated_cell([[.2, -.1], [.3, .1]])
    with torch.no_grad(): cell.W_in.weight.fill_(.4)
    u = torch.tensor([[.1]], dtype=torch.float64, requires_grad=True)
    h = torch.tensor([[.2, .3]], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda u, h: cell(u, h, dt=.01), (u, h))


def test_versioned_checkpoint_roundtrip_through_weights_only_loader(tmp_path):
    cell = isolated_cell([[0, 1], [0, 0]], mode="masked")
    path = tmp_path / "weights.pt"
    torch.save(cell.state_dict(), path)
    other = isolated_cell([[0, 1], [0, 0]], mode="masked")
    other.load_state_dict(torch.load(path, weights_only=True))
    assert other.get_extra_state() == cell.get_extra_state()


def test_solver_alias_does_not_override_explicit_clock():
    cell = BiologicalCfCCell(2, 1, solver_type="Euler_dt_0.02", dt=.004, input_routes={})
    assert cell.default_dt == .004


@pytest.mark.parametrize("tau", [0., -1., float("nan"), float("inf")])
def test_invalid_tau_is_rejected(tau):
    with pytest.raises(ValueError, match="tau"):
        BiologicalCfCCell(2, 1, tau_init=tau)


def test_same_shape_checkpoint_with_different_activation_is_rejected():
    original = BiologicalCfCCell(2, 1, backbone_act="tanh", connectivity="unconstrained")
    changed = BiologicalCfCCell(2, 1, backbone_act="relu", connectivity="unconstrained")
    with pytest.raises(RuntimeError, match="contract"):
        changed.load_state_dict(original.state_dict())
