"""Policy contracts run on tiny temporary graphs, independent of local models."""
import numpy as np
import pytest
import torch

from simulation.policy import (
    ChongFlyMSPPolicy, SensorInputLayer, DNProjectionHead, PWMOutputLayer,
    SENSOR_DIM, N_CONTROLS,
)


@pytest.fixture
def policy(tiny_model_meta):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        return ChongFlyMSPPolicy.from_meta(str(tiny_model_meta), mode='masked')


def assert_pwm_bounds(pwm):
    assert torch.isfinite(pwm).all()
    assert (pwm >= 1000).all() and (pwm <= 2000).all()


def test_pwm_ranges_and_decode():
    layer = PWMOutputLayer()
    for logit, expected in ((-10., 1000.), (0., 1500.), (10., 2000.)):
        pwm = layer(torch.full((1, 4), logit))
        torch.testing.assert_close(pwm, torch.full((1, 4), expected), atol=1., rtol=0)
    decoded = layer.decode_np(layer(torch.zeros(1, 4))[0].numpy())
    assert decoded['throttle_pct'] == pytest.approx(50.)
    assert decoded['roll_norm'] == pytest.approx(0.)


def test_sensor_packing_and_learnable_scale():
    layer = SensorInputLayer(SENSOR_DIM, learnable_scale=True)
    flow = np.array([.3, -.1], dtype=np.float32)
    tof = np.linspace(0, 1, 64, dtype=np.float32)
    packed = layer.from_numpy(flow, tof)
    assert packed.shape == (1, 66)
    np.testing.assert_array_equal(packed[0].numpy(), np.concatenate([flow, tof]))
    obs = torch.ones(4, SENSOR_DIM)
    with torch.no_grad():
        layer.gain.fill_(2.)
    torch.testing.assert_close(layer(obs), obs * 2 + layer.bias)


def test_motor_head_masks_and_explicit_dense_fallback():
    mapping = {'throttle': [4], 'roll': [5], 'pitch': [6], 'yaw': [7]}
    head = DNProjectionHead(8, mapping)
    logits = head(torch.ones(4, 8))
    assert logits.shape == (4, N_CONTROLS) and torch.isfinite(logits).all()
    assert head._proj_mask.shape == (4, 8)
    assert head._proj_mask.sum().item() == 4
    full = DNProjectionHead(8, {}, allow_dense_fallback=True)
    assert full(torch.ones(4, 8)).shape == (4, N_CONTROLS)


def test_policy_single_step_and_sequence(policy):
    pwm, hx = policy(torch.ones(1, SENSOR_DIM))
    assert pwm.shape == (1, N_CONTROLS)
    assert hx.shape == (1, 8)
    assert_pwm_bounds(pwm)
    sequence, hx = policy(torch.ones(4, 15, SENSOR_DIM))
    assert sequence.shape == (4, 15, N_CONTROLS)
    assert hx.shape == (4, 8)
    assert_pwm_bounds(sequence)


def test_numpy_interface_state_carry_and_reset(policy):
    policy.reset_state()
    flow = np.array([.05, -.02], dtype=np.float32)
    tof = np.full(64, .5, dtype=np.float32)
    first = policy.step_np(flow, tof)
    state = policy._hx.clone()
    policy.step_np(flow, tof)
    assert first.shape == (4,) and first.dtype == np.float32
    assert np.isfinite(first).all() and (first >= 1000).all() and (first <= 2000).all()
    assert torch.count_nonzero(state) > 0
    assert not torch.allclose(state, policy._hx)
    policy.reset_state()
    assert policy._hx is None
    np.testing.assert_array_equal(policy.step_np(flow, tof), first)


def test_post_optimizer_step_restores_forbidden_edges(policy):
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    pwm, _ = policy(torch.ones(2, 8, SENSOR_DIM))
    pwm.sum().backward()
    optimizer.step()
    cell = policy.cfc_network.cell
    # Even accidental parameter writes must not open forbidden routes.
    with torch.no_grad():
        cell._w_macro_param[~cell._synapse_mask] = 1
        cell.W_in.weight[~cell.W_in.route_mask] = 1
        policy.dn_head.proj.weight[~policy.dn_head._proj_mask] = 1
    policy.post_step()
    assert torch.count_nonzero(cell._w_macro_param[~cell._synapse_mask]) == 0
    assert torch.count_nonzero(cell.W_in.weight[~cell.W_in.route_mask]) == 0
    assert torch.count_nonzero(policy.dn_head.proj.weight[~policy.dn_head._proj_mask]) == 0


def test_gradient_flow_through_active_policy_parameters(policy):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(77)
        obs = torch.randn(4, 8, SENSOR_DIM)
        state = torch.randn(4, 8)
    pwm, _ = policy(obs, hx=state)
    pwm.sum().backward()
    for name, parameter in policy.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
            assert parameter.grad.abs().sum() > 0, name
