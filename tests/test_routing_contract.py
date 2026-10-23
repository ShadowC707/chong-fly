"""Causal tests: forbidden inputs must not influence states or outputs."""
import numpy as np
import pytest
import torch

from core.models import BiologicalCfCCell, BiologicalCfCNetwork
from simulation.policy import DNProjectionHead, ChongFlyMSPPolicy


def cell():
    torch.manual_seed(7)
    model = BiologicalCfCCell(4, 74, sensor_indices={"lptc_flow": [0], "lc_looming": [1]},
                              motor_indices={"throttle": [0], "roll": [1], "pitch": [2], "yaw": [3]})
    model.set_w_macro(np.zeros((4, 4), dtype=np.float32))
    with torch.no_grad():
        model.W_in.weight.fill_(.2)
        model.W_in.bias.zero_()
        model.b.zero_()
    return model


@pytest.mark.parametrize("channel,recipient", [(0, 0), (1, 0), (2, 1), (65, 1)])
def test_sensor_perturbation_reaches_only_declared_population(channel, recipient):
    model = cell()
    h, u = torch.zeros(1, 4), torch.zeros(1, 74)
    baseline = model(u, h)
    u[0, channel] = .5
    delta = model(u, h) - baseline
    assert delta[0, recipient] > 0
    assert torch.count_nonzero(delta) == 1


def test_no_edges_means_no_cross_neuron_influence_without_disabling_backbone():
    model = cell()
    initial, perturbed = torch.zeros(1, 4), torch.zeros(1, 4)
    perturbed[0, 2] = .5
    for _ in range(10):
        initial = model(torch.zeros(1, 74), initial)
        perturbed = model(torch.zeros(1, 74), perturbed)
    torch.testing.assert_close(initial[:, [0, 1, 3]], perturbed[:, [0, 1, 3]], rtol=0, atol=0)


def test_unmapped_memory_cannot_become_a_hidden_control_path():
    model = cell()
    u, h = torch.zeros(1, 74), torch.zeros(1, 4)
    memory = u.clone(); memory[:, 66:] = 1
    for _ in range(8):
        torch.testing.assert_close(model(u, h), model(memory, h), rtol=0, atol=0)
        h = model(u, h)


def test_memory_route_requires_explicit_receivers():
    model = BiologicalCfCCell(3, 74, sensor_indices={"lptc_flow": [0], "lc_looming": [1], "memory": [2]})
    model.set_w_macro(np.zeros((3, 3), dtype=np.float32))
    with torch.no_grad(): model.W_in.weight.fill_(.1)
    u, h = torch.zeros(1, 74), torch.zeros(1, 3)
    baseline = model(u, h); u[:, 66:] = .5
    delta = model(u, h)-baseline
    assert delta[0, 2] > 0 and torch.count_nonzero(delta) == 1


def test_unreachable_neuron_stays_unaffected_after_optimizer_updates():
    model = cell()
    opt = torch.optim.AdamW(model.parameters(), lr=.01)
    u, h = torch.ones(2, 74), torch.zeros(2, 4)
    for _ in range(4):
        opt.zero_grad()
        model(u, h).sum().backward()
        assert torch.count_nonzero(model.W_in.weight.grad[2:]) == 0
        opt.step(); model.apply_topology_mask()
    changed = u.clone(); changed[:, :66] *= 2
    torch.testing.assert_close(model(u, h)[:, 2:], model(changed, h)[:, 2:], rtol=0, atol=0)


@pytest.mark.parametrize("mapping", [{}, {"lptc_flow": [-1], "lc_looming": [1]},
                                     {"lptc_flow": [4], "lc_looming": [1]},
                                     {"lptc_flow": [0], "lc_loooming": [1]}])
def test_bad_sensor_mapping_fails_closed(mapping):
    with pytest.raises(ValueError, match="(?i)routing|mapping|indices|population"):
        BiologicalCfCCell(4, 74, sensor_indices=mapping)


def test_free_connectivity_requires_explicit_unconstrained_mode():
    with pytest.raises(ValueError, match="(?i)free|unconstrained"):
        BiologicalCfCCell(4, 74, mode="free", sensor_indices={"lptc_flow": [0], "lc_looming": [1]})


@pytest.mark.parametrize("mapping", [{}, {"throttle": [0]},
    {"throttle": [-1], "roll": [1], "pitch": [2], "yaw": [3]},
    {"throttle": [0], "roll": [1], "pitch": [2], "yaw": [4]}])
def test_missing_or_invalid_motor_mapping_never_enables_dense_fallback(mapping):
    with pytest.raises(ValueError, match="(?i)mapping|indices|channel"):
        DNProjectionHead(4, mapping)


def test_motor_masks_block_unauthorized_state_influence():
    head = DNProjectionHead(4, {"throttle": [0], "roll": [1], "pitch": [2], "yaw": [3]})
    with torch.no_grad(): head.proj.weight.fill_(1)
    base = head(torch.zeros(1, 4))
    for i in range(4):
        h = torch.zeros(1, 4); h[0, i] = 1
        torch.testing.assert_close(head(h)-base, h, rtol=0, atol=0)


def test_generic_network_motor_head_obeys_same_mapping():
    network = BiologicalCfCNetwork(cell(), return_sequences=True)
    with torch.no_grad(): network.motor_head.weight.fill_(1)
    h = torch.zeros(1, 4); h[:, 3] = 1
    base, _ = network(torch.zeros(1, 1, 74), hx=torch.zeros(1, 4), dt=0)
    out, _ = network(torch.zeros(1, 1, 74), hx=h, dt=0)
    torch.testing.assert_close((out-base)[0, 0], h[0], rtol=0, atol=0)


def test_policy_does_not_store_unused_dense_motor_head():
    policy = ChongFlyMSPPolicy(BiologicalCfCNetwork(cell()), sensor_dim=74)
    assert not any("motor_head" in name for name, _ in policy.named_parameters())


def test_routing_change_rejects_same_shape_checkpoint():
    original = cell()
    changed = BiologicalCfCCell(4, 74, sensor_indices={"lptc_flow": [2], "lc_looming": [1]})
    with pytest.raises(RuntimeError, match="contract"):
        changed.load_state_dict(original.state_dict(), strict=False)


def test_motor_routing_change_rejects_same_shape_checkpoint():
    original = DNProjectionHead(4, {"throttle": [0], "roll": [1], "pitch": [2], "yaw": [3]})
    changed = DNProjectionHead(4, {"throttle": [1], "roll": [0], "pitch": [2], "yaw": [3]})
    with pytest.raises(RuntimeError, match="contract"):
        changed.load_state_dict(original.state_dict(), strict=False)


def test_legacy_pitch_roll_mapping_reports_lost_output_dimensions():
    with pytest.warns(UserWarning, match="2/4"):
        head = DNProjectionHead(4, {"throttle": [0], "pitch_roll": [0], "yaw": [0, 1]})
    assert head.structural_rank == 2
    assert head._proj_mask[1].tolist() == [True, False, False, False]


def test_dense_baseline_remains_explicitly_available():
    model = BiologicalCfCCell(4, 74, mode="free", connectivity="unconstrained")
    policy = ChongFlyMSPPolicy(BiologicalCfCNetwork(model), allow_dense_motor_fallback=True)
    pwm, _ = policy(torch.ones(1, 74))
    assert pwm.shape == (1, 4) and torch.isfinite(pwm).all()
    assert policy.routing_diagnostics["connectivity"] == "unconstrained"


def test_routed_policy_roundtrip_restores_outputs_without_learning_new_routes(tmp_path):
    original = ChongFlyMSPPolicy(BiologicalCfCNetwork(cell()))
    restored = ChongFlyMSPPolicy(BiologicalCfCNetwork(cell()))
    path = tmp_path / "routed.pt"
    torch.save(original.state_dict(), path)
    restored.load_state_dict(torch.load(path, weights_only=True))
    observations = torch.linspace(-.1, .1, 2*5*74).reshape(2, 5, 74)
    out, h = original(observations)
    copy_out, copy_h = restored(observations)
    torch.testing.assert_close(out, copy_out, rtol=0, atol=0)
    torch.testing.assert_close(h, copy_h, rtol=0, atol=0)


def test_sensor_signal_reaches_motor_side_only_along_directed_path():
    model = BiologicalCfCCell(4, 1, input_routes={0: [0]})
    model.set_w_macro(np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.float32))
    with torch.no_grad():
        model.W_in.weight.fill_(1); model.W_in.bias.zero_(); model.b.zero_()
    first = model(torch.ones(1, 1), torch.zeros(1, 4))
    second = model(torch.ones(1, 1), first)
    third = model(torch.ones(1, 1), second)
    assert first[0, 0] > 0 and torch.count_nonzero(first[:, 1:]) == 0
    assert second[0, 1] > 0 and second[0, 2] == 0
    assert third[0, 2] > 0 and third[0, 3] == 0


def test_reachability_distinguishes_missing_paths_from_full_motor_rank():
    from core.routing import sensor_motor_paths, motor_mask
    W = torch.zeros(6, 6, dtype=torch.bool)
    W[0, 1] = W[1, 2] = W[3, 5] = True
    motors = motor_mask(6, {'throttle': [2], 'roll': [3], 'pitch': [4], 'yaw': [1]})
    paths = sensor_motor_paths(W, {'flow': [0], 'tof': [5], 'direct': [2]}, motors)
    assert paths['flow']['minimum_hops'] == {'throttle': 2, 'roll': None, 'pitch': None, 'yaw': 1}
    assert paths['tof']['reachable_motor_counts'] == dict.fromkeys(['throttle', 'roll', 'pitch', 'yaw'], 0)
    assert paths['flow']['reachable_readout_rank'] == 2
    assert paths['direct']['minimum_hops']['throttle'] == 0


def test_policy_reports_paths_after_removing_an_edge():
    model = cell()
    W = np.zeros((4, 4), dtype=np.float32)
    W[0, 2] = 1
    model.set_w_macro(W)
    policy = ChongFlyMSPPolicy(BiologicalCfCNetwork(model))
    assert policy.routing_diagnostics['sensor_motor_paths']['lptc_flow']['minimum_hops']['pitch'] == 1
    W[0, 2] = 0
    model.set_w_macro(W)
    policy = ChongFlyMSPPolicy(BiologicalCfCNetwork(model))
    assert policy.routing_diagnostics['sensor_motor_paths']['lptc_flow']['minimum_hops']['pitch'] is None


def test_sparse_path_diagnostics_match_dense_and_ignore_stored_zeros():
    import scipy.sparse as sp
    from core.routing import sensor_motor_paths, motor_mask
    sparse = sp.csr_matrix(([1., 0.], ([0, 0], [1, 2])), shape=(4, 4))
    mask = motor_mask(4, {'throttle': [0], 'roll': [1], 'pitch': [2], 'yaw': [3]})
    dense_paths = sensor_motor_paths(torch.tensor(sparse.toarray()) != 0, {'flow': [0]}, mask)
    assert sensor_motor_paths(sparse, {'flow': [0]}, mask) == dense_paths


def test_checkpoint_with_previous_coordinate_contract_is_rejected():
    from configs.flight_config import COORDINATE_VERSION
    original = cell()
    assert original.get_extra_state()['coordinate_version'] == COORDINATE_VERSION
    state = original.state_dict()
    state['_extra_state']['coordinate_version'] = 'previous'
    with pytest.raises(RuntimeError, match='contract'):
        cell().load_state_dict(state, strict=False)
