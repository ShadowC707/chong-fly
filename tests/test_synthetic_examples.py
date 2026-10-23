"""Tiny examples are test data, never flight candidates or biological evidence."""
import numpy as np
import pytest
import torch


@pytest.mark.parametrize('sparse', [False, True])
def test_tiny_example_roundtrip_and_directed_routes(tmp_path, sparse):
    from tests.synthetic_graphs import write_tiny_model
    from generator.graph_reducer import ReducedModel
    from simulation.policy import ChongFlyMSPPolicy
    meta = write_tiny_model(tmp_path, sparse=sparse)
    model = ReducedModel.load(str(meta), allow_legacy=False)
    weights = model.W.toarray() if sparse else model.W
    assert model.k == 8
    assert model.provenance['source_kind'] == 'synthetic'
    assert model.provenance['purpose'] == 'test_fixture'
    np.testing.assert_array_equal(model.cluster_map, np.arange(8))
    assert weights[1, 3] > 0 and weights[3, 1] == 0
    assert sum(p.stat().st_size for p in tmp_path.iterdir()) < 16_384
    policy = ChongFlyMSPPolicy.from_meta(str(meta))
    paths = policy.routing_diagnostics['sensor_motor_paths']
    assert paths['lc_looming']['minimum_hops']['yaw'] == 2
    assert paths['lptc_flow']['minimum_hops']['pitch'] == 2


def test_missing_yaw_route_has_a_working_positive_control(tmp_path):
    from tests.synthetic_graphs import write_tiny_model
    from simulation.policy import ChongFlyMSPPolicy
    x = torch.ones(2, 12, 74)
    x[..., :2] = 0
    x[1, :, 2:66] = .1
    for connected in (False, True):
        meta = write_tiny_model(tmp_path / str(connected), looming_to_yaw=connected)
        policy = ChongFlyMSPPolicy.from_meta(str(meta))
        with torch.no_grad():
            cell = policy.cfc_network.cell
            cell.W_in.weight.fill_(.1)
            cell.W_in.bias.zero_()
            policy.dn_head.proj.weight.fill_(.5)
            policy.dn_head.proj.bias.zero_()
            policy.post_step()
            outputs, _ = policy(x)
        same_yaw = torch.equal(outputs[0, :, 3], outputs[1, :, 3])
        assert same_yaw == (not connected)
