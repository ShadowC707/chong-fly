import numpy as np
import pytest
import torch


def test_narrow_and_broad_encoding_matches_declared_formula():
    from simulation.policy import SensorInputLayer
    from optimizer.functional_probe import make_stimuli, WARMUP
    _, obs = make_stimuli()
    layer = SensorInputLayer(learnable_scale=False, encoding='proximity-mean-v1')
    encoded = layer(obs[:, WARMUP])
    assert encoded[1, 2:66].max().item() == pytest.approx(2.25)
    assert encoded[3, 2:66].max().item() == pytest.approx(.5625)
    assert encoded[1, 2:66].sum().item() == pytest.approx(18.)
    assert encoded[3, 2:66].sum().item() == pytest.approx(18.)
    left = encoded[1, 2:66].reshape(8, 8)
    right = encoded[2, 2:66].reshape(8, 8)
    torch.testing.assert_close(left.flip(-1), right)
    assert torch.count_nonzero(encoded[0]) == 0


def test_audit_measures_actual_forward_without_modifying_policy(tiny_model_meta):
    from simulation.policy import ChongFlyMSPPolicy
    from scripts.audit_sensor_encoding import inspect_pipeline
    from optimizer.functional_probe import make_stimuli
    torch.manual_seed(2)
    policy = ChongFlyMSPPolicy.from_meta(str(tiny_model_meta), sensor_encoding='proximity-mean-v1').eval()
    _, obs = make_stimuli()
    before = {k:v.detach().clone() for k, v in policy.named_parameters()}
    report, arrays = inspect_pipeline(policy, obs)
    assert report['captured_pipeline_matches_forward']
    assert report['direct_tof_leak_outside_looming_max_abs'] == 0
    assert arrays['projection'].shape == (9, 175, 8)
    for k, value in policy.named_parameters():
        torch.testing.assert_close(value, before[k], rtol=0, atol=0)
    assert not policy.sensor_layer._forward_hooks
    assert not policy.cfc_network.cell.W_in._forward_hooks


def test_boundary_probe_records_support_discontinuity():
    from simulation.policy import SensorInputLayer
    from scripts.audit_sensor_encoding import boundary_inputs
    obs = boundary_inputs()
    layer = SensorInputLayer(learnable_scale=False, encoding='proximity-mean-v1')
    result = layer(obs)
    assert (result[0, 2:66] > 0).sum() == 8
    assert (result[1, 2:66] > 0).sum() == 9
    # An almost inactive extra ray rescales every previously active ray by 8/9.
    assert (result[1, 2]/result[0, 2]).item() == pytest.approx(8/9)


def test_projection_summary_uses_clear_subtraction(tiny_model_meta):
    from simulation.policy import ChongFlyMSPPolicy
    from scripts.audit_sensor_encoding import inspect_pipeline
    from optimizer.functional_probe import make_stimuli
    policy = ChongFlyMSPPolicy.from_meta(str(tiny_model_meta), sensor_encoding='proximity-mean-v1').eval()
    with torch.no_grad():
        policy.sensor_layer.bias.fill_(.4)
    _, obs = make_stimuli()
    _, arrays = inspect_pipeline(policy, obs)
    assert np.array_equal(arrays['projection_delta'][0], np.zeros_like(arrays['projection_delta'][0]))
    assert np.count_nonzero(arrays['projection'][0]) > 0
