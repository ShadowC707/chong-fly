import pytest
import torch


def test_mean_proximity_keeps_distance_mass_across_obstacle_coverage():
    from simulation.policy import SensorInputLayer
    layer = SensorInputLayer(74, learnable_scale=False, encoding='proximity-mean-v1')
    raw = torch.ones(3, 74); raw[:, :2] = .2
    for i, columns in enumerate((1, 2, 4)):
        raw[i, 2:66].reshape(8,8)[:, :columns] = .7/3.5
    encoded = layer(raw)
    torch.testing.assert_close(encoded[:, 2:66].sum(-1), torch.full((3,), 4.))
    torch.testing.assert_close(encoded[:, :2], raw[:, :2])
    assert encoded[:, 66:].count_nonzero() == 0
    assert (encoded[raw == 1] == 0).all()


def test_mean_proximity_has_finite_clear_and_single_pixel_response():
    from simulation.policy import SensorInputLayer
    layer = SensorInputLayer(74, learnable_scale=False, encoding='proximity-mean-v1')
    raw = torch.ones(2,74); raw[:,:2] = 0; raw[1,2] = 0
    encoded = layer(raw)
    assert torch.isfinite(encoded).all() and encoded[0].count_nonzero() == 0
    assert encoded[1,2] == 32 and encoded[1,3:].count_nonzero() == 0


def test_mean_encoding_keeps_neutral_origin_and_cannot_load_other_encoding(tiny_model_meta):
    from simulation.policy import ChongFlyMSPPolicy
    new = ChongFlyMSPPolicy.from_meta(str(tiny_model_meta), sensor_encoding='proximity-mean-v1', neutral_origin=True)
    old = ChongFlyMSPPolicy.from_meta(str(tiny_model_meta), sensor_encoding='proximity-v1', neutral_origin=True)
    with pytest.raises(RuntimeError, match='contract'):
        new.load_state_dict(old.state_dict(), strict=False)
    raw = torch.ones(1,10,74); raw[:,:,:2] = 0
    assert (new(raw)[0][...,3] == 1500).all()
    assert new.cfc_network.cell.get_extra_state()['architecture']['sensor_encoding_parameters']['reference_active_pixels'] == 32


def test_temporal_probe_covers_mirrored_widths_without_changing_default(tiny_model_meta):
    from simulation.policy import ChongFlyMSPPolicy
    from simulation.policy_diagnostics import temporal_response_probe
    model = ChongFlyMSPPolicy.from_meta(str(tiny_model_meta))
    default = temporal_response_probe(model)
    expanded = temporal_response_probe(model, coverage_columns=(1,2,4))
    assert len(default['cases']) == 4 and len(expanded['cases']) == 12
    assert {r['active_columns'] for r in expanded['cases']} == {1,2,4}
    assert all(r['threat_target_yaw_pwm'] == 1620 for r in expanded['cases']
               if r['normalized_distance'] == .2 and r['obstacle_side'] == 'left')
    for original, repeated in zip(default['cases'], [r for r in expanded['cases'] if r['active_columns'] == 4]):
        assert original == repeated


@pytest.mark.parametrize('columns', [(0,), (5,), (True,), (1,1)])
def test_temporal_probe_rejects_ambiguous_coverage(columns, tiny_model_meta):
    from simulation.policy import ChongFlyMSPPolicy
    from simulation.policy_diagnostics import temporal_response_probe
    with pytest.raises(ValueError, match='coverage'):
        temporal_response_probe(ChongFlyMSPPolicy.from_meta(str(tiny_model_meta)), coverage_columns=columns)
