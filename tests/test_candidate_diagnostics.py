import pytest
import torch


def test_metrics_ignore_padding_and_separate_left_and_right_turns():
    from optimizer.diagnose_candidate import prediction_metrics
    target = torch.full((2, 3, 4), 1500.)
    target[0, :2, 3] = 1700
    target[1, :2, 3] = 1300
    prediction = target.clone()
    prediction[1, :2, 3] = 1700  # a constant right-turn shortcut must fail left turns
    prediction[:, 2] = 2000
    valid = torch.tensor([[True, True, False], [True, True, False]])
    result = prediction_metrics(prediction, target, valid, ['left_obstacle', 'right_obstacle'])
    assert result['frames'] == 4
    assert result['channel_mae_pwm']['yaw'] == 200
    assert result['turn_direction_accuracy']['right'] == 1
    assert result['turn_direction_accuracy']['left'] == 0
    assert result['scenarios']['left_obstacle']['channel_mae_pwm']['yaw'] == 0


def test_response_probe_uses_control_dt_and_records_actual_float32_contrast():
    from optimizer.diagnose_candidate import response_probe
    from configs.flight_config import CONTROL_DT
    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.steps = []
        def forward(self, x, dt=None):
            self.steps.append(dt)
            tof = x[..., 2:66].reshape(*x.shape[:2], 8, 8)
            yaw = 400 * (tof[..., 4:].mean((-1,-2)) - tof[..., :4].mean((-1,-2)))
            pwm = torch.full((*x.shape[:2], 4), 1500., device=x.device, dtype=x.dtype)
            pwm[..., 3] += yaw
            return pwm, None
    policy = Policy()
    result = response_probe(policy)
    assert policy.steps == [CONTROL_DT]
    assert result['dtype'] == 'torch.float32'
    assert result['yaw_left_obstacle_pwm'] > 1500
    assert result['yaw_right_obstacle_pwm'] < 1500
    assert result['yaw_contrast_pwm'] == pytest.approx(720)


def test_research_probe_requires_explicit_opt_in_before_accessing_files(tmp_path):
    from optimizer.diagnose_candidate import run_diagnostic
    with pytest.raises(ValueError, match='research-only'):
        run_diagnostic(tmp_path/'missing', tmp_path/'missing', 'candidate',
                       tmp_path/'missing.pt', tmp_path/'output')
    assert not (tmp_path/'output').exists()


def test_unapproved_artifact_cannot_enter_pretrain_without_research_opt_in(tiny_model_meta):
    import json
    from simulation.policy import ChongFlyMSPPolicy
    from optimizer.pretrain import pretrain_policy
    from tests.test_pretrain_contract import dataset
    meta = json.loads(tiny_model_meta.read_text())
    meta['provenance']['training_ready'] = False
    tiny_model_meta.write_text(json.dumps(meta))
    policy = ChongFlyMSPPolicy.from_meta(str(tiny_model_meta))
    before = {k:p.detach().clone() for k,p in policy.named_parameters()}
    with pytest.raises(ValueError, match='research_only|admission'):
        pretrain_policy(policy, dataset(), epochs=1)
    for k,p in policy.named_parameters():
        torch.testing.assert_close(p, before[k], rtol=0, atol=0)
    pretrain_policy(policy, dataset(), epochs=1, research_only=True, seed=42)
    assert policy._pretrain_info['research_only'] is True


def test_standard_factory_rejects_explicitly_unapproved_artifact(tmp_path):
    from tests.test_role_reduction import graph
    from generator.role_reducer import RolePreservingReducer
    from optimizer.evaluate import create_model
    model = RolePreservingReducer(**graph())(16)
    model.provenance['training_ready'] = False
    model.save(str(tmp_path))
    with pytest.raises(ValueError, match='admission|training_ready'):
        create_model({'reducer':'role_degree', 'k_clusters':16}, base_dir=str(tmp_path))


def test_validator_smoke_reports_real_dt_and_native_precision(tiny_model_meta):
    from generator.validate_flywire_candidates import functional_smoke
    from configs.flight_config import CONTROL_DT
    report = functional_smoke(tiny_model_meta)
    assert report['dt'] == CONTROL_DT
    assert report['float32_response']['dt'] == CONTROL_DT
    assert report['float32_response']['dtype'] == 'torch.float32'
    assert report['behavior_validated'] is False
