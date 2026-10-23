import copy

import pytest
import torch

from optimizer.pretrain import pretrain_policy
from tests.test_pretrain_contract import TimedPolicy, dataset


def test_training_callback_runs_after_native_projection_on_every_step():
    class ProjectedPolicy(TimedPolicy):
        def __init__(self):
            super().__init__()
            self.projected = 0

        def post_step(self):
            self.projected += 1
            with torch.no_grad():
                self.bias[0] = 0

    policy = ProjectedPolicy()
    calls = []

    def after_step(model):
        assert model is policy
        assert model.projected == len(calls) + 1
        assert model.bias[0].item() == 0
        calls.append(model.projected)

    pretrain_policy(policy, dataset(), epochs=2, subset_ratio=1., batch_size=2,
                    validation_fraction=0, seed=42, post_step_callback=after_step)
    assert len(calls) == 4


def test_constraints_freeze_only_requested_parameters_and_bound_actual_updates(tiny_model_meta):
    from simulation.policy import ChongFlyMSPPolicy
    from optimizer.mvp_constraints import RecurrentConstraints
    from optimizer.probe_recurrent_dynamics import scaled_policy
    policy = scaled_policy(ChongFlyMSPPolicy.from_meta(str(tiny_model_meta),
                           preserve_signs=True, neutral_origin=True, sensor_encoding='proximity-v1', dt=.02),
                           tau_scale=.25, recurrent_scale=.75)
    cell = policy.cfc_network.cell
    tau, amplitude = cell.tau_raw.detach().clone(), cell.A.detach().clone()
    old_trainable = {name for name, value in policy.named_parameters() if value.requires_grad}
    constraint = RecurrentConstraints(policy)
    assert {name for name, value in policy.named_parameters() if value.requires_grad} == (
        old_trainable - {'cfc_network.cell.tau_raw', 'cfc_network.cell.A'})
    original = cell._effective_W().detach().clone()
    with torch.no_grad():
        cell._w_macro_param.mul_(3.)
        cell._w_macro_param[~cell._synapse_mask.bool()] = 10
        negative = (cell._synapse_sign < 0).nonzero()[0]
        cell._w_macro_param[tuple(negative)] = 10
    policy.post_step()
    constraint(policy)
    assert cell._effective_W().norm() <= constraint.w_max_norm
    assert torch.all(cell._effective_W()[~cell._synapse_mask.bool()] == 0)
    assert torch.all(cell._effective_W() * cell._synapse_sign >= 0)
    assert constraint.projections == 1
    training_data = dataset()
    training_data['X'][..., 2:34] = .1
    training_data['X'][..., :2] = .2
    pretrain_policy(policy, training_data, epochs=2, subset_ratio=1., batch_size=2,
                    validation_fraction=0, seed=42, research_only=True,
                    loss_mode='channel-v4', post_step_callback=constraint)
    assert torch.equal(cell.tau_raw, tau) and torch.equal(cell.A, amplitude)
    assert cell._effective_W().norm() <= constraint.w_max_norm
    assert not torch.equal(original, cell._effective_W())
    assert constraint.steps == 5
    restored = copy.deepcopy(policy)
    restored.cfc_network.cell.tau_raw.requires_grad_(True)
    restored.cfc_network.cell.A.requires_grad_(True)
    restored_constraint = RecurrentConstraints(restored, metadata=constraint.metadata())
    assert restored_constraint.w_max_norm == constraint.w_max_norm
    assert not restored.cfc_network.cell.tau_raw.requires_grad
    assert not restored.cfc_network.cell.A.requires_grad


@pytest.mark.parametrize('value', [0, -1, float('nan'), float('inf')])
def test_constraints_reject_invalid_saved_bound(tiny_model_meta, value):
    from simulation.policy import ChongFlyMSPPolicy
    from optimizer.mvp_constraints import RecurrentConstraints
    policy = ChongFlyMSPPolicy.from_meta(str(tiny_model_meta), preserve_signs=True)
    metadata = RecurrentConstraints(policy).metadata()
    metadata['w_max_norm'] = value
    with pytest.raises(ValueError):
        RecurrentConstraints(policy, metadata=metadata)


def valid_metrics():
    from configs.flight_config import CONTROL_DT
    from generator.reflex_contract import TEACHER_VERSION
    from optimizer.rollout import BENCHMARK_VERSION
    from simulation.benchmark_scenes import SCENES
    from simulation.control_contract import control_contract
    cases = []
    for side in ('left', 'right'):
        for distance in (.1, .2):
            for width in (1, 2, 4):
                sign = 1 if side == 'left' else -1
                cases.append(dict(obstacle_side=side, normalized_distance=distance,
                    active_columns=width, direction_correct=True, onset_s=.1,
                    recovery_settle_s=.8, clear_abs_yaw_pwm=0.,
                    threat_yaw_pwm=1500+sign*(350 if width == 4 else 80),
                    threat_target_yaw_pwm=1500+sign*350))
    return dict(spatial_response=dict(dt=CONTROL_DT, teacher_version=TEACHER_VERSION,
        tolerance_pwm=20., recovery_s=2., cases=cases), scenes=[dict(scenario=name,
        seed=20261009, benchmark_version=BENCHMARK_VERSION, control_contract=control_contract(),
        feasible=True, crashed=False, fatal_failure=False, horizon_completed=True,
        terminated=False, control_steps=500) for name in SCENES],
        functional_admission=dict(passed=False, reasons=['narrow_amplitude']))


def test_mvp_admission_accepts_exact_narrow_floor_without_old_gate_pass():
    from optimizer.finalize_mvp_candidate import assess_mvp
    result = assess_mvp(valid_metrics(), scene_seed=20261009, scene_steps=500)
    assert result['passed']
    assert result['narrow_amplitude_passed'] == 8
    assert result['broad_joint_passed'] == 4


@pytest.mark.parametrize('fault', ['weak', 'wrong', 'slow', 'nonfinite', 'broad_amplitude',
    'broad_onset', 'duplicate_case', 'duplicate_scene', 'early_stop', 'crash', 'missing_scene',
    'wrong_seed', 'wrong_tolerance'])
def test_mvp_admission_fails_closed(fault):
    from optimizer.finalize_mvp_candidate import assess_mvp
    metrics = valid_metrics()
    case = metrics['spatial_response']['cases'][0]
    broad = metrics['spatial_response']['cases'][2]
    if fault == 'weak': case['threat_yaw_pwm'] = 1579.99
    if fault == 'wrong': case['threat_yaw_pwm'] = 1420  # even if a stale flag says True
    if fault == 'slow': case['recovery_settle_s'] = .82
    if fault == 'nonfinite': case['threat_yaw_pwm'] = float('nan')
    if fault == 'broad_amplitude': broad['threat_yaw_pwm'] = 1600
    if fault == 'broad_onset': broad['onset_s'] = .22
    if fault == 'duplicate_case': metrics['spatial_response']['cases'][-1] = case.copy()
    if fault == 'duplicate_scene': metrics['scenes'][-1] = metrics['scenes'][0].copy()
    if fault == 'early_stop': metrics['scenes'][0]['horizon_completed'] = False
    if fault == 'crash': metrics['scenes'][0]['crashed'] = True
    if fault == 'missing_scene': metrics['scenes'].pop()
    if fault == 'wrong_seed': metrics['scenes'][0]['seed'] = 0
    if fault == 'wrong_tolerance': metrics['spatial_response']['tolerance_pwm'] = 30
    assert not assess_mvp(metrics, scene_seed=20261009, scene_steps=500)['passed']


def test_mvp_search_does_not_overwrite_existing_results(tmp_path):
    from optimizer.finalize_mvp_candidate import run_search
    with pytest.raises(FileExistsError):
        run_search(tmp_path/'missing', tmp_path/'missing.pt', tmp_path)


@pytest.mark.parametrize('freeze_core', [False, True])
def test_four_trial_pipeline_saves_baselines_frozen_checkpoints_and_separate_gates(tmp_path, monkeypatch, freeze_core):
    """Real tiny-graph training/export/reload; expensive physical evaluation is stubbed."""
    import json
    from generator.source_contract import file_sha256
    from optimizer import finalize_mvp_candidate as experiment
    from optimizer.benchmark_candidate import load_research_run
    from simulation.control_contract import control_contract
    from simulation.policy import ChongFlyMSPPolicy
    from tests.synthetic_graphs import write_tiny_model

    data = dataset()
    data['X'][..., 2:34] = .1
    data_path = tmp_path/'train.pt'
    torch.save(data, data_path)
    parent = tmp_path/'parent'
    parent.mkdir()
    (parent/'preflight.json').write_text(json.dumps(dict(experiment_fingerprint='synthetic-test',
        experiment=dict(dataset_sha256=file_sha256(data_path), code_sha256={}))))
    for i, name in enumerate(sorted(experiment.CANDIDATES)):
        directory = parent/f'trial_{i:05d}'
        meta = write_tiny_model(directory/'graph')
        policy = ChongFlyMSPPolicy.from_meta(str(meta), dt=.02, preserve_signs=True,
                                            sensor_encoding='proximity-v1', neutral_origin=True)
        pretrain_policy(policy, data, epochs=1, subset_ratio=1., batch_size=2,
                        seed=42, research_only=True, loss_mode='channel-v3')
        checkpoint = dict(format_version='candidate-research-checkpoint-v1', training_ready=False,
            candidate=name, dt=.02, control_contract=control_contract(), meta_sha256=file_sha256(meta),
            mode='masked', preserve_signs=True, sensor_encoding='proximity-v1', neutral_origin=True,
            state_dict=policy.state_dict(), pretrain_info=policy._pretrain_info)
        torch.save(checkpoint, directory/'checkpoint.pt')
        report = dict(checkpoint_sha256=file_sha256(directory/'checkpoint.pt'),
            experiment_fingerprint='synthetic-test', config=dict(seed=42, batch_size=2,
                yaw_balance=.7, scene_seeds=[20261009], scene_steps=500))
        (directory/'report.json').write_text(json.dumps(report))
    calls = []

    def cheap_evaluate(policy, data, ids, config):
        calls.append(policy.cfc_network.cell.tau_raw.requires_grad)
        metrics = valid_metrics()
        metrics['heldout'] = dict(channel_mae_pwm=dict(yaw=1.))
        return metrics

    monkeypatch.setattr(experiment, 'evaluate', cheap_evaluate)
    out = tmp_path/'result'
    summary = experiment.run_search(parent, data_path, out, freeze_core=freeze_core)
    assert summary['completed_trials'] == 4
    assert len(summary['feasible_pareto_trials']) == 4
    assert calls == [False]*8
    for trial in sorted(out.glob('trial_*')):
        before, saved_before = load_research_run(trial/'before')
        after, saved_after = load_research_run(trial/'after')
        assert saved_before['continuation']['epochs'] == 0
        assert saved_after['continuation']['epochs'] == 10
        assert saved_after['pretrain_info']['loss_contract']['version'] == 'channel-v4'
        assert saved_after['pretrain_info']['loss_contract']['yaw_recovery_window_s'] == 1.
        assert saved_after['training_constraints'] == saved_before['training_constraints']
        for attribute in ('tau_raw', 'A'):
            assert torch.equal(getattr(before.cfc_network.cell, attribute), getattr(after.cfc_network.cell, attribute))
            assert not getattr(after.cfc_network.cell, attribute).requires_grad
        if freeze_core:
            assert torch.equal(before.cfc_network.cell._w_macro_param, after.cfc_network.cell._w_macro_param)
            assert not after.cfc_network.cell._w_macro_param.requires_grad
            assert saved_after['frozen_parameters'] == ['tau_raw', 'A', '_w_macro_param']
        assert after.cfc_network.cell._effective_W().norm() <= saved_after['w_max_norm']
        report = json.loads((trial/'after'/'report.json').read_text())
        assert report['checkpoint_roundtrip_exact']
        assert report['mvp_admission']['passed'] and not report['functional_admission']['passed']
    assert (out/'comparison.md').is_file()


def test_full_core_training_keeps_zero_drive_and_reports_readout_gain(tiny_model_meta):
    from simulation.policy import ChongFlyMSPPolicy
    from optimizer.mvp_constraints import RecurrentConstraints
    policy = ChongFlyMSPPolicy.from_meta(str(tiny_model_meta), preserve_signs=True,
        neutral_origin=True, sensor_encoding='proximity-v1', dt=.02)
    constraint = RecurrentConstraints(policy, freeze_core=True)
    original = policy.cfc_network.cell._w_macro_param.detach().clone()
    interface = policy.cfc_network.cell.W_in.weight.detach().clone()
    data = dataset()
    data['X'][..., 2:34] = .1
    pretrain_policy(policy, data, epochs=3, subset_ratio=1., validation_fraction=0.,
        seed=42, research_only=True, loss_mode='channel-v4', post_step_callback=constraint)
    assert torch.equal(policy.cfc_network.cell._w_macro_param, original)
    assert not torch.equal(policy.cfc_network.cell.W_in.weight, interface)
    audit = constraint.audit()
    assert audit['W_unchanged'] and audit['clear_drive_max_abs'] == 0.
    assert audit['decoder_norm_policy'] == 'monitor_only'
    assert audit['decoder_yaw_norm_ratio'] > 0
    with torch.no_grad():
        policy.dn_head.proj.weight.mul_(2.)
    assert constraint.audit()['decoder_yaw_norm_ratio'] == pytest.approx(audit['decoder_yaw_norm_ratio']*2)
    assert {n for n,p in policy.named_parameters() if p.requires_grad} <= {
        n for n,p in policy.named_parameters() if n.startswith(('sensor_layer.', 'cfc_network.cell.W_in.', 'dn_head.'))}


def test_full_core_rejects_mutation_and_checkpoint_with_wrong_reference(tiny_model_meta):
    from simulation.policy import ChongFlyMSPPolicy
    from optimizer.mvp_constraints import RecurrentConstraints
    policy = ChongFlyMSPPolicy.from_meta(str(tiny_model_meta), preserve_signs=True,
        neutral_origin=True, sensor_encoding='proximity-v1')
    constraint = RecurrentConstraints(policy, freeze_core=True)
    metadata = constraint.metadata()
    with torch.no_grad():
        policy.cfc_network.cell._w_macro_param.mul_(.9)
    with pytest.raises(ValueError, match='Frozen recurrent'):
        constraint(policy)
    with pytest.raises(ValueError, match='reference'):
        RecurrentConstraints(policy, metadata=metadata)
