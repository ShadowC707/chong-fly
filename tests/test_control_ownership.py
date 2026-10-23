"""Navigation cannot learn privileged altitude targets or bypass its supervisor."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from generator.generate_reflex_dataset import generate_reflex_dataset
from optimizer.pretrain import pretrain_policy
from simulation.altitude_control import AltitudeHold


class ConstantPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(4))

    def forward(self, x, dt=None):
        return 1500 + self.bias.expand(*x.shape[:2], 4)


def test_throttle_targets_do_not_change_training_or_validation():
    data = generate_reflex_dataset(num_episodes=16, seq_len=2, seed=4)
    changed = deepcopy(data)
    data['Y'][..., 0] = 1000
    changed['Y'][..., 0] = 2000
    models = [ConstantPolicy(), ConstantPolicy()]
    for model, demos in zip(models, (data, changed)):
        pretrain_policy(model, demos, epochs=2, subset_ratio=1, seed=2)
        assert model.bias[0].item() == 0
        assert model._pretrain_info['learned_channels'] == ['roll', 'pitch', 'yaw']
        assert len(model._pretrain_info['validation_channel_mae_pwm']) == 3
    torch.testing.assert_close(models[0].bias, models[1].bias, rtol=0, atol=0)
    for key in ('loss_history', 'validation_loss_history', 'validation_channel_mae_pwm'):
        assert models[0]._pretrain_info[key] == models[1]._pretrain_info[key]


def test_validation_mse_averages_only_the_three_navigation_channels():
    data = generate_reflex_dataset(num_episodes=16, seq_len=2)
    data['Y'][..., 0] = 1000
    data['Y'][..., 1:] = torch.tensor([1550., 1600., 1650.])
    model = ConstantPolicy()
    pretrain_policy(model, data, epochs=1, subset_ratio=1, seed=0, lr=0)
    assert model._pretrain_info['validation_loss_history'][0] == pytest.approx((.1**2+.2**2+.3**2)/3)
    assert model._pretrain_info['validation_channel_mae_pwm'] == [50, 100, 150]


@pytest.mark.parametrize('fault', ['missing', 'learn_throttle', 'different_altitude', 'missing_routes'])
def test_incompatible_ownership_is_rejected_before_training(fault):
    data = generate_reflex_dataset(num_episodes=2, seq_len=2)
    metadata = data['metadata']
    if fault == 'missing': metadata.pop('control_contract', None)
    elif fault == 'learn_throttle': metadata['control_contract']['learned_channels'] = ['throttle', 'roll', 'pitch', 'yaw']
    elif fault == 'different_altitude': metadata['control_contract']['altitude']['config']['target_height_m'] = 2
    else: metadata['required_sensor_motor_paths'] = {}
    model = ConstantPolicy()
    with pytest.raises(ValueError, match='contract|routes'):
        pretrain_policy(model, data)
    assert torch.equal(model.bias, torch.zeros(4))


def test_collector_records_supervised_actions_separately_from_clean_labels(monkeypatch):
    from generator import generate_reflex_dataset as generator
    original = generator._make_scenario
    applied = []
    def scenario(*args):
        env, obs = original(*args)
        step = env.step
        def record(action, **kwargs):
            applied.append(np.asarray(action).copy())
            return step(action, **kwargs)
        env.step = record
        return env, obs
    monkeypatch.setattr(generator, '_make_scenario', scenario)
    data = generator.generate_reflex_dataset(num_episodes=1, seq_len=4, noise_std_pwm=20)
    actions = data['applied_actions'][0].numpy()
    np.testing.assert_allclose(actions, np.array(applied)[::5], atol=1e-4)
    assert data['metadata']['control_contract']['altitude'] == AltitudeHold().contract()
    assert np.any(actions[:, 1:] != data['Y'][0, :, 1:].numpy())
    assert np.all(np.abs(np.diff(actions[:, 0])) <= 3.001)  # 150 PWM/s at 50 Hz


def test_sensor_fault_aborts_collection_without_writing_dataset(monkeypatch, tmp_path):
    from generator import generate_reflex_dataset as generator
    from simulation.altitude_control import AltitudeTelemetryError
    original = generator._make_scenario
    closed = []
    def scenario(*args):
        env, obs = original(*args)
        snapshot, close = env.get_isaac_obs, env.close
        def faulty():
            result = snapshot()
            result.laser_valid = False
            return result
        def cleanup():
            closed.append(True)
            close()
        env.get_isaac_obs, env.close = faulty, cleanup
        return env, obs
    monkeypatch.setattr(generator, '_make_scenario', scenario)
    path = tmp_path/'faulty.pt'
    with pytest.raises(AltitudeTelemetryError):
        generator.generate_reflex_dataset(num_episodes=1, seq_len=2, output_path=path)
    assert closed == [True] and not path.exists()


def test_objective_uses_altitude_controller_and_refuses_raw_results(monkeypatch):
    from optimizer import evaluate
    monkeypatch.setattr(evaluate, 'create_model', lambda *a, **k: ConstantPolicy())
    def raw_rollout(**kwargs):
        assert isinstance(kwargs.get('altitude_hold'), AltitudeHold)
        return (1., 0., 1.), {'benchmark_version': 'flight-benchmark-v6', 'feasible': True}
    monkeypatch.setattr(evaluate, 'simulate_policy_rollout', raw_rollout)
    with pytest.raises(ValueError, match='contract|benchmark'):
        evaluate.objective(pretrain=False, eval_steps=1)


def test_rollout_labels_raw_standard_and_custom_controllers_separately():
    from types import SimpleNamespace
    from optimizer.rollout import simulate_policy_rollout, BENCHMARK_VERSION, RAW_BENCHMARK_VERSION
    from simulation.altitude_control import AltitudeConfig
    from tests.test_evaluation_contract import ScriptedEnv, RecordingPolicy
    for controller, expected in (
            (None, RAW_BENCHMARK_VERSION),
            (AltitudeHold(), BENCHMARK_VERSION),
            (AltitudeHold(AltitudeConfig(hover_pwm=1300)), BENCHMARK_VERSION+'+custom-altitude')):
        env = ScriptedEnv()
        env.get_isaac_obs = lambda: SimpleNamespace(laser_distance=.982, laser_valid=True,
                                                   attitude_euler=[0, 0, 0], sim_time=env.steps*env.dt)
        _, metrics = simulate_policy_rollout(RecordingPolicy(), env=env, eval_steps=2, altitude_hold=controller)
        assert metrics['benchmark_version'] == expected and metrics['feasible']
        if controller is not None:
            assert metrics['control_contract']['altitude'] == controller.contract()


@pytest.mark.parametrize('fault', ['missing', 'shape', 'nan', 'bounds'])
def test_applied_action_record_is_required_even_though_not_a_learning_target(fault):
    data = generate_reflex_dataset(num_episodes=1, seq_len=2)
    if fault == 'missing': data.pop('applied_actions')
    elif fault == 'shape': data['applied_actions'] = data['applied_actions'][..., :3]
    elif fault == 'nan': data['applied_actions'][0, 0, 0] = float('nan')
    else: data['applied_actions'][0, 0, 0] = 2500
    with pytest.raises(ValueError, match='applied_actions'):
        pretrain_policy(ConstantPolicy(), data)
