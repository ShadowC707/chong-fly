import pytest
import torch

from optimizer.pretrain import pretrain_policy
from configs.flight_config import COORDINATE_VERSION
from generator.reflex_contract import DATASET_VERSION, TEACHER_VERSION, REQUIRED_PATHS
from simulation.control_contract import control_contract


class TimedPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(4))
        self.seen_dt = []

    def forward(self, x, dt=None):
        self.seen_dt.append(dt)
        return 1500 + self.bias.expand(*x.shape[:2], 4)


def dataset():
    return {"X": torch.ones(3, 5, 74), "Y": torch.full((3, 5, 4), 1510.),
            "applied_actions": torch.full((3, 5, 4), 1510.),
            "valid": torch.ones(3, 5, dtype=torch.bool), "behavior": torch.zeros(3, 5, dtype=torch.long),
            "metadata": {"dt": .02, "dataset_version": DATASET_VERSION,
                         "teacher_version": TEACHER_VERSION, "coordinate_version": COORDINATE_VERSION,
                         "tof_max_range_m": 3.5, "episode_scenarios": ['test']*3,
                         "control_contract": control_contract(),
                         "behavior_names": ['test'], "required_sensor_motor_paths": REQUIRED_PATHS}}


def test_pretrain_uses_dataset_timebase():
    model = TimedPolicy()
    pretrain_policy(model, dataset(), epochs=1, subset_ratio=1., seed=5, validation_fraction=0)
    assert model.seen_dt == [.02]
    assert model._pretrain_info["dt"] == .02


def test_missing_dataset_does_not_silently_train_on_another_file(tmp_path):
    model = TimedPolicy()
    with pytest.raises(FileNotFoundError):
        pretrain_policy(model, str(tmp_path / "absent.pt"), epochs=1)
    assert model.seen_dt == []


def test_old_unversioned_demonstrations_are_not_accepted_silently():
    data = dataset()
    data['metadata'].pop('dataset_version')
    with pytest.raises(ValueError, match='version|regenerate'):
        pretrain_policy(TimedPolicy(), data)


def test_padded_targets_never_contribute_to_training():
    data = dataset()
    data['valid'][:, 2:] = False
    data['behavior'][:, 2:] = -1
    data['Y'][:, :2] = 1600.
    data['Y'][:, 2:] = 1000.  # unmasked loss would pull the other way
    model = TimedPolicy()
    pretrain_policy(model, data, epochs=1, subset_ratio=1., validation_fraction=0, seed=0)
    assert torch.all(model.bias[1:] > 0) and model.bias[0] == 0


def test_split_and_subset_keep_scenarios_and_holdout_episodes_separate():
    data = dataset()
    for key in ('X', 'Y', 'applied_actions', 'valid', 'behavior'): data[key] = data[key].repeat(4, 1, *([1] if data[key].ndim == 3 else []))
    data['metadata']['episode_scenarios'] = ['left']*6 + ['right']*6
    model = TimedPolicy()
    pretrain_policy(model, data, epochs=1, subset_ratio=.3, validation_fraction=.25, seed=4)
    info = model._pretrain_info
    assert not set(info['subset_indices']) & set(info['validation_indices'])
    assert info['subset_scenario_counts']['left'] and info['subset_scenario_counts']['right']
    assert len(info['validation_loss_history']) == 1


def test_rare_behavior_has_equal_total_training_weight():
    data = dataset()
    data['metadata']['behavior_names'] = ['common', 'rare']
    data['Y'][:] = 1600
    data['Y'][-1, -1] = 1400
    data['behavior'][-1, -1] = 1
    model = TimedPolicy()
    gradients = []
    model.bias.register_hook(lambda grad: gradients.append(grad.clone()))
    pretrain_policy(model, data, epochs=1, subset_ratio=1., validation_fraction=0, seed=1)
    # 14 positive targets and one negative target cancel only with balancing.
    torch.testing.assert_close(gradients[0], torch.zeros(4), atol=1e-7, rtol=0)


def test_impossible_teacher_route_is_rejected_before_parameter_update():
    model = TimedPolicy()
    model.routing_diagnostics = {'connectivity': 'structured', 'sensor_motor_paths': {
        'lptc_flow': {'minimum_hops': {'roll': 1, 'pitch': 1}},
        'lc_looming': {'minimum_hops': {'pitch': 1, 'yaw': None}}}}
    data = dataset()
    with pytest.raises(ValueError, match='lc_looming.*yaw'):
        pretrain_policy(model, data)
    assert model.seen_dt == [] and torch.all(model.bias == 0)


@pytest.mark.parametrize('fault', ['nonprefix', 'empty_row', 'wrong_coordinates', 'wrong_range', 'bad_behavior'])
def test_invalid_new_dataset_contract_fails_before_training(fault):
    data = dataset()
    if fault == 'nonprefix': data['valid'][0, 1] = False
    if fault == 'empty_row': data['valid'][0] = False
    if fault == 'wrong_coordinates': data['metadata']['coordinate_version'] = 'old'
    if fault == 'wrong_range': data['metadata']['tof_max_range_m'] = 3.
    if fault == 'bad_behavior': data['behavior'][0, 0] = 12
    with pytest.raises(ValueError): pretrain_policy(TimedPolicy(), data)


@pytest.mark.parametrize("fault", ["missing_dt", "negative_dt", "nan_inputs", "bad_targets", "empty"])
def test_invalid_dataset_fails_before_training(fault):
    data = dataset()
    if fault == "missing_dt": data.pop("metadata")
    if fault == "negative_dt": data["metadata"]["dt"] = -.02
    if fault == "nan_inputs": data["X"][0, 0, 0] = float("nan")
    if fault == "bad_targets": data["Y"] = torch.ones(3, 5, 3)
    if fault == "empty": data["X"], data["Y"] = data["X"][:0], data["Y"][:0]
    model = TimedPolicy()
    with pytest.raises(ValueError):
        pretrain_policy(model, data, epochs=1)
    assert model.seen_dt == []
