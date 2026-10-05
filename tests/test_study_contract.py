import pytest
import optuna

from optimizer import optuna_tuner as tuner
from optimizer.rollout import BENCHMARK_VERSION
from simulation.control_contract import control_contract


def test_safe_frontier_excludes_smaller_smooth_crash():
    study = tuner.create_study(storage=None)
    for feasible, values in [(False, [1., 0., 20.]), (True, [100., 2., 5.])]:
        trial = study.ask()
        trial.set_user_attr("feasible", feasible)
        trial.set_user_attr("benchmark_version", BENCHMARK_VERSION)
        trial.set_user_attr("control_contract", control_contract())
        trial.set_user_attr("constraints", [0. if feasible else 1.])
        study.tell(trial, values)
    assert [t.number for t in tuner.feasible_pareto_trials(study)] == [1]


def test_no_feasible_candidate_means_empty_frontier():
    study = tuner.create_study(storage=None)
    # Missing safety evidence must fail closed, even with excellent objective values.
    study.add_trial(optuna.trial.create_trial(values=[1., 0., 100.]))
    assert tuner.feasible_pareto_trials(study) == []


def test_legacy_study_is_rejected_without_relabelling(tmp_path):
    storage = "sqlite:///" + (tmp_path / "study.db").as_posix()
    old = optuna.create_study(study_name="legacy", storage=storage,
                              directions=["minimize", "minimize", "maximize"])
    old.add_trial(optuna.trial.create_trial(values=[1., 0., 100.]))
    with pytest.raises(ValueError, match="(?i)benchmark|version"):
        tuner.create_study(study_name="legacy", storage=storage)
    assert "benchmark_version" not in old.user_attrs
    assert len(old.trials) == 1


def test_trials_share_training_and_scenario_seed(monkeypatch):
    seeds = []
    def objective(**kwargs):
        seeds.append(kwargs["seed"])
        kwargs["trial"].set_user_attr("constraints", [0.])
        return 1., 0., 2.
    monkeypatch.setattr(tuner, "objective", objective)
    tuner.run_optuna_study(n_trials=2, storage=None, pretrain=False, seed=17)
    assert seeds == [17, 17]


def test_objective_publishes_constraints_to_optuna(monkeypatch):
    import torch
    from optimizer import evaluate
    monkeypatch.setattr(evaluate, "create_model", lambda *a, **kw: torch.nn.Linear(1, 1))
    count = 0
    def rollout(**kwargs):
        nonlocal count
        failed = count < 3
        count += 1
        return ((1., 0., 100.) if failed else (100., 1., 3.)), {
            "benchmark_version": BENCHMARK_VERSION, "control_contract": control_contract(),
            "feasible": not failed, "crashed": failed, "fatal_failure": False,
            "is_crab_flight": False, "min_clearance_m": 0., "survival_time_s": .02}
    monkeypatch.setattr(evaluate, "simulate_policy_rollout", rollout)
    study = tuner.create_study(storage=None)
    study.optimize(lambda trial: evaluate.objective(trial, pretrain=False, eval_steps=1), n_trials=2)
    assert [t.number for t in study.best_trials] == [1]
    assert [t.number for t in tuner.feasible_pareto_trials(study)] == [1]


def test_resume_with_different_seed_is_rejected(tmp_path):
    storage = "sqlite:///" + (tmp_path / "study.db").as_posix()
    tuner.create_study(storage=storage, seed=17)
    with pytest.raises(ValueError, match="seed"):
        tuner.create_study(storage=storage, seed=18)


def test_previous_neural_dynamics_cannot_resume_in_new_benchmark(tmp_path):
    storage = "sqlite:///" + (tmp_path / "old_dynamics.db").as_posix()
    old = optuna.create_study(study_name="previous", storage=storage,
                              directions=["minimize", "minimize", "maximize"])
    old.set_user_attr("benchmark_version", "flight-benchmark-v2")
    with pytest.raises(ValueError, match="version"):
        tuner.create_study(study_name="previous", storage=storage)


def test_dense_bypass_benchmark_cannot_resume_after_routing_change(tmp_path):
    storage = "sqlite:///" + (tmp_path / "dense.db").as_posix()
    old = optuna.create_study(study_name="dense", storage=storage,
                              directions=["minimize", "minimize", "maximize"])
    old.set_user_attr("benchmark_version", "flight-benchmark-v3")
    with pytest.raises(ValueError, match="version"):
        tuner.create_study(study_name="dense", storage=storage)


def test_collapsed_motor_benchmark_cannot_resume_after_reduction_change(tmp_path):
    storage = "sqlite:///" + (tmp_path / "collapsed.db").as_posix()
    old = optuna.create_study(study_name="collapsed", storage=storage,
                              directions=["minimize", "minimize", "maximize"])
    old.set_user_attr("benchmark_version", "flight-benchmark-v4")
    with pytest.raises(ValueError, match="version"):
        tuner.create_study(study_name="collapsed", storage=storage)


def test_previous_coordinate_benchmark_is_rejected(tmp_path):
    storage = 'sqlite:///' + (tmp_path / 'coordinates.db').as_posix()
    old = optuna.create_study(study_name='coordinates', storage=storage,
                              directions=['minimize', 'minimize', 'maximize'])
    old.set_user_attr('benchmark_version', 'flight-benchmark-v5')
    with pytest.raises(ValueError, match='version'):
        tuner.create_study(study_name='coordinates', storage=storage)


@pytest.mark.parametrize('version', ['flight-benchmark-v6', 'flight-benchmark-v6+range-altitude-v1'])
def test_previous_raw_and_assisted_scores_cannot_enter_navigation_frontier(version):
    study = tuner.create_study(storage=None)
    study.add_trial(optuna.trial.create_trial(values=[1., 0., 100.], user_attrs={
        'benchmark_version': version, 'control_contract': control_contract(),
        'feasible': True, 'constraints': [0.]}))
    assert tuner.feasible_pareto_trials(study) == []


@pytest.mark.parametrize('fault', ['missing', 'different_calibration'])
def test_navigation_frontier_requires_matching_controller(fault):
    study = tuner.create_study(storage=None)
    attrs = {'benchmark_version': BENCHMARK_VERSION, 'feasible': True, 'constraints': [0.]}
    if fault == 'different_calibration':
        attrs['control_contract'] = control_contract()
        attrs['control_contract']['altitude']['config']['hover_pwm'] = 1300
    study.add_trial(optuna.trial.create_trial(values=[1., 0., 100.], user_attrs=attrs))
    assert tuner.feasible_pareto_trials(study) == []


def test_study_with_changed_controller_is_rejected_on_resume(tmp_path):
    storage = 'sqlite:///' + (tmp_path / 'control.db').as_posix()
    study = tuner.create_study(storage=storage)
    changed = control_contract()
    changed['altitude']['config']['kp'] = 8
    study.set_user_attr('control_contract', changed)
    with pytest.raises(ValueError, match='control contract'):
        tuner.create_study(storage=storage)
