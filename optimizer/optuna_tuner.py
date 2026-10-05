"""
optimizer/optuna_tuner.py
=========================
Optuna Hyperparameter Tuner for Chong-Fly Policies with Behavioral Cloning.

Connects:
1. Model Generator (create_model).
2. Teacher / Pre-trainer (navigation-only pretrain_policy on reflex_dataset_v3.pt).
3. Arena / Evaluator (simulate_policy_rollout).
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Any, Dict, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    optuna = None
    HAS_OPTUNA = False

from optimizer.evaluate import objective
from optimizer.rollout import BENCHMARK_VERSION
from generator.reflex_contract import DEFAULT_DATASET_PATH
from simulation.control_contract import control_contract


def safety_constraints(trial):
    """Missing, malformed, or older safety evidence is infeasible."""
    attrs = trial.user_attrs
    if attrs.get("benchmark_version") != BENCHMARK_VERSION or attrs.get("feasible") is not True:
        return (1.0,)
    if attrs.get('control_contract') != control_contract():
        return (1.0,)
    constraints = attrs.get("constraints")
    if not isinstance(constraints, (list, tuple)) or len(constraints) != 1:
        return (1.0,)
    value = constraints[0]
    if not isinstance(value, (float, int)) or not math.isfinite(value):
        return (1.0,)
    return (float(value),)


def feasible_pareto_trials(study):
    """Compute nondominance among verified feasible trials, including older Optuna APIs.

    Filtering an unconstrained frontier afterwards is insufficient: an unsafe
    trial could have dominated and hidden every safe candidate.
    """
    candidates = [t for t in study.get_trials(deepcopy=False)
                  if t.state == optuna.trial.TrialState.COMPLETE
                  and all(c <= 0 for c in safety_constraints(t))
                  and t.values is not None and all(math.isfinite(v) for v in t.values)]
    def losses(trial):
        return tuple(v if d == optuna.study.StudyDirection.MINIMIZE else -v
                     for v, d in zip(trial.values, study.directions))
    values = [losses(t) for t in candidates]
    return [trial for i, trial in enumerate(candidates)
            if not any(all(a <= b for a, b in zip(other, values[i]))
                       and any(a < b for a, b in zip(other, values[i]))
                       for j, other in enumerate(values) if i != j)]


def create_study(
    study_name: str = "flight_benchmark_v7",
    storage: Optional[str] = "sqlite:///flight_benchmark_v7.db",
    seed: int = 42,
) -> Any:
    """
    Create a constrained three-objective study; reject incompatible history.
    """
    if not HAS_OPTUNA:
        raise ImportError("Optuna is not installed. Install via pip install optuna.")

    # Optuna 5 publishes constraints through Trial; older versions use a sampler callback.
    sampler_kwargs = {} if hasattr(optuna.trial.Trial, "set_constraint") else {"constraints_func": safety_constraints}
    sampler = optuna.samplers.NSGAIISampler(seed=seed, **sampler_kwargs)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage, # was storage.
        sampler=sampler,
        directions=["minimize", "minimize", "maximize"],
        load_if_exists=True,
    )
    expected = [optuna.study.StudyDirection.MINIMIZE, optuna.study.StudyDirection.MINIMIZE,
                optuna.study.StudyDirection.MAXIMIZE]
    if study.directions != expected:
        raise ValueError("Incompatible benchmark objective directions; use a new study")
    version = study.user_attrs.get("benchmark_version")
    if version != BENCHMARK_VERSION and (version is not None or study.trials):
        raise ValueError("Incompatible benchmark version; use a new study name/database")
    if any(t.state == optuna.trial.TrialState.COMPLETE
           and t.user_attrs.get("benchmark_version") != BENCHMARK_VERSION for t in study.trials):
        raise ValueError("Study contains trials from a different benchmark version")
    if study.user_attrs.get("comparison_seed", seed) != seed:
        raise ValueError("Comparison seed differs from this study; use a new study")
    existing_contract = study.user_attrs.get('control_contract')
    if existing_contract != control_contract() and (existing_contract is not None or version is not None or study.trials):
        raise ValueError('Incompatible control contract; use a new study')
    if any(t.state == optuna.trial.TrialState.COMPLETE
           and t.user_attrs.get('control_contract') != control_contract() for t in study.trials):
        raise ValueError('Study contains trials with a different control contract')
    study.set_user_attr("benchmark_version", BENCHMARK_VERSION)
    study.set_user_attr('control_contract', control_contract())
    study.set_user_attr("comparison_seed", seed)
    return study


def run_optuna_study(
    n_trials: int = 15,
    dataset_path: str = DEFAULT_DATASET_PATH,
    pretrain: bool = True,
    seed: int = 42,
    study_name: str = "flight_benchmark_v7",
    storage: Optional[str] = "sqlite:///flight_benchmark_v7.db",
    device: str = "auto",
) -> Any:
    """
    Executes an optimization study over Chong-Fly architectures and controllers.
    """
    if not HAS_OPTUNA:
        raise ImportError("Optuna is not installed. Run: pip install optuna")

    study = create_study(study_name=study_name, storage=storage,seed=seed)

    def _trial_obj(trial: optuna.Trial):
        return objective(
            trial=trial,
            dataset_path=dataset_path,
            pretrain=pretrain,
            seed=seed,
            device=device,
        )

    study.optimize(_trial_obj, n_trials=n_trials)
    return study


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chong-Fly Optuna Tuner with Behavioral Cloning")
    parser.add_argument("--trials", type=int, default=10, help="Number of optimization trials")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET_PATH, help="Reflex dataset path")
    parser.add_argument("--no-pretrain", action="store_true", help="Disable behavioral cloning pre-training")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--device", type=str, default="auto", help="Compute device (auto, cuda, cpu)")
    parser.add_argument("--study-name", type=str, default="flight_benchmark_v7", help="Ім'я експерименту")
    parser.add_argument("--db", type=str, default="sqlite:///flight_benchmark_v7.db", help="Шлях до БД")

    args = parser.parse_args()

    data_file = os.path.join(_ROOT, args.dataset) if not os.path.isabs(args.dataset) else args.dataset
    print(f"Starting Optuna Study: {args.trials} trials, pretrain={not args.no_pretrain}, dataset={data_file}")

    study = run_optuna_study(
        study_name=args.study_name,
        n_trials=args.trials,
        storage=args.db,
        dataset_path=data_file,
        pretrain=not args.no_pretrain,
        seed=args.seed,
        device=args.device,
    )

    print("\n" + "=" * 60)
    print("Optimization Completed!")
    if len(study.directions) > 1:
        frontier = feasible_pareto_trials(study)
        print(f"Number of feasible Pareto-optimal trials: {len(frontier)}")
        for t in frontier:
            print(f"  Trial #{t.number}: values={t.values}, params={t.params}")
    else:
        print(f"Best Trial #{study.best_trial.number}")
        print(f"Best Composite Cost: {study.best_value:.4f}")
        print("Best Hyperparameters:")
        for k, v in study.best_params.items():
            print(f"  * {k}: {v}")
    print("=" * 60)
