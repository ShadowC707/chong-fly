"""
optimizer/optuna_tuner.py
=========================
Optuna Hyperparameter Tuner for Chong-Fly Policies with Behavioral Cloning.

Connects:
1. Model Generator (create_model).
2. Teacher / Pre-trainer (pretrain_policy on reflex_dataset.pt).
3. Arena / Evaluator (simulate_policy_rollout).
"""

from __future__ import annotations

import argparse
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

from optimizer.evaluate import objective, simulate_policy_rollout, create_model
from optimizer.pretrain import pretrain_policy


def create_study(
    study_name: str = "chong_reflex_tuning",
    storage: Optional[str] = "sqlite:///chong_optuna.db", # was drone_optimization.db
    seed: int = 42,
) -> Any:
    """
    Creates an Optuna study minimizing composite cost.
    """
    if not HAS_OPTUNA:
        raise ImportError("Optuna is not installed. Install via pip install optuna.")

    sampler = optuna.samplers.NSGAIISampler(seed=seed)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage, # was storage.
        sampler=sampler,
        directions=["minimize", "minimize", "maximize"],
        load_if_exists=True,
    )
    return study


def run_optuna_study(
    n_trials: int = 15,
    dataset_path: str = "data/reflex_dataset.pt",
    pretrain: bool = True,
    seed: int = 42,
    study_name: str = "pareto_search",
    storage: str = "sqlite:///drone_optimization.db",
    device: str = "auto",
) -> Any:
    """
    Executes an optimization study over Chong-Fly architectures and controllers.
    """
    if not HAS_OPTUNA:
        raise ImportError("Optuna is not installed. Run: pip install optuna")

    study = create_study(study_name=study_name, storage=storage,seed=seed)

    def _trial_obj(trial: optuna.Trial) -> float:
        return objective(
            trial=trial,
            dataset_path=dataset_path,
            pretrain=pretrain,
            seed=seed + trial.number,
            device=device,
        )

    study.optimize(_trial_obj, n_trials=n_trials)
    return study


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chong-Fly Optuna Tuner with Behavioral Cloning")
    parser.add_argument("--trials", type=int, default=10, help="Number of optimization trials")
    parser.add_argument("--dataset", type=str, default="data/reflex_dataset.pt", help="Reflex dataset path")
    parser.add_argument("--no-pretrain", action="store_true", help="Disable behavioral cloning pre-training")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--device", type=str, default="auto", help="Compute device (auto, cuda, cpu)")
    parser.add_argument("--study-name", type=str, default="cfc_pareto_search", help="Ім'я експерименту")
    parser.add_argument("--db", type=str, default="sqlite:///drone_optimization.db", help="Шлях до БД")

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
        print(f"Number of Pareto-optimal trials: {len(study.best_trials)}")
        for t in study.best_trials:
            print(f"  Trial #{t.number}: values={t.values}, params={t.params}")
    else:
        print(f"Best Trial #{study.best_trial.number}")
        print(f"Best Composite Cost: {study.best_value:.4f}")
        print("Best Hyperparameters:")
        for k, v in study.best_params.items():
            print(f"  * {k}: {v}")
    print("=" * 60)
