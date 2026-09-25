"""
training
========
Training, evaluation, and hyperparameter optimization for Chong-Fly.
"""

from training.evaluate import evaluate_math_viability, math_screening, evaluate_simulation_behavior
from training.env import DroneSimulationEnv, simulate_policy_rollout
from training.optuna_tuner import (
    SEARCH_SPACE,
    SEARCH_SPACE_DISTRIBUTIONS,
    sample_search_space,
    evaluate_trial_math_viability,
    build_policy_from_trial,
    optuna_objective,
    optuna_multiobjective,
    compute_pareto_front,
    export_pareto_manifest,
    run_pareto_optimization,
    create_study,
    run_optuna_study,
)

__all__ = [
    "evaluate_math_viability",
    "math_screening",
    "evaluate_simulation_behavior",
    "DroneSimulationEnv",
    "simulate_policy_rollout",
    "SEARCH_SPACE",
    "SEARCH_SPACE_DISTRIBUTIONS",
    "sample_search_space",
    "evaluate_trial_math_viability",
    "build_policy_from_trial",
    "optuna_objective",
    "optuna_multiobjective",
    "compute_pareto_front",
    "export_pareto_manifest",
    "run_pareto_optimization",
    "create_study",
    "run_optuna_study",
]
