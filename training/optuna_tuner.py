"""
training/optuna_tuner.py
========================
Optuna Hyperparameter Search Space & Optimization Engine for Chong-Fly.

Task 4.1: Optuna Search Space
-----------------------------
Defines and evaluates the hyperparameter search space:
  - k_clusters       : [32, 64, 128, 256]
  - pruning_sparsity : Float(0.50, 0.95)
  - solver_type      : ['CfC', 'Euler_dt_0.02'] (Euler baseline vs CfC closed-form)
  - ablate_cx        : Categorical([True, False]) (flight hypothesis without Central Complex)

Integrates with:
  - training/evaluate.py: fast O(1) mathematical viability gate.
  - bio_pipeline/models.py: BiologicalCfCCell and BiologicalCfCNetwork.
  - simulation/policy.py: ChongFlyMSPPolicy for closed-loop drone flight loop.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import scipy.sparse as sp
import torch

try:
    import optuna
    from optuna.distributions import CategoricalDistribution, FloatDistribution
    _OPTUNA_AVAILABLE = True
except ImportError:
    optuna = None
    CategoricalDistribution = None
    FloatDistribution = None
    _OPTUNA_AVAILABLE = False

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from bio_pipeline.graph_reducer import ReducedModel
from bio_pipeline.models import BiologicalCfCCell, BiologicalCfCNetwork, build_network_from_meta
from simulation.policy import ChongFlyMSPPolicy, SENSOR_DIM, N_CONTROLS
from training.evaluate import evaluate_math_viability, math_screening, evaluate_simulation_behavior


# ─────────────────────────────────────────────────────────────────────────────
# 4.1. Search Space Specification
# ─────────────────────────────────────────────────────────────────────────────

SEARCH_SPACE: dict[str, Any] = {
    "k_clusters": [32, 64, 128, 256],
    "pruning_sparsity": {"low": 0.50, "high": 0.95},
    "solver_type": ["CfC", "Euler_dt_0.02"],
    "ablate_cx": [True, False],
}

if _OPTUNA_AVAILABLE:
    SEARCH_SPACE_DISTRIBUTIONS: dict[str, optuna.distributions.BaseDistribution] = {
        "k_clusters": CategoricalDistribution([32, 64, 128, 256]),
        "pruning_sparsity": FloatDistribution(0.50, 0.95),
        "solver_type": CategoricalDistribution(["CfC", "Euler_dt_0.02"]),
        "ablate_cx": CategoricalDistribution([True, False]),
    }
else:
    SEARCH_SPACE_DISTRIBUTIONS = {}


def sample_search_space(trial: optuna.Trial) -> dict[str, Any]:
    """
    Sample one configuration from the 4.1 Optuna Search Space:
      - k_clusters       : [32, 64, 128, 256]
      - pruning_sparsity : Float(0.50, 0.95)
      - solver_type      : ['CfC', 'Euler_dt_0.02'] (Euler baseline)
      - ablate_cx        : Categorical([True, False]) (Central Complex ablation)
    """
    if not _OPTUNA_AVAILABLE:
        raise RuntimeError("Optuna is not installed in the active environment.")

    k_clusters = trial.suggest_categorical("k_clusters", [32, 64, 128, 256])
    pruning_sparsity = trial.suggest_float("pruning_sparsity", 0.50, 0.95)
    solver_type = trial.suggest_categorical("solver_type", ["CfC", "Euler_dt_0.02"])
    ablate_cx = trial.suggest_categorical("ablate_cx", [True, False])

    return {
        "k_clusters": int(k_clusters),
        "pruning_sparsity": float(pruning_sparsity),
        "solver_type": str(solver_type),
        "ablate_cx": bool(ablate_cx),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Model Resolution & Matrix Transformations
# ─────────────────────────────────────────────────────────────────────────────

def resolve_meta_path(k: int, ablate_cx: bool, base_dir: str = "data/reduced_models") -> str:
    """
    Find metadata path corresponding to k macro-clusters and ablate_cx setting.
    Prefers _nocx pre-reduced model if ablate_cx is True.
    """
    if not os.path.isabs(base_dir):
        base_dir = os.path.join(_ROOT, base_dir)

    if ablate_cx:
        nocx_path = os.path.join(base_dir, f"meta_spectral_k{k}_nocx.json")
        if os.path.exists(nocx_path):
            return nocx_path

    standard_path = os.path.join(base_dir, f"meta_spectral_k{k}.json")
    if os.path.exists(standard_path):
        return standard_path

    raise FileNotFoundError(f"No reduced model found for k={k} in {base_dir}")


def apply_magnitude_pruning(
    W: np.ndarray | sp.csr_matrix,
    sparsity: float,
) -> tuple[np.ndarray | sp.csr_matrix, np.ndarray]:
    """
    Apply global magnitude pruning to matrix W at given sparsity quantile [0.0, 1.0).
    Returns (W_pruned, binary_mask).
    """
    is_sparse = sp.issparse(W)
    if is_sparse:
        W_dense = W.toarray().astype(np.float32)
    else:
        W_dense = np.array(W, dtype=np.float32, copy=True)

    k_pct = float(sparsity) * 100.0
    threshold = float(np.percentile(np.abs(W_dense), k_pct))
    mask = (np.abs(W_dense) >= threshold).astype(np.float32)
    W_pruned = W_dense * mask

    if is_sparse:
        return sp.csr_matrix(W_pruned), mask
    return W_pruned, mask


# ─────────────────────────────────────────────────────────────────────────────
# Fast Math Viability Gate Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_trial_math_viability(
    trial_params: dict[str, Any],
    base_dir: str = "data/reduced_models",
) -> tuple[bool, dict[str, Any]]:
    """
    Рівень 1 воронки: делегує виклик до math_screening().
    """
    return math_screening(trial_params, base_dir=base_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Policy & Network Factory from Trial
# ─────────────────────────────────────────────────────────────────────────────

def build_policy_from_trial(
    trial_params: dict[str, Any],
    base_dir: str = "data/reduced_models",
    mode: str = "masked",
    backbone_units: int = 64,
    backbone_layers: int = 2,
    sensor_dim: int = SENSOR_DIM,
) -> ChongFlyMSPPolicy:
    """
    Constructs a complete ChongFlyMSPPolicy directly from sampled trial hyperparameters.
    """
    k = trial_params["k_clusters"]
    sparsity = trial_params["pruning_sparsity"]
    solver = trial_params["solver_type"]
    ablate_cx = trial_params["ablate_cx"]

    meta_path = resolve_meta_path(k, ablate_cx, base_dir=base_dir)
    dt = 0.02 if solver == "Euler_dt_0.02" else 0.004

    policy = ChongFlyMSPPolicy.from_meta(
        meta_path=meta_path,
        mode=mode,
        pruning_sparsity=sparsity,
        solver_type=solver,
        ablate_cx=ablate_cx,
        sensor_dim=sensor_dim,
        backbone_units=backbone_units,
        backbone_layers=backbone_layers,
        dt=dt,
    )
    return policy


# ─────────────────────────────────────────────────────────────────────────────
# Optuna Objective: Багаторівнева воронка фільтрації
# ─────────────────────────────────────────────────────────────────────────────

def optuna_objective(
    trial: optuna.Trial,
    base_dir: str = "data/reduced_models",
    eval_steps: int = 100,
) -> float:
    """
    Багаторівнева воронка фільтрації (Objective Function):

    Рівень 1: Запуск math_screening.
             Якщо перевірку провалено -> виклик raise optuna.TrialPruned().

    Рівень 2: Якщо модель валідна -> запуск оцінки поведінки в симуляції (O(N)).
    """
    if not _OPTUNA_AVAILABLE:
        raise RuntimeError("Optuna is not installed.")

    params = sample_search_space(trial)

    # ── Рівень 1: Запуск math_screening ─────────────────────────────────────
    is_viable, math_metrics = math_screening(params, base_dir=base_dir)
    for key, val in math_metrics.items():
        if isinstance(val, (int, float, str, bool)):
            trial.set_user_attr(f"math_{key}", val)

    if not is_viable:
        reject_reason = math_metrics.get("reject_reason", "Math screening rejected candidate")
        trial.set_user_attr("pruned_reason", reject_reason)
        trial.set_user_attr("funnel_level", 1)
        raise optuna.TrialPruned(f"Level 1 (math_screening) failed: {reject_reason}")

    trial.set_user_attr("funnel_level_1_passed", True)

    # ── Рівень 2: Оцінка поведінки в симуляції (O(N)) ─────────────────────
    dt = 0.02 if params["solver_type"] == "Euler_dt_0.02" else 0.004
    policy = build_policy_from_trial(params, base_dir=base_dir)
    policy.eval()

    t_start = time.perf_counter()
    behavior_score, sim_metrics = evaluate_simulation_behavior(
        policy=policy,
        eval_steps=eval_steps,
        dt=dt,
        seed=42 + trial.number,
    )
    t_end = time.perf_counter()
    latency_ms = ((t_end - t_start) / eval_steps) * 1000.0

    for key, val in sim_metrics.items():
        if isinstance(val, (int, float, str, bool)):
            trial.set_user_attr(f"sim_{key}", val)

    trial.set_user_attr("latency_ms", latency_ms)

    # Якщо симуляція виявила фатальну нестабільність польоту
    if sim_metrics.get("fatal_failure", False):
        failure_reason = sim_metrics.get("failure_reason", "Simulation flight instability")
        trial.set_user_attr("pruned_reason", failure_reason)
        trial.set_user_attr("funnel_level", 2)
        raise optuna.TrialPruned(f"Level 2 simulation failed: {failure_reason}")

    trial.set_user_attr("funnel_level_2_passed", True)
    trial.set_user_attr("objective_score", behavior_score)

    return float(behavior_score)


# ─────────────────────────────────────────────────────────────────────────────
# Study Management
# ─────────────────────────────────────────────────────────────────────────────

def create_study(
    study_name: str = "chong_fly_optuna_search",
    direction: str = "minimize",
    sampler: Optional[optuna.samplers.BaseSampler] = None,
    pruner: Optional[optuna.pruners.BasePruner] = None,
) -> optuna.Study:
    """Create a configured Optuna study for Chong-Fly search space."""
    if not _OPTUNA_AVAILABLE:
        raise RuntimeError("Optuna is not installed.")

    if sampler is None:
        sampler = optuna.samplers.TPESampler(seed=42)
    if pruner is None:
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)

    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        sampler=sampler,
        pruner=pruner,
    )
    return study


def run_optuna_study(
    n_trials: int = 20,
    study_name: str = "chong_fly_optuna_search",
    base_dir: str = "data/reduced_models",
) -> optuna.Study:
    """Run an optimization study over the 4.1 search space."""
    study = create_study(study_name=study_name)

    def _obj(trial: optuna.Trial) -> float:
        return optuna_objective(trial, base_dir=base_dir)

    study.optimize(_obj, n_trials=n_trials)
    return study


# ─────────────────────────────────────────────────────────────────────────────
# Pareto Frontier Analysis & Manifest Export
# ─────────────────────────────────────────────────────────────────────────────

def optuna_multiobjective(
    trial: optuna.Trial,
    base_dir: str = "data/reduced_models",
    eval_steps: int = 100,
) -> tuple[float, float]:
    """
    Двокритеріальна оцінка для Pareto-оптимізації:
      1. survival_time_s: час виживання в симуляції (максимізація)
      2. energy_cost_j: сумарні затрати енергії (мінімізація)
    """
    if not _OPTUNA_AVAILABLE:
        raise RuntimeError("Optuna is not installed.")

    params = sample_search_space(trial)

    # ── Рівень 1: math_screening ──────────────────────────────────────────
    is_viable, math_metrics = math_screening(params, base_dir=base_dir)
    for key, val in math_metrics.items():
        if isinstance(val, (int, float, str, bool)):
            trial.set_user_attr(f"math_{key}", val)

    if not is_viable:
        reject_reason = math_metrics.get("reject_reason", "Math screening rejected candidate")
        trial.set_user_attr("pruned_reason", reject_reason)
        trial.set_user_attr("funnel_level", 1)
        raise optuna.TrialPruned(f"Level 1 (math_screening) failed: {reject_reason}")

    trial.set_user_attr("funnel_level_1_passed", True)

    # ── Рівень 2: оцінка поведінки в симуляції (O(N)) ─────────────────────
    dt = 0.02 if params["solver_type"] == "Euler_dt_0.02" else 0.004
    policy = build_policy_from_trial(params, base_dir=base_dir)
    policy.eval()

    t_start = time.perf_counter()
    _, sim_metrics = evaluate_simulation_behavior(
        policy=policy,
        eval_steps=eval_steps,
        dt=dt,
        seed=42 + trial.number,
    )
    t_end = time.perf_counter()
    latency_ms = ((t_end - t_start) / eval_steps) * 1000.0

    for key, val in sim_metrics.items():
        if isinstance(val, (int, float, str, bool)):
            trial.set_user_attr(f"sim_{key}", val)

    trial.set_user_attr("latency_ms", latency_ms)

    if sim_metrics.get("fatal_failure", False):
        failure_reason = sim_metrics.get("failure_reason", "Simulation flight instability")
        trial.set_user_attr("pruned_reason", failure_reason)
        trial.set_user_attr("funnel_level", 2)
        raise optuna.TrialPruned(f"Level 2 simulation failed: {failure_reason}")

    trial.set_user_attr("funnel_level_2_passed", True)

    survival_time_s = float(sim_metrics.get("survival_time_s", 0.0))
    energy_cost_j = float(sim_metrics.get("energy_cost_j", 999.0))

    trial.set_user_attr("survival_time_s", survival_time_s)
    trial.set_user_attr("energy_cost_j", energy_cost_j)

    return survival_time_s, energy_cost_j


def compute_pareto_front(
    candidates: list[dict[str, Any]],
    survival_key: str = "survival_time_s",
    energy_key: str = "energy_cost_j",
) -> list[dict[str, Any]]:
    """
    Визначає множину Парето-оптимальних кандидатів за двома критеріями:
      - survival_time_s (максимізація)
      - energy_cost_j   (мінімізація)
      
    Кандидат A домінує над B, якщо:
      A.survival >= B.survival І A.energy <= B.energy
      (і хоча б одна нерівність є строгою).
    """
    pareto: list[dict[str, Any]] = []
    for i, a in enumerate(candidates):
        dominated = False
        s_a = float(a[survival_key])
        e_a = float(a[energy_key])
        for j, b in enumerate(candidates):
            if i == j:
                continue
            s_b = float(b[survival_key])
            e_b = float(b[energy_key])
            if (s_b >= s_a and e_b <= e_a) and (s_b > s_a or e_b < e_a):
                dominated = True
                break
        if not dominated:
            pareto.append(a)

    # Сортування Парето-фронту за часом виживання та витратою енергії
    pareto.sort(key=lambda x: (x[survival_key], -x[energy_key]))
    return pareto


def export_pareto_manifest(
    pareto_candidates_or_study: Any,
    output_path: str = "data/filtered_models_manifest.json",
    base_dir: str = "data/reduced_models",
    total_evaluated: Optional[int] = None,
) -> dict[str, Any]:
    """
    Експортує маніфест data/filtered_models_manifest.json з моделями,
    що увійшли на Pareto-фронт (час виживання vs затрати енергії).
    """
    if not os.path.isabs(output_path):
        output_path = os.path.join(_ROOT, output_path)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 1. Витяг кандидатів
    if hasattr(pareto_candidates_or_study, "trials"):
        study = pareto_candidates_or_study
        total_eval = len(study.trials) if total_evaluated is None else total_evaluated
        raw_candidates = []
        for t in study.trials:
            if t.state == optuna.trial.TrialState.COMPLETE:
                rec = dict(t.params)
                rec["trial_number"] = t.number
                if t.values and len(t.values) >= 2:
                    rec["survival_time_s"] = float(t.values[0])
                    rec["energy_cost_j"] = float(t.values[1])
                else:
                    rec["survival_time_s"] = float(t.user_attrs.get("survival_time_s", 0.0))
                    rec["energy_cost_j"] = float(t.user_attrs.get("energy_cost_j", 999.0))

                for k, v in t.user_attrs.items():
                    if k.startswith("sim_"):
                        rec[k[4:]] = v
                    elif k.startswith("math_"):
                        rec[k[5:]] = v
                    else:
                        rec[k] = v
                raw_candidates.append(rec)
        pareto_list = compute_pareto_front(raw_candidates)
    elif isinstance(pareto_candidates_or_study, list):
        total_eval = len(pareto_candidates_or_study) if total_evaluated is None else total_evaluated
        pareto_list = compute_pareto_front(pareto_candidates_or_study)
    else:
        raise ValueError("pareto_candidates_or_study must be an optuna.Study or list of dicts")

    # 2. Формування стандартизованих записів моделей
    models_entries = []
    for idx, cand in enumerate(pareto_list):
        k = int(cand["k_clusters"])
        sparsity = float(cand["pruning_sparsity"])
        solver = str(cand["solver_type"])
        ablate_cx = bool(cand["ablate_cx"])

        meta_path = resolve_meta_path(k, ablate_cx, base_dir=base_dir)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta_json = json.load(f)

        tag = f"pareto_{idx}_k{k}_s{int(sparsity*100)}_{solver.lower()}_{'nocx' if ablate_cx else 'cx'}"
        entry = {
            "model_id": tag,
            "trial_number": cand.get("trial_number", idx),
            "k_clusters": k,
            "pruning_sparsity": sparsity,
            "solver_type": solver,
            "ablate_cx": ablate_cx,
            "objectives": {
                "survival_time_s": round(float(cand.get("survival_time_s", 0.0)), 4),
                "energy_cost_j": round(float(cand.get("energy_cost_j", 0.0)), 4),
            },
            "files": {
                "meta_file": os.path.basename(meta_path),
                "w_file": meta_json.get("w_file", f"w_spectral_k{k}.npy"),
                "cmap_file": meta_json.get("cmap_file", f"cmap_spectral_k{k}.npy"),
            },
            "metrics": {
                "survival_ratio": round(float(cand.get("survival_ratio", 1.0)), 4),
                "steps_survived": int(cand.get("steps_survived", 0)),
                "mean_altitude_error": round(float(cand.get("mean_altitude_error", 0.0)), 4),
                "mean_tilt_error": round(float(cand.get("mean_tilt_error", 0.0)), 4),
                "mean_pwm_jitter": round(float(cand.get("mean_pwm_jitter", 0.0)), 4),
                "mean_power_w": round(float(cand.get("mean_power_w", 0.0)), 2),
                "sparsity_pct": round(sparsity * 100.0, 2),
                "spectral_rho": round(float(cand.get("approx_rho", meta_json.get("metrics", {}).get("spectral_rho", 1.0))), 4),
                "platform_hint": meta_json.get("platform_hint", ""),
            },
        }
        models_entries.append(entry)

    manifest_data = {
        "manifest_name": "Chong-Fly Pareto Filtered Models Manifest",
        "criteria": {
            "objective_1": "survival_time_s (maximize)",
            "objective_2": "energy_cost_j (minimize)",
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total_evaluated_models": total_eval,
        "pareto_models_count": len(models_entries),
        "models": models_entries,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    return manifest_data


def run_pareto_optimization(
    n_trials: int = 20,
    output_manifest: str = "data/filtered_models_manifest.json",
    study_name: str = "chong_fly_pareto_search",
    base_dir: str = "data/reduced_models",
    eval_steps: int = 100,
) -> tuple[optuna.Study, dict[str, Any]]:
    """
    Запускає багатокритеріальний пошук Optuna та експортує Парето-маніфест.
    """
    if not _OPTUNA_AVAILABLE:
        raise RuntimeError("Optuna is not installed.")

    study = optuna.create_study(
        study_name=study_name,
        directions=["maximize", "minimize"],
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    def _obj(trial: optuna.Trial) -> tuple[float, float]:
        return optuna_multiobjective(trial, base_dir=base_dir, eval_steps=eval_steps)

    study.optimize(_obj, n_trials=n_trials)
    manifest = export_pareto_manifest(study, output_path=output_manifest, base_dir=base_dir)
    return study, manifest


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Optuna Search Space 4.1 for Chong-Fly Autopilot")
    parser.add_argument("--n-trials", type=int, default=15, help="Number of Optuna trials")
    parser.add_argument("--study-name", type=str, default="chong_fly_search_4_1", help="Study name")
    parser.add_argument("--base-dir", type=str, default="data/reduced_models", help="Reduced models directory")
    parser.add_argument("--pareto", action="store_true", help="Run multi-objective Pareto optimization (survival vs energy)")
    parser.add_argument("--output-manifest", type=str, default="data/filtered_models_manifest.json", help="Path to export Pareto manifest")
    args = parser.parse_args()

    print("=" * 70)
    print("  Chong-Fly: Optuna Search Space 4.1 & Pareto Optimization")
    print("=" * 70)
    print(f"Search Space:")
    print(f"  • k_clusters       : {SEARCH_SPACE['k_clusters']}")
    print(f"  • pruning_sparsity : Float(0.50, 0.95)")
    print(f"  • solver_type      : {SEARCH_SPACE['solver_type']}")
    print(f"  • ablate_cx        : {SEARCH_SPACE['ablate_cx']}")
    print(f"Mode: {'Multi-Objective Pareto (Survival vs Energy)' if args.pareto else 'Single-Objective'}")
    print(f"Running {args.n_trials} trials...\n")

    if args.pareto:
        study, manifest = run_pareto_optimization(
            n_trials=args.n_trials,
            output_manifest=args.output_manifest,
            study_name=args.study_name,
            base_dir=args.base_dir,
        )
        print("\n" + "=" * 70)
        print("  Pareto Optimization Completed")
        print("=" * 70)
        print(f"Total trials evaluated: {len(study.trials)}")
        print(f"Pareto-optimal models : {manifest['pareto_models_count']}")
        print(f"Exported manifest     : {args.output_manifest}")
        for m in manifest["models"]:
            print(f"  • {m['model_id']}: survival={m['objectives']['survival_time_s']}s, energy={m['objectives']['energy_cost_j']}J")
    else:
        study = run_optuna_study(n_trials=args.n_trials, study_name=args.study_name, base_dir=args.base_dir)

        print("\n" + "=" * 70)
        print("  Optimization Completed")
        print("=" * 70)
        print(f"Total trials: {len(study.trials)}")
        complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        print(f"  Completed: {len(complete_trials)}")
        print(f"  Pruned   : {len(pruned_trials)}")

        if study.best_trial:
            print("\nBest Trial:")
            print(f"  Value: {study.best_value:.4f}")
            print("  Params:")
            for k, v in study.best_params.items():
                print(f"    {k}: {v}")

        # Also export manifest from completed trials
        manifest = export_pareto_manifest(study, output_path=args.output_manifest, base_dir=args.base_dir)
        print(f"\nExported Pareto manifest: {args.output_manifest} ({manifest['pareto_models_count']} Pareto models)")


if __name__ == "__main__":
    main()
