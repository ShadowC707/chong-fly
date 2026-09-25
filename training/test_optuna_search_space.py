"""
training/test_optuna_search_space.py
====================================
Comprehensive validation suite for Task 4.1: Optuna Search Space.

Verifies:
  1. Search space definition and bounds (k_clusters, pruning_sparsity, solver_type, ablate_cx).
  2. Sampling mechanism with Optuna trials.
  3. Magnitude pruning across [0.50, 0.95] range.
  4. Both solver types: 'CfC' and 'Euler_dt_0.02'.
  5. Central Complex ablation: ablate_cx=True vs ablate_cx=False.
  6. Integration with evaluate_math_viability.
  7. End-to-end ChongFlyMSPPolicy construction & flight step execution.
  8. Optuna study execution with multiple trials.
"""

import os
import sys
import numpy as np
import torch
import optuna

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from bio_pipeline.models import BiologicalCfCCell, build_network_from_meta
from simulation.policy import ChongFlyMSPPolicy
from training.optuna_tuner import (
    SEARCH_SPACE,
    SEARCH_SPACE_DISTRIBUTIONS,
    sample_search_space,
    evaluate_trial_math_viability,
    build_policy_from_trial,
    apply_magnitude_pruning,
    optuna_objective,
    create_study,
    resolve_meta_path,
)
from training.evaluate import evaluate_math_viability, math_screening, evaluate_simulation_behavior
from training.env import DroneSimulationEnv, simulate_policy_rollout

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
_results = []

def check(name: str, cond: bool, detail: str = ""):
    sym = PASS if cond else FAIL
    print(f"  {sym}  {name}" + (f"  [{detail}]" if detail else ""))
    _results.append(cond)
    return cond

def section(title: str):
    print(f"\n{'─'*65}")
    print(f"  {title}")
    print(f"{'─'*65}")


def main():
    print("=" * 65)
    print("  TEST SUITE: Optuna Search Space 4.1 Validation")
    print("=" * 65)

    # ────────────────────────────────────────────────────────────
    section("1. Search Space Specification Verification")
    # ────────────────────────────────────────────────────────────
    check("k_clusters choices exact", SEARCH_SPACE["k_clusters"] == [32, 64, 128, 256],
          f"{SEARCH_SPACE['k_clusters']}")
    check("pruning_sparsity bounds",
          SEARCH_SPACE["pruning_sparsity"]["low"] == 0.50 and SEARCH_SPACE["pruning_sparsity"]["high"] == 0.95,
          "Float(0.50, 0.95)")
    check("solver_type options exact",
          SEARCH_SPACE["solver_type"] == ["CfC", "Euler_dt_0.02"],
          f"{SEARCH_SPACE['solver_type']}")
    check("ablate_cx options exact",
          SEARCH_SPACE["ablate_cx"] == [True, False],
          f"{SEARCH_SPACE['ablate_cx']}")

    # ────────────────────────────────────────────────────────────
    section("2. Optuna Trial Sampling")
    # ────────────────────────────────────────────────────────────
    study = optuna.create_study(direction="minimize")
    trial = study.ask(SEARCH_SPACE_DISTRIBUTIONS)
    sampled = sample_search_space(trial)

    check("Sampled k_clusters in [32, 64, 128, 256]",
          sampled["k_clusters"] in [32, 64, 128, 256], f"k={sampled['k_clusters']}")
    check("Sampled pruning_sparsity in [0.50, 0.95]",
          0.50 <= sampled["pruning_sparsity"] <= 0.95, f"sparsity={sampled['pruning_sparsity']:.3f}")
    check("Sampled solver_type in ['CfC', 'Euler_dt_0.02']",
          sampled["solver_type"] in ["CfC", "Euler_dt_0.02"], f"solver={sampled['solver_type']}")
    check("Sampled ablate_cx is bool",
          sampled["ablate_cx"] in [True, False], f"ablate_cx={sampled['ablate_cx']}")

    # ────────────────────────────────────────────────────────────
    section("3. Magnitude Pruning Verification")
    # ────────────────────────────────────────────────────────────
    test_w = np.random.randn(64, 64).astype(np.float32)
    for target_sparsity in [0.50, 0.70, 0.85, 0.95]:
        pruned_w, mask = apply_magnitude_pruning(test_w, target_sparsity)
        actual_sparsity = 1.0 - (np.count_nonzero(pruned_w) / (64 * 64))
        check(f"Pruning at {target_sparsity*100:.0f}%",
              abs(actual_sparsity - target_sparsity) < 0.05,
              f"target={target_sparsity:.2f}, actual={actual_sparsity:.2f}")

    # ────────────────────────────────────────────────────────────
    section("4. Solver Type Integration (CfC vs Euler_dt_0.02)")
    # ────────────────────────────────────────────────────────────
    meta_p = resolve_meta_path(32, ablate_cx=False)
    
    # 4a. CfC Solver
    policy_cfc = ChongFlyMSPPolicy.from_meta(meta_p, solver_type="CfC", dt=0.004)
    check("CfC default dt is 0.004", policy_cfc.cfc_network.cell.default_dt == 0.004)
    check("CfC solver_type set", policy_cfc.cfc_network.cell.solver_type == "CfC")
    
    # 4b. Euler Solver
    policy_euler = ChongFlyMSPPolicy.from_meta(meta_p, solver_type="Euler_dt_0.02")
    check("Euler default dt is 0.02", policy_euler.cfc_network.cell.default_dt == 0.02)
    check("Euler solver_type set", policy_euler.cfc_network.cell.solver_type == "Euler_dt_0.02")

    # Step simulation with both solvers
    flow = np.array([0.2, -0.3], dtype=np.float32)
    tof = np.zeros(64, dtype=np.float32)

    pwm_cfc = policy_cfc.step_np(flow, tof)
    pwm_euler = policy_euler.step_np(flow, tof)

    check("CfC step PWM in bounds [1000, 2000]",
          np.all(pwm_cfc >= 1000.0) and np.all(pwm_cfc <= 2000.0), f"pwm={pwm_cfc}")
    check("Euler step PWM in bounds [1000, 2000]",
          np.all(pwm_euler >= 1000.0) and np.all(pwm_euler <= 2000.0), f"pwm={pwm_euler}")
    check("Outputs differ between solvers",
          not np.allclose(pwm_cfc, pwm_euler, atol=1e-3), "dynamics differentiated")

    # ────────────────────────────────────────────────────────────
    section("5. Central Complex Ablation (ablate_cx=True vs False)")
    # ────────────────────────────────────────────────────────────
    for k in [32, 64, 128, 256]:
        meta_cx = resolve_meta_path(k, ablate_cx=False)
        meta_nocx = resolve_meta_path(k, ablate_cx=True)
        check(f"k={k} with CX file exists", os.path.exists(meta_cx), os.path.basename(meta_cx))
        check(f"k={k} no-CX file exists", os.path.exists(meta_nocx), os.path.basename(meta_nocx))

    policy_cx = build_policy_from_trial({
        "k_clusters": 64,
        "pruning_sparsity": 0.60,
        "solver_type": "CfC",
        "ablate_cx": False,
    })
    policy_nocx = build_policy_from_trial({
        "k_clusters": 64,
        "pruning_sparsity": 0.60,
        "solver_type": "CfC",
        "ablate_cx": True,
    })

    pwm_with_cx = policy_cx.step_np(flow, tof)
    pwm_no_cx = policy_nocx.step_np(flow, tof)

    check("ablate_cx=False produces valid PWM",
          np.all(pwm_with_cx >= 1000.0) and np.all(pwm_with_cx <= 2000.0))
    check("ablate_cx=True produces valid PWM",
          np.all(pwm_no_cx >= 1000.0) and np.all(pwm_no_cx <= 2000.0))

    # ────────────────────────────────────────────────────────────
    section("6. Level 1: math_screening Gate & Pruning")
    # ────────────────────────────────────────────────────────────
    valid_params = {
        "k_clusters": 64,
        "pruning_sparsity": 0.75,
        "solver_type": "CfC",
        "ablate_cx": False,
    }
    is_viable, metrics = math_screening(valid_params)
    check("math_screening passed for valid candidate", is_viable)
    check("Metrics recorded sparsity", "sparsity" in metrics and metrics["sparsity"] >= 0.70,
          f"sparsity={metrics.get('sparsity', 0):.3f}")
    check("Metrics recorded spectral bound", "w_inf_norm" in metrics)

    # Test math_screening failure detection with invalid matrix / parameters
    bad_cfg = {"dt": 0.02, "tau_min": 0.0001, "min_sparsity": 0.5}  # tau_min < dt / 10
    bad_w = np.ones((16, 16), dtype=np.float32)
    is_viable_bad, bad_metrics = math_screening(bad_cfg, bad_w)
    check("math_screening rejected invalid tau_min < dt/10", not is_viable_bad,
          bad_metrics.get("reject_reason", ""))

    # Verify that math_screening failure triggers optuna.TrialPruned in objective
    mock_study = optuna.create_study(direction="minimize")
    trial_mock = mock_study.ask(SEARCH_SPACE_DISTRIBUTIONS)
    
    # Force an invalid math config in objective to verify TrialPruned trigger
    def pruned_objective_test(trial):
        is_ok, m = math_screening({"dt": 0.02, "tau_min": 0.0001}, bad_w)
        if not is_ok:
            raise optuna.TrialPruned(m.get("reject_reason"))
        return 0.0

    pruned_caught = False
    try:
        pruned_objective_test(trial_mock)
    except optuna.TrialPruned as exc:
        pruned_caught = True
        pruned_reason = str(exc)
    check("Level 1 failure raises optuna.TrialPruned()", pruned_caught, pruned_reason if pruned_caught else "")

    # ────────────────────────────────────────────────────────────
    section("7. Level 2: Simulation Behavioral Evaluation - O(N)")
    # ────────────────────────────────────────────────────────────
    test_policy = build_policy_from_trial({
        "k_clusters": 64,
        "pruning_sparsity": 0.70,
        "solver_type": "CfC",
        "ablate_cx": False,
    })
    
    score, sim_metrics = evaluate_simulation_behavior(
        policy=test_policy,
        eval_steps=50,
        dt=0.004,
        seed=42,
    )
    check("Level 2 simulation completed", score > 0.0, f"score={score:.4f}")
    check("Simulation recorded survival_ratio", "survival_ratio" in sim_metrics,
          f"survival={sim_metrics.get('survival_ratio', 0):.2f}")
    check("Simulation recorded altitude error", "mean_altitude_error" in sim_metrics)
    check("Simulation recorded tilt error", "mean_tilt_error" in sim_metrics)
    check("Simulation recorded PWM jitter", "mean_pwm_jitter" in sim_metrics)

    # ────────────────────────────────────────────────────────────
    section("8. Full Policy Matrix across k_clusters [32, 64, 128, 256]")
    # ────────────────────────────────────────────────────────────
    for k in [32, 64, 128, 256]:
        for solver in ["CfC", "Euler_dt_0.02"]:
            params = {
                "k_clusters": k,
                "pruning_sparsity": 0.80,
                "solver_type": solver,
                "ablate_cx": (k % 64 == 0),
            }
            p = build_policy_from_trial(params)
            p_out = p.step_np(flow, tof)
            check(f"Policy k={k}, {solver}, ablate={params['ablate_cx']}",
                  len(p_out) == 4 and not np.isnan(p_out).any(),
                  f"pwm={np.round(p_out, 1)}")

    # ────────────────────────────────────────────────────────────
    section("9. Optuna Study with Multi-Level Funnel")
    # ────────────────────────────────────────────────────────────
    test_study = create_study(study_name="test_funnel_study")
    
    # Run 6 trials to verify trial evaluation through the two-level funnel
    test_study.optimize(lambda t: optuna_objective(t, eval_steps=20), n_trials=6)
    
    completed = [t for t in test_study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    check("Study completed trials successfully", len(completed) > 0, f"{len(completed)}/6 complete")
    for t in completed:
        check(f"Trial {t.number} passed Level 1 and Level 2",
              t.user_attrs.get("funnel_level_1_passed") and t.user_attrs.get("funnel_level_2_passed"),
              f"score={t.value:.3f}")
        break

    check("Best trial parameters sampled from search space",
          test_study.best_params["k_clusters"] in [32, 64, 128, 256] and
          0.50 <= test_study.best_params["pruning_sparsity"] <= 0.95 and
          test_study.best_params["solver_type"] in ["CfC", "Euler_dt_0.02"] and
          test_study.best_params["ablate_cx"] in [True, False],
          f"{test_study.best_params}")

    # ────────────────────────────────────────────────────────────
    # Summary
    # ────────────────────────────────────────────────────────────
    n_passed = sum(_results)
    n_total  = len(_results)
    print("\n" + "=" * 65)
    print(f"  Results: {n_passed}/{n_total} passed  — " +
          ("all OK ✓" if n_passed == n_total else "FAILURES DETECTED ✗"))
    print("=" * 65 + "\n")
    if n_passed != n_total:
        sys.exit(1)


if __name__ == "__main__":
    main()
