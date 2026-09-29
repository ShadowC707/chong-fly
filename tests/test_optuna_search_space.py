"""
tests/test_optuna_search_space.py
=================================
TDD Test Suite for Optuna Hyperparameter Optimization, Search Space & Evaluation:
1. create_model validation: raises FileNotFoundError on missing weights (no silent fallback).
2. create_model search space: tests categorical suggestion of available models [32, 64].
3. Policy integration dt propagation: solver_type="Euler_dt_0.02" propagates dt=0.02 to env.
4. Body-frame velocity projection: 90-degree yaw turn does not falsely trigger crab-flight.
5. Continuous obstacle evasion & respawn: active evasion increments walls_avoided and respawns obstacle.
6. Anti-hovering penalty: lazy hovering (zero walls avoided, low fwd speed) is heavily penalized.
7. Dynamic crash penalty: early crash penalty > late crash penalty, keyed on explicit is_crash.
8. Non-linear yaw chatter penalty: quadratic penalty for excessive continuous yaw oscillations.
"""

import math
import os
import sys
import numpy as np
import pytest
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from optimizer.evaluate import (
    create_model,
    compute_composite_cost,
    simulate_policy_rollout,
    DroneSimulationEnv,
    DefaultFlightPolicy,
)
from simulation.policy import ChongFlyMSPPolicy
from simulation.metrics import PhysicsTelemetryTracker, calculate_forward_ratio


# ─────────────────────────────────────────────────────────────────────────────
# 1. Search Space & Factory Validation
# ─────────────────────────────────────────────────────────────────────────────

def test_create_model_validates_weight_files_and_raises_error():
    """
    create_model MUST NOT silently fall back to DefaultFlightPolicy when
    the requested connectome matrix does not exist on disk.
    For k=16 (where w_spectral_k16.npy is missing), it must raise FileNotFoundError.
    """
    with pytest.raises(FileNotFoundError):
        create_model(
            {"k_clusters": 16, "pruning_sparsity": 0.60, "solver_type": "CfC", "ablate_cx": False},
            allow_fallback=False,
        )


def test_create_model_loads_existing_spectral_model():
    """
    create_model for k=32 and k=64 must successfully instantiate ChongFlyMSPPolicy.
    """
    policy_32 = create_model(
        {"k_clusters": 32, "pruning_sparsity": 0.60, "solver_type": "CfC", "ablate_cx": False},
        allow_fallback=False,
    )
    assert isinstance(policy_32, ChongFlyMSPPolicy)
    assert policy_32.cfc_network.cell.hidden_size == 32

    policy_64 = create_model(
        {"k_clusters": 64, "pruning_sparsity": 0.70, "solver_type": "CfC", "ablate_cx": False},
        allow_fallback=False,
    )
    assert isinstance(policy_64, ChongFlyMSPPolicy)
    assert policy_64.cfc_network.cell.hidden_size == 64


def test_create_model_optuna_trial_search_space():
    """
    When passed an Optuna Trial, suggest_categorical for k_clusters must only
    propose available models [32, 64] to prevent evaluating missing weights.
    """
    import optuna

    study = optuna.create_study(direction="minimize")
    trial = study.ask()

    # create_model should execute cleanly with trial
    policy = create_model(trial, allow_fallback=False)
    assert isinstance(policy, ChongFlyMSPPolicy)
    assert trial.params["k_clusters"] in [32, 64]
    assert 0.50 <= trial.params["pruning_sparsity"] <= 0.90
    assert trial.params["solver_type"] in ["CfC", "Euler_dt_0.02"]


def test_policy_dt_propagation_for_euler_solver():
    """
    When solver_type is 'Euler_dt_0.02', policy.default_dt must be 0.02,
    and simulate_policy_rollout must initialize env with dt=0.02.
    """
    policy = create_model(
        {"k_clusters": 32, "pruning_sparsity": 0.60, "solver_type": "Euler_dt_0.02", "ablate_cx": False},
        allow_fallback=False,
    )
    assert getattr(policy, "default_dt", None) == 0.02 or getattr(policy.cfc_network.cell, "default_dt", None) == 0.02

    # Verify rollout uses dt=0.02
    cost, metrics = simulate_policy_rollout(policy, eval_steps=10)
    # 10 steps at dt=0.02 = 0.20s
    assert np.isclose(metrics["survival_time_s"], 0.20, atol=1e-3)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Physics & Coordinate Frame Invariants (Crab Flight Fix)
# ─────────────────────────────────────────────────────────────────────────────

def test_body_frame_velocity_prevents_false_crab_flight_on_turn():
    """
    When the drone turns 90° (yaw = pi/2) and flies straight in its body heading,
    velocity in world frame is [0.0, 1.0, 0.0], but in body frame it is [1.0, 0.0, 0.0].
    The forward ratio in body frame MUST be 1.0, NOT 0.0 (no false crab flight).
    """
    psi = math.pi / 2.0  # 90 degrees
    vel_world = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    # Body frame forward and lateral velocities:
    v_fwd = vel_world[0] * math.cos(psi) + vel_world[1] * math.sin(psi)
    v_lat = -vel_world[0] * math.sin(psi) + vel_world[1] * math.cos(psi)

    assert np.isclose(v_fwd, 1.0, atol=1e-4)
    assert np.isclose(v_lat, 0.0, atol=1e-4)

    ratio_body = calculate_forward_ratio(v_fwd, v_lat)
    assert ratio_body is not None
    assert np.isclose(ratio_body, 1.0, atol=1e-3)

    # In contrast, raw world velocity gives 0.0 (false crab)
    ratio_world = calculate_forward_ratio(vel_world[0], vel_world[1])
    assert np.isclose(ratio_world, 0.0, atol=1e-3)


def test_rollout_does_not_flag_turning_drone_as_crab():
    """
    A policy that turns yaw to avoid a wall must not be flagged as is_crab_flight=True.
    """
    class TurningEvasivePolicy:
        def __init__(self):
            self.step_cnt = 0
            self.default_dt = 0.004

        def step_np(self, flow_xy, tof_8x8, memory_ring=None, dt=None):
            self.step_cnt += 1
            # Turn right by setting yaw=1800, forward pitch=1550
            return np.array([1500.0, 1500.0, 1550.0, 1800.0], dtype=np.float32)

        def reset_state(self):
            self.step_cnt = 0

    policy = TurningEvasivePolicy()
    cost, metrics = simulate_policy_rollout(policy, eval_steps=60, seed=42)

    assert not metrics.get("is_crab_flight", False), "Turning drone was falsely flagged as crab flight!"
    assert metrics.get("forward_ratio_median", 0.0) >= 0.50


# ─────────────────────────────────────────────────────────────────────────────
# 3. Obstacle Evasion, Respawn & Hovering Penalty
# ─────────────────────────────────────────────────────────────────────────────

def test_obstacle_evasion_and_respawn_during_rollout():
    """
    Tests that simulate_policy_rollout supports continuous obstacle encounters:
    When a drone turns to avoid an obstacle (< 0.8m), walls_avoided increments,
    and a new obstacle is spawned ahead.
    """
    class ExpertMockPolicy:
        def __init__(self):
            self.default_dt = 0.004

        def step_np(self, flow_xy, tof_8x8, memory_ring=None, dt=None):
            # Check center ToF distance
            center_tof = float(np.mean(tof_8x8.reshape(8, 8)[2:6, 2:6]))
            if center_tof < 0.8 / 3.0:
                # Brake and turn
                return np.array([1500.0, 1500.0, 1300.0, 1900.0], dtype=np.float32)
            else:
                # Cruise forward
                return np.array([1500.0, 1500.0, 1600.0, 1500.0], dtype=np.float32)

        def reset_state(self):
            pass

    policy = ExpertMockPolicy()
    cost, metrics = simulate_policy_rollout(policy, eval_steps=250, seed=42)

    assert metrics["walls_avoided"] >= 1, "Evasive policy should avoid at least 1 wall"
    assert not metrics.get("fatal_failure", False)


def test_anti_hovering_penalty_penalizes_zero_evasion():
    """
    A lazy hovering policy (ШІМ 1500) has walls_avoided == 0 and low mean forward speed,
    so compute_composite_cost must heavily penalize it compared to an active policy
    that survives and avoids walls.
    """
    # Hovering policy: survived 100%, but 0 walls avoided, mean fwd speed = 0.02 m/s
    cost_hover, _ = compute_composite_cost(
        survival_ratio=1.0,
        mean_power_w=15.0,
        jitter_pr=0.0,
        saccades_yaw=0,
        coverage_count=2,
        walls_avoided=0,
        mean_fwd_speed=0.02,
        is_crash=False,
    )

    # Active evasive policy: survived 100%, 2 walls avoided, mean fwd speed = 0.6 m/s, minor yaw saccades
    cost_active, _ = compute_composite_cost(
        survival_ratio=1.0,
        mean_power_w=20.0,
        jitter_pr=10.0,
        saccades_yaw=3,
        coverage_count=15,
        walls_avoided=2,
        mean_fwd_speed=0.60,
        is_crash=False,
    )

    assert cost_active < cost_hover, f"Active policy (cost={cost_active}) must beat lazy hover (cost={cost_hover})"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Crash Gradient & Non-linear Yaw Chatter Penalty
# ─────────────────────────────────────────────────────────────────────────────

def test_dynamic_crash_penalty_scales_smoothly():
    """
    A model crashing at 90% of simulation time must receive a significantly smaller
    penalty than a model crashing at 10% of simulation time.
    """
    cost_early_crash, _ = compute_composite_cost(
        survival_ratio=0.10,
        mean_power_w=15.0,
        jitter_pr=0.0,
        saccades_yaw=0,
        is_crash=True,
    )

    cost_late_crash, _ = compute_composite_cost(
        survival_ratio=0.90,
        mean_power_w=15.0,
        jitter_pr=0.0,
        saccades_yaw=0,
        is_crash=True,
    )

    assert cost_early_crash > cost_late_crash + 700.0, (
        f"Early crash ({cost_early_crash}) should be much worse than late crash ({cost_late_crash})"
    )


def test_nonlinear_yaw_chatter_penalty():
    """
    Isolated purposeful saccades (1-3) receive an agility bonus.
    Continuous high-frequency yaw chatter (> 4 saccades) triggers quadratic penalty.
    """
    # 2 saccades (e.g. 1 obstacle turn) -> rewarded
    cost_clean, _ = compute_composite_cost(
        survival_ratio=1.0, mean_power_w=15.0, jitter_pr=0.0, saccades_yaw=2, walls_avoided=1, mean_fwd_speed=0.5
    )
    cost_no_saccades, _ = compute_composite_cost(
        survival_ratio=1.0, mean_power_w=15.0, jitter_pr=0.0, saccades_yaw=0, walls_avoided=1, mean_fwd_speed=0.5
    )
    assert cost_clean < cost_no_saccades, "Purposeful saccades for evasion should receive a slight bonus"

    # 30 saccades (continuous 250 Hz vibration) -> heavy quadratic penalty
    cost_chatter, _ = compute_composite_cost(
        survival_ratio=1.0, mean_power_w=15.0, jitter_pr=0.0, saccades_yaw=30, walls_avoided=1, mean_fwd_speed=0.5
    )
    assert cost_chatter > cost_clean + 500.0, f"Chatter cost ({cost_chatter}) must heavily penalize continuous vibration"
