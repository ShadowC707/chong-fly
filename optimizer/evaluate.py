"""Model construction, pretraining, and versioned multi-seed flight evaluation."""
from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from configs.flight_config import (
    SENSOR_DIM, FLOW_DIM, TOF_DIM, MEMORY_DIM, N_CONTROLS,
    PWM_MIN, PWM_MAX, PWM_HOVER, MEMORY_DEFAULT_DISTANCE,
    DEFAULT_DT, CONTROL_DT, MAX_SIM_TIME_S,
)
from optimizer.pretrain import pretrain_policy
from generator.reflex_contract import DEFAULT_DATASET_PATH
from optimizer.rollout import simulate_policy_rollout, BENCHMARK_VERSION
from simulation.altitude_control import AltitudeHold
from simulation.control_contract import control_contract

class DefaultFlightPolicy(torch.nn.Module):
    """
    Agile 74-D recurrent flight policy using GRU temporal integration.
    Used for Optuna trials or when specific connectome ReducedModel artifacts
    are being synthesized.
    """

    def __init__(self, sensor_dim: int = SENSOR_DIM, hidden_dim: int = 32):
        super().__init__()
        self.sensor_dim = sensor_dim
        self.hidden_dim = hidden_dim
        self.fc_in = torch.nn.Linear(sensor_dim, hidden_dim)
        self.gru = torch.nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc_out = torch.nn.Linear(hidden_dim, N_CONTROLS)
        torch.nn.init.constant_(self.fc_out.bias, PWM_HOVER)
        self._hx: Optional[torch.Tensor] = None

    def reset_state(self) -> None:
        """Resets recurrent hidden state."""
        self._hx = None

    def forward(
        self,
        x: torch.Tensor,
        hx: Optional[torch.Tensor] = None,
        dt: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        squeeze_batch = False
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, 1, 74]
            squeeze_batch = True

        h_feat = torch.tanh(self.fc_in(x))
        out, h_last = self.gru(h_feat, hx)
        pwm = self.fc_out(out)
        pwm = torch.clamp(pwm, PWM_MIN, PWM_MAX)

        if squeeze_batch:
            return pwm.squeeze(1), h_last
        return pwm, h_last

    @torch.no_grad()
    def step_np(
        self,
        flow_xy: np.ndarray,
        tof_8x8: np.ndarray,
        memory_ring: Optional[np.ndarray] = None,
        dt: Optional[float] = None,
    ) -> np.ndarray:
        parts = [np.asarray(flow_xy).ravel()[:FLOW_DIM], np.asarray(tof_8x8).ravel()[:TOF_DIM]]
        if memory_ring is not None:
            parts.append(np.asarray(memory_ring).ravel()[:MEMORY_DIM])
        elif self.sensor_dim == SENSOR_DIM:
            parts.append(np.full(MEMORY_DIM, MEMORY_DEFAULT_DISTANCE, dtype=np.float32))

        raw = np.concatenate(parts).astype(np.float32)
        t_in = torch.from_numpy(raw).unsqueeze(0).unsqueeze(0).to(next(self.parameters()).device)  # [1, 1, 74]
        pwm_t, self._hx = self.forward(t_in, self._hx, dt=dt)
        return pwm_t.squeeze().cpu().numpy()

    def post_step(self) -> None:
        pass


def create_model(
    trial_or_params: Any = None,
    sensor_dim: int = SENSOR_DIM,
    base_dir: str = "data/reduced_models",
    allow_fallback: bool = False,
) -> Any:
    """
    Constructs a flight policy from an Optuna Trial or dictionary parameters.
    Attempts ChongFlyMSPPolicy.from_meta if metadata and weight matrices are available.
    If allow_fallback is False, raises FileNotFoundError if metadata or weights are missing.
    """
    params: Dict[str, Any] = {}
    if trial_or_params is not None:
        if hasattr(trial_or_params, "suggest_categorical"):
            params["reducer"] = trial_or_params.suggest_categorical("reducer", ["role_degree"])
            params["k_clusters"] = trial_or_params.suggest_categorical("k_clusters", [128, 256])
            # All-entry percentiles below the existing zero fraction remove no
            # edges. Keep pruning off until path-preserving pruning is evaluated.
            params["pruning_sparsity"] = trial_or_params.suggest_categorical("pruning_sparsity", [0.0])
            params["solver_type"] = trial_or_params.suggest_categorical("solver_type", ["exponential_euler"])
            params["ablate_cx"] = trial_or_params.suggest_categorical("ablate_cx", [False])
        elif isinstance(trial_or_params, dict):
            params = dict(trial_or_params)

    reducer = params.get("reducer", "role_degree")
    if reducer not in {"role_degree", "spectral", "centrality"}:
        raise ValueError(f"Unknown reducer: {reducer}")
    k = params.get("k_clusters", 128 if reducer == "role_degree" else 64)
    sparsity = params.get("pruning_sparsity", 0.0)
    solver = params.get("solver_type", "exponential_euler")
    connectivity = params.get("connectivity", "structured")
    ablate_cx = params.get("ablate_cx", False)

    _ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    suffix = "_nocx" if ablate_cx else ""
    meta_name = f"meta_{reducer}_k{k}{suffix}.json"
    meta_path = os.path.join(_ROOT_DIR, base_dir, meta_name)

    if not os.path.exists(meta_path):
        if not allow_fallback:
            hint = " Generate it with: python -m generator.role_reducer --k 128 256" if reducer == "role_degree" else ""
            raise FileNotFoundError(f"Connectome metadata file '{meta_path}' not found for k={k}, ablate_cx={ablate_cx}.{hint}")
        return DefaultFlightPolicy(sensor_dim=sensor_dim)

    # Валідація наявності бінарної матриці ваг на диску (без мовчазного проковтування)
    import json
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_dict = json.load(f)
    if meta_dict.get("k") != k or meta_dict.get("reducer") != reducer:
        raise ValueError("Metadata does not match the requested reducer and k")
    if reducer == "role_degree":
        from generator.role_reducer import REDUCTION_VERSION
        if meta_dict.get("provenance", {}).get("reduction_version") != REDUCTION_VERSION:
            raise ValueError("Incompatible role-preserving reduction contract; regenerate artifacts")
    w_file = meta_dict.get("w_file")
    w_path = os.path.join(os.path.dirname(meta_path), w_file) if w_file else ""
    if not os.path.exists(w_path):
        if not allow_fallback:
            raise FileNotFoundError(f"Weight matrix file '{w_path}' referenced in '{meta_name}' not found on disk")
        return DefaultFlightPolicy(sensor_dim=sensor_dim)

    try:
        from simulation.policy import ChongFlyMSPPolicy
        policy = ChongFlyMSPPolicy.from_meta(
            meta_path=meta_path,
            sensor_dim=sensor_dim,
            solver_type=solver,
            pruning_sparsity=sparsity,
            ablate_cx=ablate_cx,
            dt=CONTROL_DT,
            connectivity=connectivity,
        )
        if reducer == "role_degree" and policy.routing_diagnostics["motor_structural_rank"] < N_CONTROLS:
            raise ValueError("Source motor mapping cannot support four independent readout channels")
        provenance = meta_dict.get("provenance", {})
        policy.reduction_diagnostics = {"reducer": reducer, "k": k,
            **{key: provenance.get(key) for key in
               ("reduction_version", "source_kind", "source_sha256", "source_manifest_sha256",
                "acquisition", "polarity_contract", "minimum_k", "weight_scaling")}}
        return policy
    except Exception as e:
        if not allow_fallback:
            raise
        pass

    return DefaultFlightPolicy(sensor_dim=sensor_dim)


def objective(
    trial: Any = None,
    dataset_path: Any = DEFAULT_DATASET_PATH,
    pretrain: bool = True,
    pretrain_epochs: int = 15,
    subset_ratio: float = 0.7,
    seed: int = 42,
    device: str = "auto",
    eval_steps: Optional[int] = None,
) -> Tuple[float, float, float]:
    """Compare architectures under common training and evaluation random seeds.

    Objectives: stored tensor bytes, roll/pitch command jitter, visited voxels.
    Every seed must complete the horizon without a crash or invalid state/action.
    Full per-seed metrics are retained; safety is never averaged away.
    """
    torch.manual_seed(seed)
    policy = create_model(trial, sensor_dim=SENSOR_DIM, allow_fallback=False)
    target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        target_device = torch.device("cpu")
    policy = policy.to(target_device)
    if pretrain:
        policy = pretrain_policy(
            policy=policy, dataset_path=dataset_path, epochs=pretrain_epochs,
            subset_ratio=subset_ratio, seed=seed, device=target_device,
        )
    total_steps = int(round(MAX_SIM_TIME_S / CONTROL_DT)) if eval_steps is None else eval_steps
    seeds = [seed, seed + 100, seed + 200]
    scores, rollouts = [], []
    for scenario_seed in seeds:
        score, metrics = simulate_policy_rollout(
            policy=policy, eval_steps=total_steps, dt=CONTROL_DT, seed=scenario_seed,
            altitude_hold=AltitudeHold(),
        )
        if (metrics.get('benchmark_version') != BENCHMARK_VERSION
                or metrics.get('control_contract') != control_contract()):
            raise ValueError('Rollout benchmark/control contract differs from the navigation study')
        scores.append(score)
        rollouts.append({"seed": scenario_seed, **metrics})

    aggregate = {}
    for key in rollouts[0]:
        values = [r[key] for r in rollouts]
        if key != "seed" and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
            aggregate[key] = float(np.mean(values))
    feasible = all(r["feasible"] for r in rollouts)
    aggregate.update({
        "benchmark_version": BENCHMARK_VERSION,
        "control_contract": control_contract(),
        "feasible": feasible,
        "crashed": any(r["crashed"] for r in rollouts),
        "crash_rate": sum(r["crashed"] for r in rollouts)/len(rollouts),
        "fatal_failure": any(r["fatal_failure"] for r in rollouts),
        "is_crab_flight": any(r["is_crab_flight"] for r in rollouts),
        "min_clearance_m": min(r["min_clearance_m"] for r in rollouts),
        "min_survival_time_s": min(r["survival_time_s"] for r in rollouts),
        "constraints": [sum(not r["feasible"] for r in rollouts)/len(rollouts)],
        "training_seed": seed,
        "evaluation_seeds": seeds,
        "rollouts": rollouts,
    })
    cell = getattr(getattr(policy, "cfc_network", None), "cell", None)
    if cell is not None and hasattr(cell, "get_extra_state"):
        aggregate["neural_contract"] = cell.get_extra_state()
    if hasattr(policy, "routing_diagnostics"):
        aggregate["routing_diagnostics"] = policy.routing_diagnostics
    if hasattr(policy, "reduction_diagnostics"):
        aggregate["reduction_diagnostics"] = policy.reduction_diagnostics
    if pretrain and hasattr(policy, '_pretrain_info'):
        aggregate['pretrain_info'] = policy._pretrain_info
    if trial is not None and hasattr(trial, "set_user_attr"):
        for key, value in aggregate.items():
            trial.set_user_attr(key, value)
        if hasattr(trial, "set_constraint"):
            trial.set_constraint("flight_safety", aggregate["constraints"][0])
    # Deliberately no automatic "champion" promotion from a fabricated avoidance
    # count. A feasible trial still needs held-out evaluation and exact-weight export.
    return tuple(float(v) for v in np.mean(scores, axis=0))
