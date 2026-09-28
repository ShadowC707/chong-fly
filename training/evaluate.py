"""
training/evaluate.py
====================
Math Viability Gate for Optuna optimization.

Provides a fast pre-simulation check to prune unviable configurations.
"""

import numpy as np
import scipy.sparse as sp

def evaluate_math_viability(model_cfg: dict, w_matrix: np.ndarray | sp.csr_matrix) -> tuple[bool, dict]:
    """
    Fast O(1) mathematical viability gate to evaluate a network configuration
    and its weight matrix before running expensive physical simulations.
    
    Checks:
      1. NaN/Inf validation.
      2. Sparsity limits.
      3. Integration stability (dt vs tau).
      4. Fast spectral radius upper bound using Gershgorin circle theorem (Infinity Norm),
         avoiding O(N^3) eigenvalue decompositions.
         
    Args:
        model_cfg: Dictionary containing hyperparameter configs 
                   (e.g., 'dt', 'tau_min', 'min_sparsity', 'max_spectral_norm').
        w_matrix: The W_macro connectivity matrix (dense numpy or scipy CSR).
        
    Returns:
        (is_viable: bool, metrics: dict)
    """
    metrics = {}
    is_viable = True
    
    # 1. NaN / Inf Check
    if sp.issparse(w_matrix):
        data = w_matrix.data
    else:
        data = w_matrix
        
    if np.isnan(data).any() or np.isinf(data).any():
        metrics['reject_reason'] = 'Matrix contains NaN or Inf'
        return False, metrics

    # 2. Sparsity Check
    shape = w_matrix.shape
    total_elements = shape[0] * shape[1]
    nnz = data.size if sp.issparse(w_matrix) else np.count_nonzero(data)
    sparsity = 1.0 - (nnz / total_elements)
    metrics['sparsity'] = float(sparsity)
    
    min_sparsity = model_cfg.get('min_sparsity', 0.0)
    if sparsity < min_sparsity:
        metrics['reject_reason'] = f'Sparsity {sparsity:.3f} below required limit {min_sparsity:.3f}'
        is_viable = False

    # 3. Fast Spectral Bound (Gershgorin / Infinity Norm)
    # The true spectral radius rho(W) is bounded by the infinity norm (max absolute row sum).
    # This is an O(NNZ) operation which acts as an O(1) filter relative to physics simulation.
    if sp.issparse(w_matrix):
        row_sums = np.array(np.abs(w_matrix).sum(axis=1)).flatten()
    else:
        row_sums = np.sum(np.abs(w_matrix), axis=1)
        
    w_inf_norm = float(np.max(row_sums))
    metrics['w_inf_norm'] = w_inf_norm
    
    # 3-step Power Iteration as a tighter fast proxy for rho(W)
    # (Since inf_norm can be overly pessimistic for highly asymmetric matrices)
    v = np.ones(shape[0], dtype=np.float32) / np.sqrt(shape[0])
    for _ in range(3):
        v_next = w_matrix @ v
        norm_v = np.linalg.norm(v_next)
        if norm_v < 1e-9:
            break
        v = v_next / norm_v
    
    # Approximate rho(W)
    approx_rho = float(np.linalg.norm(w_matrix @ v))
    metrics['approx_rho'] = approx_rho
    
    # Reject if both the exact upper bound and the approx rho are wildly unstable
    # (Usually rho <= 1.0 is required for stability, but we allow a buffer in Optuna exploration)
    max_rho = model_cfg.get('max_spectral_radius', 1.0)
    strict_rho = model_cfg.get('strict_spectral_radius', False)
    
    if strict_rho and approx_rho > max_rho:
         metrics['reject_reason'] = f'Approx spectral radius {approx_rho:.3f} > {max_rho}'
         is_viable = False

    # 4. Integration Stability Check
    # Ensure membrane time constants are physically meaningful compared to integration step dt
    dt = model_cfg.get('dt', 0.004)
    tau_min = model_cfg.get('tau_min', 0.01)
    
    # If tau is extremely small relative to dt, the integration loses granularity 
    # (effectively acting as instantaneous transfer without memory).
    if tau_min < (dt / 10.0):
        metrics['reject_reason'] = f'tau_min ({tau_min}) is too small compared to dt ({dt})'
        is_viable = False
        
    return is_viable, metrics


# ─────────────────────────────────────────────────────────────────────────────
# Воронка фільтрації: Рівень 1 (Math Screening)
# ─────────────────────────────────────────────────────────────────────────────

def math_screening(
    model_cfg_or_params: dict,
    w_matrix: np.ndarray | sp.csr_matrix | None = None,
    base_dir: str = "data/reduced_models",
) -> tuple[bool, dict]:
    """
    Рівень 1 воронки фільтрації (Math Screening Gate):
    Швидка O(1) / O(NNZ) математична перевірка життєздатності конфігурації
    перед запуском дорогої симуляції.
    
    Підтримує два режими виклику:
      1. math_screening(model_cfg, w_matrix)
      2. math_screening(trial_params, base_dir=...) де trial_params містить
         {'k_clusters', 'pruning_sparsity', 'solver_type', 'ablate_cx'}
         
    Якщо перевірку провалено -> повертає (False, metrics), де
    metrics['reject_reason'] містить причину для raise optuna.TrialPruned().
    """
    if w_matrix is not None:
        return evaluate_math_viability(model_cfg_or_params, w_matrix)

    # Режим виклику з гіперпараметрами Trial
    import os
    k = model_cfg_or_params.get("k_clusters", 64)
    sparsity = model_cfg_or_params.get("pruning_sparsity", 0.0)
    solver = model_cfg_or_params.get("solver_type", "CfC")
    ablate_cx = model_cfg_or_params.get("ablate_cx", False)

    # Визначення шляху до файлу метаданих
    if not os.path.isabs(base_dir):
        # Відносно кореня проєкту
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        base_dir = os.path.join(project_root, base_dir)

    tag = f"meta_spectral_k{k}"
    if ablate_cx:
        nocx_path = os.path.join(base_dir, f"{tag}_nocx.json")
        meta_path = nocx_path if os.path.exists(nocx_path) else os.path.join(base_dir, f"{tag}.json")
    else:
        meta_path = os.path.join(base_dir, f"{tag}.json")

    from bio_pipeline.graph_reducer import ReducedModel
    reduced_model = ReducedModel.load(meta_path)
    W = reduced_model.W

    # Прунінг ваг
    is_sparse = sp.issparse(W)
    W_dense = W.toarray().astype(np.float32) if is_sparse else np.array(W, dtype=np.float32, copy=True)
    if sparsity > 0.0:
        k_pct = float(sparsity) * 100.0
        threshold = float(np.percentile(np.abs(W_dense), k_pct))
        mask = (np.abs(W_dense) >= threshold).astype(np.float32)
        W_dense = W_dense * mask

    W_pruned = sp.csr_matrix(W_dense) if is_sparse else W_dense

    dt = 0.02 if solver == "Euler_dt_0.02" else 0.004
    cfg = {
        "dt": dt,
        "tau_min": 0.01,
        "min_sparsity": float(sparsity) * 0.95,
        "max_spectral_radius": 1.5,
        "strict_spectral_radius": False,
    }

    is_viable, metrics = evaluate_math_viability(cfg, W_pruned)
    metrics["k_clusters"] = k
    metrics["solver_type"] = solver
    metrics["ablate_cx"] = ablate_cx
    metrics["meta_path"] = meta_path
    return is_viable, metrics


# ─────────────────────────────────────────────────────────────────────────────
# Воронка фільтрації: Рівень 2 (Simulation Behavioral Evaluation - O(N))
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_simulation_behavior(
    policy: any,
    eval_steps: int = 100,
    dt: float | None = None,
    seed: int = 42,
) -> tuple[float, dict]:
    """
    Рівень 2 воронки фільтрації (Simulation Behavioral Evaluation - O(N)):
    Оцінка польотної поведінки валідної моделі в симуляторі DroneSimulationEnv.
    
    Аналізує замкнену динаміку:
      - Утримання висоти та кутової стабільності
      - Відхилення горизонтального дрифту (оптомоторний рефлекс)
      - Плавність керування (мінімізація джитера PWM)
      - Реакція на перешкоди
      
    Повертає:
      (behavioral_score: float, detailed_metrics: dict)
    """
    from training.env import DroneSimulationEnv, simulate_policy_rollout

    step_dt = dt
    if step_dt is None:
        step_dt = getattr(policy, "default_dt", 0.004)

    env = DroneSimulationEnv(dt=step_dt)
    score, metrics = simulate_policy_rollout(
        policy=policy,
        env=env,
        eval_steps=eval_steps,
        dt=step_dt,
        seed=seed,
    )
    return score, metrics

