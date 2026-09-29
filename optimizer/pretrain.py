"""
optimizer/pretrain.py
=====================
Fast Behavioral Cloning Pre-training for Chong-Fly Policies.

Trains empty or randomly-initialized biological CfC policies to mimic expert
reflexes in 2-3 seconds prior to Optuna evaluation.

Key features:
- Bagging: randomly samples 70% of dataset episodes per trial for policy diversity.
- Recurrent sequence training: processes [batch, T, 74] -> [batch, T, 4] via BPTT.
- Normalized MSE Loss: calculates error on normalized [-1, 1] actuation targets.
- Gradient Clipping & Biological Sparsity: clips gradients (max_norm=1.0) and
  enforces W_macro synaptic masks via policy.post_step().
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def pretrain_policy(
    policy: Any,
    dataset_path: Union[str, Dict[str, Any]] = "data/reflex_dataset.pt",
    epochs: int = 3,
    subset_ratio: float = 0.7,
    lr: float = 0.005,
    batch_size: int = 16,
    seed: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> Any:
    """
    Rapidly pre-trains a policy using Behavioral Cloning on reflex demonstration sequences.

    Parameters
    ----------
    policy : ChongFlyMSPPolicy or PyTorch module supporting .forward(obs_seq).
    dataset_path : path to .pt dataset or pre-loaded dataset dict with 'X' and 'Y'.
    epochs : number of training epochs (default: 3).
    subset_ratio : fraction of demonstration episodes to sample via bagging (default: 0.7).
    lr : Adam learning rate (default: 0.005).
    batch_size : sequence batch size (default: 16).
    seed : optional RNG seed for reproducible bagging.
    device : torch device (default: cpu or policy device).

    Returns
    -------
    policy with updated reflex weights.
    """
    if policy is None or not isinstance(policy, torch.nn.Module):
        # Non-PyTorch dummy policy pass-through
        return policy

    # 1. Load dataset (with defensive fallbacks)
    data: Optional[Dict[str, Any]] = None
    if isinstance(dataset_path, dict):
        data = dataset_path
    elif isinstance(dataset_path, str):
        candidates = [
            dataset_path,
            os.path.join(_ROOT, dataset_path),
            os.path.join(_ROOT, "data", "reflex_dataset.pt"),
        ]
        for path in candidates:
            if os.path.exists(path):
                try:
                    data = torch.load(path, map_location="cpu", weights_only=False)
                    break
                except Exception:
                    pass

    if data is None or "X" not in data or "Y" not in data:
        # Fallback: synthesize a minimal dataset on the fly to prevent crashes
        from generator.generate_reflex_dataset import generate_reflex_dataset
        data = generate_reflex_dataset(num_episodes=5, seq_len=50, noise_std_pwm=10.0, seed=42)

    X_all = data["X"]  # [N, T, 74]
    Y_all = data["Y"]  # [N, T, 4]

    num_total = X_all.shape[0]
    if num_total == 0:
        return policy

    # 2. Bagging: sample a random subset of episodes (e.g. 70%)
    rng = np.random.default_rng(seed)
    num_subset = max(1, int(round(num_total * float(subset_ratio))))
    indices = rng.choice(num_total, size=num_subset, replace=False)

    X_sub = X_all[indices]
    Y_sub = Y_all[indices]

    # Resolve target device
    if device is None:
        try:
            device = next(policy.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    X_sub = X_sub.to(device)
    Y_sub = Y_sub.to(device)

    # 3. Setup optimizer and training mode
    policy.train()
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    loss_history: List[float] = []

    # 4. Fast training loop over sequences
    for epoch in range(epochs):
        perm = rng.permutation(num_subset)
        epoch_losses: List[float] = []

        for start_idx in range(0, num_subset, batch_size):
            batch_indices = perm[start_idx : start_idx + batch_size]
            X_batch = X_sub[batch_indices]  # [B, T, 74]
            Y_batch = Y_sub[batch_indices]  # [B, T, 4]

            optimizer.zero_grad()

            # Forward pass: policy.forward(X_batch) returns (pwm, h_last)
            out = policy.forward(X_batch)
            pwm_pred = out[0] if isinstance(out, tuple) else out  # [B, T, 4]

            # Normalized MSE Loss on [-1, 1] range:
            # PWM in [1000, 2000] -> (PWM - 1500) / 500
            pred_norm = (pwm_pred - 1500.0) / 500.0
            target_norm = (Y_batch - 1500.0) / 500.0

            loss = F.mse_loss(pred_norm, target_norm)

            if loss.requires_grad:
                loss.backward()
                # Stability: gradient clipping
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
                optimizer.step()

                # Enforce biological synapse connectivity mask on W_macro
                if hasattr(policy, "post_step") and callable(policy.post_step):
                    policy.post_step()


            epoch_losses.append(float(loss.item()))

        if epoch_losses:
            loss_history.append(float(np.mean(epoch_losses)))

    policy.eval()

    # Store metadata for introspection / debugging
    setattr(
        policy,
        "_pretrain_info",
        {
            "epochs": epochs,
            "subset_size": num_subset,
            "subset_indices": indices.tolist(),
            "loss_history": loss_history,
            "final_loss": loss_history[-1] if loss_history else None,
        },
    )


    return policy