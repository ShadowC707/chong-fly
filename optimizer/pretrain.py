"""
optimizer/pretrain.py
=====================
Behavioral cloning with versioned data, valid-prefix masks, scenario-stratified
episode splits, behavior-balanced training loss and unweighted held-out metrics.
Rejects teacher routes absent from a structured graph before changing parameters.
"""

from __future__ import annotations

import os
import inspect
import hashlib
import json
from collections import Counter
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from configs.flight_config import PWM_MID, PWM_HALF, PWM_MIN, PWM_MAX, SENSOR_DIM, COORDINATE_VERSION, TOF_RAYCASTER_MAX_RANGE_M
from generator.reflex_contract import DATASET_VERSION, TEACHER_VERSION, DEFAULT_DATASET_PATH, REQUIRED_PATHS
from simulation.control_contract import control_contract, NAVIGATION_INDICES, NAVIGATION_CHANNELS


def pretrain_policy(
    policy: Any,
    dataset_path: Union[str, os.PathLike, Dict[str, Any]] = DEFAULT_DATASET_PATH,
    epochs: int = 3,
    subset_ratio: float = 0.7,
    lr: float = 0.005,
    batch_size: int = 16,
    seed: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    validation_fraction: float = 0.2,
) -> Any:
    """
    Rapidly pre-trains a policy using Behavioral Cloning on reflex demonstration sequences.

    Parameters
    ----------
    policy : ChongFlyMSPPolicy or PyTorch module supporting .forward(obs_seq).
    dataset_path : path to .pt dataset or pre-loaded dataset dict with 'X' and 'Y'.
    epochs : number of training epochs (default: 3).
    subset_ratio : fraction of training episodes to sample; retain each scenario.
    lr : Adam learning rate (default: 0.005).
    batch_size : sequence batch size (default: 16).
    seed : optional RNG seed for reproducible splitting/subsampling.
    validation_fraction : held-out fraction within each scenario (default: 0.2).
    device : torch device (default: cpu or policy device).

    Returns
    -------
    policy with updated reflex weights.
    """
    if policy is None or not isinstance(policy, torch.nn.Module):
        # Non-PyTorch dummy policy pass-through
        return policy

    # A requested dataset is part of the experiment, never an optional hint.
    if isinstance(dataset_path, dict):
        data = dataset_path
    else:
        path = os.fspath(dataset_path)
        if not os.path.isabs(path) and not os.path.isfile(path):
            path = os.path.join(_ROOT, path)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Requested reflex dataset does not exist: {path}")
        data = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(data, dict) or "X" not in data or "Y" not in data:
        raise ValueError("Dataset must contain X and Y tensors")
    metadata = data.get('metadata', {})
    if (metadata.get('dataset_version') != DATASET_VERSION
            or metadata.get('teacher_version') != TEACHER_VERSION):
        raise ValueError('Incompatible reflex dataset/teacher version; regenerate with generator.generate_reflex_dataset')
    if (metadata.get('coordinate_version') != COORDINATE_VERSION
            or metadata.get('tof_max_range_m') != TOF_RAYCASTER_MAX_RANGE_M):
        raise ValueError('Dataset coordinate or ToF range contract does not match the simulator')
    if metadata.get('control_contract') != control_contract():
        raise ValueError('Dataset control contract differs from navigation/altitude ownership')
    if metadata.get('required_sensor_motor_paths') != REQUIRED_PATHS:
        raise ValueError('Dataset required routes contract is missing or incompatible')

    X_all = data["X"]  # [N, T, 74]
    Y_all = data["Y"]  # [N, T, 4]

    sensor_dim = getattr(policy, "sensor_dim", SENSOR_DIM)
    if not isinstance(X_all, torch.Tensor) or not isinstance(Y_all, torch.Tensor):
        raise ValueError("X and Y must be torch tensors")
    if (X_all.ndim != 3 or Y_all.ndim != 3 or X_all.shape[-1] != sensor_dim
            or Y_all.shape[-1] != 4 or X_all.shape[:2] != Y_all.shape[:2]
            or X_all.shape[0] == 0 or X_all.shape[1] == 0):
        raise ValueError("Expected nonempty aligned X[N,T,sensor_dim] and Y[N,T,4]")
    if not torch.isfinite(X_all).all() or not torch.isfinite(Y_all).all():
        raise ValueError("Dataset contains non-finite values")
    if not X_all.is_floating_point() or not Y_all.is_floating_point():
        raise ValueError("Dataset tensors must have floating-point dtype")
    if torch.any(Y_all < PWM_MIN) or torch.any(Y_all > PWM_MAX):
        raise ValueError("Target PWM is outside [1000, 2000]")
    applied = data.get('applied_actions')
    if (not isinstance(applied, torch.Tensor) or applied.shape != Y_all.shape
            or not applied.is_floating_point() or not torch.isfinite(applied).all()
            or torch.any(applied < PWM_MIN) or torch.any(applied > PWM_MAX)):
        raise ValueError('Dataset requires finite aligned applied_actions in PWM bounds')
    valid, behavior = data.get('valid'), data.get('behavior')
    if (not isinstance(valid, torch.Tensor) or valid.dtype != torch.bool
            or valid.shape != X_all.shape[:2] or not valid[:, 0].all()
            or torch.any(valid[:, 1:] & ~valid[:, :-1])):
        raise ValueError('valid must mark a nonempty contiguous prefix in every sequence')
    names = metadata.get('behavior_names', [])
    if (not isinstance(names, list) or not names or len(set(names)) != len(names)
            or not isinstance(behavior, torch.Tensor) or behavior.dtype != torch.long
            or behavior.shape != valid.shape or torch.any(behavior[valid] < 0)
            or torch.any(behavior[valid] >= len(names)) or torch.any(behavior[~valid] != -1)):
        raise ValueError('Invalid behavior labels for valid/padded frames')
    scenarios = metadata.get('episode_scenarios', [])
    if (not isinstance(scenarios, list) or len(scenarios) != len(X_all)
            or any(not isinstance(s, str) or not s for s in scenarios)):
        raise ValueError('Dataset requires one scenario label per episode')
    if not np.isfinite(validation_fraction) or not 0 <= validation_fraction < 1:
        raise ValueError('validation_fraction must be in [0, 1)')
    diagnostics = getattr(policy, 'routing_diagnostics', {})
    if diagnostics.get('connectivity') == 'structured':
        paths = diagnostics.get('sensor_motor_paths', {})
        for group, channels in REQUIRED_PATHS.items():
            for channel in channels:
                if paths.get(group, {}).get('minimum_hops', {}).get(channel) is None:
                    raise ValueError(f'Teacher requires missing directed route {group} -> {channel}; '
                                     'resolve graph/mapping before pretraining')
    try:
        dataset_dt = float(data["metadata"]["dt"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Dataset requires metadata.dt in seconds") from exc
    if not np.isfinite(dataset_dt) or dataset_dt <= 0:
        raise ValueError("Dataset dt must be positive and finite")
    if not np.isfinite(subset_ratio) or not 0 < subset_ratio <= 1:
        raise ValueError("subset_ratio must be in (0, 1]")
    if not isinstance(batch_size, int) or batch_size <= 0 or not isinstance(epochs, int) or epochs < 0:
        raise ValueError("batch_size must be positive and epochs nonnegative integers")
    signature = inspect.signature(policy.forward)
    accepts_dt = "dt" in signature.parameters or any(
        p.kind == p.VAR_KEYWORD for p in signature.parameters.values())
    forward_kwargs = {"dt": dataset_dt} if accepts_dt else {}
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True, allow_nan=False).encode())
    for tensor in (X_all, Y_all, applied, valid, behavior):
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    dataset_sha256 = digest.hexdigest()

    # Split entire episodes within each scenario, never adjacent time windows.
    rng = np.random.default_rng(seed)
    train_groups, validation_indices = [], []
    for scenario in sorted(set(scenarios)):
        members = rng.permutation([i for i, label in enumerate(scenarios) if label == scenario])
        n_val = min(len(members)-1, max(1, int(round(len(members)*validation_fraction)))) if validation_fraction else 0
        validation_indices.extend(members[:n_val].tolist())
        train_groups.append(members[n_val:])
    training_pool = np.concatenate(train_groups)
    num_subset = max(len(train_groups), int(round(len(training_pool)*subset_ratio)))
    # At least one sequence of every available scenario survives subsampling.
    selected = [int(rng.choice(group)) for group in train_groups]
    remaining = np.setdiff1d(training_pool, selected)
    selected.extend(rng.choice(remaining, size=num_subset-len(selected), replace=False).tolist())
    indices = np.asarray(selected)

    X_sub = X_all[indices]
    Y_sub = Y_all[indices]
    valid_sub = valid[indices]
    labels_sub = behavior[indices]
    counts = torch.bincount(labels_sub[valid_sub], minlength=len(names)).float()
    class_weights = torch.zeros_like(counts)
    present = counts > 0
    class_weights[present] = counts.sum() / (present.sum()*counts[present])
    frame_weights = class_weights[labels_sub.clamp_min(0)] * valid_sub

    # Resolve target device
    if device is None:
        try:
            device = next(policy.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    X_sub = X_sub.to(device)
    Y_sub = Y_sub.to(device)
    frame_weights = frame_weights.to(device)

    # 3. Setup optimizer and training mode
    policy.train()
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    loss_history: List[float] = []
    validation_history = []
    validation_mae = None
    validation_behavior_mae = {}

    # 4. Fast training loop over sequences
    for epoch in range(epochs):
        policy.train()
        perm = rng.permutation(num_subset)
        loss_numerator, loss_denominator = 0., 0.

        for start_idx in range(0, num_subset, batch_size):
            batch_indices = perm[start_idx : start_idx + batch_size]
            X_batch = X_sub[batch_indices]  # [B, T, 74]
            Y_batch = Y_sub[batch_indices]  # [B, T, 4]

            optimizer.zero_grad()

            # Forward pass: policy.forward(X_batch) returns (pwm, h_last)
            out = policy(X_batch, **forward_kwargs)
            pwm_pred = out[0] if isinstance(out, tuple) else out  # [B, T, 4]
            if pwm_pred.shape != Y_batch.shape or not torch.isfinite(pwm_pred).all():
                raise ValueError("Policy produced malformed or non-finite training outputs")

            # Normalized MSE Loss on [-1, 1] range:
            # The policy cannot observe altitude telemetry. Never teach it the
            # supervisor's throttle; retain only navigation channels in the loss.
            pred_norm = (pwm_pred[..., list(NAVIGATION_INDICES)] - PWM_MID) / PWM_HALF
            target_norm = (Y_batch[..., list(NAVIGATION_INDICES)] - PWM_MID) / PWM_HALF

            weights = frame_weights[batch_indices]
            errors = (pred_norm - target_norm).square().mean(dim=-1)
            loss = (errors * weights).sum() / weights.sum()

            if loss.requires_grad:
                loss.backward()
                # Stability: gradient clipping
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0, error_if_nonfinite=True)
                optimizer.step()

                # Enforce biological synapse connectivity mask on W_macro
                if hasattr(policy, "post_step") and callable(policy.post_step):
                    policy.post_step()


            loss_numerator += float((errors.detach()*weights).sum())
            loss_denominator += float(weights.sum())

        loss_history.append(loss_numerator / loss_denominator)
        if validation_indices:
            policy.eval()
            val_squared, val_count = 0., 0
            val_abs = torch.zeros(len(NAVIGATION_INDICES), device=device)
            by_behavior = {name: [0., 0] for name in names}
            with torch.no_grad():
                for start in range(0, len(validation_indices), batch_size):
                    ids = validation_indices[start:start+batch_size]
                    out = policy(X_all[ids].to(device), **forward_kwargs)
                    pred = out[0] if isinstance(out, tuple) else out
                    target, mask = Y_all[ids].to(device), valid[ids].to(device)
                    if pred.shape != target.shape or not torch.isfinite(pred).all():
                        raise ValueError('Invalid validation outputs')
                    error = (pred - target)[..., list(NAVIGATION_INDICES)]
                    val_squared += float((error[mask]/PWM_HALF).square().sum())
                    val_count += int(mask.sum())
                    val_abs += error[mask].abs().sum(dim=0)
                    labels = behavior[ids].to(device)
                    for label, name in enumerate(names):
                        selected_mask = mask & (labels == label)
                        by_behavior[name][0] += float(error[selected_mask].abs().sum())
                        by_behavior[name][1] += int(selected_mask.sum())*len(NAVIGATION_INDICES)
            validation_history.append(val_squared/(val_count*len(NAVIGATION_INDICES)))
            validation_mae = (val_abs/val_count).cpu().tolist()
            validation_behavior_mae = {name: total/count for name, (total, count) in by_behavior.items() if count}

    policy.eval()

    if len(loss_history) > 1:
        print(f"    [BC Pretrain] Initial Loss: {loss_history[0]:.4f} | Final Loss: {loss_history[-1]:.4f}")
    elif len(loss_history) == 1:
        print(f"    [BC Pretrain] Final Loss: {loss_history[0]:.4f}")

    # Store metadata for introspection / debugging
    setattr(
        policy,
        "_pretrain_info",
        {
            "dt": dataset_dt,
            "epochs": epochs,
            "subset_size": num_subset,
            "subset_indices": indices.tolist(),
            "validation_indices": validation_indices,
            "subset_scenario_counts": dict(Counter(scenarios[i] for i in indices)),
            "training_behavior_counts": {name: int(counts[i]) for i, name in enumerate(names)},
            "validation_loss_history": validation_history,
            "validation_channel_mae_pwm": validation_mae,
            "validation_behavior_mae_pwm": validation_behavior_mae,
            "learned_channels": list(NAVIGATION_CHANNELS),
            "control_contract": control_contract(),
            "dataset_version": DATASET_VERSION,
            "teacher_version": TEACHER_VERSION,
            "coordinate_version": COORDINATE_VERSION,
            "dataset_sha256": dataset_sha256,
            "loss_history": loss_history,
            "final_loss": loss_history[-1] if loss_history else None,
        },
    )


    return policy
