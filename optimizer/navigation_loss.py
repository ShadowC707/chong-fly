"""Train-only channel weighting with a fixed, partition-independent normalizer."""
import math
import torch

from configs.flight_config import PWM_MID, PWM_HALF
from simulation.control_contract import NAVIGATION_INDICES


def _balanced(labels, valid, groups):
    counts = torch.bincount(labels[valid], minlength=groups).to(torch.float32)
    present = counts > 0
    scale = torch.zeros_like(counts)
    scale[present] = counts.sum() / (present.sum()*counts[present])
    return scale[labels.clamp_min(0)] * valid


def yaw_recovery_mask(target, valid, *, steps):
    """Neutral targets within `steps` frames of a turn in the same episode.

    Uses only past teacher commands, never model predictions or future frames.
    The teacher's latched turn is still active, not prematurely labeled recovery.
    """
    if type(steps) is not int or steps < 1:
        raise ValueError('recovery steps must be a positive integer')
    if (target.ndim != 3 or target.shape[-1] != 4 or valid.shape != target.shape[:2]
            or valid.dtype != torch.bool or not torch.isfinite(target).all()):
        raise ValueError('Invalid recovery target/mask')
    active = (target[..., 3]-PWM_MID).abs() > 1
    age = torch.full((len(target),), steps+1, device=target.device, dtype=torch.long)
    result = torch.zeros_like(valid)
    for t in range(target.shape[1]):
        age = torch.where(valid[:, t] & active[:, t], 0, age+1)
        age = torch.where(valid[:, t], age, steps+1)
        result[:, t] = valid[:, t] & ~active[:, t] & (age <= steps)
    return result


def navigation_weights(target, valid, behavior, num_behaviors, *, yaw_balance=.5,
                       amplitude_bins=False, recovery_steps=None):
    """Roll/pitch retain behavior balance; yaw balances neutral/right/left.

    yaw_balance mixes empirical frequencies with equal weight per present yaw
    group. Calculate once from the selected training episodes, never holdouts.
    Each channel's total mass equals the number of valid training frames.
    """
    if not math.isfinite(yaw_balance) or not 0 <= yaw_balance <= 1:
        raise ValueError('yaw_balance must be in [0, 1]')
    if type(amplitude_bins) is not bool:
        raise ValueError('amplitude_bins must be an explicit boolean')
    if recovery_steps is not None and (type(recovery_steps) is not int or recovery_steps < 1):
        raise ValueError('recovery_steps must be a positive integer or None')
    if (target.ndim != 3 or target.shape[-1] != 4 or valid.shape != target.shape[:2]
        or valid.dtype != torch.bool or behavior.shape != valid.shape
        or behavior.dtype != torch.long or not valid.any()
        or not torch.isfinite(target).all() or num_behaviors < 1
        or (behavior[valid] < 0).any() or (behavior[valid] >= num_behaviors).any()):
        raise ValueError('Invalid navigation weight contract')
    base = _balanced(behavior, valid, num_behaviors)
    delta = target[..., 3] - PWM_MID
    yaw_group = torch.zeros_like(behavior)
    yaw_group[delta > 1] = 1
    yaw_group[delta < -1] = 2
    if amplitude_bins:
        yaw_group[delta > 200] = 3
        yaw_group[delta < -200] = 4
    groups = 5 if amplitude_bins else 3
    if recovery_steps is not None:
        yaw_group[yaw_recovery_mask(target, valid, steps=recovery_steps)] = groups
        groups += 1
    yaw = (1-yaw_balance)*valid + yaw_balance*_balanced(yaw_group, valid, groups)
    return torch.stack([base, base, yaw], dim=-1)


def navigation_loss(prediction, target, weights, *, normalizer):
    if not math.isfinite(normalizer) or normalizer <= 0:
        raise ValueError('Loss normalizer must be positive and finite')
    error = (prediction-target)[..., list(NAVIGATION_INDICES)] / PWM_HALF
    if weights.shape != error.shape:
        raise ValueError('Navigation weights must match the three navigation channels')
    return (error.square()*weights).sum()/normalizer
