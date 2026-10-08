"""Controlled sensor contrasts at the real controller timestep."""
import numpy as np
import torch

from configs.flight_config import CONTROL_DT


@torch.no_grad()
def temporal_response_probe(policy, *, tolerance_pwm=20., coverage_columns=(4,)):
    """Clear -> left/right threat -> clear, without resetting recurrent state.

    By default four cases combine mirrored threats at 0.35/0.70 m. Optional
    coverage_columns expands the contrast to narrow and broad obstacles.
    Settling requires
    remaining inside the tolerance for the rest of the two-second recovery.
    These controlled sensor stimuli are diagnostics, not physical trajectories.
    """
    from generator.generate_reflex_dataset import ExpertReflexPolicy
    if (not isinstance(coverage_columns, (tuple, list)) or not coverage_columns or
        any(type(c) is not int or not 1 <= c <= 4 for c in coverage_columns) or
        len(set(coverage_columns)) != len(coverage_columns)):
        raise ValueError('coverage_columns must contain distinct integers from 1 to 4')
    parameter = next(policy.parameters())
    warmup, threat, recovery = 50, 25, 100
    cases = [(side, distance, width) for distance in (.1, .2) for width in coverage_columns
             for side in ('left', 'right')]
    obs = torch.zeros(len(cases), warmup+threat+recovery, 74,
                      dtype=parameter.dtype, device=parameter.device)
    obs[..., 2:] = 1
    for i, (side, distance, width) in enumerate(cases):
        grid = obs[i, warmup:warmup+threat, 2:66].reshape(threat, 8, 8)
        columns = slice(0, width) if side == 'left' else slice(8-width, 8)
        grid[..., columns] = distance
    policy.eval()
    pwm, _ = policy(obs, dt=CONTROL_DT)
    if not torch.isfinite(pwm).all():
        raise ValueError('Nonfinite temporal response')
    rows = []
    for i, (side, distance, width) in enumerate(cases):
        expert = ExpertReflexPolicy(noise_std_pwm=0)
        targets = torch.from_numpy(np.stack([expert.step(x, dt=CONTROL_DT)
                                             for x in obs[i].cpu().numpy()]))
        yaw = pwm[i, :, 3].cpu()
        err = yaw-targets[:, 3]
        sign = 1 if side == 'left' else -1
        active = yaw[warmup:warmup+threat]-1500
        onset = (active*sign > tolerance_pwm).nonzero().flatten()
        tail = yaw[warmup+threat:]-1500
        outside = (tail.abs() > tolerance_pwm).nonzero().flatten()
        settle = int(outside[-1])+1 if len(outside) else 0
        rows.append({'obstacle_side':side, 'normalized_distance':distance, 'active_columns':width,
            'clear_abs_yaw_pwm':float((yaw[:warmup]-1500).abs().mean()),
            'threat_yaw_pwm':float(yaw[warmup+threat-1]),
            'threat_target_yaw_pwm':float(targets[warmup+threat-1, 3]),
            'threat_yaw_mae_pwm':float(err[warmup:warmup+threat].abs().mean()),
            'threat_pitch_mae_pwm':float((pwm[i, warmup:warmup+threat, 2].cpu()
                                         -targets[warmup:warmup+threat, 2]).abs().mean()),
            'direction_correct':bool(active[-1]*sign > tolerance_pwm),
            'onset_s':float(onset[0]*CONTROL_DT) if len(onset) else None,
            'recovery_settle_s':settle*CONTROL_DT if settle < recovery else None,
            'recovery_final_abs_yaw_pwm':float(tail[-1].abs())})
    return {'dt':CONTROL_DT, 'dtype':str(parameter.dtype), 'tolerance_pwm':tolerance_pwm,
            'warmup_s':warmup*CONTROL_DT, 'threat_s':threat*CONTROL_DT,
            'recovery_s':recovery*CONTROL_DT, 'cases':rows}


@torch.no_grad()
def response_probe(policy, *, steps=25):
    parameter = next(policy.parameters())
    obs = torch.zeros(3, steps, 74, dtype=parameter.dtype, device=parameter.device)
    obs[..., 2:] = 1
    left = torch.ones(8, 8, dtype=parameter.dtype, device=parameter.device)
    left[:, :4] = .1
    obs[1, :, 2:66] = left.flatten()
    obs[2, :, 2:66] = left.flip(1).flatten()
    policy.eval()
    pwm, _ = policy(obs, dt=CONTROL_DT)
    if not torch.isfinite(pwm).all():
        raise ValueError('Nonfinite response probe')
    return {'dt':CONTROL_DT, 'steps':steps, 'dtype':str(parameter.dtype),
            'yaw_clear_pwm':float(pwm[0, -1, 3]),
            'yaw_left_obstacle_pwm':float(pwm[1, -1, 3]),
            'yaw_right_obstacle_pwm':float(pwm[2, -1, 3]),
            'yaw_contrast_pwm':float(pwm[1, -1, 3]-pwm[2, -1, 3]),
            'max_abs_yaw_contrast_pwm':float((pwm[1, :, 3]-pwm[2, :, 3]).abs().max())}
