"""Train-only active-fast mixture, preserving total loss mass per channel."""
import math

import torch


def active_fast_weights(base, observations, valid, *, mix):
    """Reserve `mix` of Pitch/Yaw loss mass for close-ToF/high-flow frames.

    Call ONLY after selecting training episodes. Targets are unchanged; groups
    use current observations, never predictions, future frames or holdout data.
    Zero mix preserves the original tensor values exactly.
    """
    if type(mix) not in (int,float) or not math.isfinite(mix) or not 0<=mix<=1:
        raise ValueError('active-fast mix must be finite in [0,1]')
    if (observations.ndim!=3 or observations.shape[-1]<66 or valid.dtype!=torch.bool
            or valid.shape!=observations.shape[:2] or base.shape!=(*valid.shape,3)
            or not valid.any() or not torch.isfinite(observations).all()
            or not torch.isfinite(base).all() or (base<0).any()
            or (base[~valid]!=0).any()):
        raise ValueError('Invalid training-only active-fast weight inputs')
    mask=valid & (observations[...,2:66].amin(-1)*3.5<.8) & (observations[...,:2].norm(dim=-1)>=.12)
    count=int(mask.sum())
    info=dict(version='active-fast-mixture-v1',mix=mix,channels=['pitch','yaw'],
        min_tof_below_m=.8,flow_norm_at_least=.12,frames=count,
        episodes=int(mask.any(-1).sum()),valid_frames=int(valid.sum()),train_only=True)
    if mix==0:return base.clone(),info
    if not count:raise ValueError('No active-fast frames in selected training episodes')
    result=base.clone()
    for channel in (1,2):
        mass=base[...,channel].sum()
        result[...,channel]=(1-mix)*base[...,channel]+mix*(mass/count)*mask
    info['base_group_mass_fraction']=[float(base[...,i][mask].sum()/base[...,i].sum()) for i in (1,2)]
    info['result_group_mass_fraction']=[float(result[...,i][mask].sum()/result[...,i].sum()) for i in (1,2)]
    return result,info
