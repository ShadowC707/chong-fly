"""Replay one parameterized teacher on immutable recorded inputs, not physics."""
from collections import Counter
from copy import deepcopy
import math
import numpy as np
import torch
from generator.generate_reflex_dataset import make_teacher, _behavior


def replay_cruise_targets(data, *, cruise_pwm):
    if not math.isfinite(cruise_pwm) or not 1500 < cruise_pwm <= 1530:
        raise ValueError('Research cruise target must be in (1500,1530]')
    params=deepcopy(data['metadata']['teacher_parameters'])
    if params['noise_std_pwm'] != 0:
        raise ValueError('Exact label replay requires noise-free targets')
    result=deepcopy(data)
    version=data['metadata']['teacher_version'];dt=data['metadata']['dt']
    changed=0;worst=0.
    for i in range(len(data['X'])):
        original=make_teacher(version,**params)
        revised=make_teacher(version,**{**params,'pwm_cruise_pitch':float(cruise_pwm)})
        for t in range(int(data['valid'][i].sum())):
            x=data['X'][i,t].numpy()
            expected=original.step(x,dt=dt)
            if not np.allclose(expected,data['Y'][i,t].numpy(),rtol=0,atol=.002):
                raise ValueError(f'Original teacher replay differs at episode {i}, frame {t}')
            y=revised.step(x,dt=dt)
            if not np.array_equal(y[[0,1,3]],expected[[0,1,3]]):
                raise ValueError('Cruise revision changed a non-Pitch channel')
            result['Y'][i,t]=torch.from_numpy(y)
            result['behavior'][i,t]=_behavior(x,y)
            delta=abs(float(y[2]-expected[2]));changed+=int(delta>0);worst=max(worst,delta)
    meta=result['metadata']
    meta['teacher_parameters']={**params,'pwm_cruise_pitch':float(cruise_pwm)}
    meta['behavior_counts']=dict(Counter(meta['behavior_names'][int(b)] for b in result['behavior'][result['valid']]))
    meta['target_revision']=dict(version='v5-cruise1530-replay-v1',original_teacher_parameters=params,
        pitch_changed_frames=changed,max_pitch_change_pwm=worst,
        method='Fresh teacher per complete episode; original labels replay-verified; no new physical rollout',
        execution='X and applied_actions remain recorded behavior, not revised-teacher trajectories')
    return result
