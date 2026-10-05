"""Paired raw-teacher / range-altitude validation; no training or model promotion.

Run: python -m optimizer.validate_altitude --duration 30 --seeds 42 142 242
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from configs.flight_config import CONTROL_DT
from generator.generate_reflex_dataset import ExpertReflexPolicy
from generator.reflex_contract import TEACHER_VERSION
from optimizer.rollout import simulate_policy_rollout
from simulation.altitude_control import AltitudeHold


class TeacherPolicy:
    """Adapt the same observation-only teacher to the common rollout interface."""
    def __init__(self):
        self.teacher = ExpertReflexPolicy()

    def reset_state(self):
        self.teacher.reset()

    def step_np(self, flow, tof, memory_ring, dt):
        return self.teacher.step(np.concatenate([flow, tof, memory_ring]), dt=dt)


def validate(duration_s=30., seeds=(42, 142, 242)):
    ratio = duration_s / CONTROL_DT
    if not math.isfinite(ratio) or ratio < 1 or not math.isclose(ratio, round(ratio), abs_tol=1e-8):
        raise ValueError('Duration must be a positive integer number of control periods')
    results = []
    for seed in seeds:
        for enabled in (False, True):
            _, metrics = simulate_policy_rollout(TeacherPolicy(), eval_steps=round(ratio),
                dt=CONTROL_DT, seed=seed, altitude_hold=AltitudeHold() if enabled else None)
            row = {'seed': seed, 'profile': 'range_altitude' if enabled else 'raw_teacher', **metrics}
            results.append(row)
            print(f"{row['profile']} seed={seed}: {metrics['survival_time_s']:.3f}s, "
                  f"feasible={metrics['feasible']}, failure={metrics['failure_reason']}", flush=True)
    return {'teacher_version': TEACHER_VERSION, 'requested_duration_s': duration_s,
            'scope': 'Standalone simulation reference controller, not connectome policy or hardware qualification',
            'rollouts': results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--duration', type=float, default=30.)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 142, 242])
    parser.add_argument('--output', type=Path, default=Path('data/altitude_validation_v7.json'))
    args = parser.parse_args()
    result = validate(args.duration, args.seeds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
