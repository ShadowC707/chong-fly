"""Round 5: actual transport error of remembered points against path, turn and range since capture.

Reruns one working-set case of a transport-memory profile. Every `every_s` it re-anchors all
remembered points at the true pose (as the failure cards do) and records, per point within
`max_range_m` of the drone: path and heading turn since capture, current range, the error the
memory model assigns, and the actual error = distance to the nearest true surface (all obstacles and
the room). The surface distance is the error component normal to the surface, which is what moves
the gap to the drone for the nearest surface; displacement along a surface is not visible here.
Truth geometry is used only for this diagnosis, never by the controller.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import argparse
from contextlib import closing
from pathlib import Path

import numpy as np


def run(profile, scenario_id, every_s=.5, max_range_m=1.5):
    from edge.survey_memory import rotation
    from evaluate.failure_card import obstacles, surface_distance
    from evaluate.reiteration_baseline import build_runner, scenario_list
    scenario = next(s for s in scenario_list() if s['id'] == scenario_id)
    out = []
    with closing(build_runner(profile, scenario)) as runner:
        runner.resume()
        runner.event(dict(type='mode', mode='AUTO'))
        items = obstacles(runner.env)
        memory = runner.system.runtime.coordinator.memory
        next_s = every_s
        while not runner.terminal:
            runner.advance(5)
            t = runner.time_s
            if t+1e-9 < next_s or not len(getattr(memory, 'points', [])) or memory.faulted:
                continue
            next_s += every_s
            o = memory.observer
            row = runner.rows[-1]
            pos, yaw = np.asarray(row['position_m']), row['attitude_rad'][2]
            local = (memory.points-o.position) @ rotation(0., 0., memory.report['heading_rad'])
            rng = np.linalg.norm(local, axis=1)
            keep = rng <= max_range_m
            if not keep.any():
                continue
            world = pos + local[keep] @ rotation(0., 0., yaw).T
            actual = np.min([surface_distance(k, ob, world) for _, k, ob in items], axis=0)
            path = o.distance-memory.path0[keep]
            turn = memory.turn-memory.turn0[keep]
            model = memory.transport_error(rng)[keep]
            out.append(np.c_[np.full(keep.sum(), t), path, turn, rng[keep], model, actual, t-memory.captured[keep]])
    return np.concatenate(out) if out else np.zeros((0, 7))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile', default='pipeline_v1_r4')
    p.add_argument('--scenario', required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    data = run(a.profile, a.scenario)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(a.out, data)
    print(a.scenario, data.shape)
