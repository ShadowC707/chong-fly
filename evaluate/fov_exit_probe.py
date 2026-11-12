"""Passive check of the reviewer's hypothesis: where do obstacles leave the ToF frustum, and does a
profile act while the obstacle is still in view?

For every tick and every obstacle: true clearance, whether the obstacle's nearest point lies in the
ToF frustum (45 x 45 deg with the full attitude; geometric proxy, not ray-exact), and whether any rule
was intervening (pipeline: any active stage; phased profiles: a phase other than CRUISE, or any reason).
An exit event is a tick where an obstacle leaves the frustum with clearance < EXIT_MAX_M. For each exit:
clearance at exit, minimum clearance over the next FOLLOW_S while it stays out of view, and whether a
rule intervened during the last IN_VIEW_S before the exit (obstacle still in view). For phased profiles
the ToF distance at each BRAKE start is recorded as well. Nothing here reaches the controller.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import argparse
from contextlib import closing
import json
from pathlib import Path

import numpy as np

EXIT_MAX_M = .60
FOLLOW_S = 5.
IN_VIEW_S = 1.
CLOSE_M = .10


def intervening(decision):
    """Names of the rules acting this tick (empty: the network command passed untouched)."""
    stages = decision.get('stages')
    if stages is not None:
        names = {r['stage'] for r in stages if r['active']}
        return names | {x for x in decision.get('reasons', []) if x not in names}
    phase = decision.get('phased_phase')
    return ({'phase_'+phase} if phase not in (None, 'CRUISE') else set()) | set(decision.get('reasons', []))


def run(profile, scenario):
    import evaluate.reiteration_baseline as rb
    from evaluate.failure_card import body_rotation, in_frustum, nearest_point, obstacles
    from evaluate.spread_runs import build_live_perturbed
    if not hasattr(rb, 'ORIGINAL_BUILD_RUNNER'):
        rb.ORIGINAL_BUILD_RUNNER = rb.build_runner
    build = build_live_perturbed if scenario['kind'] == 'live' else rb.ORIGINAL_BUILD_RUNNER
    T, C, F, I, brake_starts = [], [], [], [], []
    with closing(build(profile, scenario)) as runner:
        runner.resume()
        runner.event(dict(type='mode', mode='AUTO'))
        items = [x for x in obstacles(runner.env) if x[1] != 'room']
        coordinator = runner.system.runtime.coordinator
        while not runner.terminal:
            t, n = runner.time_s, len(runner.rows)
            runner.advance(5)
            if len(runner.rows) <= n:
                break
            row = runner.rows[n]
            pos, rot = np.asarray(row['position_m']), body_rotation(np.asarray(row['attitude_rad']))
            d = coordinator.last_decision
            clr, fov = [], []
            for _, k, o in items:
                q = nearest_point(k, o, pos)
                clr.append(float(np.linalg.norm(q-pos))-runner.env.collision_radius_m)
                fov.append(in_frustum(rot.T @ (q-pos))['in_frustum'])
            T.append(t); C.append(clr); F.append(fov); I.append(sorted(intervening(d)))
            if d.get('phased_event') == 'started':
                brake_starts.append(dict(t=t, tof_min_m=d.get('tof_min_m')))
        names = [x[0] for x in items]
    T, C, F = np.array(T), np.array(C), np.array(F)
    exits = []
    for j, name in enumerate(names):
        idx = np.flatnonzero(F[:-1, j] & ~F[1:, j])+1
        for i in idx:
            if C[i, j] >= EXIT_MAX_M:
                continue
            end = i
            while end+1 < len(T) and not F[end+1, j] and T[end+1]-T[i] <= FOLLOW_S:
                end += 1
            view = np.flatnonzero((T >= T[i]-IN_VIEW_S) & (T < T[i]))
            rules = sorted({x for v in view for x in I[v]})
            exits.append(dict(obstacle=name, t=float(T[i]), exit_clearance_m=float(C[i, j]),
                              min_after_m=float(C[i:end+1, j].min()), rules_in_view=rules))
    # Close events, analysed backwards: each contiguous interval below CLOSE_M per obstacle; the last tick
    # before it where the obstacle was in view, the clearance then, the time since, and the rules acting
    # in the last IN_VIEW_S of view.
    close = []
    for j, name in enumerate(names):
        below = C[:, j] < CLOSE_M
        starts = np.flatnonzero(below & ~np.r_[False, below[:-1]])
        for i in starts:
            end = i
            while end+1 < len(T) and below[end+1]:
                end += 1
            seen = np.flatnonzero(F[:i+1, j])
            e = dict(obstacle=name, start_s=float(T[i]), min_clearance_m=float(C[i:end+1, j].min()),
                     contact=bool(C[i:end+1, j].min() <= 1e-9))
            if len(seen):
                k = seen[-1]
                view = np.flatnonzero((T >= T[k]-IN_VIEW_S) & (T <= T[k]))
                e.update(in_view_at_start=bool(k == i), last_view_s=float(T[k]), blind_s=float(T[i]-T[k]),
                         clearance_at_last_view_m=float(C[k, j]),
                         rules_in_view=sorted({x for v in view for x in I[v]}))
            else:
                e.update(in_view_at_start=False, last_view_s=None, blind_s=None, clearance_at_last_view_m=None,
                         rules_in_view=[])
            close.append(e)
    return dict(profile=profile, scenario=scenario['id'], exits=exits, close=close, brake_starts=brake_starts)


if __name__ == '__main__':
    from concurrent.futures import ProcessPoolExecutor
    from evaluate.reiteration_baseline import scenario_list
    from evaluate.spread_runs import perturbed, PERTURBATIONS
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--profiles', nargs='+', default=['pipeline_v1_r5', 'phased_side_guard'])
    p.add_argument('--workers', type=int, default=6)
    a = p.parse_args()
    tasks = [(prof, perturbed(s, k)) for s in scenario_list() for k in range(len(PERTURBATIONS)) for prof in a.profiles]
    tasks.sort(key=lambda t: t[1]['kind'] != 'live')
    with ProcessPoolExecutor(a.workers) as pool:
        results = list(pool.map(run, *zip(*tasks)))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(results, indent=1), encoding='utf-8')
    print(len(results), 'runs')
