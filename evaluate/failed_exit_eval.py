"""Step 9: exit from FAILED (phased_failed_exit) against phased_side_guard on fresh demo seeds.

Scene of the browser demo: random_room, start (-0.7, 0, 1), yaw 0, 180 s. Clustered rooms at 20 %,
divided rooms with the held-out recipe (clustered 16 % plus a wall x 1.4, y 0.2, 2.6 m long). The
author's seed 3934218795 is flown with AUTO at 0 s and at 6 s (the FAILED reproduction). Same metric
code as the working set (evaluate/reiteration_baseline.run_case, section-2 verdict). The 20-run spread
regression is evaluate/spread_runs.py --profiles phased_failed_exit.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import argparse
from contextlib import closing
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path

PROFILES = ('phased_side_guard', 'phased_failed_exit')
CLUSTERED = dict(obstacle_density_pct=20, layout_style='clustered')
DIVIDED = dict(obstacle_density_pct=16, layout_style='clustered', wall_enabled=True, wall_x=1.4, wall_y=.2,
               wall_length_m=2.6)
START, YAW, HORIZON = [-.7, 0., 1.], 0., 180.


def cases():
    out = [('author_3934218795_auto0', 3934218795, 'clustered', 0.), ('author_3934218795_auto6', 3934218795, 'clustered', 6.),
           ('demo_20261025', 20261025, 'clustered', 0.)]
    out += [(f'clustered_{s}', s, 'clustered', 0.) for s in range(2026112001, 2026112011)]
    out += [(f'divided_{s}', s, 'divided', 0.) for s in range(2026112011, 2026112021)]
    return out


def scenario(case_id, seed, kind, auto_s):
    from demo.runner import DemoRunner
    options = CLUSTERED if kind == 'clustered' else DIVIDED
    with closing(DemoRunner(controller_mode='phased_side_guard', scene='random_room', seed=seed, start=tuple(START), yaw=YAW,
                    scene_options=dict(options), max_duration_s=HORIZON)) as r:
        geometry = r.manifest()['geometry']
    return dict(id=case_id, kind='live', horizon_s=HORIZON, auto_request_s=auto_s, room_kind=kind,
                fixture=dict(scene='random_room', seed=seed, start=START, yaw=YAW, scene_options=dict(options),
                             geometry=geometry))


def task(out, profile, case):
    from evaluate.reiteration_baseline import run_case, verdict
    s = scenario(*case)
    m = run_case(out, profile, s)
    return dict(m, case=s['id'], room_kind=s['room_kind'], auto_request_s=s['auto_request_s'],
                verdict=verdict(m, s['horizon_s']))


def main(out, workers, profiles=PROFILES):
    from evaluate.mvp_selection import write
    out.mkdir(parents=True, exist_ok=False)
    write(out/'protocol.json', dict(version='failed-exit-eval-v1', profiles=list(profiles), cases=cases(),
        horizon_s=HORIZON, criterion='see output/pipeline_v1_step9/report_uk.md (written before the runs)'))
    reports = []
    with ProcessPoolExecutor(workers) as pool:
        futures = {pool.submit(task, out, p, c): (p, c[0]) for c in cases() for p in profiles}
        for f in as_completed(futures):
            p, c = futures[f]
            try:
                reports.append(f.result())
            except Exception as e:  # keep the other runs; report the failure
                reports.append(dict(profile=p, case=c, error=repr(e)))
            print(p, c, 'error' if 'error' in reports[-1] else 'ok', flush=True)
    write(out/'summary.json', sorted(reports, key=lambda r: (r['case'], r['profile'])))


if __name__ == '__main__':
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--out', type=Path, default=Path('output/pipeline_v1_step9/seeds'))
    a.add_argument('--workers', type=int, default=6)
    a.add_argument('--profiles', nargs='+', default=list(PROFILES))
    x = a.parse_args()
    main(x.out, x.workers, x.profiles)
