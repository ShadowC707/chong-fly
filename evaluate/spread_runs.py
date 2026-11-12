"""Spread of section-2 metrics under start perturbations (author decision after round 5).

Each working-set scenario is flown from 5 perturbed starts (+-2 cm in x/y, +-2 deg yaw; a fixed,
sign-balanced set, below). Everything else is identical to evaluate/reiteration_baseline.py: same
seeds, same physics, same metric code. The table gives the median and the worst value over the
5 perturbations for every metric of section 2.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np

# (dx m, dy m, dyaw deg): four axis-aligned corners with opposite yaw signs and one diagonal.
PERTURBATIONS = [(.02, 0., 2.), (-.02, 0., -2.), (0., .02, -2.), (0., -.02, 2.), (-.0141, .0141, 2.)]


def perturbed(scenario, k):
    dx, dy, dyaw = PERTURBATIONS[k]
    s = deepcopy(scenario)
    target = s['fixture'] if s['kind'] == 'live' else s['scene']
    target['start'] = [target['start'][0]+dx, target['start'][1]+dy, target['start'][2]]
    target['yaw'] = float(target['yaw'])+math.radians(dyaw)
    if s['kind'] == 'live':  # the random room is generated around the nominal start: keep it, move the drone
        s['nominal_start'], s['nominal_yaw'] = list(scenario['fixture']['start']), float(scenario['fixture']['yaw'])
    s['id'] = f"{scenario['id']}_p{k+1}"
    s['base_id'] = scenario['id']
    s['perturbation'] = dict(dx_m=dx, dy_m=dy, dyaw_deg=dyaw)
    return s


def build_live_perturbed(profile, scenario):
    """Generate the live room from the nominal start (so its geometry matches the recording), then
    place the drone at the perturbed start and yaw, as build_runner does for MVP scenes."""
    from demo.recording import plain
    import evaluate.reiteration_baseline as rb
    f = scenario['fixture']
    nominal = deepcopy(scenario)
    nominal['fixture']['start'], nominal['fixture']['yaw'] = scenario['nominal_start'], scenario['nominal_yaw']
    runner = rb.ORIGINAL_BUILD_RUNNER(profile, nominal)
    try:
        start, yaw = tuple(float(v) for v in f['start']), float(f['yaw'])
        env = runner.env
        runner.start, runner.yaw = start, yaw
        env.physics.pos[:] = start
        env.physics.vel[:] = 0
        env.physics.quat[:] = env.physics.euler_to_quaternion(0, 0, yaw)
        hit, gap = runner.contact(start, start, env.room, env.boxes, env.cylinders, env.collision_radius_m)
        if hit or gap < .2:
            raise ValueError('Invalid perturbed start clearance')
        runner.obs = env._sample_observation(0.)
        runner._manifest = plain(runner._build_manifest())
        if runner.manifest()['geometry'] != f['geometry']:
            raise ValueError('Generated room differs from the recorded geometry')
    except BaseException:
        runner.close()
        raise
    return runner


def task(out, profile, scenario):
    import evaluate.reiteration_baseline as rb
    from evaluate.reiteration_baseline import run_case, verdict
    if not hasattr(rb, 'ORIGINAL_BUILD_RUNNER'):
        rb.ORIGINAL_BUILD_RUNNER = rb.build_runner
    rb.build_runner = (lambda p, s: build_live_perturbed(p, s)) if scenario['kind'] == 'live' else rb.ORIGINAL_BUILD_RUNNER
    m = run_case(out, profile, scenario)
    return dict(m, base_scenario=scenario['base_id'], perturbation=scenario['perturbation'],
                verdict=verdict(m, scenario['horizon_s']))


METRICS = [  # key, label, worst is min (True) or max (False), format
    ('network_share', 'net', True, '{:.3f}'), ('generated_share', 'gen', False, '{:.3f}'),
    ('contacts', 'контакти', False, '{:.0f}'), ('min_clearance_m', 'зазор, м', True, '{:.3f}'),
    ('max_height_error_m', 'висота, м', False, '{:.3f}'),
    ('reverse_outside_corridor_m', 'відкат поза коридором, м', False, '{:.3f}'),
    ('reverse_distance_body_m', 'відкат (старий), м', False, '{:.3f}'),
]


def cell(values, worst_min, pattern):
    v = [x for x in values if x is not None]
    if not v:
        return '—'
    worst = min(v) if worst_min else max(v)
    return f"{pattern.format(float(np.median(v)))} / {pattern.format(worst)}"


def all_thresholds(r):
    """All section-2 groups with one rule for every profile: network_share >= 0.80 is required; an
    undefined generated_share (profiles without a stage log) is not counted as a failure."""
    v, gen = r['verdict'], r.get('generated_share')
    progress = v['progress'] if r['base_scenario'] == 'live_room' else True
    return bool((r.get('network_share') or 0) >= .80 and (gen is None or gen <= .10)
                and v['safety'] and v['delivery'] and progress)


def table(reports, profiles):
    head = ['Сценарій', 'Профіль', *[m[1] for m in METRICS], 'поступ: без нової, с', 'поступ: за 60 с',
            'безпека ✓', 'доставка ✓', 'усі пороги ✓']
    lines = ['| ' + ' | '.join(head) + ' |', '|' + '---|'*len(head)]
    for s in sorted({r['base_scenario'] for r in reports}):
        for p in profiles:
            R = [r for r in reports if r['base_scenario'] == s and r['profile'] == p]
            if not R:
                continue
            row = [s, p] + [cell([r.get(k) for r in R], w, f) for k, _, w, f in METRICS]
            if s == 'live_room':
                done = [r for r in R if r['completed']]
                row += [cell([r['longest_no_new_cell']['duration_s'] for r in done], False, '{:.1f}') +
                        (f' ({len(R)-len(done)} перерв.)' if len(done) < len(R) else ''),
                        cell([r['new_cells_last_60s'] for r in done], True, '{:.0f}')]
            else:
                row += ['—', '—']
            v = [r['verdict'] for r in R]
            allok = sum(all_thresholds(r) for r in R)
            row += [f"{sum(r['safety'] for r in v)}/{len(R)}", f"{sum(r['delivery'] for r in v)}/{len(R)}",
                    f'{allok}/{len(R)}']
            lines.append('| ' + ' | '.join(row) + ' |')
    return '\n'.join(lines)


def main(out, workers, profiles, only_live=False):
    from evaluate.reiteration_baseline import scenario_list
    out.mkdir(parents=True, exist_ok=True)
    tasks = [(p, perturbed(s, k)) for s in scenario_list() for k in range(len(PERTURBATIONS)) for p in profiles
             if not only_live or s['kind'] == 'live']
    previous = []
    if only_live and (out/'summary.json').exists():  # keep the MVP runs, replace the live ones
        previous = [r for r in json.loads((out/'summary.json').read_text(encoding='utf-8'))
                    if not str(r.get('scenario', '')).startswith('live_room')]
    (out/'protocol.json').write_text(json.dumps(dict(profiles=profiles, perturbations=PERTURBATIONS,
        scenarios=[t[1]['id'] for t in tasks[::len(profiles)]]), indent=1), encoding='utf-8')
    reports = list(previous)
    # long live-room flights first, so the pool does not end on them
    tasks.sort(key=lambda t: t[1]['kind'] != 'live')
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(task, out, p, s): (p, s['id']) for p, s in tasks}
        for f in as_completed(futures):
            p, sid = futures[f]
            try:
                reports.append(f.result())
                print('done', sid, p, flush=True)
            except Exception as error:
                print('FAILED', sid, p, repr(error), flush=True)
                reports.append(dict(profile=p, scenario=sid, error=repr(error)))
    (out/'summary.json').write_text(json.dumps(reports, indent=1, default=float), encoding='utf-8')
    ok = [r for r in reports if 'error' not in r]
    (out/'table.md').write_text(table(ok, profiles), encoding='utf-8')
    print(len(ok), 'of', len(reports), 'runs ok')


if __name__ == '__main__':
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--out', type=Path, required=True)
    a.add_argument('--workers', type=int, default=6)
    a.add_argument('--profiles', nargs='+', default=['pipeline_v1_r5', 'pipeline_v1', 'phased_side_guard'])
    a.add_argument('--table-only', action='store_true')
    a.add_argument('--only-live', action='store_true')
    args = a.parse_args()
    if args.table_only:
        reports = [r for r in json.loads((args.out/'summary.json').read_text(encoding='utf-8')) if 'error' not in r]
        print(table(reports, args.profiles))
    else:
        main(args.out, args.workers, args.profiles, args.only_live)
