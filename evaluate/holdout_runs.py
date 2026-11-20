"""Plan step 6: the held-out set, flown ONCE (6 starts of holdout.json, 180 s, AUTO at 0 s).

The official candidate is fixed in the journal before this run (phased_side_guard, by the selection
rule); other profiles may be flown alongside for reference only. Each start uses the recorded
geometry of its room (as the MVP scenes do) and the same metric code as the working set
(evaluate/reiteration_baseline.run_case, section-2 verdict). The script refuses to run when
holdout.json already says holdout_flights_run: true, and sets it to true after the flights.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOLDOUT = ROOT/'data/scenarios/reiteration_v1/holdout.json'


def scenarios():
    h = json.loads(HOLDOUT.read_text(encoding='utf-8'))
    out = []
    for name, case in h['cases'].items():
        g = case['geometry']
        for i, start in enumerate(case['starts']):
            out.append(dict(id=f'holdout_{name}_s{i+1}', kind='mvp', horizon_s=float(h['horizon_s']),
                            scene=dict(seed=case['seed'], start=list(start['position_m']), yaw=float(start['yaw_rad']),
                                       room=g['room'], boxes=g['boxes'], cylinders=g['cylinders']),
                            static_initial_clearance_m=start['static_initial_clearance_m']))
    return h, out


def task(out, profile, scenario):
    from evaluate.reiteration_baseline import run_case, verdict
    m = run_case(out, profile, scenario)
    return dict(m, verdict=verdict(m, scenario['horizon_s']))


def check_build(profiles):
    """Build (do not fly) every start with every profile: geometry, clearance and profile assembly."""
    from evaluate.reiteration_baseline import build_runner
    _, S = scenarios()
    for s in S:
        for p in profiles:
            r = build_runner(p, s)
            r.close()
    return len(S)


def main(out, profiles, official, workers):
    h, S = scenarios()
    if h.get('holdout_flights_run'):
        raise SystemExit('holdout.json says the held-out flights were already run: refusing to fly them again')
    out.mkdir(parents=True, exist_ok=False)
    (out/'protocol.json').write_text(json.dumps(dict(official_candidate=official, reference_profiles=[p for p in profiles if p != official],
        starts=[s['id'] for s in S], horizon_s=h['horizon_s'], rule='candidate fixed before the run; no tuning after'),
        indent=1), encoding='utf-8')
    reports = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(task, out, p, s): (p, s['id']) for s in S for p in profiles}
        for f in as_completed(futures):
            p, sid = futures[f]
            try:
                reports.append(f.result())
                print('done', sid, p, flush=True)
            except Exception as error:  # reported, never hidden or retried
                print('FAILED', sid, p, repr(error), flush=True)
                reports.append(dict(profile=p, scenario=sid, error=repr(error)))
    (out/'summary.json').write_text(json.dumps(reports, indent=1, default=float), encoding='utf-8')
    flown = [r for r in reports if 'error' not in r]
    if not flown:  # nothing was flown, so nothing about the held-out set was seen: do not mark it as used
        raise SystemExit('no held-out flight completed (see summary.json); holdout_flights_run left unchanged')
    text = HOLDOUT.read_text(encoding='utf-8')
    marker = '"holdout_flights_run": false'
    if text.count(marker) != 1:
        raise SystemExit('could not mark holdout_flights_run in holdout.json; set it to true by hand')
    HOLDOUT.write_text(text.replace(marker, '"holdout_flights_run": true'), encoding='utf-8')  # one field only
    print(len([r for r in reports if 'error' not in r]), 'of', len(reports), 'flights ok; holdout_flights_run set to true')


if __name__ == '__main__':
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--out', type=Path, required=True)
    a.add_argument('--official', default='phased_side_guard')
    a.add_argument('--profiles', nargs='+', default=['phased_side_guard', 'pipeline_v1_r7', 'pipeline_v1'])
    a.add_argument('--workers', type=int, default=6)
    a.add_argument('--check-build', action='store_true')
    args = a.parse_args()
    if args.check_build:
        print(check_build(args.profiles), 'starts build for', args.profiles)
    else:
        main(args.out, args.profiles, args.official, args.workers)
