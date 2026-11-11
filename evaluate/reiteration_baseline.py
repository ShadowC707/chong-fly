"""Plan step 2: section-2 metrics for control profiles on the working set (live room + 4 MVP starts).

Every case is one DemoRunner flight with AUTO requested at t=0, no scheduled STOP. The same code
computes the metrics for every profile, so the table is a like-for-like baseline. No tuning.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import closing
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ROOT/'data/scenarios/reiteration_v1'
PROFILES = ('candidate_a', 'phased_side_guard', 'phased_swath')
MVP_CASES = ('9d5431c1', 'b0a9f212')
# Section 2 thresholds, copied from the plan and acceptance.json; never edited after results.
LIMITS = dict(network_share=.80, generated_share=.10, contacts=0, min_clearance_m=.10,
              max_height_error_m=.10, reverse_distance_body_m=.25, max_no_new_cell_s=25.,
              min_new_cells_last_60s=10, faults=0)


def scenario_list():
    live = json.loads((SCENARIOS/'user_room.json').read_text(encoding='utf-8'))
    mvp = json.loads((SCENARIOS/'mvp_rooms.json').read_text(encoding='utf-8'))
    out = [dict(id='live_room', kind='live', horizon_s=float(live['horizon_s']), fixture=live)]
    for case in MVP_CASES:
        native = deepcopy(mvp['cases'][case]['scene'])
        mirror = deepcopy(native)
        mirror['start'][1] *= -1
        mirror['yaw'] *= -1
        for b in [mirror['room'], *mirror['boxes']]:
            b['y_min'], b['y_max'] = -b['y_max'], -b['y_min']
        for c in mirror['cylinders']:
            c['center_y'] *= -1
        out += [dict(id=f'mvp_{case}_native', kind='mvp', horizon_s=45., scene=native),
                dict(id=f'mvp_{case}_mirror', kind='mvp', horizon_s=45., scene=mirror)]
    return out


def build_runner(profile, scenario):
    from demo.controllers import install
    from demo.recording import plain
    from demo.runner import DemoRunner
    from simulation.drone_env import BoxObstacle, CylinderObstacle, RoomBoundaries
    horizon = scenario['horizon_s']
    if scenario['kind'] == 'live':
        f = scenario['fixture']
        runner = DemoRunner(controller_mode=profile, scene=f['scene'], seed=f['seed'], start=f['start'],
            yaw=f['yaw'], scene_options=f['scene_options'], collision_radius_m=f['geometry']['collision_radius_m'],
            max_duration_s=horizon)
        if runner.manifest()['geometry'] != f['geometry']:
            raise ValueError('Generated room differs from the recorded geometry')
        return runner
    s = scenario['scene']
    runner = DemoRunner(controller_mode='candidate_a', scene='room', seed=s['seed'], start=s['start'],
        yaw=s['yaw'], run_id=scenario['id'], collision_radius_m=.2, max_duration_s=horizon,
        contact_model='sphere-sweep-v1')
    try:
        runner.env.room = RoomBoundaries(**s['room'])
        runner.env.boxes = [BoxObstacle(**v) for v in s['boxes']]
        runner.env.cylinders = [CylinderObstacle(**v) for v in s['cylinders']]
        hit, gap = runner.contact(runner.start, runner.start, runner.env.room, runner.env.boxes,
                                  runner.env.cylinders, .2)
        if hit or gap < .2:
            raise ValueError('Invalid initial clearance')
        runner.obs = runner.env._sample_observation(0.)
        runner.env.demo_geometry_spec = dict(version='reiteration-baseline-v1', scene=s)
        if profile != 'candidate_a':
            runner.system = install(runner.system, profile=profile)
        runner._manifest = plain(runner._build_manifest())
    except BaseException:
        runner.close()
        raise
    return runner


def run_case(out, profile, scenario):
    from evaluate.compare_maneuver_repair import summarize
    from evaluate.mvp_selection import write
    from simulation.navigation_metrics import progress_metrics, share_metrics
    horizon = scenario['horizon_s']
    out = Path(out)/scenario['id']/profile
    with closing(build_runner(profile, scenario)) as runner:
        out.mkdir(parents=True, exist_ok=False)
        write(out/'protocol.json', dict(profile=profile, scenario=scenario['id'], horizon_s=horizon,
            auto_request_s=0., stop_scheduled=False, limits=LIMITS, manifest=runner.manifest()))
        runner.resume()
        runner.event(dict(type='mode', mode='AUTO'))
        decisions = []
        while not runner.terminal:
            tick_time = runner.time_s
            runner.advance(5)
            decisions.append(dict(time_s=tick_time,
                decision=runner.system.runtime.coordinator.last_decision.copy()))
        rows = runner.rows
        metrics = summarize(rows, runner.time_s)
        time = np.array([r['time_s'] for r in rows])
        xy = np.array([r['position_m'][:2] for r in rows])
        controls = [r for r in rows if r['plant_input_pwm'] is not None
                    and abs(r['time_s']/.02-round(r['time_s']/.02)) < 1e-7]
        samples = np.r_[np.arange(0, len(time)-1, 5), len(time)-1]
        auto = [r['plant_source'] for r in rows if r['plant_input_pwm'] is not None]
        first_auto = next((i for i, v in enumerate(auto) if v == 'auto'), None)
        completed = runner.pause_reason == 'duration_limit' and abs(runner.time_s-horizon) < 1e-6
        faults = [e for e in runner.events if e['kind'] in ('bridge_fault', 'simulation_fault')]
        metrics.update(progress_metrics(time, xy, samples, runner.time_s), pause_reason=runner.pause_reason,
            completed=completed, contacts=int(sum(r['collision'] is not None for r in rows)),
            continuous_auto=first_auto is not None and all(v == 'auto' for v in auto[first_auto:]),
            faults=faults, phase_counts=Counter(d['decision'].get('phased_phase') or 'none' for d in decisions),
            reasons=Counter(x for d in decisions for x in d['decision'].get('reasons', [])))
        network = [r['trace']['network_pwm'] or [np.nan]*4 for r in controls]
        coordinated = [r['trace']['coordinated_pwm'] or [np.nan]*4 for r in controls]
        flags = ([r['plant_source'] == 'auto' for r in controls], [r['stop_requested'] for r in controls])
        np.savez_compressed(out/'trace.npz', time=time, xy=xy, clearance=[r['clearance_m'] for r in rows],
            control_time=[r['time_s'] for r in controls], auto=flags[0], stop_requested=flags[1],
            network=network, coordinated=coordinated,
            accepted=[r['trace']['accepted_pwm'] or [np.nan]*4 for r in controls])
        try:  # no stage log before the pipeline (plan step 3): generated_share stays null
            metrics.update(share_metrics(network, coordinated, *flags))
        except ValueError as error:
            metrics.update(network_share=None, generated_share=None, share_error=str(error))
        write(out/'decisions.json', decisions)
        metrics.update(profile=profile, scenario=scenario['id'])
        write(out/'report.json', metrics)
        return metrics


def verdict(m, horizon):
    """Per-group checks of section 2 for one report; None means a metric is missing."""
    def ok(value, test):
        return None if value is None else bool(test(value))
    share = [ok(m.get('network_share'), lambda v: v >= LIMITS['network_share']),
             ok(m.get('generated_share'), lambda v: v <= LIMITS['generated_share'])]
    safety = [m['contacts'] == 0, m['min_clearance_m'] >= LIMITS['min_clearance_m'],
              m['max_height_error_m'] <= LIMITS['max_height_error_m'],
              m['reverse_distance_body_m'] <= LIMITS['reverse_distance_body_m']]
    progress = None
    if horizon >= 180.:
        progress = (m['longest_no_new_cell']['duration_s'] <= LIMITS['max_no_new_cell_s']
                    and m['new_cells_last_60s'] >= LIMITS['min_new_cells_last_60s'])
    delivery = not m['faults'] and m['continuous_auto'] and m['completed']
    return dict(share=None if None in share else all(share), share_parts=share, safety=all(safety),
                progress=progress, delivery=bool(delivery))


def main(out, workers, only_profiles, only_scenarios):
    from evaluate.mvp_selection import write
    out.mkdir(parents=True, exist_ok=False)
    scenarios = [s for s in scenario_list() if not only_scenarios or s['id'] in only_scenarios]
    tasks = [(p, s) for s in scenarios for p in only_profiles]
    write(out/'protocol.json', dict(version='reiteration-baseline-v1', profiles=only_profiles,
        scenarios=[s['id'] for s in scenarios], limits=LIMITS))
    reports = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_case, out, p, s): (p, s) for p, s in tasks}
        for future in as_completed(futures):
            p, s = futures[future]
            try:
                reports.append((s, future.result()))
                print('done', s['id'], p, flush=True)
            except Exception as error:  # a failed case is reported, never hidden
                print('FAILED', s['id'], p, repr(error), flush=True)
                reports.append((s, dict(profile=p, scenario=s['id'], error=repr(error))))
    table = []
    for s, m in sorted(reports, key=lambda x: (x[0]['id'], PROFILES.index(x[1]['profile']))):
        table.append(m if 'error' in m else dict(m, verdict=verdict(m, s['horizon_s'])))
    write(out/'summary.json', table)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--profiles', nargs='+', default=list(PROFILES), choices=PROFILES)
    parser.add_argument('--scenarios', nargs='*', default=[])
    a = parser.parse_args()
    main(a.out, a.workers, a.profiles, a.scenarios)
