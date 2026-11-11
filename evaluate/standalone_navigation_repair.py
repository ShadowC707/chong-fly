"""Run one coordinator profile on the exact recorded Standalone user room."""
import argparse
from collections import Counter
from contextlib import closing
import json
from pathlib import Path

import numpy as np

from demo.runner import DemoRunner
from edge.runtime import sha256
from evaluate.compare_maneuver_repair import summarize
from evaluate.mvp_selection import write
from simulation.navigation_metrics import progress_metrics, waiting_metrics

USER_ROOM = Path(__file__).resolve().parents[1]/'data/scenarios/reiteration_v1/user_room.json'


def run(out, profile, *, auto_at_start=False):
    fixture = json.loads(USER_ROOM.read_text(encoding='utf-8'))
    horizon = fixture['horizon_s']
    with closing(DemoRunner(
        controller_mode=profile, scene=fixture['scene'], seed=fixture['seed'], start=fixture['start'],
        yaw=fixture['yaw'], scene_options=fixture['scene_options'],
        collision_radius_m=fixture['geometry']['collision_radius_m'], max_duration_s=horizon,
    )) as runner:
        manifest = runner.manifest()
        if manifest['geometry'] != fixture['geometry']:
            raise ValueError('Generated room differs from the recorded geometry')
        out.mkdir(parents=True, exist_ok=False)
        auto_step = 3 if profile.startswith('phased_') else 5
        if auto_at_start:
            auto_step = round(fixture['auto_request_s']/runner.step_dt)
        write(out/'protocol.json', dict(profile=profile, scenario=fixture['version'], scenario_sha256=sha256(USER_ROOM),
            auto_request_s=auto_step*runner.step_dt, horizon_s=horizon, manifest=manifest))
        runner.resume()
        runner.advance(auto_step)
        runner.event(dict(type='mode', mode='AUTO'))
        decisions = []
        runner.advance(5-auto_step)
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
        auto_start = next((i for i, value in enumerate(auto) if value == 'auto'), None)
        metrics.update(progress_metrics(time, xy, samples, runner.time_s))
        metrics.update(waiting_metrics(decisions), pause_reason=runner.pause_reason,
            completed=runner.pause_reason == 'duration_limit' and runner.time_s == horizon,
            continuous_auto=auto_start is not None and all(value == 'auto' for value in auto[auto_start:]),
            path_m=metrics['distance_m'],
            phase_events=Counter(r['decision']['phased_event'] for r in decisions
                                 if r['decision'].get('phased_event')),
            recovery_events=Counter(r['decision']['recovery_event'] for r in decisions
                                    if r['decision'].get('recovery_event')),
            overlay_events=Counter(reason for r in decisions for reason in r['decision'].get('reasons', [])
                                   if reason in ('threat_gated_brake', 'consistent_escape', 'side_clearance_yaw')),
            bridge_faults=[e for e in runner.events if e['kind'] == 'bridge_fault'])
        np.savez_compressed(out/'trace.npz', time=time, xy=xy,
            clearance=[r['clearance_m'] for r in rows],
            control_time=[r['time_s'] for r in controls],
            network=[r['trace']['network_pwm'] or [np.nan]*4 for r in controls],
            coordinated=[r['trace']['coordinated_pwm'] or [np.nan]*4 for r in controls],
            accepted=[r['trace']['accepted_pwm'] or [np.nan]*4 for r in controls])
        write(out/'decisions.json', decisions)
        write(out/'report.json', metrics)
        print(json.dumps(metrics, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--profile', required=True)
    parser.add_argument('--auto-at-start', action='store_true',
                        help='Request AUTO at t=0, matching the interactive Start button')
    args = parser.parse_args()
    run(args.out, args.profile, auto_at_start=args.auto_at_start)
