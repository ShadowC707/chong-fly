"""Replay the fixed Stage 1 diagnostic windows without changing flight commands."""
import argparse
from contextlib import closing
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from demo.runner import DemoRunner
from edge.runtime import sha256
from evaluate.standalone_navigation_repair import USER_ROOM
from simulation import side_clearance
from simulation.drone_env import BoxObstacle


ROOT = USER_ROOM.parents[3]
STAGE0 = ROOT / 'output/reiteration_stage0_v1'
REVISIT = ROOT / 'output/standalone_navigation_live_regression_v1/phased_revisit_probe'
CASES = {
    'cylinder': dict(profile='candidate_a_brake', start_s=113.50, end_s=117., baseline=STAGE0 / 'live_brake'),
    'box': dict(profile='phased_side_guard', start_s=154.30, end_s=157., baseline=REVISIT),
    'clear': dict(profile='candidate_a', start_s=2., end_s=4., baseline=None),
    'clear_hidden': dict(profile='candidate_a', start_s=2., end_s=2.02, baseline=None),
}
WARMUP_S = .24
STOP_HORIZON_S = 5.
HIDDEN_BOX = BoxObstacle(-.26, -.16, .28, .48, .45, 1.55)


class RevisitProbe(side_clearance.SideClearanceCoordinator):
    """Reproduce the one alternate turn of the historical negative control."""

    def reset(self):
        super().reset()
        self.probe_s = 0.
        self.probe_history = []
        self.probe_pair = None

    def _choose_direction(self, image):
        direction = super()._choose_direction(image)
        signature = np.minimum(image, 2.)
        if self.probe_pair is None and not self.repeat_episode:
            for tick, earlier, side in reversed(self.probe_history):
                if (10 <= self.probe_s - tick <= 60 and side == direction
                        and np.mean(np.abs(earlier - signature)) <= .05):
                    self.probe_pair = (tick, self.probe_s)
                    direction = -direction
                    break
        self.probe_history.append((self.probe_s, signature, direction))
        return direction

    def apply(self, network_pwm, *, flow_xy, tof_m, dt, stop_requested=False):
        self.probe_s += dt
        return super().apply(network_pwm, flow_xy=flow_xy, tof_m=tof_m, dt=dt,
                             stop_requested=stop_requested)


def check_parity(rows, reference, end_s):
    prefix = [row for row in rows if row['time_s'] < end_s]
    controls = [row for row in prefix if row['plant_input_pwm'] is not None
                and abs(row['time_s'] / .02 - round(row['time_s'] / .02)) < 1e-7]
    with np.load(reference) as baseline:
        physical = baseline['time'] < end_s
        control = baseline['control_time'] < end_s
        for key, actual in (('time', [r['time_s'] for r in prefix]),
                            ('xy', [r['position_m'][:2] for r in prefix]),
                            ('clearance', [r['clearance_m'] for r in prefix])):
            np.testing.assert_array_equal(actual, baseline[key][physical])
        np.testing.assert_array_equal([r['time_s'] for r in controls], baseline['control_time'][control])
        for key in ('network', 'coordinated', 'accepted'):
            actual = [r['trace'][key + '_pwm'] or [np.nan] * 4 for r in controls]
            np.testing.assert_array_equal(actual, baseline[key][control])


def capture(name, out, *, stop=False):
    case = CASES[name]
    fixture = json.loads(USER_ROOM.read_text(encoding='utf-8'))
    start_s = case['start_s']
    end_s = start_s + STOP_HORIZON_S if stop else case['end_s']
    kwargs = (dict(scene=fixture['scene'], start=fixture['start'], yaw=fixture['yaw'],
                   scene_options=fixture['scene_options']) if case['baseline'] else
              dict(scene='clear', start=(-.7, 0., 1.), yaw=0.))
    native = side_clearance.SideClearanceCoordinator
    if name == 'box':
        side_clearance.SideClearanceCoordinator = RevisitProbe
    try:
        runner = DemoRunner(controller_mode=case['profile'], seed=fixture['seed'],
                            max_duration_s=end_s, collision_radius_m=.2, **kwargs)
    finally:
        side_clearance.SideClearanceCoordinator = native
    with closing(runner):
        manifest = runner.manifest()
        if case['baseline'] and manifest['geometry'] != fixture['geometry']:
            raise ValueError('Live geometry mismatch')
        if name == 'clear_hidden':
            runner.env.boxes.append(HIDDEN_BOX)
            manifest['geometry']['boxes'].append(asdict(HIDDEN_BOX))
        if not case['baseline']:
            # Both positive controls capture with their actual physics geometry.
            runner.obs = runner.env._sample_observation(0.)
        runner.resume()
        runner.event(dict(type='mode', mode='AUTO'))
        runner.advance(round((start_s - WARMUP_S) / runner.step_dt))
        phases = {}
        while not runner.terminal:
            t = runner.time_s
            if stop and abs(t - start_s) < 1e-8:
                if runner.previous_source != 'auto':
                    raise ValueError('STOP requires actual AUTO authority')
                runner.event(dict(type='stop'))
            runner.advance(5)
            phases[t] = runner.system.runtime.coordinator.last_decision.get('phased_phase')
        if case['baseline']:
            check_parity(runner.rows, case['baseline'] / 'trace.npz', start_s if stop else end_s)
        rows = ([row for row in runner.rows if row['time_s'] >= start_s - WARMUP_S - 1e-8]
                if case['baseline'] else runner.rows)
        out.mkdir(parents=True, exist_ok=False)
        meta = dict(case=name, profile=case['profile'], stop_requested=stop, start_s=start_s,
                    end_s=runner.time_s, reason=runner.pause_reason, phases=phases, events=runner.events,
                    geometry=manifest['geometry'], manifest=manifest,
                    exact_parity=bool(case['baseline']), fixture_sha256=sha256(USER_ROOM),
                    evaluator_sha256=sha256(__file__))
        (out / 'capture.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
        (out / 'rows.json').write_text(json.dumps(rows), encoding='utf-8')
        print(f'{name} stop={stop} end={runner.time_s:.3f} reason={runner.pause_reason}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', choices=CASES)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--stop', action='store_true')
    args = parser.parse_args()
    capture(args.case, args.out, stop=args.stop)
