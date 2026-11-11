"""Capture a bounded brake-and-look maneuver for the Stage 1 restart check."""
import argparse
from contextlib import closing
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from demo.runner import DemoRunner
from edge.runtime import sha256
from evaluate.passive_corridor import STAGE0, USER_ROOM, check_parity
from simulation.drone_env import BoxObstacle
from simulation.phased_yaw import YawRamp


SCAN_PWM = 1360.
SCAN_S = 1.25
WIDE_SCAN_S = 2.05
QUIET_FLOW = .015
QUIET_DWELL_S = .12
ASSESS_AFTER_S = 4.
END_AFTER_S = 5.


def install_probe(runner, start_s, scan_s):
    coordinator = runner.system.runtime.coordinator
    original = coordinator.apply
    yaw = YawRamp()
    state = dict(turn_start_s=None, quiet_s=0.)

    def apply(network_pwm, *, flow_xy, tof_m, dt, stop_requested=False):
        action = original(network_pwm, flow_xy=flow_xy, tof_m=tof_m, dt=dt,
                          stop_requested=stop_requested)
        if runner.time_s < start_s or stop_requested:
            yaw.synchronize(float(action[3]))
            return action
        flow = np.array([coordinator.filtered_flow, coordinator.filtered_flow_y])
        action[1:3] = 1500. + np.clip(400. * np.array([flow[1], -flow[0]]), -100., 100.)
        if state['turn_start_s'] is None:
            state['quiet_s'] = state['quiet_s'] + dt if np.linalg.norm(flow) <= QUIET_FLOW else 0.
            if state['quiet_s'] + 1e-9 >= QUIET_DWELL_S:
                state['turn_start_s'] = runner.time_s
        start = state['turn_start_s']
        turning = start is not None and runner.time_s - start < scan_s - 1e-9
        action[3] = yaw.step(SCAN_PWM if turning else 1500., turning, dt)
        coordinator.previous_pr = action[1:3].copy()
        return action

    coordinator.apply = apply
    return state


def capture(out, scene, hidden_box=None, *, wide=False):
    fixture = json.loads(USER_ROOM.read_text(encoding='utf-8'))
    live = scene == 'cylinder'
    if live and (hidden_box is not None or wide):
        raise ValueError('Paired obstacle and wide sweep are limited to the clear test scene')
    start_s = 113.5 if live else .80
    scan_s = WIDE_SCAN_S if wide else SCAN_S
    settings = dict(scene=fixture['scene'], scene_options=fixture['scene_options'],
                    start=fixture['start'], yaw=fixture['yaw']) if live else dict(scene='clear')
    with closing(DemoRunner(controller_mode='candidate_a_brake', seed=fixture['seed'],
                            max_duration_s=start_s + END_AFTER_S, **settings)) as runner:
        manifest = runner.manifest()
        if live and manifest['geometry'] != fixture['geometry']:
            raise ValueError('Recorded geometry mismatch')
        if hidden_box is not None:
            runner.env.boxes.append(hidden_box)
            manifest['geometry']['boxes'].append(asdict(hidden_box))
        if not live:
            runner.obs = runner.env._sample_observation(0.)
        state = install_probe(runner, start_s, scan_s)
        runner.resume()
        runner.event(dict(type='mode', mode='AUTO'))
        while not runner.terminal:
            runner.advance(5)
        if live:
            check_parity(runner.rows, STAGE0 / 'live_brake' / 'trace.npz', start_s)
        if state['turn_start_s'] is None or runner.time_s < start_s + ASSESS_AFTER_S:
            raise RuntimeError(f'Scan did not finish in {scene}: {runner.pause_reason}')
        rows = [r for r in runner.rows if not live or r['time_s'] >= start_s - .24 - 1e-8]
        meta = dict(scene=scene, start_s=start_s, turn_start_s=state['turn_start_s'],
                    yaw_request_s=scan_s, target_yaw_pwm=SCAN_PWM, assess_s=start_s + ASSESS_AFTER_S,
                    end_s=runner.time_s, terminal_reason=runner.pause_reason,
                    exact_live_prefix=live, manifest=manifest, geometry=manifest['geometry'],
                    events=runner.events, evaluator_sha256=sha256(__file__))
        out.mkdir(parents=True, exist_ok=False)
        (out / 'capture.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
        (out / 'rows.json').write_text(json.dumps(rows), encoding='utf-8')
        print(f'{scene} scan={state["turn_start_s"]:.2f} assess={meta["assess_s"]:.2f}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scene', choices=('cylinder', 'clear'))
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--box', type=float, nargs=6)
    parser.add_argument('--wide', action='store_true')
    args = parser.parse_args()
    capture(args.out, args.scene, BoxObstacle(*args.box) if args.box else None, wide=args.wide)
