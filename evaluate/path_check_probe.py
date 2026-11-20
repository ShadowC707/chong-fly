"""Passive estimate: would a 'remembered points vs predicted body path' check have caught each card?

The check (not part of any profile, nothing it computes reaches the controller): every tick, take
all remembered points, the body velocity estimated from flow (memory observer, filtered 0.12 s),
and the segment the body would cover while braking, L = stop distance at that speed from the
identified round-2 brake model (k = 500, full network brake). A point triggers the check when it
lies ahead along the velocity (positive projection) and within r_veto 0.30 m of the segment after
subtracting its memory error. For transport memories the error is recomputed with the calibrated
round-5 model (0.035 + 0.020*path + 0.005*range), so all rounds are judged with the same memory error.

Each card is replayed with its own profile (the runs are deterministic, the probe only reads).
For each failure event the probe reports the first trigger in the WINDOW_S before the failure, the
true clearance to the failing obstacle then, and an estimate of whether braking from that moment
would have kept the clearance >= 0.10 m: clearance_at_trigger - stop_distance * v_toward/v >= 0.10,
with v_toward the true approach speed to the obstacle. It cannot model the different trajectory a
real stage would cause, so it is an estimate of reach, not a replacement for a run.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import argparse
from contextlib import closing
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WINDOW_S = 5.
R_VETO = .30
MIN_SPEED = .03


def stop_table():
    from evaluate.brake_identification import simulate, stage1_prop
    ident = json.loads((ROOT/'output/pipeline_v1_round2/brake_identification.json').read_text(encoding='utf-8'))
    model, u = ident['model'], ident['observed']['min_network_brake_pwm']
    speeds = np.linspace(0., .4, 41)
    stops = [0.]+[simulate(model, lambda un, f: stage1_prop(un, f, 500.), v, u)['stop_m'] for v in speeds[1:]]
    return speeds, np.array(stops)


def probe_tick(memory, now_s, speeds, stops):
    from edge.survey_memory import rotation
    o = memory.observer
    if o.fault or 'heading_rad' not in memory.report:
        return None
    v = np.asarray(o.velocity, float)
    speed = float(np.linalg.norm(v))
    if speed < MIN_SPEED:
        return dict(speed=speed, trigger=False)
    if hasattr(memory, 'points'):
        if not len(memory.points):
            return dict(speed=speed, trigger=False)
        local = (memory.points-o.position) @ rotation(0., 0., memory.report['heading_rad'])
        rng = np.linalg.norm(local, axis=1)
        error = .035+.020*(o.distance-memory.path0)+.005*rng
    else:
        r = memory.remembered(now_s)
        if r is None:
            return dict(speed=speed, trigger=False)
        local, error, _ = r
    d = v/speed
    length = float(np.interp(speed, speeds, stops))
    proj = local[:, :2] @ d
    ahead = proj > 0
    if not ahead.any():
        return dict(speed=speed, trigger=False, length=length)
    p = local[ahead]
    t = np.clip(proj[ahead], 0., length)
    nearest = np.c_[t[:, None]*d[None, :], np.zeros(len(t))]
    gap = np.linalg.norm(p-nearest, axis=1)-error[ahead]
    return dict(speed=speed, trigger=bool(gap.min() < R_VETO), length=length, min_gap=float(gap.min()))


def run(profile, scenario_id, events, speeds, stops):
    from evaluate.failure_card import nearest_point, obstacles
    from evaluate.reiteration_baseline import build_runner, scenario_list
    scenario = next(s for s in scenario_list() if s['id'] == scenario_id)
    targets = {e['obstacle'] for e in events}
    ticks = []
    with closing(build_runner(profile, scenario)) as runner:
        runner.resume()
        runner.event(dict(type='mode', mode='AUTO'))
        items = {name: (k, o) for name, k, o in obstacles(runner.env)}
        memory = runner.system.runtime.coordinator.memory
        last_t = max(e['time_s'] for e in events)
        while not runner.terminal and runner.time_s <= last_t+.05:
            t = runner.time_s
            n = len(runner.rows)
            runner.advance(5)
            if len(runner.rows) <= n:
                break
            row = runner.rows[n]
            pos, vel = np.asarray(row['position_m']), np.asarray(row['velocity_mps'])
            geo = {}
            for name in targets:
                k, o = items[name]
                q = nearest_point(k, o, pos)
                dist = float(np.linalg.norm(q-pos))
                toward = float(vel @ (q-pos)/dist) if dist > 0 else 0.
                geo[name] = dict(clearance=dist-runner.env.collision_radius_m, toward=toward,
                                 speed=float(np.linalg.norm(vel[:2])))
            ticks.append(dict(t=t, probe=probe_tick(memory, t, speeds, stops), geo=geo))
    out = []
    for e in events:
        window = [k for k in ticks if e['time_s']-WINDOW_S-1e-9 <= k['t'] <= e['time_s']]
        first = next((k for k in window if k['probe'] and k['probe']['trigger']), None)
        share = float(np.mean([bool(k['probe'] and k['probe']['trigger']) for k in window])) if window else 0.
        r = dict(e, triggered=first is not None, trigger_share=share)
        if first:
            g = first['geo'][e['obstacle']]
            stop = float(np.interp(g['speed'], speeds, stops))
            approach = stop*max(g['toward'], 0.)/g['speed'] if g['speed'] > 1e-6 else 0.
            r.update(first_trigger_s=first['t'], lead_s=e['time_s']-first['t'], clearance_at_trigger=g['clearance'],
                     approach_during_stop_m=approach, closes=bool(g['clearance']-approach >= .10))
        else:
            r.update(closes=False)
        out.append(r)
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile', required=True)
    p.add_argument('--scenario', required=True)
    p.add_argument('--events', required=True, help='JSON list of {card, time_s, obstacle, min_clearance_m, mechanism}')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    speeds, stops = stop_table()
    result = run(a.profile, a.scenario, json.loads(a.events), speeds, stops)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=1), encoding='utf-8')
    print(json.dumps(result))
