"""Plan step 7: phased_side_guard in Betaflight SITL, compared with Standalone on the same scenes.

Scenes, each from the 5 start perturbations of the spread (+-2 cm, +-2 deg; evaluate/spread_runs.py):
  stop45  - the SITL long-flight arena (simulation.sitl_flight.long_scene), 45 s, STOP requested at 40 s;
  room180 - the live working-set room (user_room.json recorded geometry), 180 s, no STOP.
SITL: reiteration SITL profile (Candidate A adaptive-tau config, reiteration calibration), TWR 5,
0.25 ms physics, multi-rate sensors, reference units, phased_side_guard over the yaw-handover path.
The frozen evaluate/sitl_long_flight.py protocol is not touched or reused for this.

Modes:
  sitl        one SITL flight (run inside the chong-fly-sitl container, which has the firmware);
  standalone  one Standalone flight on the stop45 scene (room180 Standalone runs come from the spread);
  table       section-2 metrics, phase shares (FAILED separately) and the comparison table.

--profile (default phased_side_guard, the original step 7) selects the coordinator profile; step 9 item 4
flies phased_failed_exit with the same scenes, perturbations and metrics. Its room180 Standalone reference
is the step-9 spread (output/pipeline_v1_step9/spread_v2), same geometry and perturbations.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import argparse
from collections import Counter
from contextlib import closing
from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROFILE = 'phased_side_guard'  # set by --profile
SPREAD_ROOMS = {'phased_side_guard': ROOT/'output/pipeline_v1_spread/runs',
                'phased_failed_exit': ROOT/'output/pipeline_v1_step9/spread_v2'}


def scene(name, k):
    from evaluate.spread_runs import PERTURBATIONS
    from simulation.sitl_flight import long_scene
    dx, dy, dyaw = PERTURBATIONS[k]
    if name == 'stop45':
        spec, horizon, stop_at = long_scene(), 45., 40.
    elif name == 'room180':
        f = json.loads((ROOT/'data/scenarios/reiteration_v1/user_room.json').read_text(encoding='utf-8'))
        g = f['geometry']
        spec = dict(room=g['room'], boxes=g['boxes'], cylinders=g['cylinders'], start=list(f['start']),
                    yaw=float(f['yaw']), radius_m=g['collision_radius_m'])
        horizon, stop_at = 180., None
    else:
        raise ValueError('Unknown step-7 scene')
    spec = deepcopy(spec)
    spec['start'] = [spec['start'][0]+dx, spec['start'][1]+dy, spec['start'][2]]
    spec['yaw'] = float(spec['yaw'])+math.radians(dyaw)
    return spec, horizon, stop_at


def fly_sitl(out, name, k):
    from demo.profiles import CANDIDATE_A_CONFIG
    from demo.sitl import REITERATION_SOURCE, reiteration_profile
    from simulation.sitl_flight import flight_steps
    spec, horizon, stop_at = scene(name, k)
    rows, hashes = reiteration_profile()
    steps = flight_steps(out, REITERATION_SOURCE, name, 'sitl', 'reference_units', physics_dt_s=.00025,
        integration_profile='multi_rate', yaw_shaping=True, scene_override=spec, target_twr=5.,
        calibration_rows=rows, horizon_s=horizon, stop_at_s=stop_at, session_id=f'step7-{name}-p{k+1}',
        archive=True, runtime_config=CANDIDATE_A_CONFIG, controller_profile=PROFILE)
    with closing(steps):
        next(steps)
        try:
            while True:
                steps.send({})
        except StopIteration as end:
            result = end.value
    (out/'step7.json').write_text(json.dumps(dict(scene=name, perturbation=k+1, horizon_s=horizon, stop_at_s=stop_at, profile=PROFILE,
        input_hashes=hashes, result=result), indent=1, default=float), encoding='utf-8')
    return result


def fly_standalone(out, k):
    """Standalone phased_side_guard on the stop45 scene: same geometry, start and STOP time."""
    from evaluate.reiteration_baseline import build_runner
    spec, horizon, stop_at = scene('stop45', k)
    s = dict(id=f'stop45_p{k+1}', kind='mvp', horizon_s=horizon,
             scene=dict(seed=0, start=spec['start'], yaw=spec['yaw'], room=spec['room'], boxes=spec['boxes'],
                        cylinders=spec['cylinders']))
    with closing(build_runner(PROFILE, s)) as runner:
        runner.resume()
        runner.event(dict(type='mode', mode='AUTO'))
        decisions, stopped = [], False
        while not runner.terminal:
            if not stopped and runner.time_s >= stop_at-1e-9:
                runner.event(dict(type='stop'))
                stopped = True
            t = runner.time_s
            runner.advance(5)
            d = runner.system.runtime.coordinator.last_decision
            decisions.append(dict(time_s=t, phase=d.get('phased_phase'), event=d.get('phased_event')))
        rows = runner.rows
    out.mkdir(parents=True, exist_ok=True)
    controls = [r for r in rows if r['plant_input_pwm'] is not None and abs(r['time_s']/.02-round(r['time_s']/.02)) < 1e-7]
    np.savez_compressed(out/'trace.npz', time=[r['time_s'] for r in rows], pos=[r['position_m'] for r in rows],
        vel=[r['velocity_mps'] for r in rows], att=[r['attitude_rad'] for r in rows],
        clearance=[r['clearance_m'] for r in rows], collision=[r['collision'] is not None for r in rows],
        auto_row=[r['plant_source'] == 'auto' for r in rows],
        control_time=[r['time_s'] for r in controls], auto=[r['plant_source'] == 'auto' for r in controls],
        stop=[r['stop_requested'] for r in controls], min_tof=[r['min_tof_m'] for r in controls],
        network=[r['trace']['network_pwm'] or [np.nan]*4 for r in controls],
        coordinated=[r['trace']['coordinated_pwm'] or [np.nan]*4 for r in controls],
        accepted=[r['trace']['accepted_pwm'] or [np.nan]*4 for r in controls])
    (out/'decisions.json').write_text(json.dumps(decisions), encoding='utf-8')


def metrics(time, pos, vel, yaw, clearance, auto_rows, ctime, auto, stop, network, coordinated, phases, horizon,
            stop_at, collision, height_error):
    from simulation.navigation_metrics import progress_metrics, reverse_outside_corridor, share_metrics
    m = dict(share_metrics(network, coordinated, auto, stop))
    outside, total = reverse_outside_corridor(time, pos, vel, yaw, auto_rows)
    speed = np.linalg.norm(np.asarray(vel)[:, :2], axis=1)
    end = float(time[-1])
    completed = collision is None and end >= horizon-.05
    last = np.asarray(time) >= horizon-1.
    c = Counter(phases)
    n = max(sum(c.values()), 1)
    m.update(contacts=int(collision is not None), collision=collision, min_clearance_m=float(np.min(clearance)),
             max_height_error_m=float(height_error), reverse_outside_corridor_m=outside, reverse_distance_body_m=total,
             duration_s=end, completed=bool(completed), phase_share={k: v/n for k, v in c.most_common()},
             failed_share=c.get('FAILED', 0)/n, mean_speed_mps=float(speed.mean()), path_m=float(
                 np.linalg.norm(np.diff(np.asarray(pos)[:, :2], axis=0), axis=1).sum()))
    if stop_at is not None:
        m['stop_passed'] = bool(completed and last.any() and (speed[last] < .03).all())
    if horizon >= 180.:
        idx = np.arange(len(time))
        m.update(progress_metrics(np.asarray(time), np.asarray(pos)[:, :2], idx, end))
    return m


def sitl_metrics(folder, name, k):
    spec, horizon, stop_at = scene(name, k)
    report = json.loads((folder/'report.json').read_text())
    with np.load(folder/'trace.npz') as z:
        t, s = z['time_s'], z['state']
        stride = 80  # 0.25 ms physics -> 50 Hz
        idx = np.arange(0, len(t), stride)
        gaps = np.r_[z['clearance'][:1], z['clearance']]  # clearance[i] is after physics step i -> state i+1
        ra = np.asarray(z['receiver_auto'], bool)
        auto_rows = ra[np.minimum(idx, len(ra)-1)] if len(ra) else np.zeros(len(idx), bool)
        ctime = z['control_time']
        auto = ra[np.minimum(np.round(ctime/.00025).astype(int), len(ra)-1)] if len(ra) else np.zeros(len(ctime), bool)
        stop = (ctime >= stop_at-1e-9) if stop_at is not None else np.zeros(len(ctime), bool)
        net, coord = z['network_pwm'], z['coordinated_pwm']
        tof = z['tof']
    phases = [d['phase'] for d in json.loads((folder/'decisions.json').read_text())]
    m = metrics(t[idx], s[idx, :3], s[idx, 3:6], s[idx, 8], gaps[np.minimum(idx, len(gaps)-1)], auto_rows, ctime, auto,
                stop, net, coord, phases, horizon, stop_at, report.get('collision'), report['max_height_error_m'])
    m['min_clearance_m'] = float(report['min_clearance_m']) if report.get('min_clearance_m') is not None else m['min_clearance_m']
    m.update(authority_passed=report.get('authority_passed'), error=report.get('error'),
             mean_min_tof_m=float(np.mean(np.min(np.asarray(tof), axis=1)))*3.5 if len(tof) else None,
             mean_abs_pitch_cmd=float(np.nanmean(np.abs(np.asarray(coord)[:, 2]-1500))),
             mean_abs_yaw_cmd=float(np.nanmean(np.abs(np.asarray(coord)[:, 3]-1500))))
    return m


def standalone_metrics(folder, name, k):
    spec, horizon, stop_at = scene(name, k)
    with np.load(folder/'trace.npz') as z:
        if name == 'stop45':
            t, pos, vel, att = z['time'], z['pos'], z['vel'], z['att']
            clr, coll, auto_rows = z['clearance'], z['collision'], z['auto_row']
            ctime, auto, stop, net, coord, tof = (z['control_time'], z['auto'], z['stop'], z['network'],
                                                  z['coordinated'], z['min_tof'])
            collision = 'contact' if coll.any() else None
            height = float(np.max(np.abs(pos[:, 2]-1)))
            phases = [d['phase'] for d in json.loads((folder/'decisions.json').read_text())]
            m = metrics(t, pos, vel, att[:, 2], clr, auto_rows, ctime, auto, stop, net, coord, phases, horizon, stop_at,
                        collision, height)
            m.update(mean_min_tof_m=float(np.mean(tof)), mean_abs_pitch_cmd=float(np.nanmean(np.abs(coord[:, 2]-1500))),
                     mean_abs_yaw_cmd=float(np.nanmean(np.abs(coord[:, 3]-1500))))
            return m
    raise ValueError('room180 Standalone metrics come from the spread reports')


def room_standalone(k):
    folder = SPREAD_ROOMS[PROFILE]/f'live_room_p{k+1}'/PROFILE
    r = json.loads((folder/'report.json').read_text())
    d = json.loads((folder/'decisions.json').read_text())
    c = Counter(x['decision'].get('phased_phase') for x in d)
    n = max(sum(c.values()), 1)
    with np.load(folder/'trace.npz') as z:
        coord = z['coordinated']
    r.update(phase_share={k2: v/n for k2, v in c.most_common()}, failed_share=c.get('FAILED', 0)/n,
             mean_abs_pitch_cmd=float(np.nanmean(np.abs(coord[:, 2]-1500))),
             mean_abs_yaw_cmd=float(np.nanmean(np.abs(coord[:, 3]-1500))),
             mean_min_tof_m=float(np.mean([x['decision'].get('tof_min_m') or np.nan for x in d])))
    return r


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['sitl', 'standalone', 'table'])
    p.add_argument('--scene', choices=['stop45', 'room180'])
    p.add_argument('--k', type=int, help='perturbation index 0..4')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--profile', choices=['phased_side_guard', 'phased_failed_exit'], default=PROFILE)
    a = p.parse_args()
    PROFILE = a.profile
    if a.mode == 'sitl':
        print(json.dumps(fly_sitl(a.out, a.scene, a.k), default=float)[:2000])
    elif a.mode == 'standalone':
        fly_standalone(a.out, a.k)
        print('ok', a.out)
    else:
        rows = []
        for name in ('stop45', 'room180'):
            for k in range(5):
                sf = a.out/'sitl'/f'{name}_p{k+1}'
                sm = sitl_metrics(sf, name, k) if (sf/'report.json').is_file() else None
                st = (standalone_metrics(a.out/'standalone'/f'{name}_p{k+1}', name, k) if name == 'stop45'
                      else room_standalone(k))
                rows.append(dict(scene=name, k=k+1, sitl=sm, standalone=st))
        (a.out/'step7_metrics.json').write_text(json.dumps(rows, indent=1, default=float), encoding='utf-8')
        print('written', a.out/'step7_metrics.json')
