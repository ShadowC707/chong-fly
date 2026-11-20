"""Protocol 6a failure cards: rerun one working-set case with truth-side instrumentation.

For every contact or clearance below 0.10 m the card records the obstacle, whether its nearest
point lies in the ToF 8x8 frustum (45x45 deg with the body attitude) and whether any of the 64
point rays actually returns it, the ToF minimum and centre, what the hazard memory holds about it,
what every pipeline stage and the network did over the last 3 s, and a category proposal a/b/c/d.

Truth geometry is used only here, after the flight, for diagnosis; it never reaches the controller.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import argparse
from collections import Counter
from contextlib import closing
import json
import math
from pathlib import Path

import numpy as np

HALF_FOV_DEG = 22.5
WINDOW_S = 3.
CLEARANCE_LIMIT_M = .10
ON_SURFACE_M = .05  # a remembered point closer than this to the true surface belongs to it


def obstacles(env):
    out = [(f'box{i}', 'box', b) for i, b in enumerate(env.boxes)]
    out += [(f'cylinder{i}', 'cylinder', c) for i, c in enumerate(env.cylinders)]
    return out + [('room', 'room', env.room)]


def nearest_point(kind, o, p):
    """Closest surface point of one obstacle to p (for the room: the nearest wall/floor/ceiling)."""
    p = np.asarray(p, float)
    if kind == 'box':
        return np.clip(p, [o.x_min, o.y_min, o.z_min], [o.x_max, o.y_max, o.z_max])
    if kind == 'cylinder':
        radial = p[:2]-[o.center_x, o.center_y]
        n = np.linalg.norm(radial)
        xy = np.array([o.center_x, o.center_y]) + (radial/n*min(n, o.radius) if n > 0 else 0.)
        return np.r_[xy, np.clip(p[2], 0., o.height)]
    faces = [(0, o.x_min), (0, o.x_max), (1, o.y_min), (1, o.y_max), (2, o.z_min), (2, o.z_max)]
    axis, value = min(faces, key=lambda f: abs(p[f[0]]-f[1]))
    q = p.copy()
    q[axis] = value
    return q


def surface_distance(kind, o, points):
    points = np.atleast_2d(points)
    if kind == 'box':
        lo, hi = np.array([o.x_min, o.y_min, o.z_min]), np.array([o.x_max, o.y_max, o.z_max])
        outside = np.linalg.norm(np.maximum(0, np.maximum(lo-points, points-hi)), axis=1)
        inside = np.min(np.minimum(points-lo, hi-points), axis=1)
        return np.where(outside > 0, outside, np.maximum(0, inside))
    if kind == 'cylinder':
        r = np.linalg.norm(points[:, :2]-[o.center_x, o.center_y], axis=1)
        dr, dz = r-o.radius, np.maximum(0, np.maximum(-points[:, 2], points[:, 2]-o.height))
        return np.hypot(np.maximum(dr, 0), dz) + np.where((dr < 0) & (dz == 0), -dr, 0)
    lo, hi = np.array([o.x_min, o.y_min, o.z_min]), np.array([o.x_max, o.y_max, o.z_max])
    return np.min(np.abs(np.concatenate([points-lo, hi-points], axis=1)), axis=1)


def body_rotation(euler):
    from edge.survey_memory import rotation
    return rotation(*euler)


def in_frustum(local):
    x, y, z = local
    az, el = math.degrees(math.atan2(y, x)), math.degrees(math.atan2(z, math.hypot(x, y)))
    return dict(azimuth_deg=az, elevation_deg=el, in_frustum=bool(x > 0 and abs(az) <= HALF_FOV_DEG
                                                                   and abs(el) <= HALF_FOV_DEG))


def ray_owners(env, pos, rot, items):
    """Rays (of 64) whose first hit is each obstacle, from the true pose and the sensor model."""
    from simulation.drone_env import RoomBoundaries
    tof = env.tof
    full = tof.cast_rays(pos, rot, env.room, env.cylinders, env.boxes)
    huge = RoomBoundaries(x_min=-1e3, x_max=1e3, y_min=-1e3, y_max=1e3, z_min=-1e3, z_max=1e3)
    out = {}
    for name, kind, o in items:
        if kind == 'room':
            alone = tof.cast_rays(pos, rot, env.room, [], [])
        else:
            alone = tof.cast_rays(pos, rot, huge, [o] if kind == 'cylinder' else [],
                                  [o] if kind == 'box' else [])
        out[name] = int(np.count_nonzero((alone < 1.-1e-6) & (alone <= full+1e-6)))
    return out


def memory_view(memory, pos, yaw, now_s, items):
    """Remembered points re-anchored at the true pose; per obstacle: count, newest age, gap, error."""
    from edge.survey_memory import rotation
    out = {name: dict(points=0) for name, _, _ in items}
    remembered = memory.remembered(now_s)
    if remembered is None:
        return out
    local, error, captured = remembered
    world = pos + local @ rotation(0., 0., yaw).T
    gap = np.linalg.norm(local, axis=1)-error
    bearing = np.degrees(np.arctan2(local[:, 1], local[:, 0]))
    for name, kind, ob in items:
        mask = surface_distance(kind, ob, world) < ON_SURFACE_M
        if not mask.any():
            continue
        i = int(np.argmin(np.where(mask, gap, np.inf)))
        out[name] = dict(points=int(mask.sum()), newest_age_s=float(now_s-captured[mask].max()),
                         oldest_age_s=float(now_s-captured[mask].min()), min_gap_m=float(gap[i]),
                         bearing_deg=float(bearing[i]), error_model_m=float(error[i]),
                         transport_error_m=float(surface_distance(kind, ob, world[i])[0]))
    return out


def run(profile, scenario_id):
    from evaluate.reiteration_baseline import build_runner, scenario_list
    scenario = next(s for s in scenario_list() if s['id'] == scenario_id)
    ticks = []
    with closing(build_runner(profile, scenario)) as runner:
        runner.resume()
        runner.event(dict(type='mode', mode='AUTO'))
        env = runner.env
        items = obstacles(env)
        coordinator = runner.system.runtime.coordinator
        memory = getattr(coordinator, 'memory', None)
        while not runner.terminal:
            t = runner.time_s
            n = len(runner.rows)
            runner.advance(5)
            row = runner.rows[n] if len(runner.rows) > n else None
            if row is None:
                break
            d = coordinator.last_decision.copy()
            pos, euler = np.asarray(row['position_m']), np.asarray(row['attitude_rad'])
            rot = body_rotation(euler)
            dist = sorted(((float(np.linalg.norm(nearest_point(k, o, pos)-pos)), name, k, o)
                           for name, k, o in items), key=lambda x: x[0])[:3]
            near = [(name, k, o) for _, name, k, o in dist]
            geometry = {}
            for (dd, name, k, o) in dist:
                q = nearest_point(k, o, pos)
                geometry[name] = dict(distance_m=dd, clearance_m=dd-env.collision_radius_m,
                                      point=q.tolist(), **in_frustum(rot.T @ (q-pos)))
            rays = ray_owners(env, pos, rot, near)
            mem = memory_view(memory, pos, euler[2], t, near) if memory is not None else None
            ticks.append(dict(time_s=t, position_m=pos.tolist(), attitude_deg=np.degrees(euler).tolist(),
                velocity_mps=list(row['velocity_mps']), speed_mps=float(np.linalg.norm(row['velocity_mps'][:2])),
                clearance_m=row['clearance_m'], geometry=geometry, rays=rays, memory=mem,
                tof_min_m=d.get('tof_min_m'), tof_center_min_m=d.get('tof_center_min_m'),
                network=row['trace']['network_pwm'], coordinated=row['trace']['coordinated_pwm'],
                stages=d.get('stages'), reasons=list(d.get('reasons', []))))
        terminal = dict(collision=runner.rows[-1]['collision'], time_s=runner.rows[-1]['time_s'],
                        clearance_m=runner.rows[-1]['clearance_m'], pause_reason=runner.pause_reason)
        all_clearance = [(r['time_s'], r['clearance_m']) for r in runner.rows]
    return ticks, terminal, all_clearance


def failure_events(all_clearance, terminal):
    """One event per contiguous interval with clearance below the limit (the contact counts)."""
    events, start = [], None
    for i, (t, c) in enumerate(all_clearance):
        if c < CLEARANCE_LIMIT_M and start is None:
            start = i
        if (c >= CLEARANCE_LIMIT_M or i == len(all_clearance)-1) and start is not None:
            end = i if c < CLEARANCE_LIMIT_M else i-1
            seg = all_clearance[start:end+1]
            k = min(range(len(seg)), key=lambda j: seg[j][1])
            events.append(dict(start_s=seg[0][0], end_s=seg[-1][0], time_s=seg[k][0],
                               min_clearance_m=seg[k][1]))
            start = None
    if terminal['collision'] and events:
        events[-1].update(contact=terminal['collision'], time_s=terminal['time_s'])
    return events


def series(values, times, every=.5):
    out, last = [], -math.inf
    for t, v in zip(times, values):
        if t-last >= every-1e-9:
            out.append([round(t, 2), v])
            last = t
    return out


def card(ticks, event, scenario, profile):
    t_fail = event['time_s']
    window = [k for k in ticks if t_fail-WINDOW_S-1e-9 <= k['time_s'] <= t_fail]
    at = window[-1]
    obstacle = min(at['geometry'].items(), key=lambda kv: kv[1]['distance_m'])[0]
    seen = [k['time_s'] for k in ticks if k['time_s'] <= t_fail and k['rays'].get(obstacle, 0) > 0]
    in_fov = [k for k in window if obstacle in k['geometry']]
    before = next((k for k in ticks if k['time_s'] >= t_fail-WINDOW_S-1e-9), at)
    mem = at['memory'][obstacle] if at['memory'] else None
    times = [k['time_s'] for k in window]
    stage_names = [r['stage'] for r in (at['stages'] or [])]
    stages = {}
    for name in stage_names:
        recs = [next((r for r in (k['stages'] or []) if r['stage'] == name), None) for k in window]
        recs = [r for r in recs if r]
        stages[name] = dict(kind=recs[-1]['kind'], active_ticks=sum(r['active'] for r in recs), ticks=len(recs),
            reasons=Counter(r['reason'] for r in recs),
            changed=series([[round(a-b, 1) for a, b in zip(r['after'][1:], r['before'][1:])] for r in recs],
                           times[-len(recs):]))
    net = np.array([k['network'] for k in window if k['network']], float)
    braked = bool(len(net) and (net[:, 2] < 1500-10).any())
    stage_last_1s = any(r['active'] for k in window if k['time_s'] >= t_fail-1. for r in (k['stages'] or []))
    if not seen:
        category, why = 'a', 'no ToF ray ever returned this obstacle before the failure'
    elif not mem or not mem.get('points'):
        category, why = 'b', 'rays returned the obstacle earlier but the memory holds no point on it'
    elif not stage_last_1s:
        category, why = 'c', 'the memory holds points on the obstacle, no stage was active in the last 1 s'
    else:
        category, why = 'd', 'a stage was active in the last 1 s but the drone did not stop in time'
    return dict(scenario=scenario, profile=profile, event=event,
        obstacle=dict(id=obstacle, at_failure=at['geometry'][obstacle],
            in_frustum_share_last_3s=float(np.mean([k['geometry'][obstacle]['in_frustum'] for k in in_fov]))
            if in_fov else None,
            rays_at_failure=at['rays'].get(obstacle), last_ray_hit_s=seen[-1] if seen else None,
            first_ray_hit_s=seen[0] if seen else None),
        pose=dict(at_failure=dict(position_m=at['position_m'], attitude_deg=at['attitude_deg'],
            speed_mps=at['speed_mps'], velocity_mps=at['velocity_mps'])),
        sensors=dict(at_failure=dict(tof_min_m=at['tof_min_m'], tof_center_min_m=at['tof_center_min_m']),
            minus_3s=dict(time_s=before['time_s'], tof_min_m=before['tof_min_m'],
                          tof_center_min_m=before['tof_center_min_m'])),
        memory=dict(at_failure=mem, last_seen_age_s=(t_fail-seen[-1]) if seen else None,
            track=series([k['memory'][obstacle] if k['memory'] and obstacle in k['memory'] else None
                          for k in window], times)),
        geometry_track=series([dict(clearance_m=round(k['geometry'][obstacle]['clearance_m'], 3),
            az=round(k['geometry'][obstacle]['azimuth_deg'], 1), el=round(k['geometry'][obstacle]['elevation_deg'], 1),
            fov=k['geometry'][obstacle]['in_frustum'], rays=k['rays'].get(obstacle))
            if obstacle in k['geometry'] else None for k in window], times),
        stages=stages,
        network=dict(braked=braked, min_pitch=float(net[:, 2].min()) if len(net) else None,
            rpy=series([[round(v, 1) for v in k['network'][1:]] for k in window if k['network']],
                       [k['time_s'] for k in window if k['network']])),
        category_proposal=dict(category=category, rule=why))


def main(profile, scenario, out):
    ticks, terminal, clearance = run(profile, scenario)
    events = failure_events(clearance, terminal)
    cards = [card(ticks, e, scenario, profile) for e in events]
    out.mkdir(parents=True, exist_ok=True)
    (out/f'{scenario}_{profile}_cards.json').write_text(
        json.dumps(dict(terminal=terminal, events=events, cards=cards), indent=1, default=float),
        encoding='utf-8')
    print(json.dumps(dict(terminal=terminal, events=events,
        categories=[c['category_proposal'] for c in cards]), default=float))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile', default='pipeline_v1')
    p.add_argument('--scenario', required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    main(a.profile, a.scenario, a.out)
