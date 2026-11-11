"""Check restart after a bounded scan, with a physical indistinguishability pair."""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from edge.runtime import sha256
from edge.survey_memory import rotation
from evaluate.passive_corridor_readout import (
    MARGIN_M, SPEED_BOUND_MPS, geometric_qa, geometry, load, shadow,
)
from simulation.drone_env import ToFRaycaster


def assessment(name, meta, rows):
    t = meta['assess_s']
    inputs = [row for row in rows if row['time_s'] >= meta['start_s'] - .24 - 1e-8]
    history = [(row, decision) for row, decision in shadow(name, inputs, meta['start_s'])
               if row['time_s'] <= t + 1e-8]
    row, decision = history[-1]
    assert abs(row['time_s'] - t) < 1e-8
    tail = [r for r in rows if meta['start_s'] <= r['time_s'] <= t + 1e-8]
    position, times = np.array([r['position_m'] for r in inputs]), np.array([r['time_s'] for r in inputs])
    speed_bound_verified = bool(np.max(np.linalg.norm(np.diff(position, axis=0), axis=1) / np.diff(times)) <= SPEED_BOUND_MPS)
    assert speed_bound_verified, 'Displacement assumption falsified'
    settling = [r for r in tail if r['time_s'] >= t - .25 - 1e-8]
    speed = max(float(np.linalg.norm(r['velocity_mps'][:2])) for r in settling)
    yaw_rate = max(abs(math.degrees(r['angular_velocity_rad_s'][2])) for r in settling)
    aligned = {d['tof_id']: d['capture_alignment'] for _, d in history if d['capture_alignment'] is not None}
    angles = [math.degrees(item[3]) for item in aligned.values()]
    # The 45-degree nominal span describes angular reach only. The measured
    # directions remain 64 point rays and do not establish volume coverage.
    path_hit = geometric_qa(row, np.asarray(decision['path_body_m']), *geometry(meta))
    return row, dict(assessment=decision, exact_live_prefix=meta['exact_live_prefix'],
                     turn_start_s=meta['turn_start_s'], delivered_distinct_aligned_tof=len(aligned),
                     nominal_horizontal_span_deg=max(angles) - min(angles) + 45.,
                     sensor_heading_span_deg=max(angles) - min(angles),
                     min_true_scan_clearance_m=min(r['clearance_m'] for r in tail),
                     true_clearance_at_assessment_m=row['clearance_m'],
                     true_proposed_corridor_hit=path_hit,
                     horizontal_speed_max_last_250ms_mps=speed, yaw_rate_max_last_250ms_deg_s=yaw_rate,
                     settled_in_qa=speed <= .03 and yaw_rate <= 3.,
                     continuous_auto=all(r['plant_source'] == 'auto' for r in tail),
                     bridge_faults=[e for e in meta['events'] if e['kind'] == 'bridge_fault'])


def target_visibility(meta, rows, *, omit_cylinder=False):
    room, boxes, cylinders = geometry(meta)
    fewer_boxes = boxes if omit_cylinder else boxes[:-1]
    fewer_cylinders = cylinders[1:] if omit_cylinder else cylinders
    by_time = {round(r['time_s'], 8): r for r in rows}
    captures = {r['sensors']['tof']['sample_id']: r['sensors']['tof'] for r in rows
                if meta['start_s'] <= r['time_s'] <= meta['assess_s'] + 1e-8
                and r['sensors']['tof']['state'] == 'new'}
    caster = ToFRaycaster()
    hits = []
    for sample in captures.values():
        t = sample['timestamp_s']
        source = by_time[round(t, 8)]
        p, orientation = source['position_m'], rotation(*source['attitude_rad'])
        native = caster.cast_rays(p, orientation, room, cylinders, boxes)
        without = caster.cast_rays(p, orientation, room, fewer_cylinders, fewer_boxes)
        np.testing.assert_allclose(native, sample['value'], rtol=0, atol=1e-5)
        rays = int(np.count_nonzero(without > native + 1e-5))
        if rays:
            hits.append(dict(time_s=t, rays=rays))
    return dict(frames_checked=len(captures), frames_with_target=len(hits),
                first=hits[0] if hits else None, last=hits[-1] if hits else None)


def identical_pair(clear_rows, hidden_rows, cutoff_s):
    a = [r for r in clear_rows if r['time_s'] <= cutoff_s + 1e-8]
    b = [r for r in hidden_rows if r['time_s'] <= cutoff_s + 1e-8]
    assert len(a) == len(b) and abs(a[-1]['time_s'] - cutoff_s) < 1e-8
    for clear, hidden in zip(a, b):
        for key in ('time_s', 'sensors', 'position_m', 'velocity_mps', 'attitude_rad', 'plant_input_pwm'):
            assert json.dumps(clear[key]) == json.dumps(hidden[key]), (clear['time_s'], key)
    return dict(equal_physics_ticks=len(a), equal_tof_frames=len({r['sensors']['tof']['sample_id'] for r in a}),
                same_measurements_pose_and_applied_commands=True)


def wider_scan_qa(row, clear_meta, hidden_meta):
    """Stationary raycast at additional yaw angles, for picking the next experiment."""
    room, boxes, cylinders = geometry(clear_meta)
    _, altered, _ = geometry(hidden_meta)
    pos = row['position_m']
    roll, pitch, yaw = row['attitude_rad']
    caster = ToFRaycaster()
    results = []
    for offset in (0., 15., 30., 45., 60.):
        attitude = rotation(roll, pitch, yaw + math.radians(offset))
        before = caster.cast_rays(pos, attitude, room, cylinders, boxes)
        after = caster.cast_rays(pos, attitude, room, cylinders, altered)
        results.append(dict(extra_yaw_deg=offset, rays_with_added_box=int(np.count_nonzero(before > after + 1e-5))))
    assert results[0]['rays_with_added_box'] == 0 and any(r['rays_with_added_box'] for r in results)
    return results


def readout(root):
    cases, recordings, assessment_rows = {}, {}, {}
    for name in ('cylinder', 'clear', 'hidden'):
        meta, rows = load(root, 'hidden_margin' if name == 'hidden' else name)
        row, result = assessment('pair' if name in ('clear', 'hidden') else name, meta, rows)
        recordings[name], assessment_rows[name], cases[name] = (meta, rows), row, result
    cylinder_meta, cylinder_rows = recordings['cylinder']
    cases['cylinder']['cylinder_visibility'] = target_visibility(cylinder_meta, cylinder_rows, omit_cylinder=True)
    clear_meta, clear_rows = recordings['clear']
    hidden_meta, hidden_rows = recordings['hidden']
    pair = identical_pair(clear_rows, hidden_rows, clear_meta['assess_s'])
    pair['same_shadow_assessment'] = cases['clear']['assessment'] == cases['hidden']['assessment']
    pair['added_box'] = hidden_meta['geometry']['boxes'][-1]
    assert pair['same_shadow_assessment']
    assert cases['clear']['true_proposed_corridor_hit'] is None
    assert cases['hidden']['true_proposed_corridor_hit'] == 'box'
    assert cases['hidden']['min_true_scan_clearance_m'] > MARGIN_M
    assert all(c['continuous_auto'] and c['settled_in_qa'] and not c['bridge_faults'] for c in cases.values())
    pair['extra_yaw_at_fixed_pose_qa'] = wider_scan_qa(assessment_rows['clear'], clear_meta, hidden_meta)
    wide_meta, wide_rows = load(root, 'hidden_wide')
    assert wide_meta['geometry'] == hidden_meta['geometry']
    wide_row, wide = assessment('wide', wide_meta, wide_rows)
    wide['added_box_visibility'] = target_visibility(wide_meta, wide_rows)
    wide['additional_assessment_yaw_deg'] = math.degrees(wide_row['attitude_rad'][2] - assessment_rows['hidden']['attitude_rad'][2])
    assert wide['added_box_visibility']['first'] is not None and wide['min_true_scan_clearance_m'] > MARGIN_M
    assert wide['continuous_auto'] and wide['settled_in_qa'] and not wide['bridge_faults']
    report = dict(scope=__doc__, cases=cases, paired_control=pair,
                  wider_scan=wide,
                  advance_to_closed_loop=False, reason='identical_scan_history_for_clear_and_obstructed_corridors',
                  next_experiment='measure two-sided coverage in the proposed travel frame, including the added-box control',
                  source_sha256=sha256(__file__))
    (root / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    readout(parser.parse_args().root)
