"""Evaluate a forward-and-brake corridor in shadow, with geometry kept in QA."""
import argparse
from collections import Counter, deque
from dataclasses import asdict
import json
import math
from pathlib import Path

import numpy as np

from configs.flight_config import SENSORS
from demo.contacts import contact_and_clearance
from edge.runtime import sha256
from edge.sensors import Measurement
from edge.survey_heading import SurveyHeadingTracker
from edge.survey_memory import LocalObservationMemory, MemoryConfig, PoseEvidence, rotation
from evaluate.measure_stop_envelope import stopping_metrics
from simulation.drone_env import BoxObstacle, CylinderObstacle, RoomBoundaries, ToFRaycaster
from simulation.survey_heading_sensor import SimulatedHeadingSensor


CONTROL_S = .02
WARMUP_S = .24
HEADING_DELAY_S = .012
MEMORY_S = .20
SPEED_BOUND_MPS = .60
MEMORY_CONFIG = MemoryConfig(ttl_s=MEMORY_S, max_displacement_m=.12, max_tilt_rad=math.radians(5))
RANGE_SCALE_M = 3.5
FORWARD_M = .10
# Rounded up from the largest 100 ms delayed STOP in the calibration grid.
# An empirical screen, since the grid did not combine all velocities and tilts.
BRAKE_M = .415
BODY_M = .20
MARGIN_M = .10
RADIUS_M = BODY_M + MARGIN_M


def load(root, name):
    p = root / name
    return (json.loads((p / 'capture.json').read_text(encoding='utf-8')),
            json.loads((p / 'rows.json').read_text(encoding='utf-8')))


def control_tick(row):
    return abs(row['time_s'] / CONTROL_S - round(row['time_s'] / CONTROL_S)) < 1e-7


def polyline(filtered_flow):
    forward = np.array([FORWARD_M, 0.])
    norm = np.linalg.norm(filtered_flow)
    residual = filtered_flow / norm if norm > .01 else np.array([1., 0.])
    return np.stack((np.zeros(2), forward, forward + BRAKE_M * residual))


def intersects(point, error, path):
    for a, b in zip(path[:-1], path[1:]):
        direction = b - a
        fraction = np.clip((point - a) @ direction / (direction @ direction), 0., 1.)
        if np.linalg.norm(point - a - fraction * direction) <= RADIUS_M + error:
            return True
    return False


def shadow(name, rows, start_s):
    reference = 'corridor-' + name
    sensor = SimulatedHeadingSensor(reference, initial_yaw_rad=rows[0]['attitude_rad'][2])
    sensor.reset(reference + '-window', initial_yaw_rad=rows[0]['attitude_rad'][2],
                 timestamp_s=rows[0]['time_s'])
    reference = sensor.reference_id
    heading = SurveyHeadingTracker(reference)
    pending_heading = deque()
    pending_tof = None
    active = deque()
    aligned_identities = {}
    previous = None
    delivered_at_s = None
    filtered_flow = np.zeros(2)
    heading_fault = None
    for row in filter(control_tick, rows):
        if row['plant_input_pwm'] is None:
            break
        t = row['time_s']
        pending_heading.append(sensor.capture(row['attitude_rad'][2], timestamp_s=t))
        latest = None
        while pending_heading and pending_heading[0].timestamp_s + HEADING_DELAY_S <= t + 1e-8:
            delivered = pending_heading.popleft()
            if heading_fault is None:
                try:
                    latest = heading.add_heading(delivered, now_s=t)
                except ValueError as exc:
                    heading_fault = str(exc)
                    active.clear()
        new_delivery = pending_tof
        current_tof = row['sensors']['tof']
        if current_tof['state'] == 'new':
            assert abs(current_tof['timestamp_s'] - t) < 1e-8, 'Native ToF delivery changed'
            roll, pitch = row['sensors']['attitude']['value']
            pending_tof = (current_tof, roll, pitch)
        else:
            pending_tof = None
        flow = np.asarray(row['sensors']['flow']['value'])
        filtered_flow += (1. - math.exp(-CONTROL_S / .12)) * (flow - filtered_flow)
        path = polyline(filtered_flow)
        if new_delivery is not None:
            previous = new_delivery
            delivered_at_s = t
        while active and t - active[0][0] > MEMORY_S + 1e-8:
            active.popleft()
        candidates = []
        exclusions = Counter()
        aligned = None
        if heading_fault:
            exclusions[heading_fault] += 1
        if latest is not None and previous is not None and not heading_fault:
            raw, capture_roll, capture_pitch = previous
            sample = Measurement(raw['sample_id'], raw['timestamp_s'], raw['value'], raw['valid'])
            try:
                aligned = heading.align_tof(sample, reference_id=reference, now_s=t)
                identity = (aligned.timestamp_s, aligned.heading_sample_id, aligned.heading_timestamp_s,
                            aligned.heading_rad, aligned.heading_error_bound_rad)
                if aligned.state == 'held':
                    assert aligned_identities[aligned.sample_id] == identity
                else:
                    aligned_identities[aligned.sample_id] = identity
                    capture_pose = PoseEvidence(reference, aligned.timestamp_s, capture_roll,
                                                capture_pitch, origin_displacement_bound_m=0.)
                    memory = LocalObservationMemory(reference, MEMORY_CONFIG)
                    memory.remember(aligned, capture_pose, geometry_version=ToFRaycaster.geometry_version,
                                    now_s=t)
                    active.append((aligned.timestamp_s, memory))
            except ValueError as exc:
                exclusions[str(exc)] += 1
            if aligned is not None:
                roll, pitch = row['sensors']['attitude']['value']
                for capture_s, memory in active:
                    displacement_bound = min(MEMORY_CONFIG.max_displacement_m, SPEED_BOUND_MPS * (t - capture_s))
                    pose = PoseEvidence(reference, t, roll, pitch, origin_displacement_bound_m=displacement_bound)
                    report = memory.project(latest, pose, now_s=t)
                    if report['reason']:
                        exclusions[report['reason']] += 1
                    candidates.extend(report['observations'])
        if t < start_s - 1e-8:
            continue
        relevant = []
        for item in candidates:
            low, high = item['horizontal_range_min_m'], item['horizontal_range_max_m']
            error = item['position_error_bound_m']
            # r_min is clipped at zero. Recover the original projected center.
            center_range = high - error
            assert low <= center_range <= high
            angle = item['bearing_rad']
            point = center_range * np.array([math.cos(angle), math.sin(angle)])
            if intersects(point, error, path):
                relevant.append(item)
        state = 'RESTRICT' if relevant else 'UNKNOWN'
        if relevant:
            reason = 'possible_remembered_hit_in_corridor'
        elif aligned is None or exclusions:
            reason = 'unqualified_observation_or_uncovered_side_volume'
        else:
            reason = 'uncovered_side_volume'
        raw = previous[0] if previous else None
        flow_s = row['sensors']['flow']['timestamp_s']
        range_s = row['sensors']['altitude']['timestamp_s']
        downrange = row['sensors']['altitude']['value'][0]
        yield row, dict(time_s=t, state=state, reason=reason, auto=row['plant_source'] == 'auto',
                        path_body_m=path.tolist(), tof_capture_s=raw['timestamp_s'] if raw else None,
                        native_delivery_s=raw['timestamp_s'] if raw else None, shadow_delivery_s=delivered_at_s,
                        tof_age_s=t - raw['timestamp_s'] if raw else None,
                        tof_id=raw['sample_id'] if raw else None,
                        tof_state=aligned.state if aligned else None, capture_alignment=identity if aligned else None,
                        heading_reference_id=reference, heading_sample_id=latest.sample.sample_id if latest else None,
                        heading_capture_s=latest.sample.timestamp_s if latest else None,
                        heading_delivery_s=t if latest else None,
                        heading_age_s=t-latest.sample.timestamp_s if latest else None,
                        flow_age_s=t-flow_s, downrange_age_s=t-range_s,
                        flow_speed_proxy_mps=float(np.linalg.norm(filtered_flow) * downrange * SENSORS.flow_max_rate_rads),
                        motion_bound_from_latest_tof_m=SPEED_BOUND_MPS * (t - raw['timestamp_s']) if raw else None,
                        observed_hits=len(candidates), corridor_hits=len(relevant),
                        oldest_corridor_hit_age_s=max((p['age_s'] for p in relevant), default=None),
                        closest_corridor_hit=min(relevant, key=lambda p: p['horizontal_range_min_m'], default=None),
                        qualified_frames=len(active), exclusions=dict(exclusions),
                        range_only_would_allow=bool(raw and min(raw['value']) * RANGE_SCALE_M > FORWARD_M + BRAKE_M + RADIUS_M))


def geometry(meta):
    g = meta['geometry']
    return (RoomBoundaries(**g['room']), [BoxObstacle(**b) for b in g['boxes']],
            [CylinderObstacle(**c) for c in g['cylinders']])


def geometric_qa(row, path, room, boxes, cylinders):
    pos = np.array(row['position_m'])
    transform = rotation(0., 0., row['attitude_rad'][2])
    world = pos + np.c_[path, np.zeros(len(path))] @ transform.T
    hits = [contact_and_clearance(a, b, room, boxes, cylinders, RADIUS_M)[0]
            for a, b in zip(world[:-1], world[1:])]
    return next((hit for hit in hits if hit is not None), None)


def ambiguity_qa(root, meta, rows, records):
    """Two physical flights with different clearance and identical prior sensor data."""
    hidden, hidden_rows = load(root, 'clear_hidden')
    room, boxes, cylinders = geometry(meta)
    _, hidden_boxes, _ = geometry(hidden)
    assert len(hidden_boxes) == len(boxes) + 1
    row, record = next((row, record) for row, record in records if record['auto'])
    cutoff = row['time_s'] + 1e-8
    original = [r for r in rows if r['time_s'] <= cutoff]
    altered = [r for r in hidden_rows if r['time_s'] <= cutoff]
    assert len(original) == len(altered)
    for a, b in zip(original, altered):
        for key in ('time_s', 'sensors', 'position_m', 'velocity_mps', 'attitude_rad', 'plant_input_pwm'):
            assert json.dumps(a[key]) == json.dumps(b[key]), f'Paired control diverged at {a["time_s"]:.3f} on {key}'
    equal_frames = len({r['sensors']['tof']['sample_id'] for r in original})
    assert equal_frames >= 30 and altered[-1]['clearance_m'] > MARGIN_M
    path = np.asarray(record['path_body_m'])
    original_hit = geometric_qa(row, path, room, boxes, cylinders)
    altered_hit = geometric_qa(row, path, room, hidden_boxes, cylinders)
    assert original_hit is None and altered_hit == 'box'
    return dict(time_s=row['time_s'], equal_tof_frames=equal_frames, equal_physics_ticks=len(original),
                identical_prior_sensor_history_and_applied_pwm=True,
                clearance_with_added_box_m=altered[-1]['clearance_m'], corridor_clear_without_box=True,
                corridor_clear_with_box=False, assessed_path_body_m=path.tolist(), added_box_m=asdict(hidden_boxes[-1]))


def stop_qa(root, name):
    meta, rows = load(root, name + '_stop')
    after = [r for r in rows if r['time_s'] >= meta['start_s'] - 1e-8]
    first = after[0]
    time = np.array([r['time_s'] for r in after]) - first['time_s']
    result = stopping_metrics(time, [r['position_m'] for r in after], [r['velocity_mps'] for r in after])
    assert any(e['kind'] == 'stop' and e['active'] and e['time_s'] == first['time_s'] for e in meta['events'])
    return dict(request_s=first['time_s'], terminal_reason=meta['reason'],
                exact_prefix=meta['exact_parity'], min_clearance_m=min(r['clearance_m'] for r in after),
                max_horizontal_displacement_m=result['max_excursion_m'],
                initial_horizontal_speed_mps=float(np.linalg.norm(first['velocity_mps'][:2])),
                collision=next((r['collision'] for r in after if r['collision']), None),
                auto=all(r['plant_source'] == 'auto' for r in after),
                behavior_counts=dict(Counter(r['trace']['behavior_state'] for r in after)),
                bridge_faults=[e for e in meta['events'] if e['kind'] == 'bridge_fault'],
                metrics=result)


def readout(root):
    summaries = {}
    for name in ('cylinder', 'box', 'clear'):
        meta, rows = load(root, name)
        room, boxes, cylinders = geometry(meta)
        inputs = [r for r in rows if r['time_s'] >= meta['start_s'] - WARMUP_S - 1e-8]
        records = list(shadow(name, inputs, meta['start_s']))
        # Truth only verifies the declared test speed envelope and geometry.
        times = np.array([r['time_s'] for r in rows])
        pos = np.array([r['position_m'] for r in rows])
        max_step_speed = float(np.max(np.linalg.norm(np.diff(pos, axis=0), axis=1) / np.diff(times)))
        for row, record in records:
            record['phased_phase'] = meta['phases'].get(str(row['time_s']))
            record['truth_clearance_m'] = row['clearance_m']
            record['truth_corridor_hit'] = geometric_qa(row, np.asarray(record['path_body_m']), room, boxes, cylinders)
            record['truth_speed_xy_mps'] = float(np.linalg.norm(row['velocity_mps'][:2]))
        assert max_step_speed <= SPEED_BOUND_MPS, 'Declared displacement envelope falsified'
        trace = [record for _, record in records]
        before = [r for r in trace if r['truth_clearance_m'] >= MARGIN_M]
        under = next((r for r in rows if r['clearance_m'] < MARGIN_M and r['time_s'] >= meta['start_s']), None)
        threatened = [r for r in before if r['truth_corridor_hit']]
        (root / name / 'shadow.json').write_text(json.dumps(trace, indent=2), encoding='utf-8')
        first = next(r for r in trace if r['state'] != 'ALLOW')
        summaries[name] = dict(exact_parity=meta['exact_parity'], decisions=len(trace),
            states=dict(Counter(r['state'] for r in trace)), reasons=dict(Counter(r['reason'] for r in trace)),
            first_would_limit=first, first_specific_restrict=next((r for r in trace if r['state']=='RESTRICT'), None),
            max_observed_step_speed_mps=max_step_speed, max_used_tof_age_s=max(r['tof_age_s'] for r in trace),
            verified_held_ticks=sum(r['tof_state']=='held' for r in trace),
            geometric_corridor_clear_ticks=sum(r['truth_corridor_hit'] is None for r in trace),
            possible_hit_on_geometrically_clear_ticks=sum(r['state'] == 'RESTRICT' and r['truth_corridor_hit'] is None for r in trace),
            first_below_margin_s=under['time_s'] if under else None,
            first_decision_lead_to_margin_loss_s=under['time_s'] - first['time_s'] if under else None,
            first_geometric_corridor_conflict_before_margin=threatened[0] if threatened else None,
            range_only_false_allows_before_margin=sum(r['range_only_would_allow'] for r in threatened))
        if name == 'clear':
            summaries[name]['identical_history_counterexample'] = ambiguity_qa(root, meta, rows, records)
        else:
            summaries[name]['scheduled_stop_qa'] = stop_qa(root, name)
    protocol = dict(scope=__doc__, control_s=CONTROL_S, heading_delay_s=HEADING_DELAY_S,
                    heading_sampling_s=CONTROL_S, additional_shadow_tof_delay_s=CONTROL_S,
                    memory=asdict(MEMORY_CONFIG), assumed_speed_bound_mps=SPEED_BOUND_MPS,
                    body_m=BODY_M, margin_m=MARGIN_M, forward_m=FORWARD_M, empirical_stop_screen_m=BRAKE_M,
                    geometry_is_qa_only=True, source_sha256=sha256(__file__))
    decision = dict(advance_to_closed_loop=False, reason='no_safe_positive_ToF_only_authorization')
    (root / 'report.json').write_text(json.dumps(dict(protocol=protocol, cases=summaries, decision=decision), indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    readout(parser.parse_args().root)
