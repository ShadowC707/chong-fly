"""Compare teacher versions on identical recorded initial scenes, without relabeling data."""
import argparse
import math
from pathlib import Path

import numpy as np
import torch

from configs.flight_config import CONTROL_DT, DEFAULT_DT, PWM_MID
from generator import generate_reflex_dataset as generator
from generator.prepare_flywire_candidates import publication, write_json
from generator.reflex_contract import TEACHER_VERSION, LEGACY_TEACHER_VERSION, BRAKING_TEACHER_VERSION
from generator.source_contract import file_sha256
from optimizer.search_pipeline import validate_demonstration_context
from simulation.altitude_control import AltitudeHold, AltitudeTelemetryError, sample_from_observation
from simulation.memory import EgocentricMemoryWrapper

VERSIONS = {'legacy':LEGACY_TEACHER_VERSION, 'previous':BRAKING_TEACHER_VERSION, 'braking':TEACHER_VERSION}


def _rollout(record, profile, version, parameters, steps):
    env, obs = generator._make_scenario(record['scenario'], record['seed'], profile=profile)
    try:
        if record.get('scene_parameters') is not None and record['scene_parameters'] != env.scene_parameters:
            raise ValueError('Recorded scene parameters differ from reconstructed geometry')
        teacher = generator.make_teacher(version, **parameters)
        memory, altitude = EgocentricMemoryWrapper(), AltitudeHold()
        ticks, done, info, fault = 0, False, {}, None
        initial = env.physics.pos.copy()
        distance, minimum_clearance = 0., None
        turning, recovered, trace = False, False, []
        max_tilt_deg = 0.
        for step in range(steps):
            yaw = float(env.physics.quaternion_to_euler(env.physics.quat)[2])
            ring = memory.update(obs[2:66], current_yaw_rad=-yaw)
            x = np.concatenate([obs, ring]).astype(np.float32)
            target = teacher.step(x, dt=CONTROL_DT)
            turning |= bool(abs(target[3] - teacher.pwm_neutral_yaw) > 1)
            recovered |= bool(turning and target[3] == teacher.pwm_neutral_yaw
                              and target[2] > PWM_MID + 25)
            sample = sample_from_observation(env.get_isaac_obs())
            max_tilt_deg = max(max_tilt_deg, math.degrees(max(abs(sample.roll_rad),abs(sample.pitch_rad))))
            frame = {'time_s':ticks*DEFAULT_DT, 'flow':x[:2].tolist(),
                     'tof_min_m':float(obs[2:66].min())*env.tof.max_range,
                     'range_m':sample.range_m, 'attitude_rad':[sample.roll_rad,sample.pitch_rad,yaw],
                     'range_surface_diagnostic':env.last_laser_result.surface_type,
                     'range_normal_diagnostic':env.last_laser_result.hit_normal.tolist(),
                     'position_m':env.physics.pos.tolist(), 'navigation_pwm':target.tolist()}
            trace.append(frame)
            trace = trace[-8:]
            try:
                action = altitude.apply(target.copy(), sample,
                                        now_s=ticks*DEFAULT_DT)
            except AltitudeTelemetryError as exc:
                frame['altitude_fault'] = str(exc)
                fault, done = str(exc), True
                break
            frame['pwm'] = action.tolist()
            for _ in range(round(CONTROL_DT/DEFAULT_DT)):
                previous = env.physics.pos.copy()
                obs, _, done, info = env.step(action, action_type='pwm')
                ticks += 1
                distance += float(np.linalg.norm(env.physics.pos[:2]-previous[:2]))
                clearance = info.get('clearance_m')
                if clearance is not None:
                    minimum_clearance = (float(clearance) if minimum_clearance is None
                                         else min(minimum_clearance, float(clearance)))
                if done:
                    break
            if done:
                break
        return {'teacher_version':version, 'completed':not done, 'duration_s':ticks*DEFAULT_DT,
                'crashed':bool(info.get('crashed',False)), 'collision_kind':info.get('collision_kind'),
                'failure_reason':fault, 'min_clearance_m':minimum_clearance,
                'travel_m':distance, 'displacement_m':float(np.linalg.norm(env.physics.pos[:2]-initial[:2])),
                'final_position_m':env.physics.pos.tolist(), 'turned':turning,
                'resumed_cruise_after_turn':recovered, 'max_tilt_deg':max_tilt_deg, 'trace':trace}
    finally:
        env.close()


def compare_teachers(data, *, steps=500, episode_indices=None,
                     versions=('legacy','braking'), clearance_margin_m=.1, progress=None):
    if type(steps) is not int or steps <= 0:
        raise ValueError('steps must be a positive integer')
    validate_demonstration_context(data, [])
    meta = data['metadata']
    if meta.get('teacher_version') not in VERSIONS.values():
        raise ValueError('Unknown teacher version')
    if (not isinstance(versions, (list,tuple)) or not versions or len(set(versions)) != len(versions)
            or any(v not in VERSIONS for v in versions)):
        raise ValueError('versions must be distinct known labels')
    if not np.isfinite(clearance_margin_m) or clearance_margin_m <= 0:
        raise ValueError('clearance_margin_m must be positive and finite')
    indices = list(range(len(meta['episodes']))) if episode_indices is None else episode_indices
    if (not isinstance(indices, (list,tuple)) or not indices
            or any(type(i) is not int or not 0 <= i < len(meta['episodes']) for i in indices)
            or len(set(indices)) != len(indices)):
        raise ValueError('episode_indices must be distinct valid indices')
    if meta.get('execution_noise_std_pwm') != 0 or meta.get('label_noise_std_pwm') != 0:
        raise ValueError('Comparison requires zero-noise demonstrations')
    # Validate the recorded configuration before projecting shared settings.
    original = generator.make_teacher(meta['teacher_version'], **meta['teacher_parameters'])
    if original.noise_std_pwm != 0:
        raise ValueError('Recorded teacher parameters must also have zero noise')
    configurations = {}
    for label in versions:
        shared = {k:v for k,v in original.parameters.items()
                  if k in generator.make_teacher(VERSIONS[label]).parameters}
        configurations[label] = generator.make_teacher(VERSIONS[label], **shared).parameters
    rows = []
    for ep in indices:
        record = meta['episodes'][ep]
        row = {'episode':ep, 'scenario':record['scenario'], 'seed':record['seed'],
               'original_terminated':record['terminated']}
        for label in versions:
            version = VERSIONS[label]
            row[label] = _rollout(record, meta.get('scene_profile','standard-v1'), version, configurations[label], steps)
        rows.append(row)
        if progress is not None:
            progress(len(rows), len(indices))
    summary = {label:{'completed':sum(r[label]['completed'] for r in rows),
                      'crashed':sum(r[label]['crashed'] for r in rows),
                      'telemetry_faults':sum(r[label]['failure_reason'] is not None for r in rows),
                      'turned_and_resumed':sum(r[label]['resumed_cruise_after_turn'] for r in rows),
                      'completed_below_clearance_margin':sum(r[label]['completed'] and
                          r[label]['min_clearance_m'] < clearance_margin_m for r in rows)}
               for label in versions}
    return {'format_version':'teacher-comparison-v2', 'research_only':True, 'training_ready':False,
            'episode_indices':list(indices), 'clearance_margin_m':float(clearance_margin_m),
            'steps':steps, 'dt':CONTROL_DT, 'teacher_parameters':configurations, 'summary':summary, 'episodes':rows,
            'limitations':['Same recorded initial scenes; development evaluation, not an independent safety test.',
                          'Cruise recovery is an action observation, not successful obstacle clearance.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--versions', nargs='+', choices=VERSIONS, default=['legacy','braking'])
    parser.add_argument('--episodes', type=int, nargs='+')
    parser.add_argument('--failures-first', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    data = torch.load(args.dataset, map_location='cpu', weights_only=True)
    indices = args.episodes
    if args.failures_first:
        indices = list(range(len(data['metadata']['episodes']))) if indices is None else indices
        indices = sorted(indices, key=lambda i:not data['metadata']['episodes'][i]['terminated'])
    def progress(done, total):
        if done % 8 == 0 or done == total:
            print(f'Compared {done}/{total} matched scenes', flush=True)
    report = compare_teachers(data, steps=args.steps, episode_indices=indices,
                              versions=args.versions, progress=progress)
    root = Path(__file__).resolve().parents[1]
    files = ['optimizer/compare_teachers.py', 'generator/generate_reflex_dataset.py',
             'generator/reflex_contract.py', 'simulation/drone_env.py', 'simulation/drone_interface.py',
             'simulation/altitude_control.py',
             'simulation/memory.py', 'simulation/pmw3901_emulator.py', 'simulation/avionics_filter.py',
             'simulation/collisions.py', 'configs/flight_config.py']
    report.update(dataset_sha256=file_sha256(args.dataset),
                  code_sha256={name:file_sha256(root/name) for name in files})
    with publication(args.out_dir) as stage:
        write_json(stage/'report.json', report)
    print(report['summary'], flush=True)


if __name__ == '__main__':
    main()
