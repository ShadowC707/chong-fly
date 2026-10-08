"""Replay recorded teacher failures without modifying labels or physics."""
import argparse
from pathlib import Path

import numpy as np
import torch

from configs.flight_config import CONTROL_DT, DEFAULT_DT
from generator.prepare_flywire_candidates import publication, write_json
from generator.source_contract import file_sha256
from optimizer.search_pipeline import validate_demonstration_context
from simulation.altitude_control import AltitudeTelemetryError, sample_from_observation
from simulation.memory import EgocentricMemoryWrapper


def audit_failures(data):
    from generator import generate_reflex_dataset as generator
    validate_demonstration_context(data, [])
    metadata = data['metadata']
    if metadata.get('execution_noise_std_pwm') != 0 or metadata.get('label_noise_std_pwm') != 0:
        raise ValueError('Exact replay requires zero noise; episode seeds do not record execution RNG state')
    x, y, applied, valid = (data[k] for k in ('X','Y','applied_actions','valid'))
    if (valid.dtype != torch.bool or valid.shape != x.shape[:2] or not valid[:,0].all()
        or (valid[:,1:] & ~valid[:,:-1]).any() or y.shape != applied.shape
        or y.shape != (*x.shape[:2],4)):
        raise ValueError('Replay requires aligned tensors with nonempty valid prefixes')
    failures = []
    for episode, record in enumerate(metadata['episodes']):
        if not record['terminated']:
            continue
        valid_steps = int(valid[episode].sum())
        if valid_steps != record['valid_steps']:
            raise ValueError('Recorded prefix length differs from the valid mask')
        env, obs = generator._make_scenario(record['scenario'], record['seed'],
                                profile=metadata.get('scene_profile','standard-v1'))
        expert = generator.make_teacher(metadata['teacher_version'], **metadata['teacher_parameters'])
        altitude = generator.AltitudeHold()
        memory = EgocentricMemoryWrapper()
        ticks, trace, failure, done, info = 0, [], None, False, {}
        try:
            for step in range(x.shape[1]):
                yaw = float(env.physics.quaternion_to_euler(env.physics.quat)[2])
                ring = memory.update(obs[2:66], current_yaw_rad=-yaw)
                observation = np.concatenate([obs,ring]).astype(np.float32)
                target = expert.step(observation, dt=CONTROL_DT)
                sample = sample_from_observation(env.get_isaac_obs())
                try:
                    action = altitude.apply(target.copy(), sample, now_s=ticks*DEFAULT_DT)
                except AltitudeTelemetryError as exc:
                    failure, done = str(exc), True
                    if step != valid_steps:
                        raise ValueError('Replay fault disagrees with the recorded prefix') from exc
                    break
                if (step >= valid_steps or
                    not np.array_equal(observation, x[episode,step].cpu().numpy()) or
                    not np.array_equal(target, y[episode,step].cpu().numpy()) or
                    # Collection stores float32, while the altitude controller
                    # computes float64. Compare serialization, retain the
                    # original action precision when advancing physics.
                    not np.array_equal(action.astype(np.float32), applied[episode,step].cpu().numpy())):
                    raise ValueError('Replay differs from recorded observation/action prefix; do not infer causality')
                trace.append({'time_s':ticks*DEFAULT_DT, 'position_m':env.physics.pos.tolist(),
                    'world_velocity_mps':env.physics.vel.tolist(), 'yaw_rad':yaw,
                    'flow_normalized':observation[:2].tolist(),
                    'tof_min_m':float(obs[2:66].min())*env.tof.max_range,
                    'memory_min_normalized':float(ring.min()), 'pwm':action.tolist()})
                trace = trace[-8:]
                for _ in range(round(CONTROL_DT/DEFAULT_DT)):
                    obs, _, done, info = env.step(action, action_type='pwm')
                    ticks += 1
                    if done:
                        break
                if done:
                    if step+1 != valid_steps:
                        raise ValueError('Replay termination disagrees with recorded prefix')
                    break
            if (not done or bool(info.get('crashed',False)) != record['crashed'] or
                info.get('collision_kind') != record.get('collision_kind') or
                failure != record.get('failure_reason') or ticks != record['physics_ticks']):
                raise ValueError('Recorded failure was not reproduced under the current code')
            failures.append({'episode':episode, 'scenario':record['scenario'], 'seed':record['seed'],
                'scene_parameters':record.get('scene_parameters'), 'valid_steps':valid_steps,
                'prefix_exact':True, 'crashed':bool(info.get('crashed',False)),
                'collision_kind':info.get('collision_kind'), 'reproduced_failure_reason':failure,
                'failure_time_s':ticks*DEFAULT_DT, 'trace':trace})
        finally:
            env.close()
    return {'format_version':'teacher-failure-replay-v1', 'research_only':True,
            'training_ready':False, 'failures':failures,
            'limitations':['Exact reproduction is diagnostic evidence, not a safe teacher admission.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    data = torch.load(args.dataset, map_location='cpu', weights_only=True)
    report = audit_failures(data)
    root = Path(__file__).resolve().parents[1]
    files = ['optimizer/replay_demonstrations.py','optimizer/search_pipeline.py',
        'generator/generate_reflex_dataset.py','generator/reflex_contract.py',
        'simulation/drone_env.py','simulation/drone_interface.py','simulation/altitude_control.py',
        'simulation/memory.py','simulation/pmw3901_emulator.py','simulation/avionics_filter.py',
        'configs/flight_config.py']
    report.update(dataset_sha256=file_sha256(args.dataset),
                  code_sha256={name:file_sha256(root/name) for name in files})
    with publication(args.out_dir) as stage:
        write_json(stage/'report.json', report)
    print(f'Exactly reproduced {len(report["failures"])} recorded failures')


if __name__ == '__main__':
    main()
