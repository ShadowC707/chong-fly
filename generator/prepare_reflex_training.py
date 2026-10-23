"""Publish an audited, immutable v5 training bundle for the bounded pilot.

python -m generator.prepare_reflex_training --out-dir data/reflex_training_v5
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from generator.generate_reflex_dataset import generate_reflex_dataset, make_teacher
from artifacts.io import publication, write_json
from generator.reflex_contract import TEACHER_VERSION, SCENARIOS
from generator.source_contract import file_sha256
from optimizer.search_pipeline import validate_demonstration_context


def audit_labels(data):
    """Replay every valid label, independently of learned models and their loss."""
    validate_demonstration_context(data, [])
    meta = data['metadata']
    maximum, mismatches = 0., 0
    for ep, mask in enumerate(data['valid']):
        teacher = make_teacher(meta['teacher_version'], **meta['teacher_parameters'])
        for t in mask.nonzero().flatten().tolist():
            replay = teacher.step(data['X'][ep,t].numpy(), dt=meta['dt'])
            error = float(np.max(np.abs(replay-data['Y'][ep,t].numpy())))
            maximum = max(maximum,error); mismatches += int(error > 1e-4)
    counts = meta['scenario_counts']
    if mismatches or set(counts) != set(SCENARIOS):
        raise ValueError('Dataset failed exact teacher replay or scenario coverage')
    return dict(teacher_replay_max_error_pwm=maximum, mismatch_frames=mismatches,
        valid_frames=int(data['valid'].sum()), scenario_counts=counts,
        behavior_counts=meta['behavior_counts'],
        crashed_episodes=sum(r['crashed'] for r in meta['episodes']),
        terminated_episodes=sum(r['terminated'] for r in meta['episodes']))


def prepare(output, episodes=128, steps=250, seed=20261015):
    root=Path(__file__).resolve().parents[1]
    sources=['generator/prepare_reflex_training.py','generator/generate_reflex_dataset.py',
        'generator/reflex_contract.py','generator/continuous_reflex_teacher.py',
        'simulation/drone_env.py','simulation/drone_interface.py','simulation/altitude_control.py',
        'simulation/control_contract.py','simulation/memory.py','simulation/collisions.py',
        'simulation/avionics_filter.py','simulation/pmw3901_emulator.py','configs/flight_config.py']
    hashes={p:file_sha256(root/p) for p in sources}
    with publication(output) as stage:
        results={}; seen_seeds=set(); seen_scenes=set()
        for name,count,run_seed in [('train',episodes,seed),('evaluation',32,seed+1)]:
            print(f'Collecting {name}: {count} episodes x {steps} steps',flush=True)
            data=generate_reflex_dataset(num_episodes=count,seq_len=steps,seed=run_seed,
                noise_std_pwm=0,scene_profile='varied-v1',output_path=stage/f'{name}.pt')
            seeds={r['seed'] for r in data['metadata']['episodes']}
            # Exact repetitions of the sampled initial conditions are forbidden.
            import json
            scene_keys={json.dumps(dict(scenario=r['scenario'],parameters=r['scene_parameters']),sort_keys=True)
                        for r in data['metadata']['episodes']}
            if seen_seeds & seeds or seen_scenes & scene_keys:
                raise ValueError('Training and external evaluation repeat initial scenes')
            seen_seeds |= seeds; seen_scenes |= scene_keys
            results[name]=dict(audit_labels(data), file_sha256=file_sha256(stage/f'{name}.pt'),
                seed=run_seed, episodes=count, steps=steps)
            print(name,results[name],flush=True)
        if hashes != {p:file_sha256(root/p) for p in sources}:
            raise RuntimeError('Source changed during collection')
        write_json(stage/'manifest.json',dict(version='reflex-training-bundle-v1',teacher_version=TEACHER_VERSION,
            research_only=True,training_ready=False,datasets=results,code_sha256=hashes,
            evaluation_policy='Separate geometry draws in existing scenario families; not unseen scenario families. Do not use for tuning.',
            limitations=['Balanced scenarios do not imply balanced labels or complete coverage.',
                'Failed episode prefixes are retained and reported, not silently removed.',
                'No scan/light behaviors or new zonal sensor; fixed point-rays-solid-v2.']))
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-dir',type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(1)
    prepare(args.out_dir)


if __name__ == '__main__':
    main()
