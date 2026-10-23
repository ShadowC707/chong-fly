"""Visibility before behavior: paired teachers in explicit ToF geometry scenes.

python -m scripts.probe_tof_geometry --out-dir <new-directory>
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from artifacts.io import publication, write_json
from generator.source_contract import file_sha256
from optimizer.rollout import simulate_policy_rollout
from scripts.audit_tof_threats import TEACHERS, TeacherAdapter
from simulation.altitude_control import AltitudeHold
from simulation.tof_challenge_scenes import ToFChallengeScene, SCENES, SCENE_VERSION


class VisibilityAdapter(TeacherAdapter):
    def reset_state(self):
        super().reset_state()
        self.frames = []

    def step_np(self, flow, tof, memory_ring, dt=.02):
        action = super().step_np(flow, tof, memory_ring, dt)
        self.frames.append(dict(time_s=len(self.frames)*dt,
            min_tof_m=float(np.min(tof))*3.5, flow=np.asarray(flow).tolist(),
            navigation_pwm=action.tolist()))
        return action


def visibility(env):
    """Compare rays with/without objects from the same pose; GT for audit only."""
    pos, rot = env.physics.pos, env.physics.quaternion_to_rotation_matrix(env.physics.quat)
    empty = env.tof.cast_rays(pos, rot, env.room)
    scene = env.tof.cast_rays(pos, rot, env.room, boxes=env.boxes, cylinders=env.cylinders)
    mask = scene < empty-1e-6
    return dict(object_hit_rays=int(mask.sum()), near_rays=int((scene*env.tof.max_range < .8).sum()),
        minimum_visible_object_m=float(scene[mask].min()*env.tof.max_range) if mask.any() else None,
        empty_depth=empty.tolist(), scene_depth=scene.tolist())


def run_case(scenario, seed=42, steps=500):
    env = ToFChallengeScene(scenario)
    try:
        env.reset(seed=seed)
        spec = env.specification()
        initial_visibility = visibility(env)
    finally:
        env.close()
    row = dict(scenario=scenario, seed=seed, specification=spec,
        specification_sha256=hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest(),
        initial_visibility=initial_visibility, results={})
    hashes = []
    for name, cls in TEACHERS.items():
        env, policy = ToFChallengeScene(scenario), VisibilityAdapter(cls)
        try:
            _, result = simulate_policy_rollout(policy, env=env, seed=seed,
                eval_steps=steps, altitude_hold=AltitudeHold())
            result['final_position_m'] = env.physics.pos.tolist()
        finally:
            env.close()
        result['first_near_tof_time_s'] = next((f['time_s'] for f in policy.frames if f['min_tof_m'] < .8), None)
        result['navigation_trace'] = policy.frames
        hashes.append(policy.initial_observation_sha256)
        row['results'][name] = result
    if None in hashes or len(set(hashes)) != 1:
        raise ValueError('Teachers have different initial observations')
    row['initial_observation_sha256'] = hashes[0]
    return row


def plot_visibility(rows, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 4, figsize=(13, 9), layout='constrained')
    for ax, row in zip(axes.flat, rows):
        v = row['initial_visibility']
        mesh = ax.imshow(np.array(v['scene_depth']).reshape(8, 8)*3.5, vmin=0, vmax=3.5, cmap='viridis_r')
        ax.set(title=f"{row['scenario']}\nobject rays: {v['object_hit_rays']}", xlabel='left -> right', ylabel='top -> bottom')
    fig.colorbar(mesh, ax=list(axes.flat), label='Ray range (m)')
    fig.suptitle('Initial ToF visibility | 64 point rays, not a calibrated hardware sensor')
    fig.savefig(output, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    with publication(args.out_dir) as stage:
        rows = []
        for scenario in SCENES:
            row = run_case(scenario)
            rows.append(row)
            print(scenario, 'initial hits:', row['initial_visibility']['object_hit_rays'],
                  {k:dict(completed=v['horizon_completed'], crashed=v['crashed'])
                   for k, v in row['results'].items()}, flush=True)
        summary = {name:dict(runs=len(rows), completed=sum(r['results'][name]['horizon_completed'] for r in rows),
            crashed=sum(r['results'][name]['crashed'] for r in rows),
            fatal_failures=sum(r['results'][name]['fatal_failure'] for r in rows)) for name in TEACHERS}
        plot_visibility(rows, stage/'initial_visibility.png')
        root = Path(__file__).resolve().parents[1]
        sources = ['scripts/probe_tof_geometry.py', 'scripts/audit_tof_threats.py',
            'simulation/tof_challenge_scenes.py', 'simulation/drone_env.py', 'simulation/drone_interface.py',
            'simulation/benchmark_scenes.py', 'simulation/collisions.py', 'simulation/altitude_control.py',
            'simulation/pmw3901_emulator.py', 'simulation/avionics_filter.py', 'simulation/memory.py',
            'optimizer/rollout.py', 'generator/generate_reflex_dataset.py',
            'generator/continuous_reflex_teacher.py', 'configs/flight_config.py']
        write_json(stage/'report.json', dict(format_version='tof-geometry-probe-v1', research_only=True,
            training_ready=False, scene_version=SCENE_VERSION, seed=42, control_steps=500,
            summary=summary, cases=rows, code_sha256={p:file_sha256(root/p) for p in sources},
            limitations=['Point rays can miss obstacles between directions; this is not hardware detection probability.',
                'Initial visibility is a ground-truth diagnostic, not policy input or a complete detection history.',
                'First near-ToF time includes room surfaces and is not an object-specific detection time.',
                'One development seed per scene; no holdout or flight admission.',
                'Valid noiseless ToF only; dropouts, confidence, stale frames not modeled here.']))
    print(summary, flush=True)


if __name__ == '__main__':
    main()
