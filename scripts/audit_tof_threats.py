"""Separate input encoding, teacher arithmetic, and physical teacher behavior.

No checkpoint swapping, training, or promotion of the experimental teacher.
python -m scripts.audit_tof_threats --out-dir <new-directory> --physical
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
from pathlib import Path

import numpy as np
import torch

from configs.flight_config import CONTROL_DT, TOF_RAYCASTER_MAX_RANGE_M, APF_DISTANCE_THRESHOLD_M
from generator.continuous_reflex_teacher import ContinuousReflexTeacher
from generator.generate_reflex_dataset import ExpertReflexPolicy
from generator.reflex_contract import BRAKE_FIRST_TEACHER_VERSION as TEACHER_VERSION
from artifacts.io import publication, write_json
from generator.source_contract import file_sha256
from optimizer.rollout import simulate_policy_rollout
from simulation.altitude_control import AltitudeHold
from simulation.benchmark_scenes import NavigationScene, SCENES, SCENE_VERSION
from simulation.policy import SensorInputLayer

ENCODINGS = ('proximity-mean-v1', 'proximity-v1')
TEACHERS = {TEACHER_VERSION: ExpertReflexPolicy,
            ContinuousReflexTeacher.version: ContinuousReflexTeacher}


def band(depth_m, width=1, side='left'):
    x = np.ones(74, dtype=np.float32)
    x[:2] = 0.
    grid = x[2:66].reshape(8, 8)
    columns = slice(0, width) if side == 'left' else slice(8-width, 8)
    grid[:, columns] = depth_m/TOF_RAYCASTER_MAX_RANGE_M
    return x


def boundary_observations(depth_m=.7):
    offsets = np.array([-.01, -.001, -.0001, -.00001, 0., .00001, .0001, .001, .01])
    distances = APF_DISTANCE_THRESHOLD_M+offsets
    raw = np.stack([band(depth_m)]*len(distances))
    raw[:, 65] = distances/TOF_RAYCASTER_MAX_RANGE_M
    return distances, raw


def arithmetic_audit():
    distances, raw = boundary_observations()
    rows = []
    with torch.no_grad():
        encoded = {name:SensorInputLayer(learnable_scale=False, encoding=name)(torch.from_numpy(raw)).numpy()
                   for name in ENCODINGS}
    for i, distance in enumerate(distances):
        row = dict(extra_ray_m=float(distance))
        for name, values in encoded.items():
            row[name+'_near_ray'] = float(values[i, 2])
            row[name+'_sum'] = float(values[i, 2:66].sum())
        for name, cls in TEACHERS.items():
            action = cls(noise_std_pwm=0).step(raw[i])
            row[name+'_pitch_pwm'] = float(action[2])
            row[name+'_yaw_pwm'] = float(action[3])
        rows.append(row)

    # Repeated threshold crossings isolate arithmetic-induced command jitter;
    # no physical plant or model of actual ToF noise is claimed here.
    noisy = np.stack([band(.7)]*100)
    noisy[:, 65] = (APF_DISTANCE_THRESHOLD_M+np.where(np.arange(100) % 2, -.0001, .0001))/TOF_RAYCASTER_MAX_RANGE_M
    traces = {}
    for name, cls in TEACHERS.items():
        teacher = cls(noise_std_pwm=0)
        traces[name] = np.stack([teacher.step(x) for x in noisy])

    spatial = []
    for depth in (.35, .55, .7, .79, .81, 1.2):
        for width in (1, 2, 4):
            for side in ('left', 'right'):
                x = band(depth, width, side)
                row = dict(depth_m=depth, width=width, side=side)
                for name, cls in TEACHERS.items():
                    row[name] = cls(noise_std_pwm=0).step(x).tolist()
                for name in ENCODINGS:
                    with torch.no_grad():
                        y = SensorInputLayer(learnable_scale=False, encoding=name)(torch.from_numpy(x))
                    row[name] = dict(tof_sum=float(y[2:66].sum()), max_pixel=float(y[2:66].max()))
                spatial.append(row)

    # A single spurious close return remains a hazard with max aggregation.
    outlier = band(3.5); outlier[2] = .35/TOF_RAYCASTER_MAX_RANGE_M
    report = dict(boundary=rows, spatial=spatial,
        threshold_jitter={name:dict(pitch_peak_to_peak_pwm=float(np.ptp(a[:, 2])),
            yaw_peak_to_peak_pwm=float(np.ptp(a[:, 3]))) for name, a in traces.items()},
        isolated_close_return={name:cls(noise_std_pwm=0).step(outlier).tolist()
                               for name, cls in TEACHERS.items()})
    arrays = dict(boundary_raw=raw, crossing_raw=noisy,
                  **{'crossing_'+name:a for name, a in traces.items()})
    return report, arrays


class TeacherAdapter:
    """Shared rollout interface; teacher sees sensors, never ground truth."""
    def __init__(self, teacher_class):
        self.teacher = teacher_class(noise_std_pwm=0)
        self.reset_state()

    def reset_state(self):
        self.teacher.reset()
        self.initial_observation_sha256 = None

    def step_np(self, flow, tof, memory_ring, dt=CONTROL_DT):
        x = np.concatenate([flow, tof, memory_ring]).astype(np.float32)
        if self.initial_observation_sha256 is None:
            self.initial_observation_sha256 = hashlib.sha256(x.tobytes()).hexdigest()
        return self.teacher.step(x, dt=dt)


def physical_comparison(seeds=(42, 73), steps=500, progress=None):
    rows = []
    for scenario in SCENES:
        for seed in seeds:
            row = dict(scenario=scenario, seed=seed, results={})
            hashes = []
            for name, cls in TEACHERS.items():
                policy = TeacherAdapter(cls)
                env = NavigationScene(scenario)
                try:
                    _, metrics = simulate_policy_rollout(policy, env=env, seed=seed,
                        eval_steps=steps, altitude_hold=AltitudeHold())
                finally:
                    env.close()
                row['results'][name] = metrics
                hashes.append(policy.initial_observation_sha256)
            if hashes[0] is None or len(set(hashes)) != 1:
                raise ValueError('Paired teachers did not receive identical initial observations')
            row['initial_observation_sha256'] = hashes[0]
            rows.append(row)
            if progress:
                progress(len(rows), len(SCENES)*len(seeds))
    summary = {name:dict(runs=len(rows), completed=sum(r['results'][name]['horizon_completed'] for r in rows),
        crashed=sum(r['results'][name]['crashed'] for r in rows),
        fatal_failures=sum(r['results'][name]['fatal_failure'] for r in rows),
        minimum_clearance_m=min(r['results'][name]['min_clearance_m'] for r in rows)) for name in TEACHERS}
    regressions = [dict(scenario=r['scenario'], seed=r['seed']) for r in rows
        if r['results'][TEACHER_VERSION]['feasible'] and not r['results'][ContinuousReflexTeacher.version]['feasible']]
    return dict(scene_version=SCENE_VERSION, control_steps=steps, seeds=list(seeds),
                summary=summary, completion_regressions=regressions, paired_scenes=rows)


def plot_audit(report, arrays, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    rows = report['boundary']
    x = [r['extra_ray_m'] for r in rows]
    for name in ENCODINGS:
        axes[0, 0].plot(x, [r[name+'_near_ray'] for r in rows], '.-', label=name)
    axes[0, 0].set(title='Existing 0.7 m ray: unrelated ray crosses threshold',
                   xlabel='Extra ray distance (m)', ylabel='Encoded existing ray')
    for name in TEACHERS:
        axes[0, 1].plot(x, [r[name+'_pitch_pwm'] for r in rows], '.-', label=name)
        a = arrays['crossing_'+name]
        axes[1, 1].plot(np.arange(len(a))*CONTROL_DT, a[:, 2], label=name)
    axes[0, 1].set(title='Teacher: higher pitch means weaker braking here',
                   xlabel='Extra ray distance (m)', ylabel='Pitch PWM')
    spatial = [r for r in report['spatial'] if r['depth_m'] == .35 and r['side'] == 'left']
    for name in ENCODINGS:
        axes[1, 0].plot([r['width'] for r in spatial], [r[name]['tof_sum'] for r in spatial], '.-', label=name)
    axes[1, 0].set(title='Width remains information in local encoding', xlabel='Occupied columns', ylabel='Encoded ToF sum')
    axes[1, 1].set(title='One extra ray alternating 0.8001 / 0.7999 m', xlabel='Time (s)', ylabel='Pitch PWM')
    for ax in axes.flat:
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle('ToF arithmetic audit | no trained connectome predictions')
    fig.savefig(output, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', required=True, type=Path)
    parser.add_argument('--physical', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    with publication(args.out_dir) as stage:
        arithmetic, arrays = arithmetic_audit()
        physical = physical_comparison(progress=lambda n, total:print(f'Paired scenes: {n}/{total}', flush=True)) if args.physical else None
        np.savez_compressed(stage/'traces.npz', **arrays)
        with (stage/'boundary.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(arithmetic['boundary'][0]))
            writer.writeheader(); writer.writerows(arithmetic['boundary'])
        plot_audit(arithmetic, arrays, stage/'arithmetic.png')
        root = Path(__file__).resolve().parents[1]
        files = ['scripts/audit_tof_threats.py', 'generator/continuous_reflex_teacher.py',
            'generator/generate_reflex_dataset.py', 'generator/reflex_contract.py',
            'simulation/policy.py', 'optimizer/rollout.py', 'simulation/benchmark_scenes.py',
            'simulation/altitude_control.py', 'simulation/drone_env.py',
            'simulation/drone_interface.py', 'simulation/collisions.py',
            'simulation/pmw3901_emulator.py', 'simulation/avionics_filter.py',
            'simulation/memory.py', 'configs/flight_config.py']
        write_json(stage/'report.json', dict(format_version='tof-threat-audit-v1',
            created_at_utc=datetime.now(timezone.utc).isoformat(), research_only=True, training_ready=False,
            arithmetic=arithmetic, physical=physical,
            teacher_parameters={name:cls(noise_std_pwm=0).parameters for name, cls in TEACHERS.items()},
            aggregation_version=ContinuousReflexTeacher.aggregation_version,
            code_sha256={p:file_sha256(root/p) for p in files}, trace_sha256=file_sha256(stage/'traces.npz'),
            limitations=['Local encoding preserves area dependence; it does not guarantee neural yaw or braking.',
                'New teacher repairs spatial aggregation only; latching and threshold-based yaw remain.',
                'A close single-pixel outlier still triggers a response; sensor validity is not represented.',
                'Paired seeds share initial conditions, not necessarily subsequent noise values after trajectories diverge.',
                'Simple development scenes and uncalibrated physics, not independent flight admission.',
                'No model was retrained; historical checkpoints must keep their original encoding.']))
    print('Threshold jitter:', arithmetic['threshold_jitter'], flush=True)
    if physical:
        print('Physical summary:', physical['summary'], flush=True)
        print('Completion regressions:', physical['completion_regressions'], flush=True)


if __name__ == '__main__':
    main()
