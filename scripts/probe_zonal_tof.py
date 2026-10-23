"""Paired research experiment: subray aggregation before any teacher retraining.

python -m scripts.probe_zonal_tof --out-dir <new-directory>
Use --static-only for the visibility sweep without physical rollouts.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from configs.flight_config import CONTROL_DT
from generator.continuous_reflex_teacher import ContinuousReflexTeacher
from artifacts.io import publication, write_json
from generator.source_contract import file_sha256
from optimizer.rollout import BENCHMARK_VERSION, simulate_policy_rollout
from scripts.probe_tof_geometry import VisibilityAdapter, visibility
from simulation.altitude_control import AltitudeHold
from simulation.drone_env import BoxObstacle, RoomBoundaries, ToFRaycaster
from simulation.tof_challenge_scenes import ToFChallengeScene, SCENES, SCENE_VERSION
from simulation.zonal_tof import ZonalConfig, ZonalToF

EXPERIMENT_VERSION = 'zonal-tof-probe-v1'
VARIANTS = ('legacy_points', 'zone_centers', 'mean_n16', 'minimum_n16',
            'supported_n16', 'mean_n64', 'minimum_n64', 'supported_n64')
DISTANCES = (.4, .6, 1.)
WIDTHS = (.005, .01, .02, .04)
OFFSETS = np.linspace(-.2, .2, 81)


def make_sensor(name, coverage=.125, span=.05):
    if name not in VARIANTS:
        raise ValueError('Unknown sensor variant')
    if name == 'legacy_points':
        return ToFRaycaster()
    if name == 'zone_centers':
        return ZonalToF(ZonalConfig(samples_per_axis=1, rule='minimum'))
    rule, count = name.split('_n')
    return ZonalToF(ZonalConfig(samples_per_axis=4 if count == '16' else 8,
                              rule=rule, min_coverage=coverage, group_span_m=span))


def sensor_contract(sensor):
    if isinstance(sensor, ZonalToF):
        return sensor.contract()
    return dict(geometry_version=sensor.geometry_version, rows=sensor.rows, cols=sensor.cols,
                max_range_m=sensor.max_range, sampling='linspace FOV endpoints',
                body_ray_dirs=sensor.body_ray_dirs.tolist(), mount_pitch_rad=sensor.mount_pitch_rad)


def static_sweep(coverage=.125, span=.05):
    room = RoomBoundaries(-20, 20, -20, 20, 0, 4)
    pos, rot = np.array([0., 0., 1.]), np.eye(3)
    rows, maps = [], {}
    for name in VARIANTS:
        sensor = make_sensor(name, coverage, span)
        empty = sensor.cast_rays(pos, rot, room)
        for distance in DISTANCES:
            for width in WIDTHS:
                for offset in OFFSETS:
                    boxes = [BoxObstacle(distance, distance+.02, offset-width/2,
                                         offset+width/2, .2, 1.8)]
                    measured = sensor.cast_rays(pos, rot, room, boxes=boxes)
                    changed = measured < empty-1e-6
                    rows.append(dict(variant=name, distance_m=distance, width_m=width,
                        offset_m=float(offset), changed_zones=int(changed.sum()),
                        near_zones=int((measured*sensor.max_range < .8).sum()),
                        minimum_range_m=float(measured.min()*sensor.max_range)))
                    if distance == .6 and width == .01 and abs(offset) < 1e-10:
                        maps[name] = (measured*sensor.max_range).reshape(8, 8).tolist()
        print('static', name, 'complete', flush=True)
    return rows, maps, dict(room=asdict(room), position_m=pos.tolist(), rotation=rot.tolist(),
        box_x_thickness_m=.02, box_z_bounds_m=[.2, 1.8])


def physical_case(scenario, *, steps, seed, coverage, span):
    row = dict(scenario=scenario, seed=seed, results={})
    physics_hashes = []
    for name in VARIANTS:
        env, policy = ToFChallengeScene(scenario), VisibilityAdapter(ContinuousReflexTeacher)
        env.tof = make_sensor(name, coverage, span)
        started = perf_counter()
        try:
            env.reset(seed=seed)
            spec = env.specification()
            initial_state = {k: getattr(env.physics, k).tolist() for k in ('pos', 'vel', 'quat', 'omega')}
            signature = hashlib.sha256(json.dumps(dict(scene=spec, state=initial_state), sort_keys=True).encode()).hexdigest()
            physics_hashes.append(signature)
            initial_visibility = visibility(env)
            # The shared engine resets to this same scene/seed again before stepping.
            _, metrics = simulate_policy_rollout(policy, env=env, seed=seed,
                eval_steps=steps, altitude_hold=AltitudeHold())
            # Custom sensing must never masquerade as a standard v9 benchmark.
            metrics['rollout_engine_version'] = metrics['benchmark_version']
            metrics['benchmark_version'] = EXPERIMENT_VERSION
            metrics['final_position_m'] = env.physics.pos.tolist()
            metrics['sensor_contract'] = sensor_contract(env.tof)
        finally:
            env.close()
        metrics.update(initial_visibility=initial_visibility, initial_state=initial_state,
            elapsed_wall_s=perf_counter()-started,
            first_near_tof_time_s=next((f['time_s'] for f in policy.frames if f['min_tof_m'] < .8), None),
            initial_observation_sha256=policy.initial_observation_sha256,
            navigation_trace=policy.frames)
        row['results'][name] = metrics
        print(scenario, name, 'completed:', metrics['horizon_completed'],
              'crashed:', metrics['crashed'], flush=True)
    if len(set(physics_hashes)) != 1:
        raise ValueError('Sensor variants have different physical initial conditions')
    row.update(specification=spec, physical_initial_condition_sha256=physics_hashes[0])
    return row


def plot_results(rows, maps, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), layout='constrained')
    for ax, name in zip(axes.flat, VARIANTS):
        mesh = ax.imshow(maps[name], vmin=0, vmax=3.5, cmap='viridis_r')
        ax.set(title=name, xlabel='left -> right', ylabel='top -> bottom')
    fig.colorbar(mesh, ax=list(axes.flat), label='Output range (m)')
    fig.suptitle('Same centered 1 cm object at x=0.6 m | sensor output only')
    fig.savefig(output/'centered_object.png', dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout='constrained', sharex=True, sharey=True)
    colors = dict(zip(VARIANTS, plt.get_cmap('tab10').colors))
    for ax, width in zip(axes.flat, WIDTHS):
        for name in VARIANTS:
            selected = [r for r in rows if r['variant'] == name and r['distance_m'] == .6 and r['width_m'] == width]
            ax.plot([r['offset_m']*100 for r in selected], [r['minimum_range_m'] for r in selected],
                    label=name, color=colors[name], lw=1.2, alpha=.85)
        ax.axhline(.8, color='black', ls='--', label='teacher near threshold')
        ax.set(title=f'Object width: {width*1000:g} mm', xlabel='Lateral position (cm)', ylabel='Minimum output range (m)')
        ax.grid(alpha=.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=5)
    fig.suptitle('Static position sweep at x=0.6 m | jumps can be sampling/aggregation effects')
    fig.savefig(output/'position_sweep.png', dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', required=True, type=Path)
    parser.add_argument('--static-only', action='store_true')
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--min-coverage', type=float, default=.125)
    parser.add_argument('--group-span-m', type=float, default=.05)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')
    ZonalConfig(min_coverage=args.min_coverage, group_span_m=args.group_span_m)
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[1]
    sources = ['scripts/probe_zonal_tof.py', 'simulation/zonal_tof.py',
        'scripts/probe_tof_geometry.py', 'scripts/audit_tof_threats.py',
        'simulation/tof_challenge_scenes.py', 'simulation/drone_env.py', 'simulation/drone_interface.py',
        'simulation/benchmark_scenes.py', 'simulation/collisions.py', 'simulation/altitude_control.py',
        'simulation/pmw3901_emulator.py', 'simulation/avionics_filter.py', 'simulation/memory.py',
        'optimizer/rollout.py', 'generator/generate_reflex_dataset.py',
        'generator/continuous_reflex_teacher.py', 'configs/flight_config.py']
    hashes = {p: file_sha256(root/p) for p in sources}
    plan = dict(format_version=EXPERIMENT_VERSION, created_at_utc=datetime.now(timezone.utc).isoformat(),
        research_only=True, training_ready=False, rollout_engine_version=BENCHMARK_VERSION,
        teacher_version=ContinuousReflexTeacher.version,
        teacher_parameters=ContinuousReflexTeacher(noise_std_pwm=0).parameters,
        scene_version=SCENE_VERSION, seed=args.seed, control_steps=args.steps, control_dt_s=CONTROL_DT,
        physical_scenes=[] if args.static_only else list(SCENES),
        sensor_contracts={n:sensor_contract(make_sensor(n, args.min_coverage, args.group_span_m)) for n in VARIANTS},
        static_grid=dict(distances_m=DISTANCES, widths_m=WIDTHS, offsets_m=OFFSETS.tolist()), code_sha256=hashes)
    with publication(args.out_dir) as stage:
        write_json(stage/'plan.json', plan)
        static, maps, geometry = static_sweep(args.min_coverage, args.group_span_m)
        write_json(stage/'static.json', dict(geometry=geometry, cases=static, centered_object_maps_m=maps))
        plot_results(static, maps, stage)
        cases = []
        for scene in plan['physical_scenes']:
            cases.append(physical_case(scene, steps=args.steps, seed=args.seed,
                                      coverage=args.min_coverage, span=args.group_span_m))
        summary = {n:dict(runs=len(cases), completed=sum(r['results'][n]['horizon_completed'] for r in cases),
            crashed=sum(r['results'][n]['crashed'] for r in cases),
            fatal_failures=sum(r['results'][n]['fatal_failure'] for r in cases)) for n in VARIANTS}
        if hashes != {p:file_sha256(root/p) for p in sources}:
            raise RuntimeError('Source code changed while the experiment was running')
        write_json(stage/'report.json', dict(**plan, summary=summary, cases=cases,
            limitations=['Development scenes, one seed, no holdout or flight admission.',
                'Equal-angular midpoint samples, not optical or calibrated hardware response.',
                'Mean includes max-range sentinels; supported groups use geometric sample coverage, not signal strength.',
                '5 cm grouping and 12.5% default coverage are hypotheses, not sensor specifications.',
                'A support threshold can hide small targets and introduce discontinuities.',
                'No ToF noise, latency, dropouts or multi-target hardware selection modeled.',
                'First near time includes room surfaces; it is not object-specific detection.',
                'Completion only describes survival of a finite run, not exploration quality.']))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
