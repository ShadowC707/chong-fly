"""Plot an exact saved policy in the standalone benchmark, without GUI or retraining.

python -m evaluate.watch_mvp --scene obstacle_right --plot-out flight_summary.png
python -m evaluate.watch_mvp --scene all --plot-out reports/mvp/flight_summary.png
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.flight_config import CONTROL_DT, DEFAULT_DT
from generator.source_contract import file_sha256
from artifacts.research import load_research_run
from optimizer.rollout import simulate_policy_rollout
from simulation.altitude_control import AltitudeHold
from simulation.benchmark_scenes import NavigationScene, SCENES
from simulation.drone_env import DroneSimulationEnv, OpticalFlowOUWrapper
from simulation.xy_scenes import CornerApproachScene, reflect_geometry_y, DensityArena, DensityStartArena, DENSITY_SCENES

# The trial number is not the candidate size. Trial 2 is k301; trial 3 is k306.
DEFAULT_RUN = ROOT/'data/selection/mvp_frozen_core_v1/trial_00003/after'
VIEW_SCENES = (*SCENES, 'arena', 'arena_mirror', 'corner_approach', 'corner_approach_mirror', *DENSITY_SCENES)
COORDINATOR_MODES = ('baseline', 'anti_rollback', 'turn_latch', 'both')


def make_environment(scene, *, explore=False, seed=20261009, arena_start=None):
    if scene not in VIEW_SCENES:
        raise ValueError('Unknown standalone scene')
    if arena_start is not None and scene not in DENSITY_SCENES:
        raise ValueError('Custom starts require a density scene')
    if scene in DENSITY_SCENES:
        env = DensityArena(scene) if arena_start is None else DensityStartArena(scene,**arena_start)
    elif scene.startswith('corner_approach'):
        env = CornerApproachScene(mirrored=scene.endswith('_mirror'))
    else:
        env = (DroneSimulationEnv(engine='standalone', headless=True, dt=DEFAULT_DT)
               if scene in ('arena', 'arena_mirror') else NavigationScene(scene))
    if scene == 'arena_mirror':
        reflect_geometry_y(env)
    if explore:
        env = OpticalFlowOUWrapper(env, dt=DEFAULT_DT, seed=seed)
    return env


def geometry_snapshot(env):
    return dict(room=asdict(env.room), boxes=[asdict(x) for x in env.boxes],
                cylinders=[asdict(x) for x in env.cylinders],
                collision_radius_m=env.collision_radius_m)


def burst_positions(trace, yaw_bursts):
    """Interpolate physical positions at the detector's event START times."""
    physics = trace['physics']
    if not physics or not yaw_bursts.get('events'):
        return np.empty((0, 2))
    times = np.array([r['time_s'] for r in physics])
    positions = np.array([r['position_m'] for r in physics])
    events = [r['start_s'] for r in yaw_bursts['events']]
    return np.column_stack([np.interp(events, times, positions[:, i]) for i in (0, 1)])


def plot_summary(report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    trace, metrics, geometry = report['trace'], report['metrics'], report['geometry']
    fig = plt.figure(figsize=(15, 9), layout='constrained')
    grid = fig.add_gridspec(2, 2, width_ratios=[1, 1.35])
    map_ax = fig.add_subplot(grid[:, 0])
    command_ax = fig.add_subplot(grid[0, 1])
    sensor_ax = fig.add_subplot(grid[1, 1], sharex=command_ax)
    physics, controls = trace['physics'], trace['control']
    positions = np.array([r['position_m'] for r in physics]).reshape(-1, 3)
    times = np.array([r['time_s'] for r in physics])
    control_times = np.array([r['time_s'] for r in controls])
    bounds_x, bounds_y = [], []
    for i, box in enumerate(geometry['boxes']):
        map_ax.add_patch(Rectangle((box['x_min'], box['y_min']), box['x_max']-box['x_min'],
            box['y_max']-box['y_min'], facecolor='#ccd1d9', edgecolor='#555d69',
            label='Box footprint' if i == 0 else None))
        bounds_x += [box['x_min'], box['x_max']]; bounds_y += [box['y_min'], box['y_max']]
    for i, cylinder in enumerate(geometry['cylinders']):
        x,y,r = cylinder['center_x'], cylinder['center_y'], cylinder['radius']
        map_ax.add_patch(Circle((x,y), r, facecolor='#d8c5ac', edgecolor='#725c41',
                               label='Cylinder footprint' if i == 0 else None))
        bounds_x += [x-r,x+r]; bounds_y += [y-r,y+r]
    room = geometry['room']
    map_ax.add_patch(Rectangle((room['x_min'],room['y_min']), room['x_max']-room['x_min'],
        room['y_max']-room['y_min'], fill=False, linestyle='--', edgecolor='#747b86', label='Room boundary'))
    if len(positions):
        map_ax.plot(positions[:,0],positions[:,1], color='#2364aa', lw=1.8, label='Simulated trajectory')
        map_ax.scatter(*positions[0,:2], c='#198754', marker='o', s=65, zorder=5, label='Start')
        map_ax.scatter(*positions[-1,:2], c='#c7303e', marker='X', s=80, zorder=6, label='Finish / stop')
        bounds_x += positions[:,0].tolist(); bounds_y += positions[:,1].tolist()
    bursts = burst_positions(trace, metrics['yaw_bursts'])
    if len(bursts):
        map_ax.scatter(bursts[:,0], bursts[:,1], c='#e69913', edgecolors='#754a00',
                       marker='*', s=155, zorder=7, label='Yaw burst start')
    if report['scene'].startswith(('arena', 'corner_approach')):
        map_ax.set_xlim(room['x_min']-.2,room['x_max']+.2)
        map_ax.set_ylim(room['y_min']-.2,room['y_max']+.2)
    elif bounds_x:
        # NavigationScene has a 40 m room. A local view keeps the maneuver visible.
        map_ax.set_xlim(min(bounds_x)-.6,max(bounds_x)+.6)
        map_ax.set_ylim(min(bounds_y)-.6,max(bounds_y)+.6)
    map_ax.set_aspect('equal', adjustable='box')
    map_ax.set(xlabel='World X (m)', ylabel='World Y (m)',
        title=f"Top-down trajectory | yaw bursts: {metrics['yaw_bursts'].get('count', 'unavailable')}")
    map_ax.legend(loc='best', fontsize=8)
    if controls:
        commands = np.array([r['rc_pwm'] for r in controls])
        # Last command is held only until the actual end, including partial crash periods.
        end = metrics['survival_time_s']
        for index, label, color in ((1,'Roll','#8b5fbf'),(2,'Pitch','#2364aa'),
                                    (3,'Yaw','#d97706'),(0,'Throttle (AltitudeHold)','#198754')):
            command_ax.step(np.r_[control_times,end], np.r_[commands[:,index],commands[-1,index]],
                            where='post', label=label, color=color, linewidth=1.1)
        if report.get('coordinator_mode', 'baseline') != 'baseline':
            network = np.array([r['network_pwm'] for r in controls])
            for index, label, color in ((2, 'Network Pitch', '#2364aa'), (3, 'Network Yaw', '#d97706')):
                command_ax.plot(control_times, network[:, index], ls=':', lw=.8, alpha=.65,
                                color=color, label=label)
        sensor_ax.step(control_times, [r['min_tof_m'] for r in controls], where='post',
                       label='Minimum ToF range', color='#d97706')
    command_ax.axhline(1500, color='#777777', lw=.7, ls='--')
    command_ax.set(ylabel='RC command (PWM units)', title='Applied commands | controller 50 Hz')
    command_ax.legend(fontsize=8, loc='best')
    if len(positions):
        sensor_ax.plot(times, positions[:,2], color='#2364aa', label='Actual Z (simulator)')
    sensor_ax.axhline(report['target_height_m'], color='#198754', ls='--', label='Target height 1.0 m')
    sensor_ax.set(xlabel='Simulation time (s)', ylabel='Distance / height (m)',
                  title='Sensor readings and height')
    sensor_ax.legend(fontsize=8, loc='best')
    if metrics['survival_time_s'] > 0:
        sensor_ax.set_xlim(0, metrics['survival_time_s'])
    for ax in (map_ax, command_ax, sensor_ax):
        ax.grid(True, alpha=.2)
    mode = 'OU-perturbed flow' if report['explore'] else 'Unmodified sensor input'
    mode += ' | XY: ' + report.get('coordinator_mode', 'baseline')
    status = 'COMPLETE' if metrics['feasible'] else 'STOPPED / FAILED'
    fig.suptitle(f"Standalone simulation | {report['candidate']} | tau x{report['tau_scale']} | {report['scene']}\n"
                 f"{mode} | {status}: {metrics['survival_time_s']:.2f} s | source MVP admission: "
                 f"{report['checkpoint_mvp_admission'].get('passed', 'unknown')}", fontsize=12)
    if not metrics['feasible']:
        sensor_ax.text(.02,.02, metrics.get('failure_reason') or str(metrics.get('collision_kind') or 'Incomplete rollout'),
                       transform=sensor_ax.transAxes, fontsize=8, color='#a31b2b', wrap=True)
    try:
        fig.savefig(output, dpi=150)
    finally:
        plt.close(fig)


def run_view(run_dir=DEFAULT_RUN, *, scene='obstacle_right', steps=500,
             plot_out='flight_summary.png', explore=False, seed=20261009, coordinator_mode='baseline',
             coordinator_config=None, levy_explore=False, stop_at_s=None, behavior_config=None, arena_start=None):
    run_dir, plot_out = Path(run_dir), Path(plot_out)
    json_out = plot_out.with_suffix('.json')
    if plot_out.suffix.lower() != '.png':
        raise ValueError('plot-out must be a PNG path')
    if plot_out.exists() or json_out.exists():
        raise FileExistsError('Use a new output name; previous plots/reports are retained')
    if type(steps) is not int or not 1 <= steps <= 30000:
        raise ValueError('steps must be 1..30000 control periods')
    if type(explore) is not bool:
        raise ValueError('explore must be boolean')
    if coordinator_mode not in COORDINATOR_MODES:
        raise ValueError('Unknown coordinator mode')
    if coordinator_config is not None and coordinator_mode == 'baseline':
        raise ValueError('Coordinator configuration requires an enabled coordinator')
    if (levy_explore or stop_at_s is not None or behavior_config is not None) and coordinator_mode != 'both':
        raise ValueError('Search/STOP experiment requires the both avoidance coordinator')
    if stop_at_s is not None and (not np.isfinite(stop_at_s) or not 0 <= stop_at_s < steps*CONTROL_DT):
        raise ValueError('stop-at-s must lie inside the rollout')
    if explore and coordinator_mode != 'baseline':
        raise ValueError('Keep the XY experiment separate from synthetic OU exploration')
    from simulation.xy_coordinator import XYCoordinator
    coordinator = None if coordinator_mode == 'baseline' else XYCoordinator(
        anti_rollback=coordinator_mode in ('anti_rollback', 'both'),
        turn_latch=coordinator_mode in ('turn_latch', 'both'), config=coordinator_config,
        levy_explore=levy_explore, stop_hover=stop_at_s is not None, seed=seed,
        behavior_config=behavior_config)
    policy, checkpoint = load_research_run(run_dir)
    source_report = json.loads((run_dir/'report.json').read_text(encoding='utf-8'))
    digest = file_sha256(run_dir/'checkpoint.pt')
    policy.eval()
    env = make_environment(scene, explore=explore, seed=seed, arena_start=arena_start)
    controller = AltitudeHold()
    try:
        if env.engine != 'standalone' or not env.headless or env.dt != DEFAULT_DT:
            raise ValueError('Viewer requires standalone headless physics at 0.004 s')
        geometry = geometry_snapshot(env)
        scene_specification = env.specification() if hasattr(env, 'specification') else None
        exploration = env.configuration() if explore else None
        _, metrics = simulate_policy_rollout(policy, env=env, eval_steps=steps, dt=CONTROL_DT,
            seed=seed, altitude_hold=controller, record_trace=True, coordinator=coordinator, stop_at_s=stop_at_s)
    finally:
        env.close()
    trace = metrics.pop('trace')
    if explore:
        metrics['benchmark_version'] += '+ou-explore-v2'
    if file_sha256(run_dir/'checkpoint.pt') != digest:
        raise ValueError('Checkpoint changed during visualization')
    report = dict(version='standalone-mvp-viewer-v2', engine='standalone', headless=True,
        execution_path='research_policy_xy_altitude_direct_physics', receiver_included=False,
        run_dir=str(run_dir.resolve()), checkpoint_sha256=digest, candidate=checkpoint['candidate'],
        tau_scale=checkpoint.get('continuation', {}).get('selected_tau_scale'),
        checkpoint_mvp_admission=source_report.get('mvp_admission', {}),
        scene=scene, seed=seed, requested_control_steps=steps, explore=explore,
        exploration=exploration,
        coordinator_mode=coordinator_mode,
        levy_explore=levy_explore, stop_at_s=stop_at_s,
        scene_specification=scene_specification,
        benchmark_comparable=not explore and coordinator is None and scene in SCENES,
        target_height_m=controller.config.target_height_m, geometry=geometry, metrics=metrics, trace=trace,
        limitations=['Simulated motion, not hardware measurements or flight approval.',
                     'Direct research rollout: no FlightBridge, transport or receiver CommandGate.',
                     'Yaw bursts use the physical-rate detector; zero events is a valid result.',
                     'Explore adds synthetic flow; it is not learned exploration.'],
        code_sha256={name:file_sha256(ROOT/name) for name in (
            'evaluate/watch_mvp.py', 'optimizer/rollout.py', 'optimizer/benchmark_candidate.py',
            'simulation/drone_env.py', 'simulation/benchmark_scenes.py', 'simulation/metrics.py',
            'simulation/policy.py', 'simulation/altitude_control.py', 'simulation/xy_coordinator.py',
            'simulation/xy_scenes.py', 'core/models.py', 'artifacts/research.py', 'artifacts/io.py', 'core/invariants.py')})
    plot_out.parent.mkdir(parents=True, exist_ok=True)
    plot_summary(report, plot_out)
    json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', '--run', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--scene', choices=(*VIEW_SCENES, 'all'), default='obstacle_right')
    parser.add_argument('--steps', type=int, default=500, help='Control periods of 0.02 s (500 = 10 s)')
    parser.add_argument('--seed', type=int, default=20261009)
    parser.add_argument('--explore', action='store_true')
    parser.add_argument('--coordinator', choices=COORDINATOR_MODES, default='baseline')
    parser.add_argument('--levy-explore', action='store_true', help='Yaw search pulses; requires --coordinator both')
    parser.add_argument('--stop-at-s', type=float, help='Start sensor-based braking/hover; requires --coordinator both')
    parser.add_argument('--plot-out', type=Path, default=Path('flight_summary.png'))
    args = parser.parse_args()
    torch.set_num_threads(1)
    scenes = VIEW_SCENES if args.scene == 'all' else (args.scene,)
    for scene in scenes:
        path = (args.plot_out.with_name(f'{args.plot_out.stem}_{scene}{args.plot_out.suffix}')
                if args.scene == 'all' else args.plot_out)
        result = run_view(args.run_dir, scene=scene, steps=args.steps, seed=args.seed,
                          explore=args.explore, plot_out=path, coordinator_mode=args.coordinator,
                          levy_explore=args.levy_explore, stop_at_s=args.stop_at_s)
        print(json.dumps(dict(candidate=result['candidate'], scene=scene, explore=args.explore,
            feasible=result['metrics']['feasible'], seconds=result['metrics']['survival_time_s'],
            yaw_bursts=result['metrics']['yaw_bursts']['count'], plot=str(path.resolve())), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
