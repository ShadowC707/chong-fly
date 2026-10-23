"""Bounded physical diagnostics for the experimental scan teacher (no training).

python -m scripts.probe_exploration_teacher --out-dir <new-directory>
"""
import argparse
import csv
from pathlib import Path

import numpy as np

from configs.flight_config import CONTROL_DT
from generator.exploration_teacher import ExplorationTeacher, ScanConfig, ScanBlocked
from artifacts.io import publication, write_json
from generator.source_contract import file_sha256
from optimizer.rollout import simulate_policy_rollout
from simulation.altitude_control import AltitudeHold
from simulation.benchmark_scenes import NavigationScene, SCENES, SCENE_VERSION


class RecordingScene(NavigationScene):
    def __init__(self, scenario, policy):
        super().__init__(scenario)
        self.policy, self.rows = policy, []

    def reset(self, **kwargs):
        self.rows.clear()
        return super().reset(**kwargs)

    def record(self, action, mode=None):
        pos = self.physics.pos
        self.rows.append(dict(time_s=self.step_count*self.dt,
            mode=self.policy.mode if mode is None else mode,
            x_m=float(pos[0]), y_m=float(pos[1]), z_m=float(pos[2]),
            yaw_rad=float(self.physics.quaternion_to_euler(self.physics.quat)[2]),
            horizontal_speed_mps=float(np.linalg.norm(self.physics.vel[:2])),
            min_distance_m=self.policy.diagnostics['min_distance_m'],
            flow_norm=self.policy.diagnostics['flow_norm'],
            throttle_pwm=float(action[0]), roll_pwm=float(action[1]),
            pitch_pwm=float(action[2]), yaw_pwm=float(action[3])))

    def step(self, action, action_type='auto'):
        self.record(action)
        return super().step(action, action_type=action_type)


def scan_segments(rows):
    """Measure each contiguous scan, including its first post-action endpoint."""
    yaw = np.unwrap([r['yaw_rad'] for r in rows])
    segments, i = [], 0
    while i < len(rows):
        if rows[i]['mode'] != 'scan':
            i += 1
            continue
        start = i
        while i < len(rows) and rows[i]['mode'] == 'scan':
            i += 1
        end = min(i, len(rows)-1)
        window = rows[start:end+1]
        xy = np.array([[r['x_m'], r['y_m']] for r in window])
        segments.append(dict(start_s=rows[start]['time_s'],
            duration_s=rows[end]['time_s']-rows[start]['time_s'],
            exit_mode=rows[end]['mode'],
            max_drift_m=float(np.linalg.norm(xy-xy[0], axis=1).max()),
            max_speed_mps=max(r['horizontal_speed_mps'] for r in window),
            max_height_error_m=max(abs(r['z_m']-1.) for r in window),
            yaw_change_right_deg=-float(np.rad2deg(yaw[end]-yaw[start]))))
    return segments


def run_probe(config, scenario='clear', seed=42, duration_s=15.):
    if not np.isfinite(duration_s) or duration_s <= 0:
        raise ValueError('Duration must be finite and positive')
    steps = round(duration_s/CONTROL_DT)
    if steps < 1 or not np.isclose(steps*CONTROL_DT, duration_s, rtol=0, atol=1e-9):
        raise ValueError('Duration must be a multiple of the control period')
    policy = ExplorationTeacher(config)
    env = RecordingScene(scenario, policy)
    try:
        blocked, rollout = None, None
        try:
            _, rollout = simulate_policy_rollout(policy, env=env, seed=seed,
                eval_steps=steps, altitude_hold=AltitudeHold())
        except ScanBlocked as exc:
            blocked = str(exc)
        if env.rows:
            env.record([env.rows[-1][k] for k in
                ('throttle_pwm', 'roll_pwm', 'pitch_pwm', 'yaw_pwm')], mode='end')
        report = dict(teacher_version=policy.version, research_only=True,
            training_ready=False, parameters=policy.teacher_parameters,
            scenario=scenario, scene_version=SCENE_VERSION, seed=seed,
            requested_duration_s=duration_s, executed_duration_s=env.step_count*env.dt,
            rollout=rollout, blocked_reason=blocked, final_mode=policy.mode,
            events=policy.events, completed_scans=policy.completed_scans,
            interrupted_scans=policy.interrupted_scans, scans=scan_segments(env.rows))
        return report, env.rows
    finally:
        env.close()


def plot_trace(rows, path, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True, layout='constrained')
    if rows:
        t = [r['time_s'] for r in rows]
        modes = ['move', 'avoid', 'brake', 'scan', 'recover', 'blocked', 'end']
        axes[0].step(t, [modes.index(r['mode']) for r in rows], where='post')
        axes[0].set(yticks=range(len(modes)), yticklabels=modes, ylabel='Teacher mode')
        axes[1].plot(t, [r['horizontal_speed_mps'] for r in rows])
        axes[1].set_ylabel('Speed (m/s)')
        axes[2].plot(t, -np.rad2deg(np.unwrap([r['yaw_rad'] for r in rows])))
        axes[2].set_ylabel('Yaw (deg, right +)')
        axes[3].plot(t, [r['min_distance_m'] for r in rows])
        axes[3].axhline(.8, color='red', linestyle='--', label='Avoidance threshold')
        axes[3].set_ylabel('Nearest ToF (m)')
        axes[3].legend()
    for ax in axes:
        ax.grid(alpha=.2)
    axes[-1].set_xlabel('Time (s)')
    fig.suptitle(title+' | heuristic teacher, no connectome')
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--scenarios', nargs='+', choices=SCENES,
                        default=['clear', 'obstacle_left', 'obstacle_right'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--duration', type=float, default=15.)
    parser.add_argument('--interval', type=float, default=8.)
    args = parser.parse_args()
    config = ScanConfig(interval_s=args.interval)
    with publication(args.out_dir) as stage:
        reports = []
        for scenario in dict.fromkeys(args.scenarios):
            report, rows = run_probe(config, scenario, args.seed, args.duration)
            reports.append(report)
            with (stage/f'{scenario}.csv').open('w', newline='', encoding='utf-8') as f:
                if rows:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                    writer.writeheader(); writer.writerows(rows)
            plot_trace(rows, stage/f'{scenario}.png', scenario)
            print(f"{scenario}: completed_scans={report['completed_scans']}, "
                  f"interrupted={report['interrupted_scans']}, "
                  f"blocked={report['blocked_reason']}, "
                  f"horizon_completed={bool(report['rollout'] and report['rollout']['horizon_completed'])}")
        root = Path(__file__).resolve().parents[1]
        sources = ['generator/exploration_teacher.py', 'generator/generate_reflex_dataset.py',
                   'scripts/probe_exploration_teacher.py', 'optimizer/rollout.py',
                   'simulation/benchmark_scenes.py', 'configs/flight_config.py']
        write_json(stage/'report.json', dict(format_version='exploration-teacher-probe-v1',
            research_only=True, training_ready=False, cases=reports,
            source_sha256={p:file_sha256(root/p) for p in sources},
            limitations=['Uncalibrated simulation; no physical flight admission.',
                'Only static simple geometry; no independent holdout or parameter selection.',
                'Flow damping is not absolute position hold. ToF clearance is forward-only.',
                'Scan duration commands a turn, not a guaranteed angle or information gain.',
                'No connectome training, lighting cues, or novelty-based destination choice.']))


if __name__ == '__main__':
    main()
