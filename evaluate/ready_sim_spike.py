"""Bounded RC/rigid-body comparison, deliberately NOT Betaflight SITL.

Run each backend in its own environment, then compare the saved traces.
No neural policy, sensor emulation, hardware sockets or production defaults.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np

from simulation.avionics_filter import BetaflightCascadedPID, PIDConstants
from simulation.ready_sim_contract import BETAFLIGHT_COMMIT, UPSTREAM_COMMIT, make_urdf, rc_setpoints

from simulation.sitl_plant import BulletPlant, LegacyPlant

DT = .004
DURATION = 8.
CASES = ('hover', 'yaw_right', 'pitch_brake', 'roll_brake')


def command(case, tick, hover_pwm):
    """RC held for five physics steps; pulses have fixed, predeclared times."""
    t = (tick // 5) * .02
    pwm = np.array([hover_pwm, 1500., 1500., 1500.])
    if case == 'yaw_right' and 1 <= t < 2:
        pwm[3] += 140
    if case in ('pitch_brake', 'roll_brake'):
        channel = 2 if case == 'pitch_brake' else 1
        if 1 <= t < 2:
            pwm[channel] += 50
        elif 2 <= t < 3:
            pwm[channel] -= 50
    return pwm


def run(args):
    source = json.loads(args.reference.read_text(encoding='utf-8'))['uncertainty_study']
    p, fc = source['nominal_dynamics'], source['fc']
    out = args.out / args.backend
    out.mkdir(parents=True, exist_ok=True)
    asset = out / 'reference.urdf'
    make_urdf(asset, p)
    provenance = {'backend': args.backend, 'python': platform.python_version(),
        'numpy': np.__version__, 'dt_s': DT, 'rc_dt_s': .02, 'duration_s': DURATION,
        'dynamics': p, 'fc': fc, 'reference_path': str(args.reference),
        'reference_sha256': hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        'asset_sha256': hashlib.sha256(asset.read_bytes()).hexdigest(),
        'controller': 'shared project BetaflightCascadedPID, NOT firmware',
        'sensors': 'ideal true attitude/rate; no ToF or PMW', 'calibrated': False,
        'policy_loaded': False, 'altitude_hold': False,
        'betaflight_source_audit_commit': BETAFLIGHT_COMMIT, 'betaflight_executed': False,
        'implementation_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), Path('simulation/ready_sim_contract.py'),
                         Path('simulation/avionics_filter.py'), Path('simulation/drone_env.py'))}}
    if args.backend == 'pybullet':
        import gym_pybullet_drones
        import pybullet
        checkout = Path(gym_pybullet_drones.__file__).resolve().parent.parent
        commit = subprocess.check_output(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip()
        if commit != UPSTREAM_COMMIT:
            raise ValueError(f'Expected pinned upstream {UPSTREAM_COMMIT}; got {commit}')
        provenance.update(upstream_commit=commit, pybullet_api=pybullet.getAPIVersion())
    hover = 1000 + 1000 * np.sqrt(p['mass'] * p['g'] / (4 * p['thrust_coeff'])) / p['max_rpm']
    provenance['hover_pwm'] = float(hover)
    results = {}
    for case in CASES:
        plant = (LegacyPlant if args.backend == 'legacy' else BulletPlant)(p, asset)
        controller = BetaflightCascadedPID(dt=DT, angle_p_gain=fc['angle_p_gain'],
            max_rate_rads=fc['max_rate_rads'],
            roll_pid=PIDConstants(**fc['roll']), pitch_pid=PIDConstants(**fc['pitch']),
            yaw_pid=PIDConstants(**fc['yaw']))
        states, commands, motors_log = [plant.state()], [], []
        started = time.perf_counter()
        try:
            for tick in range(round(DURATION / DT)):
                s = states[-1]
                pwm = command(case, tick, hover)
                thrust, roll, pitch, yaw = rc_setpoints(pwm, p)
                motors, _ = controller.compute_motor_commands(thrust, pitch, roll, yaw, s[6:9], s[9:12])
                plant.step(motors)
                states.append(plant.state())
                commands.append(pwm)
                motors_log.append(motors)
            elapsed = time.perf_counter() - started
        finally:
            plant.close()
        states = np.array(states)
        np.savez_compressed(out / f'{case}.npz', time_s=np.arange(len(states)) * DT,
                            state=states, pwm=commands, motors=motors_log)
        results[case] = {'finite': bool(np.isfinite(states).all()),
            'final_position_m': states[-1, :3].tolist(),
            'final_yaw_deg': float(np.degrees(states[-1, 8])),
            'max_height_error_m': float(np.max(np.abs(states[:, 2] - 1))),
            'max_abs_roll_pitch_deg': np.degrees(np.max(np.abs(states[:, 6:8]), axis=0)).tolist(),
            'elapsed_wall_s': elapsed}
    provenance['results'] = results
    (out / 'report.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2))


def compare(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    reports = {b: json.loads((args.out / b / 'report.json').read_text()) for b in ('legacy', 'pybullet')}
    for field in ('dynamics', 'fc', 'reference_sha256', 'dt_s', 'rc_dt_s', 'duration_s', 'hover_pwm'):
        if reports['legacy'][field] != reports['pybullet'][field]:
            raise ValueError(f'Unmatched comparison: {field}')
    fig, axes = plt.subplots(4, 3, figsize=(13, 12), constrained_layout=True)
    rows = []
    for row, case in enumerate(CASES):
        traces = {b: np.load(args.out / b / f'{case}.npz') for b in reports}
        a, b = (traces[k]['state'] for k in ('legacy', 'pybullet'))
        if not np.array_equal(traces['legacy']['pwm'], traces['pybullet']['pwm']):
            raise ValueError('RC streams differ')
        rows.append({'case': case, 'max_position_difference_m': float(np.linalg.norm(a[:, :3]-b[:, :3], axis=1).max()),
            'max_attitude_difference_deg': float(np.degrees(np.abs(np.arctan2(np.sin(a[:, 6:9]-b[:, 6:9]), np.cos(a[:, 6:9]-b[:, 6:9]))).max()))})
        for backend, trace in traces.items():
            s, t = trace['state'], trace['time_s']
            color = 'tab:blue' if backend == 'legacy' else 'tab:orange'
            axes[row, 0].plot(t, s[:, 0], color=color, label=f'{backend} X')
            axes[row, 0].plot(t, s[:, 1], '--', color=color, label=f'{backend} Y')
            axes[row, 1].plot(t, s[:, 2], label=backend)
            axis = 8 if case in ('hover', 'yaw_right') else 7 if case == 'pitch_brake' else 6
            axes[row, 2].plot(t, np.degrees(s[:, axis]), label=backend)
        axes[row, 0].set_ylabel(case)
        axes[row, 0].set_ylim(-.55, .55)
        axes[row, 1].set_ylim(.85, 1.45)
        axes[row, 1].ticklabel_format(useOffset=False, style='plain')
        for col, title in enumerate(('Position X/Y (m)', 'Height (m); no altitude hold', 'Measured axis angle (deg)')):
            axes[row, col].set_title(title)
            axes[row, col].set_xlabel('Time (s)')
            axes[row, col].legend(fontsize=7)
            axes[row, col].grid(alpha=.25)
    fig.suptitle('Matched parameters + shared PID; independent rigid-body integration. NOT flight validation.')
    fig.savefig(args.out / 'comparison.png', dpi=140)
    plt.close(fig)
    (args.out / 'comparison.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    print(json.dumps(rows, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('legacy', 'pybullet', 'compare'), required=True)
    parser.add_argument('--reference', type=Path, default=Path('configs/ready_sim_reference_v1.json'))
    parser.add_argument('--out', type=Path, default=Path('output/ready_sim_spike_v1'))
    args = parser.parse_args()
    compare(args) if args.backend == 'compare' else run(args)


if __name__ == '__main__':
    main()
