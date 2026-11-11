"""Recompute bounded lockstep evidence from saved traces, without Docker."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

CASES = ('hover', 'roll', 'pitch', 'yaw')


def check_trace(trace, report):
    """Check the declared four-second protocol, not admission to flight."""
    state, times = trace['state'], trace['time_s']
    motors, pwm, acks = trace['motors'], trace['pwm'], trace['acks']
    checks = {
        'completed': report.get('completed') is True,
        'armed_angle': all(report.get('armed', {}).get(k) is True for k in ('armed', 'angle')),
        'shapes': state.shape == (1001, 12) and motors.shape == (1000, 4)
                  and pwm.shape == (1000, 4) and times.shape == (1001,),
        'finite': all(np.isfinite(a).all() for a in (state, times, motors, pwm, acks)),
        'firmware_horizon': report.get('firmware_elapsed_us') == 4000000,
        'ack_shape': acks.ndim == 2 and acks.shape[1] == 3 and len(acks) >= 1000,
    }
    checks['physics_clock'] = times.shape == (1001,) and np.array_equal(times, np.arange(1001)*.004)
    checks['ack_sequence_clock'] = checks['ack_shape'] and bool(
        (np.diff(acks[:, 0]) == 1).all() and (np.diff(acks[:, 1]) == 4000).all())
    checks['motor_updates'] = checks['ack_shape'] and bool((acks[-1000:, 2] == 32).all())
    checks['rc_50hz_hold'] = pwm.shape == (1000, 4) and bool(
        (np.rint(pwm.reshape(200, 5, 4)) == np.rint(pwm[::5, None, :])).all())
    checks['motor_range'] = bool(((motors >= 0) & (motors <= 1)).all())
    return checks


def summarize(root):
    source = json.loads((root/'report.json').read_text())
    rows, comparisons, traces = [], [], {}
    for prefix, repeats in (('r', 3), ('alt', 2)):
        for case in CASES:
            for repeat in range(repeats):
                name = f'{prefix}{repeat}_{case}'
                folder = root/name
                report = json.loads((folder/'report.json').read_text())
                with np.load(folder/'trace.npz', allow_pickle=False) as stored:
                    trace = {k: stored[k] for k in stored.files}
                traces[name] = trace
                checks = check_trace(trace, report)
                rows.append({'name': name, 'checks': checks, 'passed': all(checks.values()),
                    'max_height_error_cm': float(np.max(np.abs(trace['state'][:, 2]-1))*100),
                    'final_position_m': trace['state'][-1, :3].tolist()})
                if repeat:
                    ref = traces[f'{prefix}0_{case}']
                    equality = {key: bool(np.array_equal(ref[key], trace[key]))
                                for key in ('time_s', 'state', 'motors', 'pwm')}
                    comparisons.append({'reference': f'{prefix}0_{case}', 'repeat': name,
                                        'exact_equal': equality})
    # World frame: +Pitch moves +X, +Roll moves -Y, +Yaw rotates clockwise.
    signs = {prefix: {
        'pitch_forward': bool(traces[f'{prefix}0_pitch']['state'][375, 0] > 0),
        'roll_right': bool(traces[f'{prefix}0_roll']['state'][375, 1] < 0),
        'yaw_clockwise': bool(traces[f'{prefix}0_yaw']['state'][375, 8] < 0),
    } for prefix in ('r', 'alt')}
    patch_matches = hashlib.sha256((root/'firmware.patch').read_bytes()).hexdigest() == source['harness_patch_sha256']
    protocol = source['static_map']['duplicate_ack_identical'] and all(source['static_map']['conflict_and_gap_rejected'])
    admitted = (all(row['passed'] for row in rows) and patch_matches and protocol
                and all(all(c['exact_equal'].values()) for c in comparisons)
                and all(all(s.values()) for s in signs.values()))
    summary = {'scope': 'Four-second timing, axis signs and actuator-map experiment only; ideal floor sensor.',
        'timing_mapping_checks_passed': bool(admitted), 'flight_admitted': False,
        'runs': len(rows), 'runs_checks_passed': sum(row['passed'] for row in rows),
        'patch_hash_matches': patch_matches, 'duplicate_conflict_gap_checks_passed': protocol,
        'axis_signs': signs, 'comparisons': comparisons, 'results': rows,
        'firmware_sha256': source['firmware_sha256'], 'harness_patch_sha256': source['harness_patch_sha256']}
    return summary, traces


def plot(root, traces):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 3, figsize=(13, 11), constrained_layout=True)
    for i, case in enumerate(CASES):
        for prefix, label in (('r', 'Fixed throttle 1210'), ('alt', 'AltitudeHold + measured map')):
            trace = traces[f'{prefix}0_{case}']
            t, state = trace['time_s'], trace['state']
            axes[i, 0].plot(t, 100*(state[:, 2]-1), label=label)
            axis = {'hover': 8, 'roll': 6, 'pitch': 7, 'yaw': 8}[case]
            axes[i, 1].plot(t, np.degrees(state[:, axis]), label=label)
            axes[i, 2].step(t[:-1:5], np.rint(trace['pwm'][::5, 0]), where='post', label=label)
        axes[i, 0].set_ylabel(case)
        for j, title in enumerate(('Height error (cm)', 'Axis angle (deg; hover: yaw)', 'Throttle RC (integer PWM)')):
            axes[i, j].set_title(title)
            axes[i, j].set_xlabel('Model time (s)')
            axes[i, j].grid(alpha=.25)
            if case != 'hover': axes[i, j].axvspan(1, 1.5, color='grey', alpha=.12)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Patched Betaflight SITL + PyBullet | ideal floor range, no neural policy\n'
                 'All repeats identical, including injected wall-time pauses; shown once')
    fig.savefig(root/'comparison.png', dpi=145)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('output/sitl_bringup_v1/lockstep_v1'))
    args = parser.parse_args()
    summary, traces = summarize(args.root)
    (args.root/'validation.json').write_text(json.dumps(summary, indent=2)+'\n')
    plot(args.root, traces)
    print(json.dumps({k: v for k, v in summary.items() if k not in ('comparisons', 'results')}, indent=2))
    if not summary['timing_mapping_checks_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
