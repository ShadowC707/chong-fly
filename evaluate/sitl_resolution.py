"""Fixed 4/2/1-ms coupling check before introducing neural closed-loop scenes."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

CASES = ('hover', 'roll', 'pitch', 'yaw')
LIMITS = dict(position_m=.02, velocity_mps=.05, angle_deg=1.)


def compare_traces(coarse, fine):
    tc, tf = coarse['time_s'], fine['time_s']
    if (len(tc)<2 or len(tf)<2 or not np.isclose(tc[0], 0) or not np.isclose(tf[0], 0)
            or not np.isclose(tc[-1], 4) or not np.isclose(tf[-1], 4)):
        raise ValueError('Expected complete four-second horizon')
    indices = np.searchsorted(tf, tc)
    if np.any(indices >= len(tf)) or not np.allclose(tf[indices], tc, rtol=0, atol=1e-12):
        raise ValueError('Time grids do not align')
    a, b = coarse['state'], fine['state'][indices]
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Invalid state traces')
    delta = a-b
    angle = np.arctan2(np.sin(delta[:, 6:9]), np.cos(delta[:, 6:9]))
    values = dict(max_position_difference_m=float(np.linalg.norm(delta[:, :3], axis=1).max()),
                  max_velocity_difference_mps=float(np.linalg.norm(delta[:, 3:6], axis=1).max()),
                  max_angle_difference_deg=float(np.degrees(np.abs(angle)).max()))
    values['passed'] = (values['max_position_difference_m'] <= LIMITS['position_m']
                        and values['max_velocity_difference_mps'] <= LIMITS['velocity_mps']
                        and values['max_angle_difference_deg'] <= LIMITS['angle_deg'])
    return values


def read_trace(path):
    with np.load(path/'trace.npz', allow_pickle=False) as a:
        return {k: a[k] for k in a.files}


def report(out, source):
    comparisons, checks, legacy = [], [], []
    for case in CASES:
        fine = read_trace(out/f'ms1_{case}')
        for ms in (4, 2):
            comparisons.append(dict(case=case, ms=ms,
                                    **compare_traces(read_trace(out/f'ms{ms}_{case}'), fine)))
        for ms in (4, 2, 1):
            folder = out/f'ms{ms}_{case}'
            r = json.loads((folder/'report.json').read_text())
            trace = read_trace(folder)
            n = round(4000/ms)
            ack = trace['acks'][-n:]
            checks.append(dict(case=case, ms=ms, completed=r['completed'],
                clock_ok=bool(len(ack)==n and (np.diff(ack[:, 1]) == ms*1000).all()),
                motor_updates_ok=bool((ack[:, 2] == ms*8).all())))
        # Ensure default-path refactoring did not change the original numbers.
        old, new = read_trace(source/f'alt0_{case}'), read_trace(out/f'ms4_{case}')
        legacy.append(dict(case=case, exact_equal={k: bool(np.array_equal(old[k], new[k]))
                                                  for k in ('state','motors','pwm','time_s')}))
    passed = (all(v['passed'] for v in comparisons)
              and all(v['completed'] and v['clock_ok'] and v['motor_updates_ok'] for v in checks)
              and all(all(v['exact_equal'].values()) for v in legacy))
    result = dict(scope='Local coupling sensitivity, not flight admission or proof of convergence',
                  limits=LIMITS, passed=passed, comparisons=comparisons,
                  checks=checks, default_path_regression=legacy)
    (out/'comparison.json').write_text(json.dumps(result, indent=2)+'\n')
    lines=['# SITL: чутливість до фізичного кроку','',
           '| Маневр | Крок проти 1 мс | Позиція, см | Швидкість, м/с | Кут, град | Пороги |',
           '|---|---:|---:|---:|---:|---|']
    for v in comparisons:
        lines.append(f"| {v['case']} | {v['ms']} мс | {100*v['max_position_difference_m']:.4f} | {v['max_velocity_difference_mps']:.4f} | {v['max_angle_difference_deg']:.4f} | {v['passed']} |")
    (out/'summary.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 2, figsize=(11, 10), constrained_layout=True)
    for i, case in enumerate(CASES):
        for ms in (4, 2, 1):
            trace = read_trace(out/f'ms{ms}_{case}')
            t, s = trace['time_s'], trace['state']
            axes[i, 0].plot(t, 100*(s[:, 2]-1), label=f'{ms} ms')
            axis = {'hover': 8, 'roll': 6, 'pitch': 7, 'yaw': 8}[case]
            axes[i, 1].plot(t, np.degrees(s[:, axis]), label=f'{ms} ms')
        axes[i, 0].set_ylabel(case)
        for j, title in enumerate(('Height error (cm)', 'Axis angle (deg)')):
            axes[i, j].set(title=title, xlabel='Model time (s)')
            axes[i, j].grid(alpha=.25); axes[i, j].legend()
    fig.suptitle('Unchanged firmware and AltitudeHold; 50 Hz RC, ideal floor range')
    fig.savefig(out/'comparison.png', dpi=140); plt.close(fig)
    return result


def run(out, source):
    from evaluate.sitl_lockstep import physics, BIN
    previous = json.loads((source/'report.json').read_text())
    if hashlib.sha256(BIN.read_bytes()).hexdigest() != previous['firmware_sha256']:
        raise ValueError('Firmware changed; not a paired resolution experiment')
    out.mkdir(parents=True, exist_ok=False)
    files = ['evaluate/sitl_resolution.py', 'evaluate/sitl_lockstep.py', 'evaluate/ready_sim_spike.py',
             'simulation/altitude_control.py', 'simulation/ready_sim_contract.py',
             'simulation/sitl_contract.py', 'simulation/sitl_lockstep.py',
             'configs/ready_sim_reference_v1.json', 'configs/sitl_virtual_v1.cli']
    hashes = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in files}
    protocol = dict(cases=CASES, physics_steps_ms=[4,2,1], rc_period_s=.02, horizon_s=4.,
                    limits=LIMITS, no_tuning=True, network_loaded=False,
                    firmware_sha256=previous['firmware_sha256'], source_sha256=hashes)
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    for name in files:
        dest=out/'source_snapshot'/name; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(name, dest)
    for ms in (4, 2, 1):
        for case in CASES:
            result = physics(out/f'ms{ms}_{case}', source/'template', previous['hover_rc_pwm'],
                             case, altitude_map=previous['static_map'], physics_dt_s=ms/1000)
            print(ms, case, result['completed'], result.get('error'), flush=True)
    if any(hashlib.sha256(Path(name).read_bytes()).hexdigest()!=value for name,value in hashes.items()):
        raise ValueError('Executed sources changed during experiment')
    result = report(out, source)
    print(json.dumps(result, indent=2))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--report-only', action='store_true')
    args=parser.parse_args()
    result=report(args.out,args.source) if args.report_only else run(args.out,args.source)
    if not result['passed']: raise SystemExit(1)


if __name__=='__main__': main()
