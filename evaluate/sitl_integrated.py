"""Nominal offline FlightSystem -> receiver -> SITL experiment, no radio I/O.

All bridge/receiver commands retain reference PWM units. Conversion to the
virtual FC's RC curve happens ONLY after receiver selection in sitl_neural.
The pre-AUTO/fallback source is an explicit simulation hover controller, not
an assertion about physical ESP32 failsafe behavior.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from simulation.sitl_control import IntegratedControl


def run(out, source):
    from evaluate.sitl_lockstep import BIN
    from evaluate.sitl_neural import CASES, trial, scene_spec
    from edge.system import RunConfig
    expected = json.loads((source/'report.json').read_text())['firmware_sha256']
    if hashlib.sha256(BIN.read_bytes()).hexdigest() != expected:
        raise ValueError('Firmware differs from validated build')
    cfg = RunConfig.from_json('configs/demo_candidate_a.json')
    files = [p for folder in ('simulation', 'edge', 'demo', 'evaluate', 'configs')
             for p in Path(folder).glob('*.py')]
    files += [Path('configs/demo_candidate_a.json'), Path('configs/ready_sim_reference_v1.json'),
              Path('configs/sitl_virtual_v1.cli'), source/'template/eeprom.bin',
              *Path(cfg.bundle_dir).glob('*')]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()}
    out.mkdir(parents=True, exist_ok=False)
    protocol = dict(version='sitl-full-system-v1', cases=[dict(name=c, **scene_spec(c)) for c in CASES],
        sensor_profiles=['sync', 'multi_rate'], physical_dt_s=.00025, control_dt_s=.02,
        horizon_s=20., stop_at_s=17., admission='Unmodified FlightBridge; request AUTO from t=0; no bypass',
        transport='Immediate in-process delivery; no loss/jitter; receiver polled every 250 us',
        fallback='Simulation only: independent range AltitudeHold, neutral R/P/Y, 50 Hz; held between ticks',
        units='Reference PWM through gate; measured throttle map + reference_axes_to_actual AFTER selection',
        sensor_schedule='sync: all 50 Hz; multi_rate: flow/attitude 50, range 30, ToF 15 Hz; noise zero',
        assessment='Report every contact/fault, min clearance, height error, STOP < .03 m/s throughout final second; '
                   'AUTO must enter by 2 s and remain accepted through horizon. No fitting/retries.',
        limitations=['Ideal attitude and de-rotation; uncalibrated plant and FC; no radio or hardware admission',
                     '250 us is research reference, not proof of timestep convergence'],
        firmware_sha256=expected, input_source_sha256=hashes)
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    for name in ('evaluate/sitl_integrated.py', 'evaluate/sitl_neural.py', 'evaluate/sitl_lockstep.py',
                 'evaluate/ready_sim_spike.py', 'simulation/sensor_schedule.py', 'simulation/sitl_flight.py',
                 'simulation/sitl_plant.py', 'simulation/sitl_process.py', 'simulation/sitl_control.py'):
        dest = out/'source_snapshot'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(name, dest)
    results = []
    for case in CASES:
        for profile in ('sync', 'multi_rate'):
            r = trial(out/(case+'_'+profile), source, case, 'sitl', 'reference_units',
                      physics_dt_s=.00025, integration_profile=profile)
            results.append(r)
            (out/'report.json').write_text(json.dumps(results, indent=2)+'\n')
            print(json.dumps(r), flush=True)
    unchanged = all(hashlib.sha256(Path(k).read_bytes()).hexdigest() == v for k, v in hashes.items())
    clocks = {}
    for r in results:
        name = r['case']+'_'+r['integration_profile']
        with np.load(out/name/'trace.npz') as a:
            n = len(a['motors']); ack = a['acks'][-n:]
            clocks[name] = bool(n and (np.diff(ack[:,0]) == 1).all()
                and (np.diff(ack[:,1]) == 250).all() and (ack[:,2] == 2).all())
    validation = dict(sources_unchanged=unchanged, clocks=clocks, runs=len(results),
        completed=sum(r['completed'] for r in results), contacts=sum(bool(r['collision']) for r in results),
        stop_passed=sum(r['stop_passed'] for r in results),
        authority_passed=sum(r['authority_passed'] for r in results), deployment_admitted=False)
    (out/'validation.json').write_text(json.dumps(validation, indent=2)+'\n')
    render(out, results)
    print(json.dumps(validation, indent=2), flush=True)
    if not unchanged or not all(clocks.values()) or any(not r['completed'] or r['error'] for r in results):
        raise SystemExit(1)


def render(out, results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from evaluate.sitl_neural import CASES, scene_spec
    fig, axes = plt.subplots(4, 3, figsize=(14, 14), constrained_layout=True)
    lines = ['# Full system SITL: sync vs held sensors', '',
        '| Scene | Sensors | Complete | Contact | AUTO start, s | AUTO, s | Clearance, cm | Height max, cm | STOP max, m/s | Fault |',
        '|---|---|---|---|---:|---:|---:|---:|---:|---|']
    for axes_row, case in zip(axes, CASES):
        for b in scene_spec(case)['boxes']:
            axes_row[0].add_patch(Rectangle((b['x_min'], b['y_min']), b['x_max']-b['x_min'],
                                          b['y_max']-b['y_min'], color='.8'))
        for profile in ('sync', 'multi_rate'):
            r = next(v for v in results if v['case'] == case and v['integration_profile'] == profile)
            with np.load(out/(case+'_'+profile)/'trace.npz') as a:
                s, t = a['state'], a['time_s']
                axes_row[0].plot(s[:,0], s[:,1], label=profile)
                axes_row[1].plot(t, 100*(s[:,2]-1), label=profile)
                axes_row[2].plot(t, np.linalg.norm(s[:,3:5], axis=1), label=profile)
            lines.append(f"| {case} | {profile} | {r['completed']} | {r['collision']} | {r['auto_first_s']} | "
                f"{r['auto_time_s']:.2f} | {100*r['min_clearance_m']:.2f} | {100*r['max_height_error_m']:.2f} | "
                f"{r['stop_last_second_max_speed_mps']} | {r['error'] or r['authority_faults']} |")
        for ax, title in zip(axes_row, ('XY (m)', 'Height error (cm)', 'Speed (m/s)')):
            ax.set_title(case+' | '+title); ax.grid(alpha=.2); ax.legend()
        axes_row[0].set_aspect('equal', adjustable='datalim')
        axes_row[2].axvspan(17, 20, alpha=.1, color='grey')
    fig.suptitle('Pinned candidate A | full authority + receiver | patched Betaflight + PyBullet\n'
                 'Zero-delay transport, ideal attitude, no noise; uncalibrated simulation')
    fig.savefig(out/'comparison.png', dpi=140); plt.close(fig)
    (out/'summary.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    run(args.out, args.source)
