"""Complete the fixed right-obstacle 2x2 sensor-rate experiment; no tuning."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

PROFILES = ('sync', 'tof_only', 'altitude_only', 'multi_rate')
RATES = {'sync': (50, 50), 'tof_only': (15, 50),
         'altitude_only': (50, 30), 'multi_rate': (15, 30)}  # ToF, altitude


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def schedule_parity(archived_source):
    """Compare existing profiles against their archived implementation, not a rewrite."""
    from simulation.sensor_schedule import SampleSchedule
    from edge.altitude_control import AltitudeSample
    namespace = {}
    exec(compile(Path(archived_source).read_text(), str(archived_source), 'exec'), namespace)
    checks = {}
    for profile in ('sync', 'multi', 'multi_rate'):
        old, new = namespace['SampleSchedule'](profile), SampleSchedule(profile)
        same = True
        for i in range(1000):
            sample = AltitudeSample(.982+i/10000, i/10000, -i/10000, i*.02, True)
            args = (i, np.array([i/1000, -i/1000]), np.full(64, i/1000), sample)
            a, b = [json.dumps(asdict(s.frame(*args, invalid_altitude=i%37==0)),
                               sort_keys=True, default=lambda v: v.tolist()) for s in (old, new)]
            same = same and a == b
        checks[profile] = same
    return checks


def inspect_run(folder, profile, mapping):
    from evaluate.sitl_neural import actuator_command
    from simulation.sitl_rc_units import reference_axes_to_actual
    result = json.loads((folder/'report.json').read_text())
    rows = json.loads((folder/'command_trace.json').read_text())
    decisions = json.loads((folder/'decisions.json').read_text())
    with np.load(folder/'trace.npz') as a:
        state, t = a['state'], a['time_s']
        issued = np.asarray([r['selected']['applied_pwm'] for r in rows])
        expected = np.array([reference_axes_to_actual(actuator_command(p, mapping)) for p in issued])
        n = len(a['motors']); ack = a['acks'][-n:]
        peak = int(np.argmax(abs(state[:,2]-1)))
        diagnostics = dict(
            clock_passed=bool(n and (np.diff(ack[:,0])==1).all()
                and (np.diff(ack[:,1])==250).all() and (ack[:,2]==2).all()),
            actual_rc_max_mapping_error_pwm=float(abs(expected-a['pwm'][::80]).max()),
            max_yaw_rate_deg_s=float(np.degrees(abs(state[:,11])).max()),
            max_upward_speed_m_s=float(state[:,5].max()),
            height_peak_time_s=float(t[peak]), height_peak_z_m=float(state[peak,2]),
            network_yaw_abs_pwm_s=float(abs(a['network_pwm'][:,3]-1500).sum()*.02),
            coordinated_yaw_abs_pwm_s=float(abs(a['coordinated_pwm'][:,3]-1500).sum()*.02),
            issued_yaw_abs_pwm_s=float(abs(issued[:,3]-1500).sum()*.02))
    diagnostics['phase_events'] = [r for r in decisions if r['event']]
    diagnostics['accepted_equals_issued'] = all(r['accepted_pwm']==r['issued_pwm'] for r in rows)
    diagnostics['sensor_counts'] = {}
    for name in ('flow', 'tof', 'altitude', 'attitude'):
        channel = [r['sensor_channels'][name] for r in rows]
        diagnostics['sensor_counts'][name] = dict(new=sum(r['state']=='new' for r in channel),
            held=sum(r['state']=='held' for r in channel), max_age_s=max(r['age_s'] for r in channel))
    return dict(profile=profile, rates_hz=RATES[profile], result=result, diagnostics=diagnostics)


def summarize(out, folders, mapping):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    reports = [inspect_run(folders[p], p, mapping) for p in PROFILES]
    (out/'comparison.json').write_text(json.dumps(reports, indent=2)+'\n')
    fig, axes = plt.subplots(3, 2, figsize=(13, 12), constrained_layout=True)
    lines = ['# Right obstacle: sensor-rate controls', '',
        '| Profile | ToF/range Hz | Height max cm | Clearance cm | Yaw rate max deg/s | AUTO | STOP | Contact |',
        '|---|---|---:|---:|---:|---|---|---|']
    for row in reports:
        p, r, d = row['profile'], row['result'], row['diagnostics']
        lines.append(f"| {p} | {RATES[p]} | {100*r['max_height_error_m']:.2f} | "
            f"{100*r['min_clearance_m']:.2f} | {d['max_yaw_rate_deg_s']:.2f} | "
            f"{r['authority_passed']} | {r['stop_passed']} | {r['collision']} |")
        with np.load(folders[p]/'trace.npz') as a:
            t, s, ct = a['time_s'], a['state'], a['control_time']
            axes[0,0].plot(t, 100*(s[:,2]-1), label=p)
            axes[0,1].plot(t, np.degrees(s[:,11]), label=p)
            axes[1,0].plot(ct, a['network_pwm'][:,3]-1500, label=p)
            axes[1,1].plot(ct, a['coordinated_pwm'][:,3]-1500, label=p)
            axes[2,0].plot(ct, a['supervised_pwm'][:,0], label=p)
            axes[2,1].plot(t, np.linalg.norm(s[:,3:5], axis=1), label=p)
    titles = ('Height error (cm)', 'Body yaw rate (deg/s)', 'Network yaw (reference PWM offset)',
              'Coordinated yaw (reference PWM offset)', 'Supervised throttle (reference PWM)', 'XY speed (m/s)')
    for ax, title in zip(axes.flat, titles):
        ax.set(title=title, xlabel='Time (s)', xlim=(0, 10)); ax.grid(alpha=.25); ax.legend()
    fig.suptitle('Same candidate / controllers / right obstacle | first 10 s of 20 s runs\n'
                 'sync and multi_rate reused; only tof_only and altitude_only are new')
    fig.savefig(out/'comparison.png', dpi=145); plt.close(fig)
    (out/'summary.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return reports


def run(out, source, reference):
    from evaluate.sitl_neural import trial
    from evaluate.sitl_lockstep import BIN
    old = json.loads((reference/'protocol.json').read_text())
    if digest(BIN) != old['firmware_sha256']:
        raise ValueError('Firmware mismatch')
    allowed = {'simulation/sensor_schedule.py', 'evaluate/sitl_integrated.py'}
    mismatches = [k for k,v in old['input_source_sha256'].items() if k not in allowed and digest(k)!=v]
    if mismatches: raise ValueError('Changed input outside declared scheduling edits: '+str(mismatches))
    parity = schedule_parity(reference/'source_snapshot/simulation/sensor_schedule.py')
    if not all(parity.values()): raise ValueError('Old acquisition profiles changed')
    folders = {p:reference/('obstacle_right_'+p) for p in ('sync', 'multi_rate')}
    prior_hashes = {str(folder/name):digest(folder/name) for folder in folders.values()
                    for name in ('trace.npz', 'command_trace.json', 'decisions.json', 'report.json', 'manifest.json')}
    for folder in folders.values():
        r = json.loads((folder/'report.json').read_text())
        m = json.loads((folder/'manifest.json').read_text())
        if not (r['completed'] and r['authority_passed'] and not r['error']
                and m['dt_s']==.00025 and m['axis_contract']=='reference_units'):
            raise ValueError('Reference is not a complete compatible run')
    hashes = {k:digest(k) for k in old['input_source_sha256']}
    hashes['evaluate/sitl_sensor_factorial.py'] = digest(__file__)
    out.mkdir(parents=True, exist_ok=False)
    protocol = dict(version='sitl-sensor-factorial-v1', case='obstacle_right', profiles=RATES,
        new_runs=['tof_only', 'altitude_only'], reused_runs=['sync', 'multi_rate'],
        reference=str(reference), horizon_s=20., stop_at_s=17., physical_dt_s=.00025,
        control_dt_s=.02, firmware_sha256=old['firmware_sha256'], changed_inputs=sorted(allowed),
        legacy_schedule_parity=parity, input_source_sha256=hashes, reference_sha256=prior_hashes,
        rule='No tuning. All outcomes retained. Compare phase transitions, yaw stages/rate, height, '
             'clearance, STOP and AUTO. Single fixed start; closed-loop effects and interactions, '
             'not isolation of a neuron or proof of real-world validity.')
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    for name in sorted(allowed|{'evaluate/sitl_sensor_factorial.py', 'evaluate/sitl_neural.py'}):
        dest = out/'source_snapshot'/name; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(name, dest)
    new_results = []
    for profile in ('tof_only', 'altitude_only'):
        folders[profile] = out/('obstacle_right_'+profile)
        r = trial(folders[profile], source, 'obstacle_right', 'sitl', 'reference_units',
                  physics_dt_s=.00025, integration_profile=profile)
        new_results.append(r)
        (out/'report.json').write_text(json.dumps(new_results, indent=2)+'\n')
        print(json.dumps(r), flush=True)
    mapping = json.loads((source/'report.json').read_text())['static_map']['rows']
    reports = summarize(out, folders, mapping)
    validation = dict(sources_unchanged=all(digest(k)==v for k,v in hashes.items()),
        references_unchanged=all(digest(k)==v for k,v in prior_hashes.items()),
        legacy_schedule_parity=parity, new_runs=2, reused_runs=2,
        all_clocks_passed=all(r['diagnostics']['clock_passed'] for r in reports),
        all_command_maps_exact=all(r['diagnostics']['actual_rc_max_mapping_error_pwm']==0 for r in reports),
        all_accepted_equal_issued=all(r['diagnostics']['accepted_equals_issued'] for r in reports),
        deployment_admitted=False)
    (out/'validation.json').write_text(json.dumps(validation, indent=2)+'\n')
    print(json.dumps(validation), flush=True)
    if (not all(validation[k] for k in ('sources_unchanged','references_unchanged','all_clocks_passed',
                                       'all_command_maps_exact','all_accepted_equal_issued'))
            or any(not r['completed'] or r['error'] for r in new_results)):
        raise SystemExit(1)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    a = p.parse_args(); run(a.out, a.source, a.reference)
