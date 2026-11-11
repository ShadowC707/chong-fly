"""Preregistered left/right pairs of event-driven yaw shaping; no fitting."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np

CASES = ('obstacle_right', 'obstacle_left')
VARIANTS = ('control', 'smooth')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def recovery_metrics(t, signal, start, end, band):
    from evaluate.sitl_yaw_profiles import settled_delay
    mask = (t >= start-1e-9) & (t < end-1e-9)
    if not mask.any() or t[mask][-1]-start < .8-1e-9:
        return dict(censored=True, settle_s=None)
    delay = settled_delay(t[mask], signal[mask], end_s=start, band=band)
    # A terminal isolated sample does not prove stable settling.
    if delay is not None and t[mask][-1]-(start+delay) < .1-1e-9: delay = None
    return dict(censored=False, settle_s=delay)


def inspect(folder):
    result = json.loads((folder/'report.json').read_text())
    decisions = json.loads((folder/'decisions.json').read_text())
    rows = json.loads((folder/'command_trace.json').read_text())
    with np.load(folder/'trace.npz') as a:
        t, s, ct = a['time_s'], a['state'], a['control_time']
        issued = np.array([r['selected']['applied_pwm'] for r in rows])
        auto = np.array([r['accepted_pwm'] is not None for r in rows])
        recovery = []
        for i, d in enumerate(decisions):
            if d['event'] != 'clear': continue
            end = next((v['time_s'] for v in decisions[i+1:]
                        if v['phase'] in ('BRAKE', 'TURN', 'STOP', 'FAILED', 'CRITICAL', 'WAIT_CLEAR')), 20.)
            recovery.append(dict(start_s=d['time_s'], end_s=end,
                issued=recovery_metrics(ct, issued[:,3]-1500., d['time_s'], end, 20.),
                actual=recovery_metrics(t, np.degrees(s[:,11]), d['time_s'], end, 8.)))
        n=len(a['motors']); ack=a['acks'][-n:]
        result.update(clock_passed=bool(n and (np.diff(ack[:,0])==1).all()
            and (np.diff(ack[:,1])==250).all() and (ack[:,2]==2).all()),
            max_yaw_rate_deg_s=float(np.degrees(abs(s[:,11])).max()),
            yaw_final_deg=float(-np.degrees(np.unwrap(s[:,8]))[-1]),
            min_height_m=float(s[:,2].min()), max_height_m=float(s[:,2].max()),
            seconds_below_10cm=float(np.sum(a['clearance']<.1)*.00025),
            seconds_below_5cm=float(np.sum(a['clearance']<.05)*.00025),
            max_issued_yaw_step_pwm=float(abs(np.diff(issued[:,3]))[auto[1:] & auto[:-1]].max()),
            issued_yaw_abs_integral_deg=float(abs(issued[:,3]-1500.).sum()*.02*.4),
            accepted_equals_issued=all(r['accepted_pwm']==r['issued_pwm'] for r in rows),
            recoveries=recovery, phase_events=[d for d in decisions if d['event']])
    return result


def render(out, results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from evaluate.sitl_neural import scene_spec
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)
    for row, case in zip(axes, CASES):
        for b in scene_spec(case)['boxes']:
            row[0].add_patch(Rectangle((b['x_min'],b['y_min']), b['x_max']-b['x_min'],
                                     b['y_max']-b['y_min'], color='.8'))
        for variant in VARIANTS:
            folder=out/(case+'_'+variant)
            with np.load(folder/'trace.npz') as a:
                t,s,ct=a['time_s'],a['state'],a['control_time']
                row[0].plot(s[:,0],s[:,1],label=variant)
                row[1].plot(t,100*(s[:,2]-1),label=variant)
                row[2].plot(t,np.degrees(s[:,11]),label=variant)
                rows=json.loads((folder/'command_trace.json').read_text())
                row[3].plot(ct,[r['selected']['applied_pwm'][3]-1500 for r in rows],label=variant)
        row[0].set_aspect('equal', adjustable='datalim')
        for ax, title in zip(row, ('XY (m)', 'Height error (cm)', 'Actual yaw rate (deg/s)', 'Issued yaw offset (PWM)')):
            ax.set_title(case+' | '+title);ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Pinned candidate A | event-driven Yaw shaping | ToF 15 / range 30 Hz\n'
                 'Full authority + receiver; uncalibrated Betaflight/PyBullet, ideal attitude, no radio noise')
    fig.savefig(out/'comparison.png',dpi=140);plt.close(fig)


def run(out, source, reference):
    from evaluate.sitl_neural import trial
    from evaluate.sitl_lockstep import BIN
    from edge.system import RunConfig
    cfg=RunConfig.from_json('configs/demo_candidate_a.json')
    old=json.loads((reference/'protocol.json').read_text())
    if digest(BIN)!=old['firmware_sha256']: raise ValueError('Firmware mismatch')
    files=[p for f in ('simulation','edge','demo','evaluate') for p in Path(f).glob('*.py')]
    files += [Path('configs/demo_candidate_a.json'), Path('configs/ready_sim_reference_v1.json'),
              Path('configs/sitl_virtual_v1.cli'),source/'template/eeprom.bin',*Path(cfg.bundle_dir).glob('*')]
    hashes={str(p):digest(p) for p in files if p.is_file()}
    out.mkdir(parents=True,exist_ok=False)
    protocol=dict(version='sitl-phased-yaw-v1', cases=CASES, variants=VARIANTS,
        physical_dt_s=.00025, control_dt_s=.02, horizon_s=20.,stop_at_s=17.,
        sensors='flow/attitude 50 Hz, ToF 15 Hz, altitude 30 Hz; ideal sensors; zero noise',
        unchanged='Candidate A weights, pinned bundle, FC/PID, plant/TWR, ToF thresholds, transport',
        treatment='Cosine .24s nominal edge; 916.30 PWM/s bound incl moving CRUISE target; '
            'neutral RECOVER tail then live base; timeout reserves tail + one tick; issued AUTO sync',
        priority='Immediate native STOP/critical and external revoke bypass smoothing',
        criteria='Both pairs: complete, no contact/fault, clearance >=.10m, STOP <.03m/s, '
            'continuous AUTO; lower height error; every observed clear recovery issued <=20PWM '
            'and actual <=8deg/s within .8s. Censored windows do not pass. No fitting/retries.',
        recovery='From TURN->RECOVER until next BRAKE/TURN/FAILED/STOP/CRITICAL/WAIT_CLEAR, '
            'settle through remainder of window with >=.1s terminal dwell and >=.8s observation',
        reference=str(reference),input_source_sha256=hashes,firmware_sha256=digest(BIN),
        deployment_admitted=False)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    for name in hashes:
        p=Path(name)
        if not p.is_absolute() and p.suffix=='.py':
            dest=out/'source_snapshot'/p;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    results=[];baseline={}
    for case in CASES:
        for variant in VARIANTS:
            folder=out/(case+'_'+variant)
            trial(folder,source,case,'sitl','reference_units',physics_dt_s=.00025,
                  integration_profile='multi_rate',yaw_shaping=variant=='smooth')
            r=inspect(folder) | dict(variant=variant);results.append(r)
            (out/'comparison.json').write_text(json.dumps(results,indent=2)+'\n')
            print(json.dumps(r),flush=True)
            if variant=='control':
                with np.load(folder/'trace.npz') as a, np.load(reference/(case+'_multi_rate')/'trace.npz') as b:
                    baseline[case]={k:bool(np.array_equal(a[k],b[k],equal_nan=True))
                                    for k in ('state','motors','pwm','network_pwm','coordinated_pwm','supervised_pwm')}
    pairs=[]
    for case in CASES:
        a,b=[next(r for r in results if r['case']==case and r['variant']==v) for v in VARIANTS]
        recovery_ok=bool(b['recoveries']) and all(not w[c]['censored'] and w[c]['settle_s'] is not None
            and w[c]['settle_s']<=.8 for w in b['recoveries'] for c in ('issued','actual'))
        passed=(b['completed'] and not b['error'] and not b['collision'] and b['min_clearance_m']>=.10
                and b['stop_passed'] and b['authority_passed'] and b['max_height_error_m']<a['max_height_error_m']
                and recovery_ok)
        pairs.append(dict(case=case,recovery_passed=recovery_ok,research_continuation_passed=bool(passed)))
    validation=dict(sources_unchanged=all(digest(p)==v for p,v in hashes.items()),
        baseline_array_equal=baseline, clocks_passed=all(r['clock_passed'] for r in results),
        accepted_equals_issued=all(r['accepted_equals_issued'] for r in results),pairs=pairs,
        deployment_admitted=False)
    (out/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    render(out,results)
    print(json.dumps(validation,indent=2),flush=True)
    if (not validation['sources_unchanged'] or not validation['clocks_passed']
            or not all(all(v.values()) for v in baseline.values()) or any(r['error'] for r in results)):
        raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    a=p.parse_args();run(a.out,a.source,a.reference)
