"""One preregistered 45 s multi-obstacle SITL flight; no tuning or retries."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from simulation.sitl_flight import long_scene


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def trace_metrics(t, state, gaps, acks):
    if len(t)<2 or len(gaps)!=len(t)-1:
        raise ValueError('Incomplete physics trace')
    dt=np.diff(t)
    if not np.allclose(dt, .00025, atol=1e-12, rtol=0):
        raise ValueError('Unexpected physics clock')
    n=len(gaps); ack=acks[-n:]; idx=int(np.argmin(gaps))
    body_x=state[:-1, 3]*np.cos(state[:-1, 8])+state[:-1, 4]*np.sin(state[:-1, 8])
    out=dict(min_clearance_time_s=float(t[idx+1]), min_clearance_position_m=state[idx+1,:3].tolist(),
        reverse_distance_m=float(np.sum(np.where(body_x<-.03, -body_x, 0)*dt)),
        clock_passed=bool(len(ack)==n and (np.diff(ack[:,0])==1).all()
            and (np.diff(ack[:,1])==250).all() and (ack[:,2]==2).all()))
    for cm in (5,10):
        low=gaps<cm/100
        out[f'seconds_below_{cm}cm']=float(np.sum(dt[low]))
        out[f'episodes_below_{cm}cm']=int(np.sum(low & ~np.r_[False,low[:-1]]))
    return out


def run(out, source, calibration_source):
    from evaluate.sitl_lockstep import BIN
    from evaluate.sitl_neural import trial
    from edge.system import RunConfig
    cal_protocol=json.loads((calibration_source/'protocol.json').read_text())
    cal=json.loads((calibration_source/'calibration/report.json').read_text())
    source_report=json.loads((source/'report.json').read_text())
    if not (digest(BIN)==cal_protocol['firmware_sha256']==source_report['firmware_sha256']):
        raise ValueError('Firmware/calibration mismatch')
    for path in ('configs/ready_sim_reference_v1.json','configs/sitl_virtual_v1.cli','simulation/sitl_twr.py'):
        if digest(path)!=cal_protocol['input_source_sha256'][path]:
            raise ValueError('Calibration input changed: '+path)
    if not all(b['bracketed'] for b in cal['hover_brackets']) or max(abs(v) for v in cal['old_anchor_errors'].values())>1e-8:
        raise ValueError('Unverified calibration')
    cfg=RunConfig.from_json('configs/demo_candidate_a.json')
    files=[p for folder in ('simulation','edge','demo','evaluate') for p in Path(folder).glob('*.py')]
    files += [Path('configs/demo_candidate_a.json'),Path('configs/ready_sim_reference_v1.json'),
        Path('configs/sitl_virtual_v1.cli'),source/'template/eeprom.bin',source/'report.json',
        calibration_source/'calibration/report.json',*Path(cfg.bundle_dir).glob('*')]
    hashes={str(p):digest(p) for p in files if p.is_file()}
    out.mkdir(parents=True,exist_ok=False)
    protocol=dict(version='sitl-long-flight-v1',scene=long_scene(),horizon_s=45.,stop_at_s=40.,
        target_twr=5.,physics_dt_s=.00025,control_dt_s=.02,
        controller='Frozen Candidate A + SmoothPhasedCoordinator; unchanged weights/thresholds/FC PID',
        sensors='Ideal noise-free flow/attitude 50 Hz, range 30 Hz, ToF 15 Hz',
        transport='FlightSystem -> SessionReceiver, immediate delivery, polled every physics step',
        criteria=dict(full_horizon_no_contact=True,min_clearance_m=.10,max_height_error_m=.10,
            stop_last_second_max_speed_mps=.03,continuous_auto_after_admission=True,clock_verified=True),
        limitations=['Uncalibrated plant and ideal sensors; no physical flight admission',
            'Geometric sphere contacts terminate flight; no impact response',
            'One fixed scene/start, no route goal, no claim of scene coverage'],
        no_training=True,no_parameter_search=True,no_promotion=True,
        input_sha256=hashes,firmware_sha256=digest(BIN))
    write(out/'protocol.json',protocol)
    for name in hashes:
        p=Path(name)
        if not p.is_absolute() and p.suffix=='.py':
            dest=out/'source_snapshot'/p;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    print('Frozen long-flight protocol: 45s / STOP40 / TWR5 / one scene',flush=True)
    r=trial(out/'flight',source,'long_arena','sitl','reference_units',physics_dt_s=.00025,
        integration_profile='multi_rate',yaw_shaping=True,scene_override=long_scene(),
        target_twr=5.,calibration_rows=cal['rows'],horizon_s=45.,stop_at_s=40.)
    with np.load(out/'flight/trace.npz') as a:
        r.update(trace_metrics(a['time_s'],a['state'],a['clearance'],a['acks']))
    rows=json.loads((out/'flight/command_trace.json').read_text())
    accepted=[row for row in rows if row['accepted_pwm'] is not None]
    r['accepted_equals_issued']=bool(accepted) and all(row['accepted_pwm']==row['issued_pwm'] for row in accepted)
    r['sources_unchanged']=all(digest(k)==v for k,v in hashes.items())
    r['checks']=dict(full_horizon=bool(r['completed'] and not r['error'] and not r['collision']),
        clearance=r['min_clearance_m'] is not None and r['min_clearance_m']>=.10,
        height=r['max_height_error_m']<=.10,stop=r['stop_passed'],authority=r['authority_passed'],
        clock=r['clock_passed'],accepted_equals_issued=r['accepted_equals_issued'],sources=r['sources_unchanged'])
    r['passed']=all(r['checks'].values());r['deployment_admitted']=False
    write(out/'report.json',r)
    print(json.dumps(r,indent=2),flush=True)
    # A behavioral failure is a result, not a reason to retry or withhold replay.
    if r['error'] or not r['sources_unchanged'] or not r['clock_passed']:
        raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--calibration-source',type=Path,required=True)
    a=p.parse_args();run(a.out,a.source,a.calibration_source)
