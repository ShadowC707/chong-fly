"""Four fixed cosine-Yaw probes: nominal/TWR5 x both signs. No network or tuning."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from evaluate.sitl_yaw_profiles import trial, DT, CONTROL_DT, HORIZON, ENDS, yaw_offset


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def assess_pair(nominal,lower):
    settle=lower.get('actual_rate_settle_from_step_end_s')
    command=lower.get('command_settle_from_step_end_s')
    checks=dict(complete=lower['completed'],direction=lower['final_directed_yaw_deg']>0,
        height_10cm=lower['max_height_error_m']<=.1,no_motor_limit=lower.get('saturated_motor_ticks',1)==0,
        command_recovery=command is not None and command<=.8,
        physical_recovery=settle is not None and settle<=.8)
    return dict(checks=checks,passed=all(checks.values()),
        physical_recovery_margin_s=None if settle is None else .8-settle,
        angle_change_fraction=(lower['final_directed_yaw_deg']-nominal['final_directed_yaw_deg'])/max(abs(nominal['final_directed_yaw_deg']),1e-12))


def run(out,source,reference,calibration_source):
    from evaluate.sitl_lockstep import BIN
    from simulation.sitl_twr import thrust_to_weight
    p=json.loads(Path('configs/ready_sim_reference_v1.json').read_text())['uncertainty_study']['nominal_dynamics']
    old=json.loads((reference/'protocol.json').read_text())
    if digest(BIN)!=old['firmware_sha256']:raise ValueError('Firmware changed')
    calibration_protocol=json.loads((calibration_source/'protocol.json').read_text())
    # These later revisions were already pinned in the completed TWR series.
    # Their navigation/integration paths are not called by this fixed-profile
    # experiment. The shared actuator helper is additionally covered by exact
    # nominal trace reproduction below; arbitrary new drift is still rejected.
    known_later={'simulation/phased_coordinator.py','evaluate/sitl_integrated.py','evaluate/sitl_neural.py'}
    changed=[k for k,v in old['input_source_sha256'].items() if k!='evaluate/sitl_yaw_profiles.py' and digest(k)!=v]
    if any(k not in known_later or digest(k)!=calibration_protocol['input_source_sha256'][k] for k in changed):
        raise ValueError('Prior fixed-profile physics inputs changed: '+str(changed))
    cal=json.loads((calibration_source/'calibration/report.json').read_text())
    if not all(b['bracketed'] for b in cal['hover_brackets']) or max(abs(v) for v in cal['old_anchor_errors'].values())>1e-8:
        raise ValueError('Unverified actuator calibration')
    if calibration_protocol['firmware_sha256']!=digest(BIN):raise ValueError('Calibration firmware mismatch')
    for path in ('configs/ready_sim_reference_v1.json','configs/sitl_virtual_v1.cli','simulation/sitl_twr.py'):
        if digest(path)!=calibration_protocol['input_source_sha256'][path]:raise ValueError('Calibration inputs changed: '+path)
    files=[v for folder in ('simulation','evaluate','configs') for v in Path(folder).glob('*.py')]
    files += [Path('configs/ready_sim_reference_v1.json'),Path('configs/sitl_virtual_v1.cli'),
        source/'template/eeprom.bin',source/'report.json',calibration_source/'calibration/report.json',reference/'protocol.json',
        reference/'cosine_-1/trace.npz',reference/'cosine_+1/trace.npz']
    hashes={str(f):digest(f) for f in files if f.is_file()}
    out.mkdir(parents=True,exist_ok=False)
    protocol=dict(version='isolated-yaw-twr-v1',targets=[thrust_to_weight(p),5.],signs=[-1,1],
        profile='Existing cosine: +/-140 PWM; .24s rise, .06s plateau, .24s fall; area 16.8deg',
        start_s=1.,fall_start_s=ENDS['step'],profile_end_s=ENDS['cosine'],horizon_s=HORIZON,
        physics_dt_s=DT,control_dt_s=CONTROL_DT,sensors='Ideal fresh floor range/attitude at 50Hz',
        treatment='max_rpm only plus verified collective thrust adapter; same mass, inertia, lag, drag, FC/PID and yaw input',
        excluded='No network, coordinator, room obstacles, transport or multirate sensor scheduling',
        primary_recovery='Actual rate enters +/-8deg/s and remains to trace end within .8s after falling-edge start at 1.30s',
        secondary_recovery='Also report from end of full cosine at 1.54s; do not shift primary deadline',
        criteria='Full 6s, correct directed turn, height<=.10m, no motor limit, command +/-20PWM and actual rate +/-8deg/s by primary deadline',
        controls='Nominal arrays must exactly equal original yaw_profiles_v1 cosine traces; paired yaw/roll/pitch commands identical',
        later_modules_verified_against_twr_series=changed,
        no_search=True,no_training=True,no_promotion=True,input_sha256=hashes,firmware_sha256=digest(BIN))
    write(out/'protocol.json',protocol)
    for path in ('evaluate/sitl_yaw_profiles.py','evaluate/sitl_yaw_twr.py','simulation/sitl_twr.py'):
        dest=out/'source_snapshot'/path;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
    print('Protocol frozen',flush=True)
    results=[];validation={};pairs=[]
    for sign in (-1,1):
        for mode in ('nominal','twr5'):
            folder=out/f'{mode}_{sign:+d}'
            kwargs={} if mode=='nominal' else dict(target_twr=5.,calibration_rows=cal['rows'])
            r=trial(folder,source,'cosine',sign,**kwargs)
            r.update(mode=mode,twr=thrust_to_weight(p) if mode=='nominal' else 5.)
            results.append(r);write(out/'report.json',results)
            with np.load(folder/'trace.npz') as a:
                n=len(a['motors']);ack=a['acks'][-n:]
                checks=dict(clock=bool(n and (np.diff(ack[:,0])==1).all() and (np.diff(ack[:,1])==250).all() and (ack[:,2]==2).all()))
                if mode=='nominal':
                    with np.load(reference/f'cosine_{sign:+d}'/'trace.npz') as b:
                        checks['baseline_exact']={key:bool(np.array_equal(a[key],b[key])) for key in ('time_s','state','motors','rpm','forces','pwm','reference_pwm')}
                else:
                    with np.load(out/f'nominal_{sign:+d}'/'trace.npz') as b:
                        checks['same_axes_reference']=bool(np.array_equal(a['reference_pwm'][:,1:],b['reference_pwm'][:,1:]))
                        checks['same_axes_actual_rc']=bool(np.array_equal(a['pwm'][:,1:],b['pwm'][:,1:]))
            validation[folder.name]=checks
        pairs.append(dict(sign=sign,**assess_pair(results[-2],results[-1])))
    summary=dict(sources_unchanged=all(digest(k)==v for k,v in hashes.items()),runs=validation,pairs=pairs,deployment_admitted=False)
    write(out/'validation.json',summary);render(out,results)
    print(json.dumps(summary,indent=2),flush=True)
    if not summary['sources_unchanged'] or any(not r['completed'] for r in results) or any(not c['clock'] for c in validation.values()):raise SystemExit(1)
    if any(not all(c.get('baseline_exact',{}).values()) or not c.get('same_axes_reference',True) or not c.get('same_axes_actual_rc',True) for c in validation.values()):raise SystemExit('Paired control differs')


def render(out,results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    for row,sign in zip(axes,(-1,1)):
        for mode in ('nominal','twr5'):
            with np.load(out/f'{mode}_{sign:+d}'/'trace.npz') as a:
                t,s=a['time_s'],a['state']
                row[0].plot(t,-sign*np.rad2deg(s[:,11]),label=mode)
                row[1].plot(t,100*(s[:,2]-1),label=mode)
                row[2].plot(t,-sign*np.rad2deg(np.unwrap(s[:,8])),label=mode)
                target_yaw=sign*(a['reference_pwm'][:,3]-1500)*.4
        row[0].plot(t[:-1],target_yaw,ls=':',color='black',label='same rate target')
        row[0].axhline(8,color='.6',ls='--');row[0].axhline(-8,color='.6',ls='--')
        row[0].axvline(2.1,color='red',ls=':',label='primary deadline')
        for ax,title in zip(row,('Directed Yaw rate (deg/s)','Height error (cm)','Directed angle (deg)')):
            ax.set(title=f'sign {sign:+d} | '+title,xlabel='time (s)',xlim=(0,4));ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle('Identical cosine Yaw | nominal vs TWR5 | live altitude, frozen FC | no neural decisions')
    fig.savefig(out/'comparison.png',dpi=145);plt.close(fig)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--source',type=Path,required=True);ap.add_argument('--reference',type=Path,required=True)
    ap.add_argument('--calibration-source',type=Path,required=True)
    a=ap.parse_args();run(a.out,a.source,a.reference,a.calibration_source)
