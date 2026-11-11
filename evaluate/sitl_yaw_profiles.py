"""Four fixed Yaw profiles, live altitude feedback, unchanged SITL plant/FC.

No neural decisions, XY coordinator, receiver or deployment admission.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np

DT = .00025
CONTROL_DT = .02
HORIZON = 6.
START = 1.
ENDS = {'step': 1.30, 'cosine': 1.54}


def yaw_offset(profile, control_tick, sign):
    if profile not in ENDS or type(control_tick) is not int or control_tick < 0 or sign not in (-1, 1):
        raise ValueError('Invalid fixed yaw profile arguments')
    k = control_tick-50
    if profile == 'step':
        factor = 1. if 0 <= k < 15 else 0.
    elif k < 0 or k >= 27:
        factor = 0.
    elif k < 12:
        factor = (1-np.cos(np.pi*k/12))/2
    elif k < 15:
        factor = 1.
    else:
        factor = (1+np.cos(np.pi*(k-15)/12))/2
    return float(sign*140*factor)


def settled_delay(time, signal, *, end_s, band):
    """First in-band sample that stays in band for the REMAINDER of the trace."""
    t, value = np.asarray(time), np.asarray(signal)
    eligible = np.flatnonzero(t >= end_s-1e-10)
    if not len(eligible): return None
    outside = np.flatnonzero((t >= end_s-1e-10) & (abs(value) > band))
    index = max(int(eligible[0]), int(outside[-1])+1 if len(outside) else 0)
    return None if index >= len(t) else float(max(0., t[index]-end_s))


def motor_limit_counts(motors):
    # Pinned PWM profile: mincommand=1000, motor_idle=550, max_throttle=2000.
    # initEscEndpoints: minThrottle = mincommand + motorIdle * .1.
    # SITL output = (PWM - mincommand)/1000. These are sampled limit hits,
    # not an observation of every internal firmware clipping operation.
    m = np.asarray(motors)
    lower = np.any(m <= .055+1e-6, axis=1)
    upper = np.any(m >= 1-1e-6, axis=1)
    return dict(lower_limit_ticks=int(lower.sum()), upper_limit_ticks=int(upper.sum()),
                saturated_motor_ticks=int((lower | upper).sum()))


def trial(out, source, profile, sign, *, target_twr=None, calibration_rows=None):
    if (target_twr is None) != (calibration_rows is None):
        raise ValueError('TWR target and measured calibration rows must be supplied together')
    from evaluate.sitl_lockstep import Session
    from evaluate.ready_sim_spike import BulletPlant
    from evaluate.sitl_neural import actuator_command
    from evaluate.sitl_yaw_diagnosis import split_thrust
    from simulation.altitude_control import AltitudeHold, AltitudeSample
    from simulation.ready_sim_contract import make_urdf
    from simulation.sitl_rc_units import reference_axes_to_actual, validate_virtual_profile

    p = json.loads(Path('configs/ready_sim_reference_v1.json').read_text())['uncertainty_study']['nominal_dynamics']
    calibration = json.loads((source/'report.json').read_text())
    validate_virtual_profile(Path('configs/sitl_virtual_v1.cli'), p)
    adapter = None
    if target_twr is not None:
        from simulation.sitl_twr import ThrustAdapter
        adapter = ThrustAdapter(p,target_twr,calibration_rows)
        p = adapter.dynamics
    altitude = AltitudeHold()
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(source/'template/eeprom.bin', out/'eeprom.bin')
    make_urdf(out/'body.urdf', p)
    (out/'manifest.json').write_text(json.dumps(dict(profile=profile, sign=sign,
        dynamics=p, altitude_config=asdict(altitude.config), physics_dt_s=DT,
        control_dt_s=CONTROL_DT, horizon_s=HORIZON,
        scope='Scripted Yaw + live range altitude at 50 Hz; no neural/XY/authority loop',
        sensor='Ideal floor range with capture-aligned attitude; no noise'), indent=2)+'\n')
    if adapter is not None:
        manifest = json.loads((out/'manifest.json').read_text())
        manifest['twr_adapter'] = adapter.contract()
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    plant = BulletPlant(p, out/'body.urdf', dt=DT)
    session = None
    states = [plant.state()]; motors = []; rpms = []; forces = []; commands = []; references = []
    error = None
    try:
        session = Session(out)
        session.arm(); session.pwm[0] = calibration['hover_rc_pwm'] if adapter is None else adapter.hover_rc_pwm; session.pump(250)
        while session.seq % 5: session.step()
        for tick in range(round(HORIZON/DT)):
            old = states[-1]
            if tick % 80 == 0:
                i = tick//80; now = i*CONTROL_DT
                r, pitch = old[6:8]
                sample = AltitudeSample(float(old[2]/(np.cos(r)*np.cos(pitch))
                    -altitude.config.emitter_below_center_m), float(r), float(pitch), now, True)
                throttle = altitude.update(sample, now_s=now, allow_hold=True).throttle_pwm
                reference = np.array([throttle, 1500., 1500., 1500.+yaw_offset(profile, i, sign)])
                command = reference_axes_to_actual(actuator_command(reference, calibration['static_map']['rows'])
                    if adapter is None else adapter.command(reference))
            session.pwm[:] = command
            motor = session.step(rc_new=tick%80==0, duration_us=250)
            plant.step(motor); state = plant.state()
            f = split_thrust(plant.rpm, p['thrust_coeff'])
            states.append(state); motors.append(motor); rpms.append(plant.rpm.copy())
            forces.append([f[k] for k in ('total_thrust_n','mean_rpm_thrust_n','split_thrust_n')])
            commands.append(command.copy()); references.append(reference.copy())
            _, quat = plant.pb.getBasePositionAndOrientation(plant.body, physicsClientId=plant.client)
            rotation = np.array(plant.pb.getMatrixFromQuaternion(quat)).reshape(3,3)
            session.pos, session.vel = state[:3], state[3:6]
            session.q = np.array(quat)[[3,0,1,2]]; session.omega = state[9:12]
            session.force = rotation.T@((state[3:6]-old[3:6])/DT+[0,0,p['g']])
            if not np.isfinite(state).all() or not .25 < state[2] < 3 or abs(state[6:8]).max()>np.radians(45):
                raise RuntimeError('Physical envelope exceeded')
    except Exception as exc:
        error = repr(exc)
    finally:
        if session: session.close()
        plant.close()
    s = np.asarray(states); t = np.arange(len(s))*DT; motor = np.asarray(motors)
    f = np.asarray(forces); ref = np.asarray(references)
    np.savez_compressed(out/'trace.npz', time_s=t, state=s, motors=motor, rpm=rpms,
        forces=f, pwm=commands, reference_pwm=ref, acks=[] if session is None else session.acks)
    n = len(motors)
    result = dict(profile=profile, sign=sign, completed=n==round(HORIZON/DT) and error is None,
        error=error, duration_s=float(t[-1]), max_height_error_m=float(abs(s[:,2]-1).max()),
        final_directed_yaw_deg=float(-sign*np.degrees(np.unwrap(s[:,8])[-1]-s[0,8])),
        max_yaw_rate_deg_s=float(np.degrees(abs(s[:,11])).max()))
    if n:
        rate = np.degrees(s[:,11])
        ref_rate = -(ref[:,3]-1500)*.4
        result.update(commanded_angle_deg=float(-sign*ref_rate.sum()*DT),
            max_thrust_over_weight=float(f[:,0].max()/(p['mass']*p['g'])),
            max_split_thrust_over_weight=float(f[:,2].max()/(p['mass']*p['g'])),
            min_motor_output=float(motor.min()), max_motor_output=float(motor.max()),
            **motor_limit_counts(motor),
            command_settle_from_step_end_s=settled_delay(t[:-1],ref[:,3]-1500,end_s=ENDS['step'],band=20),
            actual_rate_settle_from_step_end_s=settled_delay(t,rate,end_s=ENDS['step'],band=8),
            actual_rate_settle_from_profile_end_s=settled_delay(t,rate,end_s=ENDS[profile],band=8))
    (out/'report.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)
    return result


def run(out, source, reference):
    from evaluate.sitl_lockstep import BIN
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    prior = json.loads((reference/'protocol.json').read_text())
    if digest(BIN) != prior['firmware_sha256']: raise ValueError('Firmware changed')
    hashes = prior['input_source_sha256'].copy()
    if any(digest(k)!=v for k,v in hashes.items()): raise ValueError('Source changed since sensor control study')
    hashes['evaluate/sitl_yaw_profiles.py'] = digest(__file__)
    out.mkdir(parents=True, exist_ok=False)
    budget = {p:sum(yaw_offset(p,i,1) for i in range(300))*.4*.02 for p in ENDS}
    if not all(abs(v-16.8)<1e-10 for v in budget.values()): raise ValueError('Discrete angle budget mismatch')
    protocol = dict(version='yaw-profiles-v1', signs=[-1,1], profiles=list(ENDS),
        start_s=START, ends_s=ENDS, commanded_angle_deg=budget, peak_reference_pwm=140,
        smooth_ramp_s=.24, smooth_plateau_s=.06, physics_dt_s=DT, control_dt_s=CONTROL_DT,
        horizon_s=HORIZON, actual_rate_settle_band_deg_s=8., command_settle_band_pwm=20.,
        recovery_deadline_s=.8, sensor_mode='ideal fresh range/attitude at 50 Hz',
        criteria='Research continuation only: both signs complete; smooth height and peak yaw rate lower; '
                 'directed actual angle differs from paired step by <=10%; actual rate settles within '
                 '.8 s of original STEP end; no motor saturation. Not a physical flight admission.',
        firmware_sha256=prior['firmware_sha256'], input_source_sha256=hashes)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    for name in ('evaluate/sitl_yaw_profiles.py','evaluate/sitl_lockstep.py','evaluate/ready_sim_spike.py'):
        dest=out/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(name,dest)
    results=[];clocks={}
    for sign in (-1,1):
        for profile in ENDS:
            name=f'{profile}_{sign:+d}'
            results.append(trial(out/name,source,profile,sign))
            (out/'report.json').write_text(json.dumps(results,indent=2)+'\n')
            with np.load(out/name/'trace.npz') as a:
                n=len(a['motors']);ack=a['acks'][-n:]
                clocks[name]=bool(n and (np.diff(ack[:,0])==1).all() and (np.diff(ack[:,1])==250).all()
                                  and (ack[:,2]==2).all())
    pairs=[]
    for sign in (-1,1):
        step=next(r for r in results if r['sign']==sign and r['profile']=='step')
        smooth=next(r for r in results if r['sign']==sign and r['profile']=='cosine')
        angle_diff=abs(smooth['final_directed_yaw_deg']-step['final_directed_yaw_deg'])/max(abs(step['final_directed_yaw_deg']),1e-12)
        settle=smooth.get('actual_rate_settle_from_step_end_s')
        passed=bool(step['completed'] and smooth['completed'] and step['final_directed_yaw_deg']>0
            and smooth['final_directed_yaw_deg']>0 and angle_diff<=.1 and settle is not None and settle<=.8
            and smooth['max_height_error_m']<step['max_height_error_m']
            and smooth['max_yaw_rate_deg_s']<step['max_yaw_rate_deg_s']
            and smooth.get('saturated_motor_ticks',1)==0)
        pairs.append(dict(sign=sign,angle_relative_difference=angle_diff,research_continuation_passed=passed))
    validation=dict(sources_unchanged=all(digest(k)==v for k,v in hashes.items()),clocks=clocks,pairs=pairs,
                    deployment_admitted=False)
    (out/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    render(out,results)
    print(json.dumps(validation,indent=2),flush=True)
    if not validation['sources_unchanged'] or not all(clocks.values()) or any(not r['completed'] for r in results):
        raise SystemExit(1)


def render(out, results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
    for row,sign in zip(axes,(-1,1)):
        for profile in ENDS:
            with np.load(out/f'{profile}_{sign:+d}'/'trace.npz') as a:
                t,s=a['time_s'],a['state']
                row[0].plot(t,100*(s[:,2]-1),label=profile)
                row[1].plot(t,-sign*np.degrees(s[:,11]),label=profile)
                row[1].plot(t[:-1],sign*(a['reference_pwm'][:,3]-1500)*.4,ls=':',alpha=.6)
                row[2].plot(t,-sign*np.degrees(np.unwrap(s[:,8])),label=profile)
        for ax,title in zip(row,('Height error (cm)','Directed Yaw rate (deg/s), target dotted','Directed angle (deg)')):
            ax.set(title=f'sign {sign:+d} | '+title,xlabel='Time (s)',xlim=(0,6));ax.grid(alpha=.25);ax.legend()
    fig.suptitle('Same peak and discrete commanded angle | live altitude | unchanged virtual FC/plant\n'
                 'No network/coordinator/transport; ideal sensors; research only')
    fig.savefig(out/'comparison.png',dpi=140);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    a=p.parse_args();run(a.out,a.source,a.reference)
