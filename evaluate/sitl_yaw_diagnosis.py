"""Bounded strong-yaw diagnosis: identical RC replay and rotor thrust accounting."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np

HORIZON=10.
STEPS_US=(1000,500,250)


def split_thrust(rpm,kf):
    rpm=np.asarray(rpm,dtype=float)
    if rpm.shape!=(4,) or not np.isfinite(rpm).all() or np.any(rpm<0) or not np.isfinite(kf) or kf<=0:
        raise ValueError('Expected four nonnegative rotor speeds and positive kf')
    return dict(total_thrust_n=float(kf*np.sum(rpm**2)),
                mean_rpm_thrust_n=float(4*kf*np.mean(rpm)**2),
                split_thrust_n=float(4*kf*np.var(rpm)))


def load_commands(path,horizon):
    with np.load(path,allow_pickle=False) as a:
        t,commands=a['time_s'].copy(),a['pwm'].copy()
    if len(t)<2 or commands.shape!=(len(t)-1,4) or not np.isfinite(commands).all():
        raise ValueError('Malformed recording')
    step_us=round((t[1]-t[0])*1e6)
    if step_us<=0 or not np.allclose(t,np.arange(len(t))*step_us/1e6,rtol=0,atol=1e-10):
        raise ValueError('Expected uniform model time')
    if t[-1]+1e-10<horizon:raise ValueError('Recording does not cover horizon')
    if 20000%step_us:raise ValueError('Recording grid does not divide 50 Hz')
    count=round(horizon*1e6/step_us);commands=commands[:count]
    stride=20000//step_us
    if not np.array_equal(commands,commands[(np.arange(count)//stride)*stride]):
        raise ValueError('Commands change between 50 Hz frames')
    return commands,step_us


def replay(dest,source,commands,record_us,step_us,*,neutral_yaw=False):
    from evaluate.sitl_lockstep import Session
    from evaluate.ready_sim_spike import BulletPlant
    from simulation.ready_sim_contract import make_urdf
    if record_us%step_us:raise ValueError('Physics must divide recording step')
    dt=step_us/1e6
    p=json.loads(Path('configs/ready_sim_reference_v1.json').read_text())['uncertainty_study']['nominal_dynamics']
    previous=json.loads((source/'report.json').read_text())
    dest.mkdir(parents=True,exist_ok=False)
    shutil.copy2(source/'template/eeprom.bin',dest/'eeprom.bin');make_urdf(dest/'body.urdf',p)
    plant=BulletPlant(p,dest/'body.urdf',dt=dt);session=Session(dest)
    states=[plant.state()];motors=[];rotors=[];forces=[];input_log=[]
    error=None
    try:
        session.arm();session.pwm[0]=previous['hover_rc_pwm'];session.pump(250)
        while session.seq%5:session.step()
        for tick in range(round(HORIZON/dt)):
            pwm=commands[(tick*step_us)//record_us].copy()
            if neutral_yaw:pwm[3]=1500.
            session.pwm[:]=pwm
            motor=session.step(rc_new=tick%(20000//step_us)==0,duration_us=step_us)
            old=states[-1];plant.step(motor);state=plant.state()
            thrust=split_thrust(plant.rpm,p['thrust_coeff'])
            cosine=float(np.cos(old[6])*np.cos(old[7]))
            drag_z=-p['drag_linear']*old[5]*np.linalg.norm(old[3:6])
            predicted=(thrust['total_thrust_n']*cosine+drag_z)/p['mass']-p['g']
            observed=(state[5]-old[5])/dt
            states.append(state);motors.append(motor);rotors.append(plant.rpm.copy());input_log.append(pwm)
            forces.append([thrust[k] for k in ('total_thrust_n','mean_rpm_thrust_n','split_thrust_n')]
                          +[cosine,drag_z,predicted,observed])
            _,q=plant.pb.getBasePositionAndOrientation(plant.body,physicsClientId=plant.client)
            rotation=np.array(plant.pb.getMatrixFromQuaternion(q)).reshape(3,3)
            session.pos,session.vel=state[:3],state[3:6];session.q=np.array(q)[[3,0,1,2]];session.omega=state[9:12]
            session.force=rotation.T@((state[3:6]-old[3:6])/dt+[0,0,p['g']])
            if not np.isfinite(state).all() or not .1<state[2]<6:raise RuntimeError('Replay envelope exceeded')
    except Exception as exc:error=repr(exc)
    finally:session.close();plant.close()
    s=np.asarray(states);f=np.asarray(forces);n=len(motors)
    np.savez_compressed(dest/'trace.npz',time_s=np.arange(len(s))*dt,state=s,motors=motors,
        rpm=rotors,pwm=input_log,forces=f,acks=session.acks)
    result=dict(step_us=step_us,neutral_yaw=neutral_yaw,error=error,completed=n==round(HORIZON/dt),
        max_height_error_m=float(abs(s[:,2]-1).max()),max_height_m=float(s[:,2].max()),
        min_height_m=float(s[:,2].min()),
        max_force_balance_error_mps2=float(abs(f[:,5]-f[:,6]).max()) if n else None,
        max_thrust_over_weight=float(f[:,0].max()/(p['mass']*p['g'])) if n else None,
        max_split_thrust_over_weight=float(f[:,2].max()/(p['mass']*p['g'])) if n else None,
        max_mean_rpm_thrust_over_weight=float(f[:,1].max()/(p['mass']*p['g'])) if n else None)
    (dest/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    return result


def summarize(out,recording):
    results=[];traces={}
    for name in ('us1000','us500','us250','us250_neutral_yaw'):
        results.append(json.loads((out/name/'report.json').read_text()))
        with np.load(out/name/'trace.npz') as a:traces[name]={k:a[k] for k in a.files}
    comparisons=[];fine=traces['us250']
    for us in (1000,500):
        a,b=traces[f'us{us}']['state'],fine['state'][::us//250]
        if a.shape!=b.shape:raise ValueError('Incomplete replay horizon')
        delta=a-b
        v=dict(step_us=us,max_position_difference_m=float(np.linalg.norm(delta[:,:3],axis=1).max()),
            max_velocity_difference_mps=float(np.linalg.norm(delta[:,3:6],axis=1).max()),
            max_angle_difference_deg=float(np.degrees(abs(np.arctan2(np.sin(delta[:,6:9]),np.cos(delta[:,6:9])))).max()))
        v['within_previous_tolerances']=v['max_position_difference_m']<=.02 and v['max_velocity_difference_mps']<=.05 and v['max_angle_difference_deg']<=1
        comparisons.append(v)
    with np.load(recording/'trace.npz') as a:
        exact=all(np.array_equal(a[k][:len(traces['us1000'][k])],traces['us1000'][k]) for k in ('state','motors','pwm'))
    clocks={}
    for name,a in traces.items():
        n=len(a['motors']);us=round((a['time_s'][1]-a['time_s'][0])*1e6);ack=a['acks'][-n:]
        clocks[name]=bool((np.diff(ack[:,0])==1).all() and (np.diff(ack[:,1])==us).all() and (ack[:,2]==us//125).all())
    report=dict(scope='Fixed RC replay, not a neural rollout or height-control improvement',
        results=results,comparisons=comparisons,original_1ms_exact=exact,clocks=clocks,
        completed_runs=sum(r['completed'] for r in results),total_runs=len(results),
        all_runs_completed=all(r['completed'] for r in results),
        force_columns=['total_thrust_n','mean_rpm_thrust_n','split_thrust_n','old_cosine','drag_z_n','predicted_az','observed_az'])
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,1,figsize=(11,12),constrained_layout=True)
    for name,a in traces.items():
        t,s=a['time_s'],a['state'];ls='--' if 'neutral' in name else '-'
        label=name+(' (aborted at height limit)' if 'neutral' in name and t[-1]<HORIZON else '')
        axes[0].plot(t,100*(s[:,2]-1),ls,label=label)
        axes[1].plot(t,np.degrees(s[:,11]),ls,label=label)
    dynamics=json.loads((out/'source_snapshot/configs/ready_sim_reference_v1.json').read_text())['uncertainty_study']['nominal_dynamics']
    mass_g=dynamics['mass']*dynamics['g'];t=fine['time_s'][:-1];f=fine['forces']
    for i,label in enumerate(('total','mean RPM contribution','rotor split contribution')):
        axes[2].plot(t,f[:,i]/mass_g,label=label)
    for i,label in ((0,'throttle'),(3,'yaw')):axes[3].plot(t,fine['pwm'][:,i],label=label)
    for ax,title in zip(axes,('Height error (cm)','Yaw rate (deg/s)','Thrust / weight, 250 us replay','Recorded RC')):
        ax.set(title=title,xlabel='Model time (s)');ax.grid(alpha=.25);ax.legend()
    fig.suptitle('Identical recorded commands, except explicitly neutral-yaw control\nNo network/PI feedback during replay')
    fig.savefig(out/'comparison.png',dpi=140);plt.close(fig)
    print(json.dumps(report,indent=2))
    return report


def main():
    from evaluate.sitl_lockstep import BIN
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--recording',type=Path,required=True)
    a=p.parse_args();commands,record_us=load_commands(a.recording/'trace.npz',HORIZON)
    prior=json.loads((a.source/'report.json').read_text())
    if hashlib.sha256(BIN.read_bytes()).hexdigest()!=prior['firmware_sha256']:raise ValueError('Firmware mismatch')
    a.out.mkdir(parents=True,exist_ok=False)
    files=['evaluate/sitl_yaw_diagnosis.py','evaluate/sitl_lockstep.py','evaluate/ready_sim_spike.py',
           'simulation/sitl_contract.py','simulation/sitl_lockstep.py','configs/ready_sim_reference_v1.json']
    hashes={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in files}
    (a.out/'protocol.json').write_text(json.dumps(dict(horizon_s=HORIZON,steps_us=STEPS_US,
        recording=str(a.recording),recording_sha256=hashlib.sha256((a.recording/'trace.npz').read_bytes()).hexdigest(),
        neutral_yaw='Only yaw set to 1500 at 250us; other saved commands unchanged; no feedback.',
        firmware_sha256=prior['firmware_sha256'],source_sha256=hashes),indent=2)+'\n')
    for name in files:
        dest=a.out/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(name,dest)
    for us in STEPS_US:replay(a.out/f'us{us}',a.source,commands,record_us,us)
    replay(a.out/'us250_neutral_yaw',a.source,commands,record_us,250,neutral_yaw=True)
    result=summarize(a.out,a.recording)
    unchanged=all(hashlib.sha256(Path(k).read_bytes()).hexdigest()==v for k,v in hashes.items())
    (a.out/'validation.json').write_text(json.dumps(dict(sources_unchanged=unchanged,
        original_exact=result['original_1ms_exact'],all_clocks_valid=all(result['clocks'].values()),
        completed_runs=result['completed_runs'],all_runs_completed=result['all_runs_completed'],
        deployment_admitted=False),indent=2)+'\n')
    if (not unchanged or not result['original_1ms_exact'] or not all(result['clocks'].values())
            or not result['all_runs_completed']):raise SystemExit(1)


if __name__=='__main__':main()
