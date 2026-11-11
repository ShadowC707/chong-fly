"""Replay a saved RC stream unchanged at 4/2/1 ms to isolate plant coupling.

No policy, coordinator, height feedback or command rescaling during replay.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np


def run(out,source,recording):
    from evaluate.sitl_lockstep import Session,BIN
    from evaluate.ready_sim_spike import BulletPlant
    from simulation.ready_sim_contract import make_urdf
    with np.load(recording/'trace.npz') as a:
        commands=a['pwm'].copy();old_state=a['state'].copy();old_motors=a['motors'].copy()
    if len(commands)%5 or not np.isfinite(commands).all():raise ValueError('Expected full 50Hz command frames')
    expected=json.loads((source/'report.json').read_text())
    if hashlib.sha256(BIN.read_bytes()).hexdigest()!=expected['firmware_sha256']:raise ValueError('Firmware mismatch')
    out.mkdir(parents=True,exist_ok=False)
    p=json.loads(Path('configs/ready_sim_reference_v1.json').read_text())['uncertainty_study']['nominal_dynamics']
    protocol=dict(recording=str(recording),recording_sha256=hashlib.sha256((recording/'trace.npz').read_bytes()).hexdigest(),
        steps_ms=[4,2,1],duration_s=len(commands)*.004,firmware_sha256=expected['firmware_sha256'],
        scope='Open-loop identical RC replay, not a new neural rollout; contacts do not alter replay.',
        source_sha256={str(q):hashlib.sha256(q.read_bytes()).hexdigest() for q in
            map(Path,['evaluate/sitl_command_replay.py','evaluate/sitl_lockstep.py','evaluate/ready_sim_spike.py'])})
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    traces={};results=[]
    for ms in (4,2,1):
        dt=ms/1000;ratio=4//ms;dest=out/f'ms{ms}';dest.mkdir()
        shutil.copy2(source/'template/eeprom.bin',dest/'eeprom.bin');make_urdf(dest/'body.urdf',p)
        plant=BulletPlant(p,dest/'body.urdf',dt=dt);session=Session(dest)
        states=[plant.state()];motors=[]
        try:
            session.arm();session.pwm[0]=expected['hover_rc_pwm'];session.pump(250)
            while session.seq%5:session.step()
            for tick in range(len(commands)*ratio):
                session.pwm[:]=commands[tick//ratio]
                motor=session.step(rc_new=tick%(5*ratio)==0,duration_us=ms*1000)
                old=states[-1];plant.step(motor);s=plant.state();states.append(s);motors.append(motor)
                _,q=plant.pb.getBasePositionAndOrientation(plant.body,physicsClientId=plant.client)
                rotation=np.array(plant.pb.getMatrixFromQuaternion(q)).reshape(3,3)
                session.pos,session.vel=s[:3],s[3:6];session.q=np.array(q)[[3,0,1,2]];session.omega=s[9:12]
                session.force=rotation.T@((s[3:6]-old[3:6])/dt+[0,0,p['g']])
                if not np.isfinite(s).all() or not .1<s[2]<6:raise RuntimeError('Replay envelope exceeded')
        finally:session.close();plant.close()
        state=np.array(states);traces[ms]=state
        np.savez_compressed(dest/'trace.npz',time_s=np.arange(len(state))*dt,state=state,motors=motors,
                            acks=session.acks)
        result=dict(ms=ms,final_position=state[-1,:3].tolist(),max_height_error_m=float(abs(state[:,2]-1).max()))
        if ms==4:result['original_exact']=bool(np.array_equal(old_state,state) and np.array_equal(old_motors,motors))
        results.append(result);print(json.dumps(result),flush=True)
    fine=traces[1]
    for result in results:
        ms=result['ms'];d=traces[ms]-fine[::ms]
        result['max_position_delta_to_1ms_m']=float(np.linalg.norm(d[:,:3],axis=1).max())
        result['max_angle_delta_to_1ms_deg']=float(np.degrees(abs(np.arctan2(np.sin(d[:,6:9]),np.cos(d[:,6:9])))).max())
    (out/'report.json').write_text(json.dumps(results,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(10,9),constrained_layout=True)
    for ms,state in traces.items():
        t=np.arange(len(state))*ms/1000
        axes[0].plot(t,state[:,2],label=f'{ms} ms')
        axes[1].plot(t,np.degrees(state[:,11]),label=f'{ms} ms')
    axes[2].step(np.arange(len(commands))*.004,commands[:,3],where='post')
    for ax,title in zip(axes,('Height (m)','Yaw rate (deg/s)','Identical commanded yaw RC')):
        ax.set(title=title,xlabel='Model time (s)');ax.grid(alpha=.25)
    axes[0].legend();axes[1].legend()
    fig.suptitle('Fixed recorded commands; STOP requested at 17 s in original run')
    fig.savefig(out/'comparison.png',dpi=140);plt.close(fig)
    print(json.dumps(results,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--recording',type=Path,required=True)
    a=p.parse_args();run(a.out,a.source,a.recording)
