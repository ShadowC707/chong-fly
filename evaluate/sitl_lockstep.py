"""Bounded single-thread-clock SITL experiment; no neural policy or hardware."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import time

import numpy as np

from simulation.sitl_lockstep import infer_hover

from simulation.sitl_process import BIN, Session
DT=.004


def configure(out):
    if (out/'eeprom.bin').exists():raise ValueError('Use fresh output directory')
    s=Session(out); transcript=[]
    try:
        s.msp(1); transcript.append(s.cli('#'))
        for line in Path('/work/configs/sitl_virtual_v1.cli').read_text().splitlines():
            if not line or line.startswith('#'):continue
            response=s.cli(line);transcript.append(response)
            if 'ERROR' in response or 'INVALID' in response:raise RuntimeError(response)
        transcript.append(s.cli('diff all'))
        transcript.append(s.cli('save'))
    finally:
        (out/'configuration.txt').write_text('\n'.join(transcript));s.close()
    # Calibration is retained only in the template, then copied per trial.
    s=Session(out)
    try:
        s.pump(1000);s.msp(205);s.pump(1500)
        s.msp(250) # EEPROM_WRITE, virtual board only.
        s.pump(250)
    finally:s.close()


def static_map(out,template):
    out.mkdir(parents=True,exist_ok=False)
    shutil.copy2(template/'eeprom.bin',out/'eeprom.bin')
    s=Session(out);rows=[]
    try:
        armed=s.arm()
        for pwm in (1150,1180,1200,1210,1220,1250,1300):
            s.pwm[0]=pwm
            s.pump(250)
            samples=np.array(s.pump(125))
            rows.append({'rc_pwm':pwm,'mean_motor':float(samples.mean()),
                'motor_range':[float(samples.min()),float(samples.max())]})
        # Confirm duplicate requests are cached and do not advance firmware.
        s.udp.send(s.last_frame); duplicate=s.udp.recv(1024)
        duplicate_ok=duplicate==s.last_raw
        if not duplicate_ok:raise ValueError('Duplicate altered state/response')
        # A same-sequence request with changed contents must be rejected,
        # and a missing sequence must not advance the simulated clock.
        rejected=[]
        for offset,value,fmt in ((164,1501,'<H'),(4,s.seq+2,'<Q')):
            invalid=bytearray(s.last_frame);struct.pack_into(fmt,invalid,offset,value)
            s.udp.send(invalid);reply=s.udp.recv(1024)
            _,clock,_,status=struct.unpack_from('<QQII',reply)
            rejected.append(status==1 and clock==s.clock)
        if not all(rejected):raise ValueError('Rejected sequence advanced the clock')
        s.step()
        p=json.loads(Path('/work/configs/ready_sim_reference_v1.json').read_text())['uncertainty_study']['nominal_dynamics']
        target=np.sqrt(p['mass']*p['g']/(4*p['thrust_coeff']))/p['max_rpm']
        hover=infer_hover(rows,target)
        result={'rows':rows,'target_normalized_rpm':float(target),'inferred_hover_rc_pwm':hover,
            'actuator_assumption':'normalized motor output linearly maps to RPM; not hardware calibration',
            'duplicate_ack_identical':duplicate_ok,'conflict_and_gap_rejected':rejected,'armed':armed}
        # Integer RC, verify the two bracketing values; no adaptive PID tuning.
        result['hover_bracket']=[]
        for pwm in (int(np.floor(hover)),int(np.ceil(hover))):
            s.pwm[0]=pwm;s.pump(250)
            values=np.array(s.pump(125))
            result['hover_bracket'].append({'rc_pwm':pwm,'mean_motor':float(values.mean())})
        (out/'report.json').write_text(json.dumps(result,indent=2))
        return result
    finally:
        np.savez_compressed(out/'acks.npz',acks=s.acks,wall_ms=s.wall_ms);s.close()


def physics(out,template,hover,case,wall_pause=False,altitude_map=None,*,physics_dt_s=DT):
    from evaluate.ready_sim_spike import BulletPlant
    from simulation.ready_sim_contract import make_urdf
    if physics_dt_s not in (.004,.002,.001): raise ValueError('Unsupported physics step')
    dt=physics_dt_s; rc_stride=round(.02/dt); ticks=round(4/dt)
    out.mkdir(parents=True,exist_ok=False);shutil.copy2(template/'eeprom.bin',out/'eeprom.bin')
    p=json.loads(Path('/work/configs/ready_sim_reference_v1.json').read_text())['uncertainty_study']['nominal_dynamics']
    make_urdf(out/'reference.urdf',p)
    plant=BulletPlant(p,out/'reference.urdf',dt=dt);s=Session(out)
    altitude=None
    if altitude_map is not None:
        from simulation.altitude_control import AltitudeConfig,AltitudeHold,AltitudeSample
        # Existing controller operates in its documented reference RPM² space.
        # The measured inverse map then converts that demand into firmware RC.
        altitude=AltitudeHold(AltitudeConfig(hover_pwm=1000+1000*altitude_map['target_normalized_rpm']))
    states=[];motors=[];commands=[]
    result={'case':case,'wall_pause':wall_pause,'hover_rc_pwm':hover,'completed':False,
            'altitude_hold':altitude is not None,'network_loaded':False,'physics_dt_s':dt,
            'altitude_sensor':'ideal floor range and matching current attitude at 50Hz' if altitude else None}
    try:
        result['armed']=s.arm()
        s.pwm[0]=hover;s.pump(250)
        # Keep live model time and release exactly on the 50Hz RC boundary.
        while s.seq%5:s.step()
        states.append(plant.state())
        start_clock=s.clock
        for tick in range(ticks):
            s.pwm[:]=[hover,1500,1500,1500]
            if altitude is not None:
                if tick%rc_stride==0:
                    state=states[-1];r,pitch=state[6:8]
                    cosine=np.cos(r)*np.cos(pitch)
                    sample=AltitudeSample(float(state[2]/cosine-altitude.config.emitter_below_center_m),
                                          float(r),float(pitch),tick*dt,True)
                    demand=altitude.update(sample,now_s=tick*dt).throttle_pwm
                    altitude_rc=infer_hover(altitude_map['rows'],(demand-1000)/1000)
                s.pwm[0]=altitude_rc
            if round(1/dt)<=tick<round(1.5/dt) and case!='hover':s.pwm[{'roll':1,'pitch':2,'yaw':3}[case]]+=50
            if wall_pause and tick%round(.5/dt)==0:time.sleep(.03)
            motor=s.step(rc_new=tick%rc_stride==0,duration_us=round(dt*1e6))
            old=states[-1]
            plant.step(motor)
            state=plant.state()
            _,qxyzw=plant.pb.getBasePositionAndOrientation(plant.body,physicsClientId=plant.client)
            rotation=np.array(plant.pb.getMatrixFromQuaternion(qxyzw)).reshape(3,3)
            s.pos,s.vel,s.q,s.omega=state[:3],state[3:6],np.array(qxyzw)[[3,0,1,2]],state[9:12]
            s.force=rotation.T@((state[3:6]-old[3:6])/dt+[0,0,p['g']])
            states.append(state);motors.append(motor);commands.append(s.pwm.copy())
            if not np.isfinite(state).all() or not .25<state[2]<3 or np.max(np.abs(state[6:8]))>np.radians(45):
                raise RuntimeError('Physical envelope exceeded')
        result.update(completed=True,firmware_elapsed_us=s.clock-start_clock)
    except Exception as exc:
        result['error']=repr(exc)
    finally:
        s.close();plant.close()
        np.savez_compressed(out/'trace.npz',time_s=np.arange(len(states))*dt,state=states,motors=motors,pwm=commands,
                            acks=s.acks,wall_ms=s.wall_ms)
        if states:
            a=np.array(states)
            result.update(final_position_m=a[-1,:3].tolist(),final_rpy_deg=np.degrees(a[-1,6:9]).tolist(),
                max_abs_rpy_deg=np.degrees(np.max(np.abs(a[:,6:9]),axis=0)).tolist(),
                max_height_error_m=float(np.max(np.abs(a[:,2]-1))))
        (out/'report.json').write_text(json.dumps(result,indent=2))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('/results/lockstep_v1'))
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    configure(args.out/'template')
    mapping=static_map(args.out/'static_map',args.out/'template')
    # Choose nearest integer from a measured bracket, not a behavioral search.
    chosen=min(mapping['hover_bracket'],key=lambda x:abs(x['mean_motor']-mapping['target_normalized_rpm']))['rc_pwm']
    results=[]
    for repeat in range(3):
        for case in ('hover','roll','pitch','yaw'):
            results.append(physics(args.out/f'r{repeat}_{case}',args.out/'template',chosen,case,repeat==2))
    altitude_results=[]
    for repeat in range(2):
        for case in ('hover','roll','pitch','yaw'):
            altitude_results.append(physics(args.out/f'alt{repeat}_{case}',args.out/'template',chosen,case,repeat==1,altitude_map=mapping))
    summary={'static_map':mapping,'hover_rc_pwm':chosen,'results':results,
        'completed':sum(r['completed'] for r in results),'runs':len(results),
        'altitude_results':altitude_results,'altitude_completed':sum(r['completed'] for r in altitude_results),
        'firmware_sha256':hashlib.sha256(BIN.read_bytes()).hexdigest(),
        'harness_patch_sha256':hashlib.sha256(Path('/opt/betaflight/chong-lockstep.patch').read_bytes()).hexdigest()}
    shutil.copy2('/opt/betaflight/chong-lockstep.patch',args.out/'firmware.patch')
    (args.out/'report.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
    if summary['completed']!=summary['runs'] or summary['altitude_completed']!=len(altitude_results):raise SystemExit(1)


if __name__=='__main__':main()
