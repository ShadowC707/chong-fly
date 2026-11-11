"""Fixed-step flight execution for the interactive and batch SITL front ends."""
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import shutil

import numpy as np

from simulation.sitl_lockstep import infer_hover


CASES=('clear','obstacle_left','obstacle_right','obstacle_center')
DT=.004
HORIZON=20.
STOP_AT=17.
RADIUS=.2


def scene_spec(case):
    if case not in CASES: raise ValueError('Unknown fixed scene')
    boxes=[]
    if case!='clear':
        lo,hi={'obstacle_left':(.05,.8),'obstacle_right':(-.8,-.05),
               'obstacle_center':(-.4,.4)}[case]
        boxes=[dict(x_min=1.5,x_max=1.7,y_min=lo,y_max=hi,z_min=0.,z_max=2.5)]
    return dict(room=dict(x_min=-4.,x_max=4.,y_min=-4.,y_max=4.,z_min=0.,z_max=3.),
                boxes=boxes,cylinders=[],start=[0.,0.,1.],yaw=0.,radius_m=RADIUS)


class SceneSensors:
    """Reuse existing sensor formulas. World state is never passed to the policy."""
    def __init__(self,case,*,spec=None):
        from simulation.drone_env import RoomBoundaries,BoxObstacle,CylinderObstacle,ToFRaycaster
        from simulation.drone_interface import DownwardLaserSensor
        from simulation.pmw3901_emulator import PMW3901FlowSensor
        spec=scene_spec(case) if spec is None else spec
        start=np.asarray(spec['start'],dtype=float)
        if (start.shape!=(3,) or not np.isfinite(start).all() or start[2]!=1.
                or not np.isfinite(spec['yaw']) or spec['radius_m']!=RADIUS):
            raise ValueError('SITL requires a finite start at 1 m and radius .2 m')
        self.room=RoomBoundaries(**spec['room'])
        self.boxes=[BoxObstacle(**b) for b in spec['boxes']]
        self.cylinders=[CylinderObstacle(**c) for c in spec['cylinders']]
        self.tof=ToFRaycaster()
        self.laser=DownwardLaserSensor(noise_std=0.)
        self.flow=PMW3901FlowSensor(noise_std=0.)

    def sample(self,state,time_s):
        from simulation.drone_env import QuadcopterDynamics
        from simulation.altitude_control import AltitudeSample
        rotation=QuadcopterDynamics.quaternion_to_rotation_matrix(
            QuadcopterDynamics.euler_to_quaternion(*state[6:9]))
        laser=self.laser.measure(state[:3],rotation,floor_z=self.room.z_min,
                                 boxes=self.boxes,cylinders=self.cylinders)
        flow=self.flow.compute_flow(state[3:6],rotation,laser.distance,omega_body=state[9:12])
        tof=self.tof.cast_rays(state[:3],rotation,self.room,self.cylinders,self.boxes)
        altitude=AltitudeSample(float(laser.distance),float(state[6]),float(state[7]),time_s,bool(laser.is_valid))
        return flow,tof,altitude


def actuator_command(reference_pwm,rows):
    pwm=np.asarray(reference_pwm,dtype=float).copy()
    if pwm.shape!=(4,) or not np.isfinite(pwm).all() or np.any((pwm<1000)|(pwm>2000)):
        raise ValueError('Invalid reference command')
    pwm[0]=infer_hover(rows,(pwm[0]-1000)/1000)
    return pwm


def validate_duration(horizon_s,stop_at_s):
    if (not np.isfinite([horizon_s,stop_at_s]).all() or horizon_s<=0
            or not 0<=stop_at_s<=horizon_s-1
            or any(abs(v/.02-round(v/.02))>1e-8 for v in (horizon_s,stop_at_s))):
        raise ValueError('Duration/STOP must align to 20 ms; require at least one full STOP second')


@dataclass
class FlightView:
    state: np.ndarray
    system: object
    integrated: object
    manifest: dict
    clearance: float
    tick: int = 0
    time_s: float = 0.
    collision: str | None = None
    control_state: np.ndarray | None = None
    control_gap: float | None = None
    observed: dict | None = None
    decision: dict | None = None
    fc_pwm: tuple | None = None
    error: str | None = None


def flight_steps(out,source,case,backend,axis_contract='native_rc',*,physics_dt_s=DT,integration_profile=None,yaw_shaping=False,
          scene_override=None,tof_probe=None,target_twr=None,calibration_rows=None,
          horizon_s=HORIZON,stop_at_s=STOP_AT,session_id=None,archive=True,
          runtime_config='configs/demo_candidate_a.json'):
    """Yield at control boundaries. Each send advances a complete block or to contact."""
    if backend not in ('sitl','legacy'): raise ValueError('Unknown physics backend')
    from edge.system import RunConfig
    from edge.pinned_system import build_pinned_system
    from demo.controllers import install
    from demo.contacts import contact_and_clearance
    from simulation.sitl_plant import LegacyPlant,BulletPlant
    from simulation.sitl_process import Session
    from simulation.ready_sim_contract import make_urdf,rc_setpoints
    from simulation.avionics_filter import BetaflightCascadedPID,PIDConstants

    dt=physics_dt_s
    if dt not in (.004,.001,.0005,.00025) or (backend=='legacy' and dt!=DT):
        raise ValueError('Neural probe supports 4/1/0.5/0.25 ms SITL and 4 ms legacy')
    rc_stride=round(.02/dt)

    config=RunConfig.from_json(runtime_config)
    if session_id is not None: config=replace(config,session_id=session_id)
    system=install(build_pinned_system(config))
    if type(yaw_shaping) is not bool: raise ValueError('Explicit yaw_shaping flag required')
    if yaw_shaping:
        if integration_profile is None: raise ValueError('Yaw experiment requires authority integration')
        from demo.yaw_experiment import install as install_yaw
        system=install_yaw(system)
    runtime=system.runtime
    integrated=None
    if integration_profile is not None:
        from simulation.sitl_control import IntegratedControl
        integrated=IntegratedControl(system,integration_profile)
    source_report=json.loads((source/'report.json').read_text())
    nominal=json.loads(Path('configs/ready_sim_reference_v1.json').read_text())['uncertainty_study']
    p,fc=nominal['nominal_dynamics'],nominal['fc']
    thrust_adapter=None
    if target_twr is not None:
        if backend!='sitl' or axis_contract!='reference_units' or calibration_rows is None:
            raise ValueError('TWR sensitivity requires SITL reference units and measured calibration')
        from simulation.sitl_twr import ThrustAdapter
        thrust_adapter=ThrustAdapter(p,target_twr,calibration_rows,
            min_pwm=runtime.altitude.config.min_pwm,max_pwm=runtime.altitude.config.max_pwm)
        p=thrust_adapter.dynamics
    elif calibration_rows is not None:
        raise ValueError('Calibration override requires explicit TWR profile')
    def map_throttle(reference):
        return (thrust_adapter.command(reference) if thrust_adapter is not None
                else actuator_command(reference,source_report['static_map']['rows']))
    if axis_contract not in ('native_rc','reference_units'):raise ValueError('Unknown axis contract')
    if axis_contract=='reference_units':
        from simulation.sitl_rc_units import reference_axes_to_actual,validate_virtual_profile
        validate_virtual_profile(Path('configs/sitl_virtual_v1.cli'),p)
    spec=scene_spec(case) if scene_override is None else scene_override
    sensors=SceneSensors(case,spec=spec)
    hit,gap=contact_and_clearance(spec['start'],spec['start'],sensors.room,sensors.boxes,sensors.cylinders,RADIUS)
    if hit: raise ValueError('Initial scene intersects collision shell')
    nondefault_pose=list(spec['start'])!=[0.,0.,1.] or spec['yaw']!=0.
    if nondefault_pose and backend=='legacy':
        raise ValueError('Legacy probe requires origin hover and zero yaw')
    out.mkdir(parents=True,exist_ok=False)
    make_urdf(out/'reference.urdf',p)
    session=None
    controller=BetaflightCascadedPID(dt=DT,angle_p_gain=fc['angle_p_gain'],max_rate_rads=fc['max_rate_rads'],
        roll_pid=PIDConstants(**fc['roll']),pitch_pid=PIDConstants(**fc['pitch']),yaw_pid=PIDConstants(**fc['yaw']))
    manifest=dict(system=system.manifest,scene=spec,backend=backend,
        authority='Direct runtime experiment; FlightBridge, CommandGate and handover not executed',
        sensors='Existing point-rays-solid-v2 ToF, PMW with exact derotation, laser; noise=0; all 50 Hz',
        contact='Existing swept sphere, radius .2 m; terminate at first contact; no impact response',
        dynamics=p,legacy_fc=fc,dt_s=dt,control_dt_s=.02,horizon_s=horizon_s,stop_at_s=stop_at_s,
        altitude_config=asdict(runtime.altitude.config),neural_weights_changed=False,
        axis_contract=axis_contract,
        comparison='Different flight controllers AND rigid-body implementations; not an isolated network ablation')
    if integrated:
        manifest.update(authority='FlightSystem + SessionReceiver; immediate delivery; reference units until gate output',
            sensors='Existing ideal sensor formulas; '+integration_profile+' acquisition; noise=0',
            integration_profile=integration_profile,
            fallback='Simulation range AltitudeHold + neutral R/P/Y at 50 Hz; not physical fallback validation')
    if tof_probe is not None: manifest['diagnostic_tof_override']=tof_probe.contract()
    if thrust_adapter is not None:
        manifest['thrust_adapter']=thrust_adapter.contract()
        manifest['actuator_calibration_rows']=calibration_rows
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    plant=(BulletPlant(p,out/'reference.urdf',dt=dt,start=spec['start'],yaw=spec['yaw'])
           if backend=='sitl' else LegacyPlant(p,out/'reference.urdf'))
    view=FlightView(plant.state(),system,integrated,manifest,gap)
    states=[view.state]; motors=[];commands=[];control=[];decisions=[];gaps=[];receiver_auto=[];receiver_tripped=[]
    result=dict(case=case,backend=backend,completed=False,collision=None,error=None)
    try:
        inputs=yield view
        if backend=='sitl':
            shutil.copy2(source/'template/eeprom.bin',out/'eeprom.bin')
            session=Session(out,archive=archive)
            if nondefault_pose:
                # Feed the same pose during FC warm-up as at the first Bullet step.
                session.pos=view.state[:3]
                _,quat=plant.pb.getBasePositionAndOrientation(plant.body,physicsClientId=plant.client)
                session.q=np.array(quat)[[3,0,1,2]]
            result['firmware_arm']=session.arm()
            session.pwm[0]=(round(thrust_adapter.hover_rc_pwm) if thrust_adapter is not None
                            else source_report['hover_rc_pwm']);session.pump(250)
            while session.seq%5:session.step()
        for tick in range(round(horizon_s/dt)):
            now=tick*dt
            if integrated: now=(tick//rc_stride)*.02+(tick%rc_stride)*dt
            old=view.state
            if tick%rc_stride==0:
                if tick:
                    inputs=yield view
                view.control_state,view.control_gap=old,view.clearance
                view.fc_pwm=None
                stop_requested=inputs.get('stop_requested',stop_at_s is not None and now>=stop_at_s)
                flow,tof,altitude=sensors.sample(old,now)
                if tof_probe is not None:
                    raw_tof=np.array(tof,copy=True)
                    tof=tof_probe.apply(raw_tof,now)
                if integrated:
                    observed=integrated.tick(tick//rc_stride,flow,tof,altitude,stop_requested=stop_requested,
                        request=inputs.get('request','AUTO'),manual_input=inputs.get('manual_input'))
                    view.observed=observed
                    stages=runtime.last_stages.copy()
                    stages['coordinator']=runtime.coordinator.last_decision if stages else {}
                    reference=np.asarray(observed['selected']['applied_pwm'],dtype=float)
                else:
                    stages=runtime.step(flow,tof,altitude,now_s=now,stop_requested=stop_requested)
                    reference=np.asarray(stages['supervised_pwm'],dtype=float)
                command=map_throttle(reference) if backend=='sitl' else reference.copy()
                if backend=='sitl' and axis_contract=='reference_units':command=reference_axes_to_actual(command)
                decision=stages['coordinator']
                decisions.append(dict(time_s=now,phase=decision.get('phased_phase'),
                    event=decision.get('phased_event'),owned=decision.get('phased_owned_channels',[False]*4)))
                if yaw_shaping: decisions[-1]['yaw_smoothing']=decision.get('yaw_smoothing')
                if tof_probe is not None:
                    decisions[-1]['tof_probe']=dict(tof_probe.last,raw_tof=raw_tof.tolist())
                view.decision=decisions[-1]
                if archive: control.append((now,np.array(flow),np.array(tof),
                                np.array(stages.get('network_pwm',[np.nan]*4)),
                                np.array(stages.get('coordinated_pwm',[np.nan]*4)),
                                np.array(stages.get('supervised_pwm',reference)),altitude.range_m))
            if integrated:
                selected=integrated.output(now)
                if archive:
                    receiver_auto.append(selected['source']=='auto');receiver_tripped.append(selected['tripped'])
                if not np.array_equal(reference,selected['applied_pwm']):
                    reference=np.asarray(selected['applied_pwm'],dtype=float)
                    command=map_throttle(reference) if backend=='sitl' else reference.copy()
                    if backend=='sitl' and axis_contract=='reference_units':command=reference_axes_to_actual(command)
            if session:
                session.pwm[:]=command
                motor=session.step(rc_new=tick%rc_stride==0,duration_us=round(dt*1e6))
                if dt==.00025 and session.last_ack['motor_updates']!=2:
                    raise RuntimeError('Expected two firmware motor updates per 250 us')
                if tick%rc_stride==0:
                    view.fc_pwm=tuple(round(x) for x in command)
            else:
                thrust,roll,pitch,yaw=rc_setpoints(command,p)
                motor,_=controller.compute_motor_commands(thrust,pitch,roll,yaw,old[6:9],old[9:12])
            plant.step(motor)
            state=plant.state()
            if archive:
                states.append(state);motors.append(motor);commands.append(command.copy())
            if session:
                _,quat=plant.pb.getBasePositionAndOrientation(plant.body,physicsClientId=plant.client)
                rotation=np.array(plant.pb.getMatrixFromQuaternion(quat)).reshape(3,3)
                session.pos,session.vel=state[:3],state[3:6]
                session.q=np.array(quat)[[3,0,1,2]];session.omega=state[9:12]
                session.force=rotation.T@((state[3:6]-old[3:6])/dt+[0,0,p['g']])
            hit,gap=contact_and_clearance(old[:3],state[:3],sensors.room,sensors.boxes,sensors.cylinders,RADIUS)
            view.tick=tick+1
            view.time_s=(view.tick//rc_stride)*.02+(view.tick%rc_stride)*dt if integrated else view.tick*dt
            view.state,view.collision,view.clearance=state,hit,gap
            if archive: gaps.append(gap)
            if not np.isfinite(state).all():
                raise RuntimeError('Nonfinite physics state')
            if hit:
                result['collision']=hit;break
            if np.max(np.abs(state[6:8]))>np.radians(45):
                raise RuntimeError('Nonfinite state or 45-degree tilt envelope exceeded')
        result['completed']=view.tick==round(horizon_s/dt) and not result['collision']
    except Exception as exc:
        view.error=result['error']=repr(exc)
    finally:
        try:
            if session:session.close()
        finally:
            plant.close()
        if archive:
            s=np.asarray(states); t=np.arange(len(s))*dt
            np.savez_compressed(out/'trace.npz',time_s=t,state=s,motors=motors,pwm=commands,clearance=gaps,
                control_time=[v[0] for v in control],flow=[v[1] for v in control],tof=[v[2] for v in control],
                network_pwm=[v[3] for v in control],coordinated_pwm=[v[4] for v in control],
                supervised_pwm=[v[5] for v in control],laser_range_m=[v[6] for v in control],
                receiver_auto=receiver_auto,receiver_tripped=receiver_tripped,
                acks=[] if not session else session.acks)
            if integrated:
                (out/'command_trace.json').write_text(json.dumps(integrated.rows,indent=2)+'\n')
            (out/'decisions.json').write_text(json.dumps(decisions,indent=2)+'\n')
            last=(t>=horizon_s-1)
            speed=np.linalg.norm(s[:,3:5],axis=1)
            owned=np.array([v['owned'] for v in decisions],dtype=bool)
            result.update(duration_s=float(t[-1]),min_clearance_m=min(gaps) if gaps else None,
                max_height_error_m=float(np.max(np.abs(s[:,2]-1))),
                path_m=float(np.linalg.norm(np.diff(s[:,:2],axis=0),axis=1).sum()),
                final_speed_mps=float(speed[-1]),
                stop_last_second_max_speed_mps=float(speed[last].max()) if last.any() else None,
                stop_passed=bool(result['completed'] and last.any() and (speed[last]<.03).all()),
                coordinator_owned_fraction=float(owned.any(axis=1).mean()) if owned.size else 0.,
                coordinator_owned_by_channel=owned.mean(axis=0).tolist() if owned.size else [0.]*4,
                timeout_events=sum(v['event'] in ('brake_timeout','turn_timeout','episode_timeout') for v in decisions))
            # Phase ownership includes commands that happen to equal the network.
            # Separately count actual changes by the COMPLETE XY coordinator, including STOP.
            changed=np.array([abs(v[3]-v[4])>1e-6 for v in control])
            result['xy_changed_fraction']=float(changed.any(axis=1).mean()) if changed.size else 0.
            if integrated:
                indices=np.flatnonzero(receiver_auto)
                first=float(indices[0]*dt) if len(indices) else None
                result.update(integration_profile=integration_profile,auto_first_s=first,
                    auto_time_s=float(sum(receiver_auto)*dt),receiver_trip_ticks=int(sum(receiver_tripped)),
                    authority_faults=sorted({r['fault'] for r in integrated.rows if r['fault']}),
                    authority_passed=bool(result['completed'] and first is not None and first<=2.
                        and all(receiver_auto[indices[0]:]) and not any(receiver_tripped)),
                    accepted_frames=sum(r['accepted_pwm'] is not None for r in integrated.rows))
            (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def long_scene():
    return dict(room=dict(x_min=-6., x_max=6., y_min=-6., y_max=6., z_min=0., z_max=3.),
        boxes=[dict(x_min=2.2, x_max=2.8, y_min=-.65, y_max=.25, z_min=0., z_max=2.5),
               dict(x_min=-3.2, x_max=-2.4, y_min=1.6, y_max=2.8, z_min=0., z_max=2.5)],
        cylinders=[dict(center_x=2., center_y=3.3, radius=.45, height=2.5),
                   dict(center_x=-1.8, center_y=-2.8, radius=.55, height=2.5)],
        start=[0., 0., 1.], yaw=0., radius_m=.2)
