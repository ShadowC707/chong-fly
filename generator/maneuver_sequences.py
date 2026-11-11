"""Bounded XY maneuver collection; unchanged v5 teacher, no deployment promotion.

Geometry and initial motion vary; targets are recomputed from actual sensor
history, never randomly assigned maneuver labels or world-state oracle actions.
Mirrored scene pairs belong to the same explicit data partition.
"""
import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from artifacts.io import file_sha256, publication, write_json
from configs.flight_config import CONTROL_DT, DEFAULT_DT, COORDINATE_VERSION
from generator.generate_reflex_dataset import make_teacher, _behavior
from generator.reflex_contract import DATASET_VERSION, TEACHER_VERSION, BEHAVIORS, REQUIRED_PATHS
from simulation.drone_env import DroneSimulationEnv, RoomBoundaries, BoxObstacle, CylinderObstacle
from simulation.altitude_control import AltitudeHold, AltitudeTelemetryError, sample_from_observation
from simulation.control_contract import control_contract
from simulation.memory import EgocentricMemoryWrapper
from demo.contacts import step_with_contact, contact_and_clearance

FAMILIES = ('offset_box', 'oblique_wall', 'staggered', 'corner')
MOTIONS = {'slow':(.25,0.), 'fast':(.9,0.), 'lateral':(.35,.22), 'reverse':(-.25,.08)}


def make_plan(split):
    if split not in ('train', 'evaluation'):
        raise ValueError('Unknown split')
    base = 202611040 if split == 'train' else 202611140
    motions = tuple(MOTIONS) if split == 'train' else ('fast','lateral')
    rows = []
    for family in FAMILIES:
        for motion in motions:
            group = base + len(rows)//2
            for side in (1,-1):
                rows.append(dict(seed=base*10+len(rows), scene_seed=group,
                    pair_id=f'{split}-{group}', split=split, family=family,
                    motion=motion, side=side, scenario=f'{family}_{motion}'))
    return rows


def make_environment(record):
    family, motion, side = record['family'], record['motion'], record['side']
    if family not in FAMILIES or motion not in MOTIONS or side not in (-1,1):
        raise ValueError('Invalid maneuver specification')
    rng = np.random.default_rng(record['scene_seed'])
    d, width, offset = rng.uniform(1.1,1.65), rng.uniform(.35,.65), rng.uniform(.12,.28)
    boxes, cylinders = [], []
    angle = 0.
    if family == 'offset_box':
        boxes = [BoxObstacle(d,d+.25,offset-width/2,offset+width/2,0.,2.5)]
    elif family == 'oblique_wall':
        boxes = [BoxObstacle(d,d+.2,-.9,.9,0.,2.5)]
        angle = float(rng.uniform(.15,.30))
    elif family == 'staggered':
        cylinders = [CylinderObstacle(d,.25,.28,2.5), CylinderObstacle(d+1.8,-.35,.32,2.5)]
    else:
        boxes = [BoxObstacle(d,d+.2,-.65,1.1,0.,2.5),
                 BoxObstacle(d-1.0,d+.2,1.1,1.3,0.,2.5)]
        angle = float(rng.uniform(.08,.18))
    if side == -1:
        boxes = [BoxObstacle(b.x_min,b.x_max,-b.y_max,-b.y_min,b.z_min,b.z_max) for b in boxes]
        cylinders = [CylinderObstacle(c.center_x,-c.center_y,c.radius,c.height) for c in cylinders]
    angle *= side
    env = DroneSimulationEnv(engine='standalone', headless=True, dt=DEFAULT_DT,
        room=RoomBoundaries(-6.,6.,-6.,6.,0.,4.), collision_radius_m=.2)
    env.boxes, env.cylinders = boxes, cylinders
    env.reset(initial_pos=np.array([0.,0.,1.]), seed=record['seed'])
    env.physics.quat[:] = env.physics.euler_to_quaternion(0.,0.,angle)
    forward, lateral = MOTIONS[motion]
    body = np.array([forward,side*lateral,0.])
    rotation = env.physics.quaternion_to_rotation_matrix(env.physics.quat)
    env.physics.vel[:] = rotation@body
    obs = env._sample_observation(elapsed_dt=0.)
    env.scene_parameters = dict(room=asdict(env.room),boxes=[asdict(b) for b in boxes],
        cylinders=[asdict(c) for c in cylinders], initial_yaw_rad=angle,
        initial_velocity_mps=env.physics.vel.tolist(), collision_radius_m=.2)
    kind,_ = contact_and_clearance(env.physics.pos,env.physics.pos,env.room,boxes,cylinders,.2)
    if kind is not None:
        env.close(); raise ValueError('Initial body is in contact')
    return env, obs


def collect(plan, steps=600, progress=None, actor=None):
    if type(steps) is not int or steps <= 0 or not plan:
        raise ValueError('Expected nonempty plan and positive steps')
    if len({r['seed'] for r in plan}) != len(plan):
        raise ValueError('Episode seeds must be distinct')
    x = np.zeros((len(plan),steps,74),np.float32)
    y = np.full((len(plan),steps,4),1500.,np.float32)
    applied = y.copy(); valid = np.zeros((len(plan),steps),bool)
    behavior = np.full((len(plan),steps),-1,np.int64)
    position = np.zeros((len(plan),steps,3),np.float32)
    gap = np.zeros((len(plan),steps),np.float32)
    records = []
    for ep, spec in enumerate(plan):
        env, obs = make_environment(spec)
        teacher = make_teacher(TEACHER_VERSION,noise_std_pwm=0)
        if actor is not None: actor.reset_state()
        memory, altitude = EgocentricMemoryWrapper(), AltitudeHold()
        done, ticks, fault, info = False, 0, None, {}
        try:
            for t in range(steps):
                yaw = float(env.physics.quaternion_to_euler(env.physics.quat)[2])
                ring = memory.update(obs[2:66],current_yaw_rad=-yaw)
                features = np.r_[obs,ring].astype(np.float32)
                target = teacher.step(features,dt=CONTROL_DT)
                executed = (target.copy() if actor is None else
                    np.asarray(actor.step_np(features[:2],features[2:66],features[66:],dt=CONTROL_DT),dtype=np.float32))
                try:
                    action = altitude.apply(executed.copy(),sample_from_observation(env.get_isaac_obs()),
                                            now_s=ticks*DEFAULT_DT)
                except AltitudeTelemetryError as exc:
                    if t == 0: raise
                    fault, done = str(exc), True
                    break
                x[ep,t], y[ep,t], applied[ep,t] = features, target, action
                valid[ep,t] = True; behavior[ep,t] = _behavior(features,target)
                position[ep,t] = env.physics.pos
                _,gap[ep,t] = contact_and_clearance(env.physics.pos,env.physics.pos,env.room,
                    env.boxes,env.cylinders,env.collision_radius_m)
                for _ in range(5):
                    obs,_,done,info = step_with_contact(env,action)
                    ticks += 1
                    if done: break
                if done: break
            records.append(dict(**spec,scene_parameters=env.scene_parameters,
                valid_steps=int(valid[ep].sum()),physics_ticks=ticks,duration_s=ticks*DEFAULT_DT,
                terminated=bool(done),crashed=bool(info.get('crashed',False)),
                collision_kind=info.get('collision_kind'),failure_reason=fault,
                final_position_m=env.physics.pos.tolist(),
                final_clearance_m=float(info.get('clearance_m',gap[ep,0]))))
            if progress: progress(records[-1])
        finally:
            env.close()
    names = [r['scenario'] for r in records]
    meta = dict(dataset_version=DATASET_VERSION,teacher_version=TEACHER_VERSION,
        teacher_aggregation_version=teacher.aggregation_version,teacher_parameters=teacher.parameters,
        coordinate_version=COORDINATE_VERSION,tof_max_range_m=3.5,sensor_dim=74,action_dim=4,
        dt=CONTROL_DT,physics_dt=DEFAULT_DT,control_contract=control_contract(),
        required_sensor_motor_paths=deepcopy(REQUIRED_PATHS),behavior_names=list(BEHAVIORS),
        episode_scenarios=names,scenario_counts=dict(Counter(names)),episodes=records,
        behavior_counts=dict(Counter(BEHAVIORS[int(b)] for b in behavior[valid])),
        sensor_geometry_version=env.tof.geometry_version,observation_source='geometric_raycast',
        scene_profile='xy-maneuver-sequences-v1',num_episodes=len(plan),seq_len=steps,
        label_noise_std_pwm=0.,execution_noise_std_pwm=0.,training_admitted=False,
        collection_contract=dict(contact_model='sphere-sweep-v1',body_radius_m=.2,
            execution_actor='teacher' if actor is None else 'provided_policy',
            labels='v5 teacher on actor observation history',
            coordinator=False,sensor_sampling='Fresh observation at 50 Hz; not integrated 50/30/15 Hz replay',
            split_rule='Explicit train/evaluation; keep each pair_id together in any later split',
            failed_episodes='Retained and marked; no automatic inclusion as successful demonstrations'))
    data = dict(X=torch.from_numpy(x),Y=torch.from_numpy(y),applied_actions=torch.from_numpy(applied),
        valid=torch.from_numpy(valid),behavior=torch.from_numpy(behavior),metadata=meta)
    # Visualization-only truth never enters X or the teacher.
    data['diagnostics'] = dict(position=torch.from_numpy(position),clearance=torch.from_numpy(gap))
    return data


def audit(data):
    m = data['metadata']; mismatches = 0; max_error = 0.
    for ep in range(len(data['X'])):
        teacher = make_teacher(m['teacher_version'],**m['teacher_parameters'])
        for t in data['valid'][ep].nonzero().flatten().tolist():
            error=float(np.max(abs(teacher.step(data['X'][ep,t].numpy())-data['Y'][ep,t].numpy())))
            max_error=max(max_error,error); mismatches+=int(error>1e-4)
    if mismatches: raise ValueError('Teacher labels do not replay')
    traces=[]
    for ep,r in enumerate(m['episodes']):
        x=data['X'][ep,data['valid'][ep]].numpy(); y=data['Y'][ep,data['valid'][ep]].numpy()
        delta=y[:,3]-1500
        signs=np.where(delta>50,1,np.where(delta< -50,-1,0))
        turns=np.flatnonzero((signs!=0)&np.r_[True,signs[1:]!=signs[:-1]])
        close=x[:,2:66].min(1)*3.5<.8
        recent=np.zeros(len(x),bool); age=1000
        for t,c in enumerate(close):
            age=0 if c else age+1; recent[t]=0<age<=50
        traces.append(dict(episode=ep,pair_id=r['pair_id'],family=r['family'],motion=r['motion'],
            side=r['side'],completed=not r['terminated'],crashed=r['crashed'],
            duration_s=r['duration_s'],turn_bursts=int(len(turns)),
            right_frames=int((signs>0).sum()),left_frames=int((signs<0).sum()),
            brake_after_clear_frames=int((recent&(y[:,2]<1495)).sum()),
            min_clearance_m=min(float(data['diagnostics']['clearance'][ep,data['valid'][ep]].min()),r['final_clearance_m'])))
    return dict(episodes=traces,valid_frames=int(data['valid'].sum()),label_max_error_pwm=max_error,
        label_mismatch_frames=mismatches,completed=sum(r['completed'] for r in traces),
        crashed=sum(r['crashed'] for r in traces),training_admitted=False)


def prepare(output, actor_bundle=None):
    root=Path(__file__).resolve().parents[1]
    paths=[Path(__file__),root/'generator/generate_reflex_dataset.py',
        root/'generator/continuous_reflex_teacher.py',root/'simulation/drone_env.py',
        root/'demo/contacts.py',root/'simulation/altitude_control.py']
    hashes={str(p.relative_to(root)):file_sha256(p) for p in paths}
    actor = None; actor_hashes = {}
    if actor_bundle is not None:
        from edge.runtime import NumpyPolicy
        actor_bundle=Path(actor_bundle).resolve()
        actor=NumpyPolicy(actor_bundle)
        actor_hashes={str(p):file_sha256(p) for p in
            (actor_bundle/'model.npz',actor_bundle/'metadata.json',root/'edge/runtime.py')}
    plans={s:make_plan(s) for s in ('train','evaluation')}
    with publication(output) as stage:
        write_json(stage/'protocol.json',dict(version='xy-maneuver-sequences-v1',plans=plans,
            steps=600,source_sha256=hashes,actor_sha256=actor_hashes,
            teacher_changed=False,automatic_training=False,
            purpose='Test behavioral coverage before mixing data; no deployment or teacher safety claim.'))
        reports={}
        for split,plan in plans.items():
            data=collect(plan,actor=actor,progress=lambda r:print(split,r['family'],r['motion'],r['side'],
                round(r['duration_s'],3),'crash' if r['crashed'] else 'end',flush=True))
            reports[split]=audit(data)
            torch.save(data,stage/(split+'.pt'))
            write_json(stage/(split+'_report.json'),dict(metadata=data['metadata'],audit=reports[split]))
        if hashes != {str(p.relative_to(root)):file_sha256(p) for p in paths}:
            raise RuntimeError('Collection source changed during run')
        if any(file_sha256(p)!=v for p,v in actor_hashes.items()):
            raise RuntimeError('Actor changed during collection')
        write_json(stage/'manifest.json',dict(reports=reports,source_sha256=hashes,
            actor_sha256=actor_hashes,
            files={s+'.pt':file_sha256(stage/(s+'.pt')) for s in plans},training_admitted=False))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--actor-bundle',type=Path,help='Optional verified frozen NumPy policy for corrective collection')
    torch.set_num_threads(1)
    args=p.parse_args();prepare(args.out,args.actor_bundle)
