"""Read-only diagnosis: teacher fidelity, sensor age and collision-object visibility.

World geometry is used ONLY for hindsight diagnostics, never as policy input.
No controller parameters, weights or live configuration are modified.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def dataset_audit(out):
    import torch
    from edge.runtime import NumpyPolicy, sha256
    torch.set_num_threads(1)
    trained=json.loads((ROOT/'data/selection/mvp_frozen_core_v1/trial_00003/after/report.json').read_text())
    policy=NumpyPolicy(ROOT/'data/edge/k306_tau025_integrated_v1')
    result={'scope':'Open-loop replay of labeled episodes; not a flight comparison.', 'datasets':{}}
    for name in ('train','evaluation'):
        path=ROOT/'data/reflex_training_v5_pilot'/f'{name}.pt'
        data=torch.load(path,map_location='cpu',weights_only=True)
        digest=hashlib.sha256(json.dumps(data['metadata'],sort_keys=True,allow_nan=False).encode())
        for key in ('X','Y','applied_actions','valid','behavior'):
            tensor=data[key]
            digest.update(str((tuple(tensor.shape),tensor.dtype)).encode())
            digest.update(tensor.contiguous().numpy().tobytes())
        if name=='train':
            assert digest.hexdigest()==trained['pretrain']['dataset_sha256'], 'Training data identity differs'
        x,y,valid=(data[k].numpy() for k in ('X','Y','valid'))
        predicted=np.full(y.shape,np.nan,np.float32)
        recent=np.zeros(valid.shape,bool)
        for ep in range(len(x)):
            policy.reset_state();last=-10000
            for t in np.flatnonzero(valid[ep]):
                obs=x[ep,t]
                predicted[ep,t]=policy.step_np(obs[:2],obs[2:66],obs[66:74])
                if obs[2:66].min()*3.5 < .8: last=t
                recent[ep,t]=0<t-last<=50
        distance=x[:,:,2:66].min(axis=2)*3.5
        flow=np.linalg.norm(x[:,:,:2],axis=2)
        masks=dict(all=valid,
            clear_still=valid & (distance>=.95) & (flow<.03),
            clear_fast_forward=valid & (distance>=.95) & (x[:,:,0]>=.12) & (abs(x[:,:,1])<.03),
            recently_cleared=valid & recent & (distance>=.8),
            teacher_braking_after_clear=valid & recent & (distance>=.8) & (y[:,:,2]<1495),
            active_fast=valid & (distance<.8) & (flow>=.12),
            teacher_neutral_yaw_active_fast=valid & (distance<.8) & (flow>=.12) & (abs(y[:,:,3]-1500)<=20))
        splits={'external_evaluation':range(len(x))} if name=='evaluation' else {
            'fitted_episodes':trained['pretrain']['subset_indices'],
            'internal_validation':trained['pretrain']['validation_indices']}
        reports={}
        for split,indices in splits.items():
            reports[split]={}
            select=np.zeros(valid.shape,bool);select[list(indices)]=True
            for label,mask in masks.items():
                mask=mask & select;n=int(mask.sum())
                reports[split][label]=dict(count=n,episodes=int(mask.any(axis=1).sum()),
                    mae_roll_pitch_yaw=np.abs(predicted[mask]-y[mask]).mean(axis=0)[1:].tolist() if n else None,
                    teacher_pitch_mean=float(y[mask,2].mean()) if n else None,
                    network_pitch_mean=float(predicted[mask,2].mean()) if n else None,
                    network_cruise_over_1510=int((predicted[mask,2]>1510).sum()),
                    network_yaw_over_50=int((abs(predicted[mask,3]-1500)>50).sum()))
        result['datasets'][name]=dict(file_sha256=sha256(path),semantic_sha256=digest.hexdigest(),
                                      teacher_parameters=data['metadata']['teacher_parameters'],reports=reports)
    (out/'dataset_fidelity.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


def box_ranges(origin,directions,box):
    """Positive first-hit ranges; infinity for misses, including parallel misses."""
    lo=np.array([box.x_min,box.y_min,box.z_min]);hi=np.array([box.x_max,box.y_max,box.z_max])
    parallel=abs(directions)<=1e-6
    inv=1./np.where(parallel,1.,directions)
    t0,t1=(lo-origin)*inv,(hi-origin)*inv
    near=np.where(parallel,-np.inf,np.minimum(t0,t1)).max(axis=1)
    far=np.where(parallel,np.inf,np.maximum(t0,t1)).min(axis=1)
    outside=(parallel & ((origin<lo)|(origin>hi))).any(axis=1)
    return np.where(~outside & (near<=far) & (far>0) & (near>0),near,np.inf)


def nearest_point(pos,kind,obj):
    if kind=='box':
        return np.clip(pos,[obj.x_min,obj.y_min,obj.z_min],[obj.x_max,obj.y_max,obj.z_max])
    if kind=='wall':
        point=np.array(pos,copy=True);point[obj[0]]=obj[1];return point
    radial=np.asarray(pos[:2])-[obj.center_x,obj.center_y]
    length=float(np.linalg.norm(radial))
    point=np.array(pos,copy=True)
    if length>obj.radius: point[:2]=[obj.center_x,obj.center_y]+radial*obj.radius/length
    point[2]=np.clip(pos[2],0.,obj.height)
    return point


def nearest_surface(pos,room,boxes,cylinders):
    candidates=[('box',i,b) for i,b in enumerate(boxes)]+[('cylinder',i,c) for i,c in enumerate(cylinders)]
    walls=[(0,room.x_min),(0,room.x_max),(1,room.y_min),(1,room.y_max),(2,room.z_min),(2,room.z_max)]
    candidates += [('wall',i,w) for i,w in enumerate(walls)]
    return min(candidates,key=lambda item:np.linalg.norm(nearest_point(pos,item[0],item[2])-pos))


def trace_diagnosis(data,name,out):
    from simulation.drone_env import ToFRaycaster, QuadcopterDynamics,RoomBoundaries,BoxObstacle,CylinderObstacle
    from generator.generate_reflex_dataset import make_teacher
    g=data['manifest']['geometry'];frames=data['frames']
    room=RoomBoundaries(**g['room']);boxes=[BoxObstacle(**b) for b in g['boxes']]
    cylinders=[CylinderObstacle(**c) for c in g['cylinders']]
    event_index=int(np.argmin([r['clearance_m'] for r in frames]))
    event=frames[event_index];event_pos=np.asarray(event['position_m'])
    kind,target_index,target=nearest_surface(event_pos,room,boxes,cylinders)
    caster=ToFRaycaster()
    meta=json.loads((ROOT/'data/reflex_training_v5_pilot/train_report.json').read_text())['metadata']
    teacher=make_teacher(meta['teacher_version'],**meta['teacher_parameters'])
    last_tick=None;last_id=None;seen=None;seen_threat=None;captured_hits=0;capture_error=[];rows=[]
    for row in frames:
        if row['time_s']>event['time_s']+1e-8:break
        tick=row['trace'].get('tick_time_s')
        if tick==last_tick:continue
        last_tick=tick
        if row['trace'].get('network_pwm') is None:
            teacher.reset();continue
        pos=np.asarray(row['position_m']);angles=row['attitude_rad']
        rotation=QuadcopterDynamics.quaternion_to_rotation_matrix(QuadcopterDynamics.euler_to_quaternion(*angles))
        fresh=caster.cast_rays(pos,rotation,room,cylinders,boxes)*3.5
        measured=np.asarray(row['sensors']['tof']['value'])*3.5
        directions=caster.body_ray_dirs@rotation.T
        if kind=='box':
            ranges=box_ranges(pos,directions,target)
        elif kind=='wall':
            axis,coordinate=target
            delta=directions[:,axis]
            ranges=(coordinate-pos[axis])/np.where(abs(delta)>1e-6,delta,1.)
            ranges=np.where((abs(delta)>1e-6)&(ranges>0),ranges,np.inf)
        else:
            ranges=caster.cast_rays(pos,rotation,RoomBoundaries(-1e6,1e6,-1e6,1e6,-1e6,1e6),[target],[])*3.5
            ranges=np.where(ranges<caster.max_range-1e-5,ranges,np.inf)
        visible=(ranges<caster.max_range-1e-5)&(abs(ranges-fresh)<1e-4)
        sample=row['sensors']['tof']
        if sample['sample_id']!=last_id:
            last_id=sample['sample_id'];captured_hits=int(visible.sum())
            capture_error.append(float(abs(fresh-measured).max()))
            if visible.any():seen=row['time_s']
            if np.any(visible & (ranges<.8)):seen_threat=row['time_s']
        q=nearest_point(pos,kind,target)
        relative=rotation.T@(q-pos)
        velocity=np.asarray(row['velocity_mps']);vbody=rotation.T@velocity
        target_pwm=teacher.step(np.r_[row['flow_xy'],measured/3.5,np.ones(8)])
        applied=row.get('plant_input_pwm') or [np.nan]*4
        rows.append(dict(time_s=row['time_s'],position_m=pos.tolist(),yaw_deg=float(np.degrees(angles[2])),
            speed_mps=float(np.linalg.norm(velocity[:2])),forward_mps=float(vbody[0]),
            clearance_m=row['clearance_m'],target_gap_m=float(np.linalg.norm(q-pos)-g['collision_radius_m']),
            target_bearing_deg=float(np.degrees(np.arctan2(relative[1],relative[0]))),
            fresh_tof_min_m=float(fresh.min()),held_tof_min_m=float(measured.min()),
            tof_age_s=row['time_s']-sample['timestamp_s'],visible_target_rays=int(visible.sum()),
            captured_target_rays=captured_hits,last_seen_target_s=seen,last_seen_threat_s=seen_threat,
            network_pitch=row['trace']['network_pwm'][2],applied_pitch=applied[2],
            network_yaw=row['trace']['network_pwm'][3],applied_yaw=applied[3],
            teacher_pitch=float(target_pwm[2]),teacher_yaw=float(target_pwm[3]),
            auto=row['plant_source']=='auto',flow_xy=row['flow_xy']))
    selected=[r for r in rows if event['time_s']-3<=r['time_s']<=event['time_s']]
    summary=dict(name=name,event_time_s=event['time_s'],event_clearance_m=event['clearance_m'],
        event_collision=event.get('collision'),target_kind=kind,target_index=target_index,
        target_geometry=(g['boxes'][target_index] if kind=='box' else
                         g['cylinders'][target_index] if kind=='cylinder' else list(target)),
        capture_reconstruction_max_error_m=max(capture_error,default=0),
        last_control=rows[-1],last_3s_clear_held_but_threat_fresh_s=.02*sum(r['held_tof_min_m']>=.8 and r['fresh_tof_min_m']<.8 for r in selected),
        last_3s_close_gap_under_10cm_and_cruise_s=.02*sum(r['clearance_m']<.1 and r['applied_pitch']>1510 and r['auto'] for r in selected),
        last_3s_neural_cruise_while_teacher_brakes_s=.02*sum(r['network_pitch']>1510 and r['teacher_pitch']<1495 for r in selected))
    (out/(name+'_trace.json')).write_text(json.dumps(rows,indent=2),encoding='utf-8')
    return summary


def rebuilt_variant(case,variant):
    from demo.runner import DemoRunner
    from demo.recording import plain
    from evaluate.compare_maneuver_repair import install_experiment
    folder=ROOT/'output/maneuver_repair_v1'/(case+'_'+variant)
    m=json.loads((folder/'manifest.json').read_text());cfg=m['system']['run_config']
    r=DemoRunner(scene=m['scene'],seed=m['seed'],start=m['start']['position_m'],yaw=m['start']['yaw_rad'],
        scene_options=m['scene_options'],collision_radius_m=.2,max_duration_s=45,
        run_id='visibility-'+case+'-'+variant)
    r.system=install_experiment(r.system,variant);r._manifest=plain(r._build_manifest())
    r.resume();r.event(dict(type='mode',mode='AUTO'));r.advance(round(45/.004))
    data=r.recording();r.env.close()
    # Verify actual executed history against the previous fixed experiment.
    unique=[];last=None
    for row in data['frames']:
        tick=row['trace'].get('tick_time_s')
        if tick!=last:unique.append(row);last=tick
    with np.load(folder/'trace.npz') as a:
        assert np.array_equal(a['pos'],[row['position_m'] for row in unique])
        assert np.array_equal(a['network'],[row['trace'].get('network_pwm') or [np.nan]*4 for row in unique],equal_nan=True)
    return data


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'protocol.json').write_text(json.dumps(dict(
        scope='Diagnosis only; no controller changes or parameter search',
        original_cases=['068fe108','89d81a9f','9d5431c1','b0a9f212'],
        repair_cases=[['b0a9f212','both'],['9d5431c1','release_only'],['89d81a9f','both']],
        hypotheses=['fresh vs held ToF','previous visibility of collision object','neural vs teacher braking after loss of sight'],
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2),encoding='utf-8')
    dataset_audit(args.out);print('Dataset fidelity complete',flush=True)
    summaries=[]
    for case in ('068fe108','89d81a9f','9d5431c1','b0a9f212'):
        path=next((ROOT/'flight simulations/changed contract').glob('*'+case+'*.json'))
        data=json.loads(path.read_text(encoding='utf-8'))
        summaries.append(trace_diagnosis(data,case+'_original',args.out))
        del data
        print('Original diagnosis:',case,flush=True)
    for case,variant in (('b0a9f212','both'),('9d5431c1','release_only'),('89d81a9f','both')):
        data=rebuilt_variant(case,variant)
        summaries.append(trace_diagnosis(data,case+'_'+variant,args.out));del data
        print('Replayed variant diagnosis:',case,variant,flush=True)
    (args.out/'diagnosis.json').write_text(json.dumps(summaries,indent=2),encoding='utf-8')


if __name__=='__main__': main()
