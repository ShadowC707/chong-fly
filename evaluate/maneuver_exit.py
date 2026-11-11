"""Fixed 2x2 teacher-channel intervention before the new lateral contact."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
from evaluate.practical_maneuver import ObservedPolicy

ROOT=Path(__file__).resolve().parents[1]
ARMS={'control':(), 'teacher_pitch':(2,), 'teacher_yaw':(3,), 'teacher_xy':(2,3)}
SWITCH_S=21.


class ExitIntervention(ObservedPolicy):
    def __init__(self,policies,teacher,channels):
        if channels not in ARMS.values():raise ValueError('Only pitch/yaw diagnostic interventions supported')
        self.channels=channels;self.intervene=False
        super().__init__(policies,teacher,'candidate')

    def step_np(self,*args,**kwargs):
        pwm=super().step_np(*args,**kwargs)
        if self.intervene:
            pwm[list(self.channels)]=self.records[-1]['outputs']['teacher'][list(self.channels)]
        self.records[-1]['driver_pwm']=pwm.copy()
        return pwm


def post_switch_metrics(time,pos,clearance,*,switch_s):
    mask=time>=switch_s
    if not mask.any():raise ValueError('No post-switch trajectory')
    xy=pos[mask,:2]
    return dict(min_clearance_m=float(clearance[mask].min()),
        path_m=float(np.linalg.norm(np.diff(xy,axis=0),axis=1).sum()),
        net_displacement_m=float(np.linalg.norm(xy[-1]-xy[0])),xy_span_m=np.ptp(xy,axis=0).tolist())


def run(out):
    from demo.runner import DemoRunner
    from demo.recording import plain
    from edge.system import FlightSystem
    from edge.runtime import sha256
    from generator.generate_reflex_dataset import make_teacher
    from optimizer.continue_flow_reflex import write
    from evaluate.compare_maneuver_repair import summarize
    from simulation.drone_env import RoomBoundaries
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    source=ROOT/'output/bilateral_turn_v1/magnitude'
    original=source/'flights/9d5431c1'
    m=json.loads((original/'manifest.json').read_text())
    with np.load(original/'physical.npz') as a:reference={k:a[k].copy() for k in a.files}
    files=[original/'manifest.json',original/'physical.npz',source/'checkpoint.pt',source/'runtime_config.json',
        *[p for p in (source/'numpy').iterdir() if p.is_file()],ROOT/'configs/mvp_runtime_v3.json',
        *[p for p in (ROOT/'data/edge/k306_tau025_integrated_v1').iterdir() if p.is_file()]]
    hashes={str(p):sha256(p) for p in files}
    protocol=dict(version='maneuver-exit-channel-pair-v1',case='9d5431c1',switch_s=SWITCH_S,horizon_s=45.,
        input_sha256=hashes,script_sha256=sha256(__file__),arms=ARMS,
        intervention='From 21s replace specified raw network channels with continuously observed v5/1530 teacher channels; coordinator/altitude/receiver unchanged.',
        selection='One known collision; switch at 21s before the final turn and cruise resumption, fixed before running; no time search.',
        geometry_diagnostic='Every 20ms before physics step, isolated raycast against original contact box index 6. Fresh rays, not held sensor samples; geometry never enters driver.',
        limitations='Teacher Pitch changes braking magnitude and release timing, not timing alone. Teacher Yaw changes sign, magnitude and duration. No independent generalization claim.',
        no_training=True,no_promotion=True)
    write(out/'protocol.json',protocol)
    results=[]
    for arm,channels in ARMS.items():
        r=DemoRunner(config=source/'runtime_config.json',scene=m['scene'],seed=m['seed'],
            start=m['start']['position_m'],yaw=m['start']['yaw_rad'],scene_options=m.get('scene_options'),
            collision_radius_m=.2,max_duration_s=45.,contact_model='sphere-sweep-v1',run_id=arm+'-exit-diagnosis')
        try:
            for key in ('room','boxes','cylinders'):assert r.manifest()['geometry'][key]==m['geometry'][key]
            p=ExitIntervention(dict(candidate=r.system.runtime.policy),
                make_teacher('geometry-reflex-v5',noise_std_pwm=0,pwm_cruise_pitch=1530.),channels)
            r.system.runtime.policy=p
            manifest=deepcopy(r.system.manifest)
            manifest['experiment']=dict(protocol,arm=arm,
                trace_note='network_pwm denotes substituted driver after switch; raw neural output is separately saved in decisions.npz')
            r.system=FlightSystem(r.system.config,r.system.runtime,manifest)
            r._manifest=plain(r._build_manifest())
            dest=out/arm;dest.mkdir();write(dest/'manifest.json',r.manifest())
            geometry=[];box=r.env.boxes[6]
            lo=np.array([box.x_min,box.y_min,box.z_min]);hi=np.array([box.x_max,box.y_max,box.z_max])
            r.resume();r.event(dict(type='mode',mode='AUTO'))
            for _ in range(11250):
                p.intervene=r.step_index>=round(SWITCH_S/.004)
                n=len(p.records);now=r.time_s
                if r.step_index%5==0:
                    q=r.env.physics.pos.copy();rot=r.env.physics.quaternion_to_rotation_matrix(r.env.physics.quat)
                    offset=np.clip(q,lo,hi)-q;body=rot.T@offset
                    isolated=r.env.tof.cast_rays(q,rot,RoomBoundaries(-100,100,-100,100,-100,100),boxes=[box],cylinders=[])
                    geometry.append(dict(time_s=now,box_gap_m=float(np.linalg.norm(offset)-.2),
                        nearest_body_bearing_deg=float(np.degrees(np.arctan2(body[1],body[0]))),
                        visible_rays=int(np.count_nonzero(isolated<1.)),min_box_ray_m=float(isolated.min()*3.5)))
                r.advance()
                if len(p.records)>n:p.records[-1]['time_s']=now
                if r.terminal:break
            time=np.array([row['time_s'] for row in r.rows]);pos=np.array([row['position_m'] for row in r.rows])
            vel=np.array([row['velocity_mps'] for row in r.rows]);gap=np.array([row['clearance_m'] for row in r.rows])
            prefix=time<=SWITCH_S
            old_prefix=reference['time']<=SWITCH_S
            np.testing.assert_array_equal(time[prefix],reference['time'][old_prefix])
            np.testing.assert_array_equal(pos[prefix],reference['pos'][old_prefix])
            np.testing.assert_array_equal(vel[prefix],reference['velocity'][old_prefix])
            if arm=='control':
                np.testing.assert_array_equal(time,reference['time'])
                np.testing.assert_array_equal(pos,reference['pos'])
                np.testing.assert_array_equal(vel,reference['velocity'])
            yaw=np.array([row['attitude_rad'][2] for row in r.rows]);auto=np.array([row['plant_source']=='auto' for row in r.rows])
            np.savez_compressed(dest/'physical.npz',time=time,pos=pos,velocity=vel,yaw=yaw,auto=auto,clearance=gap)
            controls=r.rows[::5]
            np.savez_compressed(dest/'trace.npz',time=[row['time_s'] for row in controls],
                driver=[row['trace'].get('network_pwm') or [np.nan]*4 for row in controls],
                coordinated=[row['trace'].get('coordinated_pwm') or [np.nan]*4 for row in controls],
                applied=[row.get('plant_input_pwm') or [np.nan]*4 for row in controls],
                min_tof=[row['min_tof_m'] for row in controls],
                tof_timestamp=[row['sensors']['tof']['timestamp_s'] for row in controls])
            np.savez_compressed(dest/'decisions.npz',time=[row['time_s'] for row in p.records],
                X=[row['X'] for row in p.records],segment=[row['segment'] for row in p.records],
                phase=[row['phase'] for row in p.records],network=[row['outputs']['candidate'] for row in p.records],
                teacher=[row['outputs']['teacher'] for row in p.records],driver=[row['driver_pwm'] for row in p.records])
            write(dest/'contact_box_visibility.json',geometry)
            after=post_switch_metrics(time,pos,gap,switch_s=SWITCH_S)
            dt=np.diff(time,append=r.time_s);speed=np.linalg.norm(vel[:,:2],axis=1)
            after['moving_s']=float(dt[(time>=SWITCH_S)&auto&(speed>.03)].sum())
            after['stationary_s']=float(dt[(time>=SWITCH_S)&auto&(speed<=.03)].sum())
            row=dict(arm=arm,metrics=summarize(r.rows,r.time_s),post_switch=after,
                prefix_exact=True,control_full_exact=arm=='control')
            write(dest/'report.json',row);results.append(row);write(out/'report.json',results)
            print(arm,row,flush=True)
        finally:r.env.close()
    assert all(sha256(Path(p))==h for p,h in hashes.items())
    write(out/'validation.json',dict(input_and_active_bundle_unchanged=True,common_prefix_exact=True,
        control_reproduced_exactly=True,no_promotion=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    run(parser.parse_args().out)
