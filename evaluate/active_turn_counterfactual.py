"""Post-hoc 2x2 intervention at t=34s of one recorded collision, no deployment."""
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


class YawIntervention(ObservedPolicy):
    intervene=False

    def step_np(self,*args,**kwargs):
        pwm=super().step_np(*args,**kwargs)
        if self.intervene: pwm[3]=self.records[-1]['outputs']['teacher'][3]
        self.records[-1]['driver_pwm']=pwm.copy()
        return pwm


def run(out):
    from demo.runner import DemoRunner
    from demo.recording import plain
    from edge.system import FlightSystem
    from edge.runtime import sha256
    from generator.generate_reflex_dataset import make_teacher
    from optimizer.continue_flow_reflex import write
    from evaluate.compare_maneuver_repair import summarize
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    source=ROOT/'output/release_cruise_v1/release_encoding'
    original=source/'flights/9d5431c1'
    m=json.loads((original/'manifest.json').read_text())
    with np.load(original/'physical.npz') as a: reference={k:a[k].copy() for k in a.files}
    files=[original/'manifest.json',original/'physical.npz',source/'checkpoint.pt',source/'runtime_config.json',
        *[p for p in (source/'numpy').iterdir() if p.is_file()]]
    hashes={str(p):sha256(p) for p in files}
    protocol=dict(version='active-turn-counterfactual-v1',case='9d5431c1',switch_s=34.,horizon_s=45.,
        input_sha256=hashes,script_sha256=sha256(__file__),
        arms=['control','teacher_yaw','raw_pitch','both'],
        intervention='At 34s replace only raw network yaw with continuously replayed v5/1530 teacher yaw, and/or disable anti_rollback; turn_latch remains enabled.',
        selection='Post-hoc time selected from known collision; no time or parameter search.',
        limitations='One case; substitute teacher has extra internal history; this is not a learned neural fix or a proposed runtime mode.',
        promotion=False)
    write(out/'protocol.json',protocol)
    results=[]
    for arm in protocol['arms']:
        r=DemoRunner(config=source/'runtime_config.json',scene=m['scene'],seed=m['seed'],
            start=m['start']['position_m'],yaw=m['start']['yaw_rad'],scene_options=m.get('scene_options'),
            collision_radius_m=.2,max_duration_s=45.,contact_model='sphere-sweep-v1',run_id=arm+'-turn-diagnosis')
        try:
            for key in ('room','boxes','cylinders'):assert r.manifest()['geometry'][key]==m['geometry'][key]
            p=YawIntervention(dict(candidate=r.system.runtime.policy),
                make_teacher('geometry-reflex-v5',noise_std_pwm=0,pwm_cruise_pitch=1530.),'candidate')
            r.system.runtime.policy=p
            manifest=deepcopy(r.system.manifest)
            manifest['experiment']=dict(protocol,arm=arm,
                trace_note='network_pwm is intervention driver after switch; separate raw network and teacher are in decisions.npz')
            r.system=FlightSystem(r.system.config,r.system.runtime,manifest)
            r._manifest=plain(r._build_manifest())
            dest=out/arm;dest.mkdir();write(dest/'manifest.json',r.manifest())
            r.resume();r.event(dict(type='mode',mode='AUTO'))
            for _ in range(11250):
                if r.step_index>=8500:
                    p.intervene=arm in ('teacher_yaw','both')
                    r.system.runtime.coordinator.anti_rollback=arm not in ('raw_pitch','both')
                n=len(p.records);now=r.time_s
                r.advance()
                if len(p.records)>n:p.records[-1]['time_s']=now
                if r.terminal:break
            time=np.array([row['time_s'] for row in r.rows]);pos=np.array([row['position_m'] for row in r.rows])
            velocity=np.array([row['velocity_mps'] for row in r.rows])
            prefix=time<34.
            np.testing.assert_array_equal(time[prefix],reference['time'][reference['time']<34.])
            np.testing.assert_array_equal(pos[prefix],reference['pos'][reference['time']<34.])
            np.testing.assert_array_equal(velocity[prefix],reference['velocity'][reference['time']<34.])
            if arm=='control':
                np.testing.assert_array_equal(time,reference['time'])
                np.testing.assert_array_equal(pos,reference['pos'])
                np.testing.assert_array_equal(velocity,reference['velocity'])
            np.savez_compressed(dest/'physical.npz',time=time,pos=pos,velocity=velocity,
                clearance=[row['clearance_m'] for row in r.rows])
            np.savez_compressed(dest/'decisions.npz',time=[row['time_s'] for row in p.records],
                X=[row['X'] for row in p.records],network=[row['outputs']['candidate'] for row in p.records],
                teacher=[row['outputs']['teacher'] for row in p.records],driver=[row['driver_pwm'] for row in p.records])
            row=dict(arm=arm,metrics=summarize(r.rows,r.time_s),prefix_exact=True,control_full_exact=arm=='control')
            write(dest/'report.json',row);results.append(row);write(out/'report.json',results)
            print(arm,row['metrics'],flush=True)
        finally:r.env.close()
    assert all(sha256(Path(p))==h for p,h in hashes.items())
    write(out/'validation.json',dict(input_unchanged=True,common_prefix_exact=True,control_reproduced_exactly=True,promotion=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    run(parser.parse_args().out)
