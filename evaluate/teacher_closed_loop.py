"""Current v5/1530 teacher through native handover and the unchanged flight stack."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
from evaluate.validate_maneuver_teacher import PolicyComparison
from evaluate.practical_maneuver import CASES, decision_metrics

ROOT=Path(__file__).resolve().parents[1]


class TeacherDriver(PolicyComparison):
    def __init__(self, policies, teacher):
        super().__init__(policies, teacher, 'teacher')

    def step_np(self, flow_xy, tof_8x8, memory_ring=None, dt=None):
        output=super().step_np(flow_xy, tof_8x8, memory_ring, dt)
        rec=self.records[-1]
        rec['phase']=('threat' if rec['input'][2:66].min()*3.5<.8 else
                      'held_brake' if self.teacher._escape_sign is not None else
                      'recovery' if self.teacher._cruise_recovery<1 else 'cruise')
        return output


def run(out):
    from demo.runner import DemoRunner
    from demo.recording import plain
    from edge.system import FlightSystem
    from edge.runtime import NumpyPolicy, sha256
    from generator.generate_reflex_dataset import make_teacher
    from evaluate.compare_maneuver_repair import summarize

    out=Path(out); out.mkdir(parents=True,exist_ok=False)
    runs=dict(before=ROOT/'output/balanced_maneuver_v1/before', candidate=ROOT/'output/balanced_maneuver_v1/plastic')
    source=ROOT/'output/practical_maneuver_v1'
    paths={case:source/(case+'_before')/'manifest.json' for case in CASES}
    teacher=make_teacher('geometry-reflex-v5',noise_std_pwm=0,pwm_cruise_pitch=1530.)
    files=[*paths.values(), ROOT/'generator/generate_reflex_dataset.py', ROOT/'generator/continuous_reflex_teacher.py',
           ROOT/'output/maneuver_teacher_check_v2/protocol.json', ROOT/'output/maneuver_teacher_check_v2/report.json',
           source/'report.json']
    for r in runs.values(): files += [r/'runtime_config.json', *[p for p in (r/'numpy').iterdir() if p.is_file()]]
    hashes={str(p):sha256(p) for p in files}
    fixed=json.loads((ROOT/'output/balanced_maneuver_v1/protocol.json').read_text())['fixed_baseline_sha256']
    assert all(sha256(ROOT/p)==h for p,h in fixed.items())
    protocol=dict(version='teacher-native-1530-v1', cases=CASES,horizon_s=45.,body_radius_m=.2,
        teacher_version='geometry-reflex-v5',teacher_parameters=teacher.parameters,
        input_sha256=hashes,script_sha256=sha256(__file__),
        driver='Teacher from the first policy call, including SHADOW; no post-admission switch',
        observation='Only normalized flow[2], ToF[64], memory[8], dt and reset history; teacher ignores memory channels',
        state='Teacher latch, brake peak, settle and recovery timers; no truth position, velocity, geometry or future input',
        runtime='Baseline config unchanged; both/original_v1, 50/30/15 Hz, AltitudeHold, bridge, receiver',
        comparisons='Saved baseline, plastic, and historical v5/1600 with baseline-assisted AUTO admission',
        criteria='Valid autonomous exposure, completed horizon, contacts, clearance, motion/reverse; not coverage optimization',
        no_training=True,no_parameter_search=True,promotion=False)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    reports=[]
    for case in CASES:
        m=json.loads(paths[case].read_text())
        r=DemoRunner(config=runs['before']/'runtime_config.json',scene=m['scene'],seed=m['seed'],
            start=m['start']['position_m'],yaw=m['start']['yaw_rad'],scene_options=m.get('scene_options'),
            collision_radius_m=.2,max_duration_s=45.,contact_model='sphere-sweep-v1',run_id=case+'-teacher1530')
        try:
            for key in ('room','boxes','cylinders'): assert r.manifest()['geometry'][key]==m['geometry'][key]
            policies={name:r.system.runtime.policy if name=='before' else NumpyPolicy(path/'numpy') for name,path in runs.items()}
            observer=TeacherDriver(policies,make_teacher('geometry-reflex-v5',noise_std_pwm=0,pwm_cruise_pitch=1530.))
            r.system.runtime.policy=observer
            manifest=deepcopy(r.system.manifest)
            manifest['experiment']=dict(version=protocol['version'],driver='teacher',
                command_stage_note='network_pwm contains teacher actions in this research overlay',
                neural_display='Passive baseline observer, not teacher state',teacher_parameters=teacher.parameters,
                source_sha256=protocol['script_sha256'])
            r.system=FlightSystem(r.system.config,r.system.runtime,manifest)
            r._manifest=plain(r._build_manifest())
            dest=out/case;dest.mkdir()
            (dest/'manifest.json').write_text(json.dumps(r.manifest(),indent=2),encoding='utf-8')
            r.resume(); r.event(dict(type='mode',mode='AUTO'))
            for _ in range(11250):
                n=len(observer.records); count=len(r.rows); now=r.time_s
                r.advance()
                if len(observer.records)>n:
                    row=next(row for row in r.rows[count:] if abs(row['time_s']-now)<1e-8)
                    observer.records[-1].update(time_s=now,plant_source=row['plant_source'])
                if r.terminal: break
            metrics=summarize(r.rows,r.time_s)
            t=np.array([row['time_s'] for row in r.rows]); dt=np.diff(t,append=r.time_s)
            auto=np.array([row['plant_source']=='auto' for row in r.rows])
            vel=np.array([row['velocity_mps'] for row in r.rows]); speed=np.linalg.norm(vel[:,:2],axis=1)
            metrics.update(valid_autonomous_trial=bool(auto.any()),first_auto_s=float(t[auto][0]) if auto.any() else None,
                mean_auto_speed_mps=float((speed*dt*auto).sum()/metrics['auto_s']) if metrics['auto_s'] else None,
                stationary_auto_s=float(dt[auto&(speed<=.03)].sum()),auto_distance_m=float((speed*dt*auto).sum()))
            np.savez_compressed(dest/'physical.npz',time=t,pos=[v['position_m'] for v in r.rows],velocity=vel,
                yaw=[v['attitude_rad'][2] for v in r.rows],auto=auto,clearance=[v['clearance_m'] for v in r.rows])
            rows=r.rows[::5]
            np.savez_compressed(dest/'trace.npz',time=[v['time_s'] for v in rows],
                **{key:np.array([v['trace'].get(field) or [np.nan]*4 for v in rows]) for key,field in
                   [('network','network_pwm'),('coordinated','coordinated_pwm'),('supervised','supervised_pwm'),
                    ('issued','issued_pwm'),('accepted','accepted_pwm')]},
                applied=[v.get('plant_input_pwm') or [np.nan]*4 for v in rows],min_tof=[v['min_tof_m'] for v in rows])
            records=observer.records
            np.savez_compressed(dest/'decisions.npz',X=[v['input'] for v in records],time=[v['time_s'] for v in records],
                segment=[v['segment'] for v in records],phase=[v['phase'] for v in records],
                auto=[v['plant_source']=='auto' for v in records],
                **{name:[v['outputs'][name] for v in records] for name in ('before','candidate','teacher')})
            # Metrics for both neural observers on the teacher-driven input sequence.
            diagnostics=decision_metrics(records,r.time_s,'before')
            diagnostics.pop('premature_forward_episodes');diagnostics.pop('continued_brake_episodes')
            report=dict(case=case,driver='teacher1530',metrics=metrics,events=r.events,decisions=diagnostics)
            (dest/'report.json').write_text(json.dumps(plain(report),indent=2,allow_nan=False),encoding='utf-8')
            reports.append(report)
            (out/'report.json').write_text(json.dumps(plain(reports),indent=2,allow_nan=False),encoding='utf-8')
            print(case,metrics,flush=True)
        finally: r.env.close()
    assert all(sha256(Path(p))==h for p,h in hashes.items())
    assert all(sha256(ROOT/p)==h for p,h in fixed.items())
    (out/'validation.json').write_text(json.dumps(dict(completed_trials=4,
        valid_autonomous_trials=sum(r['metrics']['valid_autonomous_trial'] for r in reports),
        input_and_active_baseline_unchanged=True,no_training=True,automatic_promotion=False),indent=2),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    run(p.parse_args().out)
