"""Passive replay of admission assessment; recorded commands are never changed."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from edge.sensors import Measurement,SensorSnapshot
from edge.runtime import sha256
from simulation.maneuver_admission import ManeuverAdmission

ROOT=Path(__file__).resolve().parents[1]
CASES=('068fe108','89d81a9f','9d5431c1','b0a9f212')


def snapshot_from_row(row):
    channels={}
    for name in ('flow','tof','altitude','attitude'):
        sample=row['sensors'][name]
        channels[name]=Measurement(**{k:sample[k] for k in ('sample_id','timestamp_s','value','valid')})
    return SensorSnapshot(row['trace']['sequence'],row['trace']['tick_time_s'],**channels)


def replay(path,out):
    data=json.loads(path.read_text(encoding='utf-8'))
    monitor=ManeuverAdmission();trace=[];previous=None;active=False
    for row in data['frames']:
        tick=row['trace']['tick_time_s']
        if row['plant_source']!='auto':
            if active:monitor.reset();active=False
            continue
        if tick==previous:continue
        previous=tick;active=True
        decision=monitor.observe(snapshot_from_row(row),now_s=row['time_s'])
        trace.append(dict(time_s=row['time_s'],**decision,
            recorded_network_pwm=row['trace']['network_pwm'],recorded_applied_pwm=row['plant_input_pwm']))
    cases=[r for r in trace if r['episode_pending']]
    # Durations limited to the actually recorded interval (not padded at termination).
    durations=np.minimum(.02,np.maximum(0.,data['end_time_s']-np.array([r['time_s'] for r in trace])))
    counts=Counter(r['state'] for r in trace)
    report=dict(source=str(path.relative_to(ROOT)),source_sha256=sha256(path),
        contract=monitor.contract(),scope='Passive observer. No new trajectory and no safety improvement claimed.',
        observed_auto_ticks=len(trace),states=dict(counts),invalid_ticks=counts['INVALID_SENSORS'],
        first_episode_s=cases[0]['time_s'] if cases else None,
        first_survey_required_s=next((r['time_s'] for r in trace if r['state']=='SURVEY_REQUIRED'),None),
        after_episode_recorded_forward_request_s=float(sum(dt for r,dt in zip(trace,durations)
            if r['episode_pending'] and r['recorded_applied_pwm'][2]>1510)),
        resume_permissions=sum(r['resume_allowed'] is True for r in trace))
    name=path.stem
    (out/(name+'_trace.json')).write_text(json.dumps(trace,indent=2),encoding='utf-8')
    (out/(name+'_report.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


def stop_check(path,out):
    """Known STOP intervention; observer never changes the simulated commands."""
    from demo.runner import DemoRunner
    from evaluate.compare_maneuver_repair import install_experiment
    data=json.loads(path.read_text(encoding='utf-8'));m=data['manifest']
    r=DemoRunner(scene=m['scene'],seed=m['seed'],start=m['start']['position_m'],
        yaw=m['start']['yaw_rad'],scene_options=m['scene_options'],collision_radius_m=.2,
        max_duration_s=16.,run_id='admission-stop-check')
    try:
        r.system=install_experiment(r.system,'release_only')
        monitor=ManeuverAdmission();trace=[]
        r.resume();r.event(dict(type='mode',mode='AUTO'))
        for step in range(4000):
            if step==2000:r.event(dict(type='stop'))
            r.advance()
            row=r.rows[-1]
            if step%5==0 and row['plant_source']=='auto':
                decision=monitor.observe(snapshot_from_row(row),now_s=row['time_s'])
                trace.append(dict(time_s=row['time_s'],**decision))
            if r.terminal:break
        # Preserve the earlier fixed experiment's prefix, not a silently new scene.
        folder=ROOT/'output/maneuver_repair_v1/9d5431c1_release_only'
        prefix=[row for row in r.rows if row['time_s']<8 and abs(row['time_s']/.02-round(row['time_s']/.02))<1e-8]
        with np.load(folder/'trace.npz') as a:
            assert np.array_equal(a['pos'][a['time']<8],[row['position_m'] for row in prefix])
        report=dict(scope='Observer on release_only diagnostic variant with externally scheduled STOP, not a deployment candidate.',
            source_sha256=sha256(path),stop_at_s=8.,end_s=r.time_s,prefix_position_exact=True,
            first_motion_quiet_s=next((row['time_s'] for row in trace if row['motion_quiet']),None),
            states=dict(Counter(row['state'] for row in trace)),
            collision=next((row['collision'] for row in r.rows if row['collision']),None),
            final_state=trace[-1],final_true_speed_mps=float(np.linalg.norm(r.env.physics.vel[:2])))
        (out/'stop_check_trace.json').write_text(json.dumps(trace,indent=2),encoding='utf-8')
        (out/'stop_check.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print('STOP observer check',report['states'],'quiet',report['first_motion_quiet_s'],flush=True)
    finally:r.env.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    paths=[next((ROOT/'flight simulations/changed contract').glob('*'+case+'*.json')) for case in CASES]
    protocol=dict(scope='Shadow assessment only; no controller integration or heading from world truth.',
        cases=CASES,contract=ManeuverAdmission().contract(),
        stop_check=dict(case='9d5431c1',variant='release_only',stop_at_s=8.,horizon_s=16.),
        script_sha256=sha256(__file__),observer_sha256=sha256(ROOT/'simulation/maneuver_admission.py'),
        sources={str(p.relative_to(ROOT)):sha256(p) for p in paths})
    (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    reports=[]
    for path in paths:
        report=replay(path,args.out);reports.append(report)
        print(path.stem,report['states'],'first episode',report['first_episode_s'],
              'survey',report['first_survey_required_s'],flush=True)
    (args.out/'report.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
    stop_check(paths[2],args.out)


if __name__=='__main__':main()
