"""Finite sensor-contract experiment; no flight commands or corridor claims.

Analytic rotations and one recorded simulation feed an explicit synthetic
heading sensor. Recorded world yaw is ONLY the sensor source and QA oracle.
It is not a measured hardware channel and is never passed to the tracker.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import json
import math
from pathlib import Path

import numpy as np

from edge.runtime import sha256
from edge.sensors import Measurement
from edge.survey_heading import HeadingConfig, SurveyHeadingTracker
from simulation.survey_heading_sensor import SimulatedHeadingSensor

ROOT=Path(__file__).resolve().parents[1]
SEED=20261104


def angle_error(a,b):
    return abs((a-b+math.pi)%(2*math.pi)-math.pi)


def run_case(name, *, truth_at, ticks, tof_at, tof_delay_s=0.):
    tracker=SurveyHeadingTracker(name)
    origin=truth_at(0.)
    sensor=SimulatedHeadingSensor(name,seed=SEED,initial_yaw_rad=origin)
    queue_h=[];queue_t=[];current_tof=None;latest_heading=None;rows=[];rejected=Counter()
    rng=np.random.default_rng(SEED)
    for step,t in enumerate(ticks):
        # 50 Hz acquisition, physical clock 250 Hz; latency 12 or 16 ms.
        if step%5==0:
            sample=sensor.capture(truth_at(t),timestamp_s=t)
            queue_h.append((t+.012+.004*int(rng.integers(0,2)),sample))
        new_tof=tof_at(step,t)
        if new_tof is not None:queue_t.append((t+tof_delay_s,new_tof))
        while queue_h and queue_h[0][0]<=t+1e-8:
            _,sample=queue_h.pop(0)
            latest_heading=tracker.add_heading(sample,now_s=t)
        while queue_t and queue_t[0][0]<=t+1e-8:
            _,current_tof=queue_t.pop(0)
        if step%5 or current_tof is None:continue
        try:
            aligned=tracker.align_tof(current_tof,reference_id=name,now_s=t)
        except ValueError as exc:
            rejected[str(exc)]+=1
            continue
        truth_capture=truth_at(aligned.timestamp_s)-origin
        error=angle_error(aligned.heading_rad,truth_capture)
        naive=angle_error(latest_heading.angle_rad,truth_capture)
        rows.append(dict(time_s=t,tof_capture_s=aligned.timestamp_s,state=aligned.state,
            heading_at_capture_rad=aligned.heading_rad,true_heading_at_capture_rad=truth_capture,
            latest_heading_rad=latest_heading.angle_rad,error_rad=error,naive_error_rad=naive,
            bound_rad=aligned.heading_error_bound_rad))
    truth=np.unwrap([truth_at(float(t)) for t in ticks])
    max_truth_rate=float(np.max(np.abs(np.diff(truth)/np.diff(ticks))))
    assert max_truth_rate<=tracker.config.max_rate_rad_s+1e-5
    assert rows and all(r['error_rad']<=r['bound_rad']+1e-7 for r in rows)
    assert set(rejected)<= {'No fresh heading for current observation'}
    return rows,dict(name=name,aligned_ticks=len(rows),new_tof=sum(r['state']=='new' for r in rows),
        rejected=dict(rejected),max_error_deg=math.degrees(max(r['error_rad'] for r in rows)),
        max_declared_bound_deg=math.degrees(max(r['bound_rad'] for r in rows)),
        max_naive_current_heading_error_deg=math.degrees(max(r['naive_error_rad'] for r in rows)),
        mean_error_deg=math.degrees(float(np.mean([r['error_rad'] for r in rows]))),
        mean_naive_error_deg=math.degrees(float(np.mean([r['naive_error_rad'] for r in rows]))),
        max_truth_rate_deg_s=math.degrees(max_truth_rate),
        all_errors_within_declared_bound=True)


def analytic_tof():
    last=-1
    def sample(step,t):
        nonlocal last
        epoch=int(math.floor(t*15+1e-8))
        if epoch==last:return None
        last=epoch
        # Only timing/identity are under study; these ranges do not model a room.
        return Measurement(epoch,t,tuple(np.linspace(.2,1.,64)))
    return sample


def fault_check():
    sensor=SimulatedHeadingSensor('fault')
    tracker=SurveyHeadingTracker('fault')
    tracker.add_heading(sensor.capture(0.,timestamp_s=0.),now_s=0.)
    # Captures .02 and .04 were lost; next .06 cannot bridge continuity.
    try:tracker.add_heading(sensor.capture(.1,timestamp_s=.06),now_s=.06)
    except ValueError:pass
    assert tracker.faulted
    rejected=False
    try:tracker.align_tof(Measurement(0,.06,np.ones(64)),reference_id='fault',now_s=.06)
    except ValueError:rejected=True
    assert rejected
    tracker.reset('recovered')
    sensor.reset('recovered',initial_yaw_rad=.1,timestamp_s=.1)
    tracker.add_heading(sensor.capture(.1,timestamp_s=.1),now_s=.1)
    result=tracker.align_tof(Measurement(0,.1,np.ones(64)),reference_id='recovered',now_s=.1)
    return dict(two_lost_captures_latched_fault=True,alignment_rejected_during_fault=True,
                explicit_new_reference_recovered=result.reference_id=='recovered')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    path=next((ROOT/'flight simulations/changed contract').glob('*9d5431c1*.json'))
    protocol=dict(scope=__doc__,seed=SEED,heading_hz=50,tof_hz=15,
        delivery_delay_s=[.012,.016],noise_bound_deg=.25,drift_deg_s=.1,
        config=asdict(HeadingConfig()),recording=str(path.relative_to(ROOT)),recording_sha256=sha256(path),
        files={f:sha256(ROOT/f) for f in ('edge/survey_heading.py',
            'simulation/survey_heading_sensor.py','evaluate/validate_survey_heading.py')},
        cases=['left_wrap','right_wrap','reversal','tof_delayed_40ms','recorded_9d5431c1'])
    (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    ticks=np.arange(601)*.004
    rate=math.radians(180)
    specs=[('left_wrap',lambda t:rate*t,0.),('right_wrap',lambda t:-rate*t,0.),
           ('reversal',lambda t:rate*(t if t<=1.2 else 2.4-t),0.),
           ('tof_delayed_40ms',lambda t:rate*t,.04)]
    results=[];traces={}
    for name,truth,delay in specs:
        trace,report=run_case(name,truth_at=truth,ticks=ticks,tof_at=analytic_tof(),tof_delay_s=delay)
        traces[name]=trace;results.append(report)
    recording=json.loads(path.read_text(encoding='utf-8'))['frames']
    # Use the recorded physical tick times; capture yaw and ToF at the same tick.
    recording=[r for r in recording if r['plant_source']!='terminated']
    lookup={round(r['time_s'],8):r for r in recording}
    times=[r['time_s'] for r in recording]
    assert np.allclose(np.diff(times),.004,rtol=0,atol=1e-8)
    last_id=[None]
    def recorded_tof(step,t):
        value=lookup[round(t,8)]['sensors']['tof']
        if value['sample_id']==last_id[0]:return None
        last_id[0]=value['sample_id']
        assert abs(value['timestamp_s']-t)<1e-8
        return Measurement(**{k:value[k] for k in ('sample_id','timestamp_s','value','valid')})
    name='recorded_9d5431c1'
    trace,report=run_case(name,truth_at=lambda t:lookup[round(t,8)]['attitude_rad'][2],
                          ticks=times,tof_at=recorded_tof)
    traces[name]=trace;results.append(report)
    payload=dict(results=results,fault_check=fault_check(),changes_flight_commands=False,
                 corridor_or_collision_safety_test=False)
    (args.out/'report.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    (args.out/'traces.json').write_text(json.dumps(traces,indent=2),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(len(traces),1,figsize=(11,12),layout='constrained')
    for ax,(name,rows) in zip(axes,traces.items()):
        t=[r['time_s'] for r in rows]
        ax.plot(t,[math.degrees(r['error_rad']) for r in rows],label='Capture-aligned error')
        ax.plot(t,[math.degrees(r['naive_error_rad']) for r in rows],label='Current heading on held ToF',alpha=.7)
        ax.plot(t,[math.degrees(r['bound_rad']) for r in rows],'k--',label='Declared alignment bound',linewidth=.8)
        ax.set(title=name,ylabel='Angular error (deg)',xlabel='Time (s)');ax.grid(alpha=.2)
    axes[0].legend(ncol=3,fontsize=8)
    fig.suptitle('Synthetic heading channel: ToF capture time vs current heading\nNo flight-policy changes; bounds depend on stated sensor/rate assumptions')
    fig.savefig(args.out/'comparison.png',dpi=150);plt.close(fig)
    print(json.dumps(payload,indent=2))


if __name__=='__main__':main()
