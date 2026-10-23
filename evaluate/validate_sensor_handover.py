"""Fixed Step-3 protocol: exact telemetry grid and separate physical handovers.

No gain tuning, transport, or receiver simulation. Physics applies issued lease
PWM directly, with an explicitly simulated manual altitude controller otherwise.
"""
import argparse
from simulation.sensor_schedule import SampleSchedule
from collections import Counter
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from edge.flight_bridge import SensorFrame
from edge.runtime import AltitudeSample, sha256
from edge.sensors import Measurement, SensorSnapshot
from edge.system import RunConfig, build_system


HEIGHTS=(.92,1.,1.08)
DELTAS=(0.,-15.,15.)
VARIANTS=(('legacy','sync'),('freeze_sync_v1','sync'),('freeze_sync_v1','multi'))


def make_system(bundle,mode,profile,session):
    return build_system(RunConfig(bundle_dir=str(bundle),expected_metadata_sha256=sha256(Path(bundle)/'metadata.json'),
        coordinator_mode='both',coordinator_preset='original_v1',stop_hover=True,levy_explore=False,
        sensor_mode='multi_rate_v1' if profile=='multi' else 'strict_synchronous_v1',
        handover_mode=mode,session_id=session,seed=20261021))


def metrics(rows):
    active=[r for r in rows if r['issued_pwm'] is not None]
    shadow=[r for r in rows if r['tick_time_s']<5.]
    steps=[abs(b['issued_pwm'][0]-a['issued_pwm'][0]) for a,b in zip(rows,rows[1:])
           if a['issued_pwm'] is not None and b['issued_pwm'] is not None]
    first=active[0] if active else None
    second=active[1] if len(active)>1 else None
    return dict(first_auto_s=None if first is None else first['tick_time_s'],
        shadow_integral=shadow[-1]['altitude_integral'],
        first_issue_manual_jump_pwm=None if first is None else abs(first['issued_pwm'][0]-first['manual_pwm'][0]),
        next_supervised_from_entry_pwm=None if second is None else abs(second['supervised_pwm'][0]-first['issued_pwm'][0]),
        max_issued_step_pwm=max(steps,default=0.), faults=sum(r['authority']=='FAULT' for r in rows),
        sensor_hold_ticks={ch:sum(r['sensor_channels'].get(ch,{}).get('state')=='held' for r in rows)
                           for ch in ('flow','tof','altitude','attitude')},
        final_integral=rows[-1]['altitude_integral'], states=dict(Counter(r['authority'] for r in rows)))


def telemetry_case(bundle,height,delta,mode,profile):
    name=f'telemetry_{height:.2f}_{delta:+.0f}_{mode}_{profile}'
    system=make_system(bundle,mode,profile,name)
    schedule=SampleSchedule(profile)
    rows=[]
    for i in range(400):
        t=i*.02
        sample=AltitudeSample(height-.018,0.,0.,t,True)
        f=schedule.frame(i,(0.,0.),np.ones(64),sample)
        if isinstance(f,SensorSnapshot):
            sample=AltitudeSample(*f.altitude.value,f.altitude.timestamp_s,True)
        request='SHADOW' if t<5. else 'AUTO'
        # Exact admission delta, using a disposable observer copy, never changing
        # the actual controller. These are scripted sensors, not a physical flight.
        integrate=mode=='legacy' or (system.bridge.state=='AUTO' and request=='AUTO')
        proposal=deepcopy(system.runtime.altitude).update(sample,now_s=t,
            allow_hold=profile=='multi',integrate=integrate).throttle_pwm
        manual=[proposal+delta,1500.,1500.,1500.]
        result=system.tick(f,now_s=t,request=request,manual_pwm=manual,stop_requested=True)
        rows.append(dict(result['trace'].to_dict(),manual_pwm=manual))
    m=metrics(rows)
    passed=(m['first_auto_s']==5. and m['faults']==0 and m['first_issue_manual_jump_pwm']==0.
            and m['max_issued_step_pwm']<=3.+1e-8)
    if mode!='legacy':
        passed=passed and m['shadow_integral']==0. and m['next_supervised_from_entry_pwm']<=3.+1e-8
    return dict(name=name,height_m=height,delta_pwm=delta,mode=mode,profile=profile,
                passed=passed,metrics=m,manifest=system.manifest,trace=rows)


def physical_case(bundle,height,trim,mode,profile):
    from simulation.benchmark_scenes import NavigationScene
    from simulation.altitude_control import AltitudeConfig,AltitudeHold,sample_from_observation
    name=f'physical_{height:.2f}_{trim:+.0f}_{mode}_{profile}'
    system=make_system(bundle,mode,profile,name)
    schedule=SampleSchedule(profile)
    env=NavigationScene('clear')
    manual_hold=AltitudeHold(AltitudeConfig(target_height_m=height))
    rows=[]; done=False
    try:
        env.reset(initial_pos=[0.,0.,height],seed=20261021)
        env.physics.vel[:]=0.
        obs=env._sample_observation(0.)
        for i in range(450):
            t=i*.02
            sample=sample_from_observation(env.get_isaac_obs())
            manual=manual_hold.apply([1213.,1500.,1500.,1500.],sample,now_s=t)
            # Apply this trim to the actual simulated manual command during the
            # last half-second before requesting AUTO; admission may rightly fail.
            if 4.5<=t<8.: manual[0]+=trim
            request='SHADOW' if t<5. else ('AUTO' if t<8. else 'MANUAL')
            f=schedule.frame(i,obs[:2],obs[2:],sample)
            result=system.tick(f,now_s=t,request=request,manual_pwm=manual,stop_requested=True)
            applied=manual if result['lease'] is None else result['lease'].pwm
            rows.append(dict(result['trace'].to_dict(),manual_pwm=manual.tolist(),
                plant_pwm=list(applied),height_m=float(env.physics.pos[2]),
                vertical_speed_mps=float(env.physics.vel[2])))
            for _ in range(5):
                obs,_,done,_=env.step(applied)
                if done: break
            if done: break
    finally:
        env.close()
    m=metrics(rows)
    active=[r for r in rows if r['issued_pwm'] is not None]
    m.update(terminated=bool(done),completed_s=len(rows)*.02,
        min_height_m=min(r['height_m'] for r in rows),max_height_m=max(r['height_m'] for r in rows),
        height_at_engage_m=active[0]['height_m'] if active else None,
        peak_abs_vertical_speed_mps=max(abs(r['vertical_speed_mps']) for r in rows),
        max_height_error_auto_m=max((abs(r['height_m']-1.) for r in active),default=None),
        final_height_m=rows[-1]['height_m'])
    return dict(name=name,height_m=height,manual_trim_pwm=trim,mode=mode,profile=profile,
                metrics=m,manifest=system.manifest,trace=rows)


def verify_synchronous(before,after,*,steps=2000):
    from edge.bundle_validation import replay_bundle
    from edge.validate_numpy import _replay_inputs
    cases=[dict(coordinator_mode=m,coordinator_preset=p,levy_explore=b,stop_hover=b,seed=20261021)
           for m,p,b in [('baseline','original_v1',False),('both','original_v1',False),
                         ('both','near_zero_v1',False),('both','original_v1',True)]]
    inputs=_replay_inputs(steps)
    stop=np.zeros((4,steps),bool); stop[-1,max(0,steps-250):]=True
    old=replay_bundle(before,inputs,cases,stop_requests=stop)
    new=replay_bundle(after,inputs,[dict(c,sensor_mode='multi_rate_v1') for c in cases],
                      stop_requests=stop,assembly=True)
    equal={k:old[k].dtype==new[k].dtype and old[k].shape==new[k].shape
           and old[k].tobytes()==new[k].tobytes() for k in ('network_pwm','supervised_pwm','hidden')}
    reasons=old['report']['reasons']==new['report']['reasons']
    return dict(passed=all(equal.values()) and reasons,bitwise_equal=equal,reasons_equal=reasons,
        total_steps=4*steps,before={k:v for k,v in old['report'].items() if k!='reasons'},
        after={k:v for k,v in new['report'].items() if k!='reasons'},
        limitation='New hold-capable altitude path, all samples fresh at 50 Hz, integral enabled. '
                    'Does not assert handover output equality; that change is intentionally behavioral.')


def plot_results(output,report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'legacy':'tab:blue','freeze_sync_v1':'tab:orange'}
    fig,axes=plt.subplots(3,3,figsize=(14,10),sharex=True)
    for r in report['telemetry']:
        if r['profile']!='sync': continue
        ax=axes[HEIGHTS.index(r['height_m']),DELTAS.index(r['delta_pwm'])]
        rows=[json.loads(line) for line in (output/(r['name']+'.jsonl')).read_text().splitlines()]
        t=np.array([row['tick_time_s'] for row in rows]); mask=(t>=4.8)&(t<=5.8)
        proposed=[row['supervised_pwm'][0] if row['supervised_pwm'] else np.nan for row in rows]
        issued=[row['issued_pwm'][0] if row['issued_pwm'] else np.nan for row in rows]
        ax.plot(t[mask],np.asarray(proposed)[mask],'--',color=colors[r['mode']],alpha=.6,label=r['mode']+' supervised')
        ax.plot(t[mask],np.asarray(issued)[mask],color=colors[r['mode']],label=r['mode']+' issued')
        ax.axvline(5,color='gray',linewidth=.6)
        ax.set_title(f"height={r['height_m']:.2f} m, exact delta={r['delta_pwm']:+.0f} PWM")
        ax.grid(alpha=.2); ax.set_ylabel('Throttle PWM'); ax.set_xlabel('Time (s)')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=4)
    fig.suptitle('Scripted telemetry: SHADOW -> AUTO (no physical motion)')
    fig.tight_layout(rect=(0,.04,1,.96)); fig.savefig(output/'handover_commands.png',dpi=150); plt.close(fig)
    if not report['physical']: return
    fig,axes=plt.subplots(3,3,figsize=(14,10),sharex=True,sharey=True)
    for r in report['physical']:
        ax=axes[HEIGHTS.index(r['height_m']),DELTAS.index(r['manual_trim_pwm'])]
        rows=[json.loads(line) for line in (output/(r['name']+'.jsonl')).read_text().splitlines()]
        color='tab:green' if r['profile']=='multi' else colors[r['mode']]
        ax.plot([x['tick_time_s'] for x in rows],[x['height_m'] for x in rows],
                color=color,label=r['mode']+'/'+r['profile'])
        ax.set_title(f"initial height={r['height_m']:.2f} m, manual trim={r['manual_trim_pwm']:+.0f}")
    for ax in axes.flat:
        ax.axhline(1.,color='gray',ls='--',linewidth=.6); ax.axvspan(5,8,alpha=.05,color='green')
        ax.grid(alpha=.2); ax.set_ylabel('Height (m)'); ax.set_xlabel('Time (s)')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=3)
    fig.suptitle('Physical simulation: AUTO requested at 5 s; admission may be denied')
    fig.tight_layout(rect=(0,.04,1,.96)); fig.savefig(output/'physical_heights.png',dpi=150); plt.close(fig)


def run(bundle,output,*,physics=True):
    output=Path(output)
    if output.exists(): raise FileExistsError('Use a new experiment directory')
    output.mkdir(parents=True)
    protocol=dict(heights_m=HEIGHTS,delta_pwm=DELTAS,variants=VARIANTS,shadow_s=5.,
        telemetry_horizon_s=8.,physical_horizon_s=9.,auto_request_s=5.,manual_return_s=8.,
        physical_manual_trim_from_s=4.5,control_dt_s=.02,physics_dt_s=.004,
        multi_rate_hz=dict(flow=50,tof=15,altitude=30,attitude=50),physics_enabled=physics,
        limits=dict(first_command_jump_pwm=0.,issued_throttle_step_pwm=3.,new_shadow_integral=0.),
        limitations=['Telemetry grid uses fixed height and exact manual-versus-supervised delta.',
                     'Physical grid applies manual trim to actual plant; safe admission denial is an outcome.',
                     'Manual fallback is a simulation controller, not ESP firmware. Gate/transport are absent.'])
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n',encoding='utf-8')
    summaries=dict(telemetry=[],physical=[])
    for category,fn in [('telemetry',telemetry_case)]+([('physical',physical_case)] if physics else []):
        for height in HEIGHTS:
            for delta in DELTAS:
                for mode,profile in VARIANTS:
                    result=fn(bundle,height,delta,mode,profile)
                    trace=result.pop('trace')
                    with (output/(result['name']+'.jsonl')).open('x',encoding='utf-8') as f:
                        for row in trace: f.write(json.dumps(row,allow_nan=False)+'\n')
                    summaries[category].append(result)
        print(json.dumps(dict(stage=category,runs=len(summaries[category]))),flush=True)
    report=dict(version='sensor-handover-v1',protocol=protocol,**summaries,
                telemetry_passed=all(r['passed'] for r in summaries['telemetry']))
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    plot_results(output,report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--no-physics',action='store_true')
    args=p.parse_args()
    report=run(args.bundle,args.out_dir,physics=not args.no_physics)
    print(json.dumps(dict(telemetry_passed=report['telemetry_passed'],out=str(args.out_dir))))
    if not report['telemetry_passed']: raise SystemExit(1)


if __name__=='__main__': main()
