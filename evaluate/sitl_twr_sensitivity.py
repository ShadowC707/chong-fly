"""Fixed TWR sensitivity, measured static actuator map, frozen smooth controller."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from evaluate.sitl_phased_yaw import digest,inspect
from simulation.sitl_twr import ThrustAdapter,thrust_to_weight

GRID=(1070,1100,1150,1180,1200,1210,1220,1250,1300,1400,1500,1600,1700,1800,1900,1950)
CASES=('obstacle_right','obstacle_left')


def hover_probe_grid(inferred):
    if not np.isfinite(inferred) or not 1002<=inferred<=1998:
        raise ValueError('Hover probe outside RC domain')
    return list(range(int(np.floor(inferred))-2,int(np.ceil(inferred))+3))


def calibrate(out,source,parameters,targets):
    from evaluate.sitl_lockstep import Session
    out.mkdir(parents=True,exist_ok=False)
    shutil.copy2(source/'template/eeprom.bin',out/'eeprom.bin')
    session=Session(out);rows=[];brackets=[]
    try:
        arm=session.arm()
        for pwm in GRID:
            session.pwm[0]=pwm;session.pump(250)
            values=np.array(session.pump(125))
            rows.append(dict(rc_pwm=pwm,mean_motor=float(values.mean()),
                             motor_range=[float(values.min()),float(values.max())]))
        for target in targets:
            adapter=ThrustAdapter(parameters,target,rows)
            samples=[]
            for pwm in hover_probe_grid(adapter.hover_rc_pwm):
                session.pwm[0]=pwm;session.pump(250)
                values=np.array(session.pump(125))
                samples.append(dict(rc_pwm=pwm,mean_motor=float(values.mean())))
            brackets.append(dict(twr=target,target_motor=adapter.hover_motor,measured=samples,
                bracketed=min(s['mean_motor'] for s in samples)<=adapter.hover_motor<=max(s['mean_motor'] for s in samples)))
        old=json.loads((source/'report.json').read_text())['static_map']['rows']
        errors={str(r['rc_pwm']):next(v['mean_motor'] for v in rows if v['rc_pwm']==r['rc_pwm'])-r['mean_motor'] for r in old}
        result=dict(rows=rows,hover_brackets=brackets,old_anchor_errors=errors,arm=arm,
            meaning='Measured virtual firmware RC -> normalized output at neutral axes; not a hardware thrust map')
        (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
        if not all(b['bracketed'] for b in brackets) or max(abs(v) for v in errors.values())>1e-8:
            raise ValueError('Static calibration changed or target hover not bracketed')
        return result
    finally:
        np.savez_compressed(out/'acks.npz',acks=session.acks)
        session.close()


def assess(folder,parameters,twr,rows):
    from simulation.sitl_rc_units import reference_axes_to_actual
    from evaluate.sitl_yaw_profiles import motor_limit_counts
    r=inspect(folder);adapter=ThrustAdapter(parameters,twr,rows)
    commands=json.loads((folder/'command_trace.json').read_text())
    expected=np.array([reference_axes_to_actual(adapter.command(c['selected']['applied_pwm'])) for c in commands])
    with np.load(folder/'trace.npz') as a:
        r.update(actual_rc_mapping_max_error=float(abs(expected-a['pwm'][::80]).max()),
            motor_min=float(a['motors'].min()),motor_max=float(a['motors'].max()),
            motor_limits=motor_limit_counts(a['motors']),
            actual_throttle_rc_min=float(a['pwm'][:,0].min()),actual_throttle_rc_max=float(a['pwm'][:,0].max()))
    recovery=bool(r['recoveries']) and all(not w[c]['censored'] and w[c]['settle_s'] is not None and w[c]['settle_s']<=.8
        for w in r['recoveries'] for c in ('issued','actual'))
    r['checks']=dict(complete=r['completed'],no_error=r['error'] is None,no_contact=r['collision'] is None,
        clearance_10cm=r['min_clearance_m']>=.1,height_10cm=r['max_height_error_m']<=.1,
        stop=r['stop_passed'],authority=r['authority_passed'],recovery=recovery,
        no_timeout=r['timeout_events']==0,no_motor_limit=r['motor_limits']['saturated_motor_ticks']==0,
        mapping_exact=r['actual_rc_mapping_max_error']==0.)
    r.update(twr=twr,max_rpm=adapter.dynamics['max_rpm'],passed=all(r['checks'].values()))
    return r


def render(out,results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from evaluate.sitl_neural import scene_spec
    fig,axes=plt.subplots(2,4,figsize=(18,9),constrained_layout=True)
    for row,case in zip(axes,CASES):
        for b in scene_spec(case)['boxes']:
            row[0].add_patch(Rectangle((b['x_min'],b['y_min']),b['x_max']-b['x_min'],b['y_max']-b['y_min'],color='.8'))
        for r in [v for v in results if v['case']==case]:
            folder=out/r['run'];label=f"TWR {r['twr']:.2f}"
            with np.load(folder/'trace.npz') as a:
                t,s=a['time_s'],a['state']
                row[0].plot(s[:,0],s[:,1],label=label)
                row[1].plot(t,100*(s[:,2]-1),label=label)
                row[2].plot(t,np.degrees(s[:,11]),label=label)
                row[3].plot(t[:-1],a['pwm'][:,0],label=label)
        row[0].set_aspect('equal',adjustable='datalim')
        for ax,title in zip(row,('XY (m)','Height error (cm)','Yaw rate (deg/s)','Actual throttle RC')):
            ax.set_title(case+' | '+title);ax.grid(alpha=.2);ax.legend()
    fig.suptitle('TWR sensitivity via max RPM only | frozen smooth controller, network and FC/PID\n'
                 'Collective thrust reference preserved by measured RC map; uncalibrated hardware model')
    fig.savefig(out/'comparison.png',dpi=140);plt.close(fig)


def run(out,source,reference):
    from evaluate.sitl_neural import trial
    from evaluate.sitl_lockstep import BIN
    from edge.system import RunConfig
    p=json.loads(Path('configs/ready_sim_reference_v1.json').read_text())['uncertainty_study']['nominal_dynamics']
    targets=[thrust_to_weight(p),10.,5.]
    old=json.loads((reference/'protocol.json').read_text())
    changed=[k for k,v in old['input_source_sha256'].items() if k!='evaluate/sitl_neural.py' and digest(k)!=v]
    if changed or digest(BIN)!=old['firmware_sha256']: raise ValueError('Frozen inputs changed: '+str(changed))
    cfg=RunConfig.from_json('configs/demo_candidate_a.json')
    files=[v for f in ('simulation','edge','demo','evaluate') for v in Path(f).glob('*.py')]
    files += [Path('configs/demo_candidate_a.json'),Path('configs/ready_sim_reference_v1.json'),
              Path('configs/sitl_virtual_v1.cli'),source/'template/eeprom.bin',*Path(cfg.bundle_dir).glob('*')]
    hashes={str(f):digest(f) for f in files if f.is_file()}
    out.mkdir(parents=True,exist_ok=False)
    protocol=dict(version='sitl-twr-sensitivity-v1',targets=targets,cases=CASES,calibration_rc_grid=GRID,
        hover_probe='Fixed integer RC grid floor(inferred)-2 .. ceil(inferred)+2; no adaptive search',
        input_source_sha256=hashes,firmware_sha256=digest(BIN),reference=str(reference),
        horizon_s=20.,stop_at_s=17.,physics_dt_s=.00025,control_dt_s=.02,
        sensor_rates_hz=dict(flow=50,attitude=50,tof=15,altitude=30),
        control='Pinned Candidate A + frozen cosine-bounded-v1; same reference AltitudeHold and FlightBridge',
        treatment='Only max_rpm changes; u_new=u_ref*n_ref/n_new then measured inverse static map. '
            'Mass, inertia, kf/km, lag, drag, FC/PID and axis RC adapter unchanged.',
        criteria='Complete/no contact/error; clearance>=.10m; height error<=.10m; STOP<.03m/s final second; '
            'continuous AUTO; observed recovery issued +/-20PWM and rate +/-8deg/s within .8s; '
            'no timeout or motor limit; exact RC mapping. All failures retained, no tuning.',
        limitations='TWR values are sensitivity hypotheses, not hardware measurements; collective compensation '
            'does not compensate FC mixing authority or rotor torque derivatives; ideal sensors/no radio noise.',
        deployment_admitted=False)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    for name in hashes:
        f=Path(name)
        if not f.is_absolute() and f.suffix=='.py':
            dst=out/'source_snapshot'/f;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dst)
    calibration=calibrate(out/'calibration',source,p,targets)
    print(json.dumps(dict(calibration=calibration)),flush=True)
    results=[];baseline={}
    for index,target in enumerate(targets):
        for case in CASES:
            name=case+'_'+('nominal' if index==0 else 'twr'+str(int(target)))
            folder=out/name
            trial(folder,source,case,'sitl','reference_units',physics_dt_s=.00025,
                  integration_profile='multi_rate',yaw_shaping=True,target_twr=target,calibration_rows=calibration['rows'])
            r=assess(folder,p,target,calibration['rows'])|dict(run=name);results.append(r)
            (out/'comparison.json').write_text(json.dumps(results,indent=2)+'\n')
            print(json.dumps({k:v for k,v in r.items() if k not in ('phase_events',)}),flush=True)
            if index==0:
                with np.load(folder/'trace.npz') as a,np.load(reference/(case+'_smooth')/'trace.npz') as b:
                    baseline[case]={k:bool(np.array_equal(a[k],b[k],equal_nan=True))
                        for k in ('state','motors','pwm','network_pwm','coordinated_pwm','supervised_pwm')}
    validation=dict(sources_unchanged=all(digest(k)==v for k,v in hashes.items()),
        nominal_array_equal=baseline,clocks_passed=all(r['clock_passed'] for r in results),
        accepted_equals_issued=all(r['accepted_equals_issued'] for r in results),
        passed={r['run']:r['passed'] for r in results},deployment_admitted=False)
    (out/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    render(out,results);print(json.dumps(validation,indent=2),flush=True)
    if not validation['sources_unchanged'] or not validation['clocks_passed'] or any(r['error'] for r in results):
        raise SystemExit(1)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--source',type=Path,required=True);ap.add_argument('--reference',type=Path,required=True)
    a=ap.parse_args();run(a.out,a.source,a.reference)
