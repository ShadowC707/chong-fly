"""Six fixed physical trials: two geometries and one explicit sensor diagnostic.

No controller, weight, PID, or plant tuning; a diagnostic is not navigation.
"""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from evaluate.sitl_neural import scene_spec
from evaluate.sitl_phased_yaw import digest, inspect


def box(x0,x1,y0,y1):
    return dict(x_min=x0,x_max=x1,y_min=y0,y_max=y1,z_min=0.,z_max=2.5)


def scenes():
    # Translate two published phased_holdout geometries by +2m in x so their
    # (-2,0,1) initial position becomes the SITL runner's fixed origin.
    # Keep the room [-4,4]; this is a new explicit SITL test, not exact old replay.
    pocket=scene_spec('clear')
    pocket['boxes']=[box(2.5,2.7,-1.2,1.2),box(1.5,2.7,1.2,1.4),box(1.5,2.7,-1.4,-1.2)]
    stagger=scene_spec('clear')
    stagger['boxes']=[box(1.8,2.2,-1.5,.1),box(3.3,3.7,.1,1.5)]
    return dict(dead_end=pocket,stagger=stagger,lifecycle=scene_spec('clear'))


class LifecycleToF:
    def contract(self):
        return dict(version='fixed-lifecycle-tof-v1',scope='Synthetic ToF ONLY; not navigation admission',
            intervals_m=[[2.,3.4,.7],[3.4,3.8,2.],[3.8,14.,.7],[14.,17.,2.]],
            otherwise='Actual scene ToF',units='distance_m / 3.5, all 64 cells; normal 15Hz hold follows',
            unchanged='Physical flow/range/attitude, FC and plant; no state/phase feedback to stimulus')

    def apply(self,tof,now_s):
        distance=next((d for a,b,d in self.contract()['intervals_m'] if a<=now_s<b),None)
        self.last=dict(time_s=now_s,overridden=distance is not None,distance_m=distance)
        return np.array(tof,copy=True) if distance is None else np.full(64,distance/3.5)


def lifecycle_check(decisions,t,issued_yaw):
    starts=[d for d in decisions if d['event']=='started']
    failures=[(i,d) for i,d in enumerate(decisions) if d['event'] in
              ('turn_timeout','episode_timeout','brake_timeout')]
    renewed=any(d['event']=='renewed_threat' for d in decisions)
    result=dict(covered=bool(starts and renewed and failures),renewed_threat=renewed,
                timeout_count=len(failures),neutral_by_deadline=False,original_deadline_s=None)
    if not result['covered']: return result
    i,failed=failures[0]; when=failed['time_s']
    if failed['event']=='episode_timeout': deadline=starts[0]['time_s']+8.
    elif failed['event']=='brake_timeout':
        beginning=next(d['time_s'] for d in reversed(decisions[:i]) if d['event'] in ('started','renewed_threat'))
        deadline=beginning+2.
    else:
        # Reconstruct cumulative TURN time from phase intervals, including
        # interrupted turns. Use mathematical deadline (conservative vs tick rounding).
        prior=0.;turn_start=None
        for j,d in enumerate(decisions[:i]):
            nxt=decisions[j+1]
            if d['phase']=='TURN':
                if turn_start is None: turn_start=d['time_s']
                if nxt['phase']=='TURN': continue
                if j+1==i: break
                prior+=nxt['time_s']-turn_start;turn_start=None
        if turn_start is None: return result
        deadline=turn_start+4.-prior
    end=next((d['time_s'] for d in decisions[i+1:] if d['phase']!='FAILED'),20.)
    mask=(t>=deadline-1e-9)&(t<end-1e-9)
    result.update(original_deadline_s=deadline,failed_at_s=when,failed_end_s=end,
        neutral_by_deadline=bool(mask.any() and np.all(abs(issued_yaw[mask]-1500.)<=1e-6)),
        neutral_tail_s=max(0.,deadline-when))
    return result


def assess(folder,case):
    r=inspect(folder)
    decisions=json.loads((folder/'decisions.json').read_text())
    rows=json.loads((folder/'command_trace.json').read_text())
    with np.load(folder/'trace.npz') as a:
        t,s,ct=a['time_s'],a['state'],a['control_time']
        yaw=np.array([v['selected']['applied_pwm'][3] for v in rows])
        idx=int(np.argmin(a['clearance']))
        dx=np.diff(s[:,:2],axis=0)
        backward=-(dx[:,0]*np.cos(s[:-1,8])+dx[:,1]*np.sin(s[:-1,8]))
        r.update(minimum=dict(time_s=float(t[idx+1]),position_m=s[idx+1,:3].tolist()),
            reverse_distance_m=float(np.maximum(backward,0).sum()),
            failed_time_s=sum(d['phase']=='FAILED' for d in decisions)*.02,
            renewed_threat_count=sum(d['event']=='renewed_threat' for d in decisions),
            lifecycle=lifecycle_check(decisions,ct,yaw) if case=='lifecycle' else None)
    r['checks']=dict(complete=r['completed'],no_contact=r['collision'] is None,
        no_error=r['error'] is None,clearance_10cm=r['min_clearance_m']>=.1,
        height_10cm=r['max_height_error_m']<=.1,stop=r['stop_passed'],authority=r['authority_passed'])
    if case=='lifecycle':
        r['checks'].update(events_covered=r['lifecycle']['covered'],
                           neutral_by_deadline=r['lifecycle']['neutral_by_deadline'],one_timeout=r['timeout_events']==1)
    else: r['checks']['no_timeout']=r['timeout_events']==0
    r['passed']=all(r['checks'].values())
    return r


def render(out,results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig,axes=plt.subplots(3,4,figsize=(18,12),constrained_layout=True)
    for row,(case,spec) in zip(axes,scenes().items()):
        for b in spec['boxes']:
            row[0].add_patch(Rectangle((b['x_min'],b['y_min']),b['x_max']-b['x_min'],b['y_max']-b['y_min'],color='.8'))
        for variant in ('control','smooth'):
            folder=out/(case+'_'+variant)
            with np.load(folder/'trace.npz') as a:
                t,s,ct=a['time_s'],a['state'],a['control_time']
                row[0].plot(s[:,0],s[:,1],label=variant)
                row[1].plot(t,100*(s[:,2]-1),label=variant)
                row[2].plot(t,np.degrees(s[:,11]),label=variant)
                rows=json.loads((folder/'command_trace.json').read_text())
                row[3].plot(ct,[r['selected']['applied_pwm'][3]-1500 for r in rows],label=variant)
        row[0].set_aspect('equal',adjustable='datalim')
        for ax,title in zip(row,('XY (m)','Height error (cm)','Actual yaw rate (deg/s)','Issued yaw offset (PWM)')):
            ax.set_title(case+' | '+title);ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Fixed yaw shaping stress pairs | ideal multi-rate sensors | unchanged FC/plant/network\n'
                 'Lifecycle row: synthetic ToF stimulus, physical remaining loop; NOT a navigation test')
    fig.savefig(out/'comparison.png',dpi=140);plt.close(fig)


def run(out,source,reference):
    from evaluate.sitl_neural import trial
    from evaluate.sitl_lockstep import BIN
    from edge.system import RunConfig
    old=json.loads((reference/'protocol.json').read_text())
    # Only the trial runner was extended; all previously executed controller,
    # bundle, physical, and adapter modules must match the accepted first pair.
    changed=[p for p,v in old['input_source_sha256'].items()
             if p!='evaluate/sitl_neural.py' and digest(p)!=v]
    if changed or digest(BIN)!=old['firmware_sha256']:
        raise ValueError('Changed fixed inputs: '+str(changed))
    cfg=RunConfig.from_json('configs/demo_candidate_a.json')
    files=[p for f in ('simulation','edge','demo','evaluate') for p in Path(f).glob('*.py')]
    files += [Path('configs/demo_candidate_a.json'),Path('configs/ready_sim_reference_v1.json'),
              Path('configs/sitl_virtual_v1.cli'),source/'template/eeprom.bin',*Path(cfg.bundle_dir).glob('*')]
    hashes={str(p):digest(p) for p in files if p.is_file()}
    out.mkdir(parents=True,exist_ok=False)
    protocol=dict(version='sitl-yaw-stress-v1',scenes=scenes(),variants=['control','smooth'],
        reference=str(reference),input_source_sha256=hashes,firmware_sha256=digest(BIN),
        horizon_s=20.,stop_at_s=17.,physical_dt_s=.00025,control_dt_s=.02,
        rates_hz=dict(flow=50,attitude=50,tof=15,altitude=30),tof_diagnostic=LifecycleToF().contract(),
        rule='No fitting or retries based on behavior; every failure retained. No promotion.',
        criteria='Complete 20s, no contact/error, clearance>=.10m, height error<=.10m, STOP<.03m/s '
            'throughout last second, continuous AUTO. Natural scenes: no timeout. Diagnostic: '
            'renewed_threat plus exactly one timeout and neutral issued yaw by ORIGINAL budget deadline.',
        interpretation='Missing natural recurrence is missing coverage, not a reason to change geometry. '
            'Synthetic ToF tests lifecycle separately, never obstacle-avoidance admission.',
        deployment_admitted=False)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    for name in hashes:
        p=Path(name)
        if not p.is_absolute() and p.suffix=='.py':
            dest=out/'source_snapshot'/p;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    results=[]
    for case,spec in scenes().items():
        for variant in ('control','smooth'):
            folder=out/(case+'_'+variant)
            trial(folder,source,case,'sitl','reference_units',physics_dt_s=.00025,integration_profile='multi_rate',
                  yaw_shaping=variant=='smooth',scene_override=spec,
                  tof_probe=LifecycleToF() if case=='lifecycle' else None)
            r=assess(folder,case)|dict(variant=variant);results.append(r)
            (out/'comparison.json').write_text(json.dumps(results,indent=2)+'\n')
            print(json.dumps({k:v for k,v in r.items() if k not in ('phase_events','recoveries')}),flush=True)
    validation=dict(sources_unchanged=all(digest(p)==v for p,v in hashes.items()),
        clocks_passed=all(r['clock_passed'] for r in results),
        accepted_equals_issued=all(r['accepted_equals_issued'] for r in results),
        passed={r['case']+'_'+r['variant']:r['passed'] for r in results},deployment_admitted=False)
    (out/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    render(out,results);print(json.dumps(validation,indent=2),flush=True)
    if not validation['sources_unchanged'] or not validation['clocks_passed'] or any(r['error'] for r in results):
        raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
    a=p.parse_args();run(a.out,a.source,a.reference)
