"""Two preregistered fine-step repeats of the known strong-yaw neural scene."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np


def run(out,source,reference):
    from evaluate.sitl_neural import trial
    from evaluate.sitl_lockstep import BIN
    expected=json.loads((source/'report.json').read_text())['firmware_sha256']
    if hashlib.sha256(BIN.read_bytes()).hexdigest()!=expected:raise ValueError('Firmware mismatch')
    previous=json.loads((reference/'protocol.json').read_text())
    allowed={'evaluate/sitl_neural.py','evaluate/sitl_lockstep.py','evaluate/ready_sim_spike.py'}
    for name,digest in previous['input_source_sha256'].items():
        if name not in allowed and hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Changed controller/config/source: '+name)
    hashes={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in
            (*previous['input_source_sha256'],__file__)}
    out.mkdir(parents=True,exist_ok=False)
    (out/'protocol.json').write_text(json.dumps(dict(case='obstacle_left',steps_us=[500,250],horizon_s=20.,
        axis_contract='reference_units',firmware_sha256=expected,input_source_sha256=hashes,
        limits=dict(position_m=.02,velocity_mps=.05,angle_deg=1.),
        acceptance='Report contacts, STOP, timeout, height and fixed numerical thresholds; no fitting.'),indent=2)+'\n')
    for name in allowed|{'evaluate/sitl_yaw_closed_loop.py'}:
        dest=out/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(name,dest)
    results=[];traces={}
    for us in (500,250):
        folder=out/f'us{us}'
        r=trial(folder,source,'obstacle_left','sitl','reference_units',physics_dt_s=us/1e6)
        r['step_us']=us;results.append(r)
        with np.load(folder/'trace.npz') as a:traces[us]={k:a[k] for k in a.files}
        (out/'report.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(r),flush=True)
    diff=traces[500]['state']-traces[250]['state'][::2]
    comparison=dict(max_position_difference_m=float(np.linalg.norm(diff[:,:3],axis=1).max()),
        max_velocity_difference_mps=float(np.linalg.norm(diff[:,3:6],axis=1).max()),
        max_angle_difference_deg=float(np.degrees(abs(np.arctan2(np.sin(diff[:,6:9]),np.cos(diff[:,6:9])))).max()))
    comparison['within_tolerances']=(comparison['max_position_difference_m']<=.02
        and comparison['max_velocity_difference_mps']<=.05 and comparison['max_angle_difference_deg']<=1.)
    clocks={}
    for us,a in traces.items():
        n=len(a['motors']);ack=a['acks'][-n:]
        clocks[us]=bool((np.diff(ack[:,0])==1).all() and (np.diff(ack[:,1])==us).all() and (ack[:,2]==us//125).all())
    unchanged=all(hashlib.sha256(Path(k).read_bytes()).hexdigest()==v for k,v in hashes.items())
    validation=dict(comparison=comparison,clocks=clocks,sources_unchanged=unchanged,
        completed=sum(r['completed'] for r in results),contacts=sum(r['collision'] is not None for r in results),
        stop_passed=sum(r['stop_passed'] for r in results),deployment_admitted=False)
    (out/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(10,10),constrained_layout=True)
    for us,a in traces.items():
        t,s=a['time_s'],a['state']
        for ax,value in zip(axes,(100*(s[:,2]-1),np.degrees(s[:,8]),np.linalg.norm(s[:,3:5],axis=1))):
            ax.plot(t,value,label=f'{us} us');ax.grid(alpha=.25);ax.legend()
    for ax,title in zip(axes,('Height error (cm)','Yaw (deg)','Horizontal speed (m/s)')):
        ax.set(title=title,xlabel='Model time (s)');ax.axvspan(17,20,color='.5',alpha=.1)
    fig.suptitle('Unchanged neural/XY/altitude controllers; obstacle_left; fine physical steps')
    fig.savefig(out/'comparison.png',dpi=140);plt.close(fig)
    print(json.dumps(validation,indent=2))
    if not unchanged or any(r['error'] for r in results):raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    a=p.parse_args();run(a.out,a.source,a.reference)
