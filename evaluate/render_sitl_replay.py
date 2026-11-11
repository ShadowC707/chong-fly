"""Offline visualization of recorded SITL states. Never reruns flight physics."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

TEMPLATE=Path(__file__).with_name('sitl_replay.html')


def replay_data(folder):
    folder=Path(folder)
    load=lambda name:json.loads((folder/(name+'.json')).read_text(encoding='utf-8'))
    manifest,report,rows,decisions=[load(n) for n in ('manifest','report','command_trace','decisions')]
    with np.load(folder/'trace.npz',allow_pickle=False) as a:
        t,s,ct=a['time_s'],a['state'],a['control_time']
        if (len(t)<2 or s.shape!=(len(t),12) or not np.isfinite(s).all()
                or not np.isfinite(t).all() or not (np.diff(t)>0).all()
                or len(ct)==0 or not (np.diff(ct)>0).all()):
            raise ValueError('Invalid physics/control trace')
        if (len(rows)!=len(ct) or len(decisions)!=len(ct)
                or not np.allclose([r['tick_time_s'] for r in rows],ct,atol=1e-9,rtol=0)
                or not np.allclose([d['time_s'] for d in decisions],ct,atol=1e-9,rtol=0)):
            raise ValueError('Misaligned command/decision times')
        if (a['tof'].shape!=(len(ct),64) or a['flow'].shape!=(len(ct),2)
                or a['pwm'].shape!=(len(t)-1,4) or len(a['clearance'])!=len(t)-1):
            raise ValueError('Misaligned sensor/physics arrays')
        indices=np.searchsorted(t,ct-1e-9)
        if np.any(indices>=len(t)) or not np.allclose(t[indices],ct,atol=1e-9,rtol=0):
            raise ValueError('Control time has no recorded physics state')
        indices=np.unique(np.r_[indices,len(t)-1])
        frames=[]
        for i in indices:
            j=int(np.searchsorted(ct,t[i]+1e-9,side='right')-1)
            if j<0:raise ValueError('No preceding command')
            row=rows[j]
            capture=row['sensor_channels']['tof']['timestamp_s']
            k=int(np.searchsorted(ct,capture-1e-9))
            if k>j or k>=len(ct) or abs(ct[k]-capture)>1e-9:
                raise ValueError('Held ToF capture has no recorded sample')
            frames.append(dict(t=float(t[i]),p=s[i,:3].tolist(),a=s[i,6:9].tolist(),
                speed=float(np.linalg.norm(s[i,3:5])),clearance=None if i==0 else float(a['clearance'][i-1]),
                flow=a['flow'][j].tolist(),tof=a['tof'][k].tolist(),tof_age_s=float(t[i]-capture),
                phase=decisions[j]['phase'],source=row['selected']['source'],
                actual_rc=a['pwm'][min(i,len(t)-2)].tolist(),trace=row))
    names=['trace.npz','manifest.json','report.json','command_trace.json','decisions.json']
    hashes={name:hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in names}
    return dict(schema='sitl-recorded-replay-v1',manifest=manifest,report=report,frames=frames,
        source_sha256=hashes,display='Recorded poses at control ticks plus terminal pose; no integration/interpolation',
        sensor_display='ToF from recorded capture timestamp, held between samples; no fresh-truth substitution',
        metrics='Report uses full 250 us physics, not display decimation')


def html_document(data):
    payload=json.dumps(data,ensure_ascii=False,allow_nan=False,separators=(',',':')).replace('<','\\u003c')
    return TEMPLATE.read_text(encoding='utf-8').replace('/*REPLAY_DATA*/null',payload)


def render_summary(folder,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle,Circle
    manifest=json.loads((folder/'manifest.json').read_text())
    report=json.loads((folder/'report.json').read_text())
    rows=json.loads((folder/'command_trace.json').read_text())
    spec=manifest['scene']
    fig,axes=plt.subplots(2,2,figsize=(13,10),layout='constrained')
    ax=axes[0,0]
    for b in spec['boxes']:
        ax.add_patch(Rectangle((b['x_min'],b['y_min']),b['x_max']-b['x_min'],b['y_max']-b['y_min'],color='.65'))
    for c in spec['cylinders']:ax.add_patch(Circle((c['center_x'],c['center_y']),c['radius'],color='.65'))
    with np.load(folder/'trace.npz') as a:
        t,s,ct=a['time_s'],a['state'],a['control_time']
        ax.plot(s[:,0],s[:,1],lw=1.4);ax.scatter(s[0,0],s[0,1],c='green',label='start')
        ax.scatter(s[-1,0],s[-1,1],c='red',marker='x',label='end')
        for second in range(5,int(t[-1])+1,5):
            i=np.searchsorted(t,second);ax.annotate(str(second)+'s',s[i,:2],fontsize=8)
        room=spec['room'];ax.set(xlim=(room['x_min'],room['x_max']),ylim=(room['y_min'],room['y_max']),
            xlabel='X (m)',ylabel='Y (m)',title='Recorded path / fixed geometry');ax.set_aspect('equal');ax.legend()
        axes[0,1].plot(t,s[:,2],label='height');axes[0,1].axhline(1,color='.4',ls='--',label='target')
        axes[0,1].set(title='Height (m)');axes[0,1].legend()
        axes[1,0].plot(t,np.linalg.norm(s[:,3:5],axis=1),label='XY speed (m/s)')
        axes[1,0].plot(t[1:],a['clearance'],label='body clearance (m)')
        axes[1,0].axhline(.1,color='red',ls=':',label='clearance criterion .10m');axes[1,0].legend()
        for key in ('network_pwm','coordinated_pwm','accepted_pwm'):
            values=[np.nan if r.get(key) is None else r[key][3] for r in rows]
            axes[1,1].step(ct,values,where='post',label=key)
        axes[1,1].set(title='Yaw reference PWM (not actual FC curve)');axes[1,1].legend(fontsize=8)
        for a in (axes[0,1],axes[1,0],axes[1,1]):
            a.axvspan(manifest['stop_at_s'],manifest['horizon_s'],alpha=.12,color='grey')
            a.set(xlim=(0,manifest['horizon_s']),xlabel='time (s)');a.grid(alpha=.2)
    fig.suptitle(f"Candidate A + smooth phases | SITL / TWR5 | {report['duration_s']:.3f}s recorded\n"
        f"contact={report['collision']} | clearance={report['min_clearance_m']*100:.2f}cm | STOP={report['stop_passed']}")
    fig.savefig(out,dpi=140);plt.close(fig)


def render(folder,out):
    data=replay_data(folder)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(html_document(data),encoding='utf-8')
    render_summary(folder,out.with_suffix('.png'))
    audit=dict(source_sha256=data['source_sha256'],renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        template_sha256=hashlib.sha256(TEMPLATE.read_bytes()).hexdigest(),
        output_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),frames=len(data['frames']),
        terminal_time_s=data['frames'][-1]['t'],physics_rerun=False)
    out.with_suffix('.audit.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(audit,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();render(a.run,a.out)
