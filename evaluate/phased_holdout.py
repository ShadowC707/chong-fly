"""Fixed geometric holdout for phased v2. No parameter fitting."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import math
import json
import numpy as np
from evaluate.phased_maneuver import trial,write,ROOT,SOURCE
from edge.runtime import sha256


def box(a,b,c,d):return dict(x_min=a,x_max=b,y_min=c,y_max=d,z_min=0.,z_max=2.5)
def cylinder(x,y,r=.3):return dict(center_x=x,center_y=y,radius=r,height=2.5)

def cases():
    stagger=[box(-.2,.2,-1.5,.1),box(1.3,1.7,.1,1.5)]
    mirror=[box(b['x_min'],b['x_max'],-b['y_max'],-b['y_min']) for b in stagger]
    pocket=[box(.5,.7,-1.2,1.2),box(-.5,.7,1.2,1.4),box(-.5,.7,-1.4,-1.2)]
    specs=[
        ('wall_normal',(1.5,0),0,[],[]),
        ('room_corner',(1.5,1.5),45,[],[]),
        ('wall_glancing',(1.5,-1.5),60,[],[]),
        ('box_front',(-2,0),0,[box(-.4,.4,-.4,.4)],[]),
        ('box_grazing',(-2,.6),0,[box(-.4,.4,-.4,.4)],[]),
        ('cylinder_front',(-2,0),0,[],[cylinder(0,0)]),
        ('cylinder_oblique',(-2,-.6),15,[],[cylinder(0,0)]),
        ('gap_1p3',(-2,0),0,[box(-.3,.3,-2,-.65),box(-.3,.3,.65,2)],[]),
        ('stagger_left',(-2,0),0,stagger,[]),
        ('stagger_right',(-2,0),0,mirror,[]),
        ('dead_end',(-2,0),0,pocket,[]),
        ('column_group',(-2,-.65),0,[],[cylinder(x,y,.25) for x in (0,1.5) for y in (-1.3,0,1.3)])]
    return [dict(name=name,seed=20261107,start=[*start,1.],yaw=math.radians(yaw),boxes=b,cylinders=c) for name,start,yaw,b,c in specs]


def longest(mask,dt):
    best=run=0.
    for yes,seconds in zip(mask,dt):
        run=run+seconds if yes else 0.;best=max(best,run)
    return float(best)


def admission(m):
    rules={'contact':not m['collision'],'horizon':m['completed'],
        'clearance_5cm':m['min_clearance_m']>=.05,
        'near_10cm_episode':m['near_10cm_longest_s']<=1.,
        'near_10cm_total':m['below_10cm_s']<=2.,
        'no_timeouts':m['timeout_events']==0,
        'no_stall':m['stationary_longest_s']<=5.,
        'motion':m['distance_m']>=4.,'auto_exposure':m['auto_s']>=40.,
        'height':m['max_height_error_m']<=.1}
    return dict(passed=all(rules.values()),checks=rules)


def evaluate(out,task):
    r=trial(out,task)
    with np.load(out/task['id']/'physical.npz') as a:
        dt=np.diff(a['time'],append=r['metrics']['duration_s'])
        speed=np.linalg.norm(a['velocity'][:,:2],axis=1)
        r['metrics']['stationary_longest_s']=longest(a['auto']&(speed<=.03),dt)
        r['metrics']['near_10cm_longest_s']=longest(a['auto']&(a['clearance']<.1),dt)
    r['admission']=admission(r['metrics']);write(out/task['id']/'report.json',r)
    return r


def report(out,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle,Circle
    fig,axes=plt.subplots(3,4,figsize=(17,13),layout='constrained')
    lines=['# Phased v2: нові фіксовані сценарії','',
        '| Сценарій | Режим | Контакт | Зазор см | <10 см с | Найдовша зупинка с | Шлях м | Допуск |',
        '|---|---|---|---:|---:|---:|---:|---|']
    for ax,c in zip(axes.flat,cases()):
        for b in c['boxes']:ax.add_patch(Rectangle((b['x_min'],b['y_min']),b['x_max']-b['x_min'],b['y_max']-b['y_min'],color='.85'))
        for b in c['cylinders']:ax.add_patch(Circle((b['center_x'],b['center_y']),b['radius'],color='.85'))
        for r in [x for x in rows if x['case']==c['name']]:
            m=r['metrics']
            with np.load(out/r['id']/'physical.npz') as a:
                line,=ax.plot(*a['pos'][:,:2].T,label=r['mode'])
                ax.scatter(*a['pos'][int(a['clearance'].argmin()),:2],marker='x',color=line.get_color())
                ax.scatter(*a['pos'][0,:2],color='green',s=15)
            fail=','.join(k for k,v in r['admission']['checks'].items() if not v)
            lines.append(f'| {c["name"]} | {r["mode"]} | {m["collision"] or "—"} | {100*m["min_clearance_m"]:.2f} | {m["below_10cm_s"]:.2f} | {m["stationary_longest_s"]:.2f} | {m["distance_m"]:.2f} | {fail or "passed"} |')
        ax.set(xlim=(-4,4),ylim=(-4,4),title=c['name']);ax.set_aspect('equal');ax.legend(fontsize=7)
    fig.savefig(out/'comparison.png',dpi=140);plt.close(fig)
    (out/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def run(out):
    out.mkdir(parents=True,exist_ok=False)
    tasks=[dict(id=c['name']+'_'+mode,case=c['name'],offset=0,mode=mode,scene_spec=c) for c in cases() for mode in ('baseline','phased')]
    files=[SOURCE/'checkpoint.pt',SOURCE/'runtime_config.json',*[p for p in (SOURCE/'numpy').iterdir() if p.is_file()],
        *[p for p in (ROOT/'data/edge/k306_tau025_integrated_v1').iterdir() if p.is_file()],
        ROOT/'simulation/phased_coordinator.py',ROOT/'evaluate/phased_maneuver.py',Path(__file__)]
    hashes={str(p.resolve()):sha256(p) for p in files}
    write(out/'protocol.json',dict(version='phased-v2-geometric-holdout-v1',tasks=tasks,input_sha256=hashes,
        horizon_s=45,collision_radius_m=.2,parameter_search=False,
        criteria=dict(min_clearance_m=.05,near_10cm_longest_s=1.,near_10cm_total_s=2.,stationary_longest_s=5.,
            min_distance_m=4.,min_auto_s=40.,max_height_error_m=.1,no_contact=True,no_timeout=True),
        meaning='New explicit geometries and start headings, same neural model/sensor physics; not sim2real.',
        promotion='Simulator opt-in only if ALL 12 phased cases pass; no automatic hardware promotion.'))
    snap=out/'executed_source';snap.mkdir()
    for p in files[-3:]:(snap/p.name).write_bytes(p.read_bytes())
    rows=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(evaluate,out,t) for t in tasks]
        for f in as_completed(futures):
            rows.append(f.result());rows.sort(key=lambda r:next(i for i,t in enumerate(tasks) if t['id']==r['id']))
            write(out/'report.json',rows)
    assert all(sha256(p)==h for p,h in hashes.items())
    phased=[r for r in rows if r['mode']=='phased']
    write(out/'validation.json',dict(trials=len(rows),inputs_unchanged=True,
        phased_passed=sum(r['admission']['passed'] for r in phased),
        simulator_admission=all(r['admission']['passed'] for r in phased)))
    report(out,rows)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.out)
