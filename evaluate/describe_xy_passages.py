"""Descriptive XY gate crossings; never changes frozen start-validation admission."""
import argparse
import json
import math
from pathlib import Path
import numpy as np


def crossing_events(physics,a,b):
    a,b=np.asarray(a,float),np.asarray(b,float)
    segment=b-a
    def cross(u,v): return float(u[0]*v[1]-u[1]*v[0])
    events=[]
    for prev,row in zip(physics,physics[1:]):
        p=np.asarray(prev['position_m'][:2],float); q=np.asarray(row['position_m'][:2],float)
        motion=q-p; denominator=cross(motion,segment)
        if abs(denominator)<1e-12: continue
        fraction=cross(a-p,segment)/denominator
        along=cross(a-p,motion)/denominator
        if 0<fraction<=1 and 0<=along<=1:
            time=prev['time_s']+fraction*(row['time_s']-prev['time_s'])
            if time>=35.: continue
            events.append(dict(time_s=time,xy_m=(p+fraction*motion).tolist()))
    return events


def describe(out_dir):
    # Segments connect the two neighbour centres, trimmed by body-expanded
    # geometry: .35/.45 m radial offsets for NE cylinders; .4 m XY for SW box.
    root=math.sqrt(2)
    gates=dict(gap_ne=([2.3-.35/root,.35/root],[.45/root,2.3-.45/root]),
               gap_sw=([-1.9,-.4],[-.45/root,-2.3+.45/root]))
    comparison=json.loads((out_dir/'comparison.json').read_text(encoding='utf-8'))
    rows=[]
    for row in comparison['runs']:
        name=row['start']['name']
        if name not in gates: continue
        report=json.loads((out_dir/row['report_file']).read_text(encoding='utf-8'))
        events=crossing_events(report['trace']['physics'],*gates[name])
        rows.append(dict(start=name,seed=row['seed'],crossed=bool(events),events=events))
    result=dict(definition='Centre XY crosses the body-trimmed segment between the two named neighbours before STOP; descriptive, not navigation success',
        gates=gates,runs=rows,admission_affected=False,
        limitation='A crossing is not a completed route; lack of crossing is not an avoidance failure. Added as descriptive analysis, not a primary predeclared criterion.')
    (out_dir/'passages.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir',type=Path,required=True)
    describe(parser.parse_args().out_dir)
