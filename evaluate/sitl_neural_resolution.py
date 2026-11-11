"""Four fixed closed-loop repeats at 1 ms; no fitting and no promotion."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np

from evaluate.sitl_neural import trial,CASES


def run(out,source,reference):
    from evaluate.sitl_lockstep import BIN
    expected=json.loads((source/'report.json').read_text())['firmware_sha256']
    if hashlib.sha256(BIN.read_bytes()).hexdigest()!=expected:raise ValueError('Firmware mismatch')
    old_protocol=json.loads((reference/'protocol.json').read_text())
    # All original model/config/shared-code inputs must still match. Only the
    # experiment runner itself gained a smaller physical step after that run.
    allowed={'evaluate/sitl_neural.py'}
    for name,digest in old_protocol['input_source_sha256'].items():
        if name not in allowed and hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Changed original experiment input: '+name)
    out.mkdir(parents=True,exist_ok=False)
    sources={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in
             (*old_protocol['input_source_sha256'],'evaluate/sitl_neural_resolution.py')}
    (out/'protocol.json').write_text(json.dumps(dict(cases=CASES,physics_dt_s=.001,
        axis_contract='reference_units',reference=str(reference),firmware_sha256=expected,
        input_source_sha256=sources,limits=dict(position_m=.02,angle_deg=1.,velocity_mps=.05),
        note='Same predeclared numerical tolerances; failures reported without retuning.'),indent=2)+'\n')
    for name in allowed|{'evaluate/sitl_neural_resolution.py'}:
        dest=out/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(name,dest)
    results=[]
    for case in CASES:
        r=trial(out/(case+'_sitl'),source,case,'sitl','reference_units',physics_dt_s=.001)
        with np.load(reference/(case+'_sitl')/'trace.npz') as coarse,np.load(out/(case+'_sitl')/'trace.npz') as fine:
            if len(coarse['state'])==5001 and len(fine['state'])==20001:
                diff=coarse['state']-fine['state'][::4]
                delta=dict(position_m=float(np.linalg.norm(diff[:,:3],axis=1).max()),
                    velocity_mps=float(np.linalg.norm(diff[:,3:6],axis=1).max()),
                    angle_deg=float(np.degrees(abs(np.arctan2(np.sin(diff[:,6:9]),np.cos(diff[:,6:9])))).max()))
                delta['within_tolerances']=delta['position_m']<=.02 and delta['velocity_mps']<=.05 and delta['angle_deg']<=1.
            else:delta=dict(within_tolerances=False,reason='Incomplete horizon')
        r['comparison_to_4ms']=delta;results.append(r)
        (out/'report.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(r),flush=True)
    unchanged=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==digest for p,digest in sources.items())
    (out/'validation.json').write_text(json.dumps(dict(sources_unchanged=unchanged,
        all_within_numerical_tolerances=all(r['comparison_to_4ms']['within_tolerances'] for r in results),
        completed=sum(r['completed'] for r in results),contacts=sum(r['collision'] is not None for r in results),
        stop_passed=sum(r['stop_passed'] for r in results),deployment_admitted=False),indent=2)+'\n')
    if not unchanged or any(r['error'] for r in results):raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    a=p.parse_args();run(a.out,a.source,a.reference)
