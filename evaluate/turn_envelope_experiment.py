"""One fixed paired revision of the 45-case uncertainty series. No tuning."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
from pathlib import Path
import shutil

import numpy as np

from edge.runtime import sha256
from evaluate.sim_uncertainty import ROOT, SOURCE, profiles, scenarios, trial, write
from simulation.phased_coordinator import PhasedConfig
from simulation.turn_envelope import TurnEnvelope

OLD=ROOT/'output/sim_uncertainty_v1'


def paired_trial(out,case,name,profile,variant):
    envelope=TurnEnvelope() if variant=='envelope' else None
    r=trial(out/variant,case,name,profile,turn_envelope=envelope)
    exact=None
    if variant=='control':
        for filename in ('physical.npz','trace.npz'):
            with np.load(OLD/r['id']/filename) as old, np.load(out/variant/r['id']/filename) as new:
                if set(old.files)!=set(new.files):
                    raise AssertionError('Trace schema changed')
                for key in old.files:
                    np.testing.assert_array_equal(new[key],old[key],err_msg=r['id']+':'+key)
        old_report=json.loads((OLD/r['id']/'report.json').read_text())
        if old_report['metrics']!=r['metrics'] or old_report['admission']!=r['admission']:
            raise AssertionError('Control metrics differ from frozen previous experiment')
        exact=True
    r.update(variant=variant,previous_control_exact=exact)
    write(out/variant/r['id']/'report.json',r)
    return r


def make_report(out,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    pairs=[]
    for case in scenarios():
        for profile in profiles():
            a=next(r for r in rows if r['case']==case['name'] and r['profile']==profile and r['variant']=='control')
            b=next(r for r in rows if r['case']==case['name'] and r['profile']==profile and r['variant']=='envelope')
            pairs.append(dict(case=case['name'],profile=profile,
                before_passed=a['admission']['passed'],after_passed=b['admission']['passed'],
                before_failed=a['admission']['failed'],after_failed=b['admission']['failed'],
                new_contact=a['metrics']['collision'] is None and b['metrics']['collision'] is not None,
                before_clearance_m=a['metrics']['min_clearance_m'],after_clearance_m=b['metrics']['min_clearance_m'],
                before_timeout=a['metrics']['timeout_events'],after_timeout=b['metrics']['timeout_events'],
                before_distance_m=a['metrics']['distance_m'],after_distance_m=b['metrics']['distance_m']))
    write(out/'pairs.json',pairs)
    lines=['# Bounded turn envelope: paired sensitivity experiment','',
        'Same frozen network; 224 deg commanded budget; assumed yaw floor 140 deg/s full scale.',
        'TURN allowance 5.714286 s, hard cap 6 s, whole episode 8 s. Not measured angular progress.','',
        '| Case | Profile | Passed before/after | Clearance before/after cm | Timeout before/after | After failures |',
        '|---|---|---|---:|---|---|']
    for p in pairs:
        lines.append(f'| {p["case"]} | {p["profile"]} | {p["before_passed"]}/{p["after_passed"]} | '
            f'{p["before_clearance_m"]*100:.2f}/{p["after_clearance_m"]*100:.2f} | '
            f'{p["before_timeout"]}/{p["after_timeout"]} | {", ".join(p["after_failed"]) or "—"} |')
    (out/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    fig,axes=plt.subplots(2,5,figsize=(20,9),layout='constrained')
    for j,case in enumerate(scenarios()):
        for i,profile in enumerate(('yaw_70pct','combined')):
            ax=axes[i,j]
            for box in case['boxes']:
                ax.add_patch(Rectangle((box['x_min'],box['y_min']),box['x_max']-box['x_min'],box['y_max']-box['y_min'],color='.85'))
            for variant in ('control','envelope'):
                with np.load(out/variant/(case['name']+'__'+profile)/'physical.npz') as data:
                    line,=ax.plot(*data['pos'][:,:2].T,label=variant)
                    ax.scatter(*data['pos'][int(data['clearance'].argmin()),:2],marker='x',color=line.get_color())
            ax.set(xlim=(-4,4),ylim=(-4,4),title=case['name']+' / '+profile)
            ax.set_aspect('equal');ax.legend(fontsize=8)
    fig.savefig(out/'comparison.png',dpi=130);plt.close(fig)
    write(out/'validation.json',dict(completed_trials=len(rows),inputs_unchanged=True,
        controls_exact=sum(r['previous_control_exact'] is True for r in rows),
        passed_before=sum(p['before_passed'] for p in pairs),passed_after=sum(p['after_passed'] for p in pairs),
        new_contacts=sum(p['new_contact'] for p in pairs),
        regressions=[p for p in pairs if p['before_passed'] and not p['after_passed']],
        fixed=[p for p in pairs if not p['before_passed'] and p['after_passed']],
        profiles={name:dict(before=sum(p['before_passed'] for p in pairs if p['profile']==name),
            after=sum(p['after_passed'] for p in pairs if p['profile']==name)) for name in profiles()},
        hardware_admission=False,default_changed=False))


def run(out,workers):
    out.mkdir(parents=True,exist_ok=False)
    for variant in ('control','envelope'): (out/variant).mkdir()
    prior=json.loads((OLD/'protocol.json').read_text())
    files=[SOURCE/'checkpoint.pt',SOURCE/'runtime_config.json',
        *[p for p in (SOURCE/'numpy').iterdir() if p.is_file()],
        *[p for p in (ROOT/'data/edge/k306_tau025_integrated_v1').iterdir() if p.is_file()],
        OLD/'protocol.json',OLD/'report.json',
        *[p for folder in ('simulation','demo','edge','evaluate','configs','core') for p in (ROOT/folder).rglob('*.py')]]
    # Verify model packages against previous frozen experiment BEFORE starting.
    for p in files:
        if p.suffix=='.py' or p.parent==OLD: continue
        key=str(p.relative_to(ROOT))
        if key not in prior['input_sha256'] or sha256(p)!=prior['input_sha256'][key]:
            raise ValueError('Frozen model/config changed since control: '+key)
    for case in scenarios():
        for name in profiles():
            files += [OLD/(case['name']+'__'+name)/f for f in ('physical.npz','trace.npz','report.json')]
    hashes={str(p.relative_to(ROOT)):sha256(p) for p in files}
    for relative in hashes:
        target=out/'frozen_inputs'/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/relative,target)
    tasks=[(c,n,p,v) for c in scenarios() for n,p in profiles().items() for v in ('control','envelope')]
    write(out/'protocol.json',dict(version='turn-envelope-paired-v1',trials=len(tasks),
        scenarios=scenarios(),profiles={n:asdict(p) for n,p in profiles().items()},
        control='Unmodified phased v2; all 45 traces/metrics must exactly reproduce previous series',
        revision=TurnEnvelope().contract(PhasedConfig()),
        criteria=prior['criteria'],horizon_s=45.,input_sha256=hashes,
        assumption='Same envelope for ALL plant profiles; never infer true yaw gain from environment',
        search=False,weights_changed=False,hardware_promotion=False,
        decision='Report paired regressions, contacts and failures; do not tune or promote automatically'))
    rows=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(paired_trial,out,*t) for t in tasks]
        for f in as_completed(futures):
            rows.append(f.result())
            rows.sort(key=lambda r:next(i for i,(c,n,p,v) in enumerate(tasks)
                if (r['case'],r['profile'],r['variant'])==(c['name'],n,v)))
            write(out/'report.json',rows)
    for relative,h in hashes.items():
        if sha256(ROOT/relative)!=h or sha256(out/'frozen_inputs'/relative)!=h:
            raise RuntimeError('Experiment inputs changed: '+relative)
    make_report(out,rows)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--workers',type=int,choices=(1,2,3),default=3)
    args=parser.parse_args()
    run(args.out,args.workers)
