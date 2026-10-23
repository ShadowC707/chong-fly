"""Audit saved behavior traces; optionally replay one run to verify telemetry-only edits."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from evaluate.compare_xy_behaviors import summarize
from evaluate.watch_mvp import run_view, ROOT
from generator.source_contract import file_sha256
from simulation.xy_coordinator import BehaviorConfig, XYConfig


def pulse_measurements(report):
    physics, controls = report['trace']['physics'], report['trace']['control']
    t = np.array([r['time_s'] for r in physics])
    yaw = np.unwrap([r['yaw_rad'] for r in physics])
    rows=[]
    for c in controls:
        if c['coordinator'].get('search_event') != 'started':
            continue
        s=c['time_s']
        rows.append(dict(start_s=s,
            yaw_change_at_028s_deg=float(np.rad2deg(np.interp(s+.28,t,yaw)-np.interp(s,t,yaw))),
            yaw_change_at_080s_deg=float(np.rad2deg(np.interp(s+.8,t,yaw)-np.interp(s,t,yaw))),
            reflex_samples_in_080s=sum(abs(r['network_pwm'][3]-1500)>=25
                                      for r in controls if s<=r['time_s']<s+.8)))
    return rows


def wall_diagnostics(report):
    p=report['trace']['physics']; c=report['trace']['control']
    t=np.array([r['time_s'] for r in p]); ct=np.array([r['time_s'] for r in c])
    pos=np.array([r['position_m'] for r in p]); yaw=np.array([r['yaw_rad'] for r in p])
    body=np.array([r['body_velocity_mps'] for r in p])
    world_vy=body[:,0]*np.sin(yaw)+body[:,1]*np.cos(yaw)
    x=np.interp(ct,t,pos[:,0]); y=np.interp(ct,t,pos[:,1]); vy=np.interp(ct,t,world_vy)
    mask=(x>3.1)&(y> -2.8)&(y<2.5)&(vy>.05)&(ct<35)
    selected=[row for row,keep in zip(c,mask) if keep]
    blockers=Counter(reason for r in selected for reason in r['coordinator']['search_blockers'])
    minimum=np.array([r['min_tof_m'] for r in selected])
    center=np.array([r['coordinator']['tof_center_min_m'] for r in selected])
    latch=np.array([r['coordinator']['turn_sign']!=0 for r in selected])
    reflex=np.array([abs(r['network_pwm'][3]-1500)>=25 for r in selected])
    return dict(selection='x>3.1, -2.8<y<2.5, world_vy>0.05 m/s, t<35 s',
        samples=len(selected), time_s=len(selected)*.02,
        range_min_m=float(minimum.min()) if len(minimum) else None,
        range_max_m=float(minimum.max()) if len(minimum) else None,
        blocker_time_s={k:v*.02 for k,v in blockers.items()},
        blockers_overlap=True,
        current_eligible_s=float(np.sum((minimum>=1.15)&~latch&~reflex)*.02),
        central_corridor_eligible_preserving_095_preemption_s=float(
            np.sum((center>=1.15)&(minimum>=.95)&~latch&~reflex)*.02),
        central_corridor_eligible_if_lowered_to_085_s=float(
            np.sum((center>=1.15)&(minimum>=.85)&~latch&~reflex)*.02),
        counterfactual='Eligibility on saved inputs only; not a closed-loop forecast or safety finding.')


def audit(source, output, *, replay=False):
    if output.exists():
        raise FileExistsError('Use a new audit directory')
    reports={p.stem:json.loads(p.read_text(encoding='utf-8')) for p in sorted(source.glob('arena_*.json'))}
    if not reports:
        raise ValueError('No saved arena traces')
    rows=[]
    for name,r in reports.items():
        summary=summarize(r)
        physics=r['trace']['physics']
        t=np.array([x['time_s'] for x in physics])
        speed=np.linalg.norm([x['body_velocity_mps'][:2] for x in physics],axis=1)
        later=np.flatnonzero((t>=35)&(speed>=.03))
        settle=(float(t[later[-1]+1]-35) if len(later) and later[-1]<len(t)-1 else
                0. if np.all(speed[t>=35]<.03) else None)
        summary.update(name=name,settled_below_003_for_rest_of_trace_s=settle,
                       pulses=pulse_measurements(r))
        rows.append(summary)
    baselines=[r for name,r in reports.items() if name.endswith('_control')]
    positions=[np.array([x['position_m'] for x in r['trace']['physics']]) for r in baselines]
    same=all(np.array_equal(positions[0],p) for p in positions[1:]) if positions else None
    result=dict(version='xy-behavior-audit-v1',sources={p.name:file_sha256(p)
        for p in sorted(source.glob('arena_*.json'))},runs=rows,baseline_trajectories_identical=same,
        ideal_yaw_setpoint_integrals_deg={f'{a}pwm_{d}s':a/500*200*d/2
            for a,d in [(110,.28),(220,.55),(250,.6)]},
        angle_caveat='Integrals of requested yaw rate, not guaranteed physical heading changes.',
        hardware_admission=False)
    output.mkdir(parents=True)
    if replay:
        old=reports['arena_sparse_levy']
        contract=old['metrics']['coordinator']['contract']
        new=run_view(old['run_dir'],scene=old['scene'],steps=old['requested_control_steps'],
            plot_out=output/'sparse_telemetry_replay.png',seed=old['seed'],coordinator_mode='both',
            levy_explore=True,stop_at_s=old['stop_at_s'],
            coordinator_config=XYConfig(**contract['config']),
            behavior_config=BehaviorConfig(**contract['behavior_config']))
        checks={}
        for kind,field in [('control','network_pwm'),('control','applied_pwm'),
                           ('physics','position_m'),('physics','yaw_rad')]:
            a=np.array([r[field] for r in old['trace'][kind]])
            b=np.array([r[field] for r in new['trace'][kind]])
            checks[field]=bool(np.array_equal(a,b))
        if not all(checks.values()):
            raise AssertionError(f'Telemetry replay changed behavior: {checks}')
        result['unchanged_behavior_replay']=checks
        result['wall_diagnostics']=wall_diagnostics(new)
    result['code_sha256']={name:file_sha256(ROOT/name) for name in (
        'evaluate/audit_xy_behaviors.py','evaluate/compare_xy_behaviors.py','simulation/xy_coordinator.py')}
    (output/'audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('data/diagnostics/xy_behaviors_density_v1'))
    parser.add_argument('--out-dir',type=Path,required=True)
    parser.add_argument('--replay',action='store_true')
    args=parser.parse_args()
    if args.replay:
        import torch
        torch.set_num_threads(1)
    result=audit(args.source,args.out_dir,replay=args.replay)
    print(json.dumps({k:v for k,v in result.items() if k not in ('runs','sources','code_sha256')},indent=2))


if __name__=='__main__':
    main()
