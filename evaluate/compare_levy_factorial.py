"""Predeclared 3x4 Levy experiment; no training or adaptive parameter search."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
import torch

from evaluate.compare_xy_behaviors import summarize, plot_grid
from evaluate.watch_mvp import DEFAULT_RUN, ROOT, run_view
from generator.source_contract import file_sha256
from simulation.xy_coordinator import BehaviorConfig, XYConfig
from simulation.xy_scenes import DENSITY_SCENES


def variant_configs():
    base=BehaviorConfig()
    directed=replace(base,search_amplitude_pwm=240.,search_duration_s=.60,target_half_gate=True)
    return dict(levy_v1=base,pause_only=replace(base,pause_search_wait=True),
                directed_turn=directed,both=replace(directed,pause_search_wait=True))


def interior_metrics(physics, *, stop_at_s):
    """Right-endpoint occupancy, weighted by physics duration; STOP excluded.

    Report post-wall occupancy too: starting in the centre is not wall escape.
    """
    t=np.array([r['time_s'] for r in physics])
    p=np.array([r['position_m'] for r in physics])
    inside=(abs(p[:,0])<2.8)&(abs(p[:,1])<2.8)
    dt=np.maximum(0.,np.minimum(t[1:],stop_at_s)-np.minimum(t[:-1],stop_at_s))
    interior=float(np.sum(dt*inside[1:]))
    total=float(dt.sum())
    first_wall=None; after=0.; dwell=0.; returns=0; seeking=False
    for i,elapsed in enumerate(dt, start=1):
        if elapsed<=0: continue
        if not inside[i]:
            if first_wall is None: first_wall=float(t[i])
            seeking=True; dwell=0.
        elif first_wall is not None:
            after+=float(elapsed)
            if seeking:
                dwell+=float(elapsed)
                if dwell+1e-9>=1.:
                    returns+=1; seeking=False
    return dict(interior_time_before_stop_s=interior,
        observed_pre_stop_s=total,interior_fraction_before_stop=interior/total if total else 0.,
        first_wall_band_s=first_wall,interior_time_after_first_wall_s=after,
        returns_from_wall_1s=returns)


def compare_old_control(report, old_path):
    if not old_path.exists():
        return dict(available=False)
    old=json.loads(old_path.read_text(encoding='utf-8'))
    if old['seed']!=report['seed'] or old['checkpoint_sha256']!=report['checkpoint_sha256']:
        return dict(available=False,reason='Seed or checkpoint differs')
    checks={field:np.array_equal(np.array([r[field] for r in old['trace'][kind]]),
                                np.array([r[field] for r in report['trace'][kind]]))
        for kind,field in [('control','network_pwm'),('control','applied_pwm'),
                           ('physics','position_m'),('physics','yaw_rad')]}
    if not all(checks.values()):
        raise AssertionError(f'levy_v1 regression: {checks}')
    return dict(available=True,bitwise_equal=checks,source_sha256=file_sha256(old_path))


def run_experiment(run_dir, out_dir, *, seed=20261021):
    if out_dir.exists(): raise FileExistsError('Use a new output directory')
    variants=variant_configs()
    source_hash=file_sha256(run_dir/'checkpoint.pt')
    plan=dict(version='levy-factorial-v2',seed=seed,seconds=40.,stop_at_s=35.,
        checkpoint_sha256=source_hash,scenes=list(DENSITY_SCENES),
        variants={k:asdict(v) for k,v in variants.items()},unchanged_xy_config=asdict(XYConfig()),
        metric_definitions=dict(interior='abs(x)<2.8 AND abs(y)<2.8; duration-weighted physical samples before STOP',
            return_from_wall='After a wall-band visit, remain in the interior >=1 second before STOP',
            clearance='Nearest physical surface minus body-envelope radius; earliest minimum at physics tick',
            target_gate='Selected half mean>=1.8 m AND min>=1.2 m, checked at pulse start only'),
        limitations=['One seed and one start per scene; not a reliability estimate.',
            'Directed-turn factor combines larger pulse and target gate; their separate effects are not isolated.',
            'Forward depth does not certify the complete swept trajectory or unseen directions.',
            'Pausing changes eligible-time accumulation, not the hard-preemption rule.',
            'RNG draw order can diverge after different events despite the same initial seed.'],
        code_sha256={name:file_sha256(ROOT/name) for name in ('simulation/xy_coordinator.py',
            'simulation/xy_scenes.py','optimizer/rollout.py','evaluate/watch_mvp.py',
            'evaluate/compare_xy_behaviors.py','evaluate/compare_levy_factorial.py')})
    out_dir.mkdir(parents=True)
    (out_dir/'plan.json').write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8')
    reports,rows=[],[]
    for scene in DENSITY_SCENES:
        for variant,config in variants.items():
            plot=out_dir/f'{scene}_{variant}.png'
            report=run_view(run_dir,scene=scene,steps=2000,seed=seed,coordinator_mode='both',
                levy_explore=True,stop_at_s=35.,behavior_config=config,plot_out=plot)
            report['experiment_variant']=variant
            row=summarize(report)
            row.update(mode=variant,min_clearance_event=report['metrics']['min_clearance_event'],
                target_gate_blocked_steps=sum(r['coordinator'].get('search_event')=='target_blocked'
                                             for r in report['trace']['control']),
                **interior_metrics(report['trace']['physics'],stop_at_s=35.))
            if variant=='levy_v1':
                row['old_control_replay']=compare_old_control(report,
                    ROOT/'data/diagnostics/xy_behaviors_density_v1'/f'{scene}_levy.json')
            report['factorial_metrics']=row
            plot.with_suffix('.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
            reports.append(report); rows.append(row)
            (out_dir/'comparison.json').write_text(json.dumps(dict(plan=plan,runs=rows),indent=2,allow_nan=False)+'\n',encoding='utf-8')
            print(json.dumps(row),flush=True)
    if file_sha256(run_dir/'checkpoint.pt')!=source_hash:
        raise AssertionError('Checkpoint changed')
    plot_grid(reports,rows,out_dir/'comparison.png',columns=4)
    lines=['# Levy v2: fixed factorial experiment','',
        '| Scene | Variant | Crash | Min clearance m | Time s | XY at minimum m | Interior % before STOP | Returns >=1s | Area % | STOP pass |',
        '|---|---|---|---:|---:|---|---:|---:|---:|---|']
    for r in rows:
        e=r['min_clearance_event']; x,y=e['position_m'][:2]
        lines.append(f"| {r['scene']} | {r['mode']} | {r['crashed']} | {r['min_clearance_m']:.4f} | "
            f"{e['time_s']:.3f} | ({x:.4f}, {y:.4f}) | {100*r['interior_fraction_before_stop']:.2f} | "
            f"{r['returns_from_wall_1s']} | {r['room_coverage_percent']:.2f} | {(r['stop'] or {}).get('speed_target_passed',False)} |")
    lines+=['',*plan['limitations']]
    (out_dir/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=DEFAULT_RUN)
    parser.add_argument('--out-dir',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=20261021)
    args=parser.parse_args(); torch.set_num_threads(1)
    run_experiment(args.run_dir,args.out_dir,seed=args.seed)


if __name__=='__main__': main()
