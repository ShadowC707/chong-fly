"""Static sensor study: compact targets, curvature, tilt, and zone boundaries.

python -m scripts.probe_tof_surfaces --out-dir <new-directory>
No teacher, training, flight physics, or collision admission is involved.
"""
import argparse
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from artifacts.io import publication, write_json
from generator.source_contract import file_sha256
from simulation.drone_env import ToFRaycaster
from simulation.tof_surfaces import GEOMETRY_VERSION, PlanePatch, Sphere, cast_surfaces
from simulation.zonal_tof import ZonalConfig, ZonalToF, aggregate_ranges

VERSION = 'tof-surface-probe-v1'
DENSITIES = (4, 8, 16, 32)
RULES = ('mean', 'minimum', 'supported')
OFFSETS = np.linspace(-.06, .06, 13)
VARIANTS = ('legacy_points', 'zone_centers') + tuple(f'{r}_n{s*s}' for s in DENSITIES for r in RULES)
MAX_RANGE, NEAR = 3.5, .8


def scenes():
    """Every scene has a known surface point at (.6,y,z) within the FOV and .8 m."""
    result = []
    for family in ('compact_patch', 'sphere'):
        sizes = (.01, .02, .04, .08) if family == 'compact_patch' else (.01, .02, .04, .08, .2, .4)
        for size in sizes:
            for y in OFFSETS:
                for z in OFFSETS:
                    target = (PlanePatch((.6,y,z), (0,1,0), (0,0,1), size/2, size/2)
                              if family == 'compact_patch' else Sphere((.6+size/2,y,z), size/2))
                    result.append((dict(family=family, size_m=size, y_m=float(y), z_m=float(z)), target))
    for size in (.08, .25, .6):
        for yaw in (-70, -50, -30, 0, 30, 50, 70):
            for pitch in (-50, 0, 50):
                a, b = np.deg2rad([yaw, pitch])
                # Rz(yaw) @ Ry(pitch), applied to initial in-plane y/z axes.
                u = (-np.sin(a), np.cos(a), 0.)
                v = (np.cos(a)*np.sin(b), np.sin(a)*np.sin(b), np.cos(b))
                for y, z in ((0.,0.), (.03,0.), (0.,.03), (.03,.03)):
                    result.append((dict(family='tilted_patch', size_m=size, yaw_deg=yaw,
                        pitch_deg=pitch, y_m=y, z_m=z), PlanePatch((.6,y,z), u, v, size/2, size/2)))
    return result


def observation_metrics(raw_m, output_m):
    sampled = (raw_m < NEAR).any(axis=1)
    reported = output_m < NEAR
    state = ('sampling_miss' if not sampled.any() else
             'aggregation_miss' if not reported.any() else 'near_reported')
    return dict(state=state, raw_near_zones=int(sampled.sum()), output_near_zones=int(reported.sum()),
        lost_near_zones=int((sampled & ~reported).sum()), minimum_output_m=float(output_m.min()),
        minimum_raw_m=float(raw_m.min()), max_near_sample_fraction=float((raw_m < NEAR).mean(axis=1).max()))


def evaluate(target, directions):
    matrices, metrics = [], {}
    for label, dirs in directions.items():
        raw = cast_surfaces([0,0,0], dirs.reshape(-1,3), [target], MAX_RANGE).reshape(64,-1)
        if label in ('legacy_points', 'zone_centers'):
            output = raw[:, 0]
            matrices.append(output)
            metrics[label] = observation_metrics(raw, output)
        else:
            for rule in RULES:
                cfg = ZonalConfig(samples_per_axis=int(label), rule=rule)
                output = np.array([aggregate_ranges(x, cfg, MAX_RANGE)[0] for x in raw])
                name = f'{rule}_n{int(label)**2}'
                matrices.append(output)
                metrics[name] = observation_metrics(raw, output)
    return np.array(matrices, dtype=np.float32), metrics


def summarize(cases, outputs):
    grouped = defaultdict(list)
    for case in cases:
        spec = case['specification']
        grouped[(spec['family'], spec['size_m'])].append(case)
    groups = []
    for (family, size), rows in grouped.items():
        summary = {}
        for variant in VARIANTS:
            values = [r['metrics'][variant] for r in rows]
            summary[variant] = {state:sum(v['state'] == state for v in values)
                for state in ('sampling_miss', 'aggregation_miss', 'near_reported')}
            summary[variant]['lost_near_zones'] = sum(v['lost_near_zones'] for v in values)
            summary[variant]['raw_near_zones'] = sum(v['raw_near_zones'] for v in values)
        groups.append(dict(family=family, size_m=size, cases=len(rows), variants=summary))
    convergence = []
    for coarse, dense in ((16,64), (64,256), (256,1024)):
        for rule in RULES:
            a, b = outputs[:, VARIANTS.index(f'{rule}_n{coarse}')], outputs[:, VARIANTS.index(f'{rule}_n{dense}')]
            max_change = np.max(np.abs(a-b), axis=1)
            convergence.append(dict(rule=rule, coarse_n=coarse, dense_n=dense,
                cases=len(cases), any_near_decision_flips=int(((a < NEAR).any(axis=1) != (b < NEAR).any(axis=1)).sum()),
                zone_near_decision_flips=int(((a < NEAR) != (b < NEAR)).sum()),
                median_case_max_change_m=float(np.median(max_change)),
                p95_case_max_change_m=float(np.quantile(max_change,.95)),
                worst_case_max_change_m=float(max_change.max())))
    return dict(groups=groups, density_sensitivity=convergence)


def plots(cases, outputs, summary, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch
    selected = ('legacy_points','zone_centers','minimum_n64','supported_n64','minimum_n1024','supported_n1024')
    fig, axes = plt.subplots(2,3,figsize=(12,8),layout='constrained')
    status = {'sampling_miss':0, 'aggregation_miss':1, 'near_reported':2}
    cmap = ListedColormap(['#c84747','#e9a63c','#2c9878'])
    norm = BoundaryNorm([-.5,.5,1.5,2.5], 3)
    rows = [c for c in cases if c['specification']['family'] == 'sphere' and c['specification']['size_m'] == .01]
    for ax, name in zip(axes.flat, selected):
        grid = np.array([status[c['metrics'][name]['state']] for c in rows]).reshape(13,13).T
        ax.imshow(grid,origin='lower',extent=(-65,65,-65,65),cmap=cmap,norm=norm)
        ax.axhline(0,color='black',ls=':',lw=.8); ax.axvline(0,color='black',ls=':',lw=.8)
        ax.set(title=name, xlabel='Lateral offset y (mm)', ylabel='Vertical offset z (mm)')
    fig.legend(handles=[Patch(color=cmap(i),label=label) for i,label in enumerate(
        ('No sampled near hit','Near hit discarded by aggregation','Near output retained'))],
        loc='outside lower center',ncol=3)
    fig.suptitle('10 mm sphere, front x=0.6 m | static offsets, not time or noise\nDotted axes cross four central zones')
    fig.savefig(output_dir/'compact_sphere.png',dpi=140); plt.close(fig)

    groups = [g for g in summary['groups'] if g['family'] != 'tilted_patch']
    data = np.array([[g['variants'][v]['near_reported']/g['cases']*100 for v in VARIANTS] for g in groups])
    fig, ax = plt.subplots(figsize=(13,7),layout='constrained')
    mesh = ax.imshow(data,vmin=0,vmax=100,cmap='YlGn')
    ax.set_xticks(range(len(VARIANTS)),VARIANTS,rotation=55,ha='right')
    ax.set_yticks(range(len(groups)),[f"{g['family']} {g['size_m']*1000:g} mm" for g in groups])
    for row in range(len(groups)):
        for col in range(len(VARIANTS)):
            ax.text(col,row,f'{data[row,col]:.0f}',ha='center',va='center',fontsize=8,
                    color='white' if data[row,col]>70 else 'black')
    ax.set_title('Positions retaining a near output (%) | 169 fixed positions per row\nGeometric sampling experiment, not hardware detection probability')
    fig.colorbar(mesh,ax=ax,label='% of tested positions')
    fig.savefig(output_dir/'compact_summary.png',dpi=140); plt.close(fig)

    chosen = next(i for i,c in enumerate(cases) if c['specification'] == dict(
        family='tilted_patch',size_m=.6,yaw_deg=70,pitch_deg=0,y_m=0.,z_m=0.))
    fig, axes = plt.subplots(2,3,figsize=(11,7),layout='constrained')
    for row,n in enumerate((64,1024)):
        for col,rule in enumerate(RULES):
            name=f'{rule}_n{n}'
            mesh=axes[row,col].imshow(outputs[chosen,VARIANTS.index(name)].reshape(8,8),
                                     vmin=0,vmax=3.5,cmap='viridis_r')
            axes[row,col].set(title=name,xlabel='left -> right',ylabel='top -> bottom')
    fig.colorbar(mesh,ax=list(axes.flat),label='Output range (m)')
    fig.suptitle('Same 60 cm square panel, yaw=70 degrees, center x=0.6 m\nA single tilted surface has a spread of ranges within each zone')
    fig.savefig(output_dir/'tilted_panel.png',dpi=140); plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir',type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    sources=['scripts/probe_tof_surfaces.py','simulation/tof_surfaces.py',
             'simulation/zonal_tof.py','simulation/drone_env.py','configs/flight_config.py',
             'generator/prepare_flywire_candidates.py','generator/source_contract.py', 'artifacts/io.py']
    hashes={p:file_sha256(root/p) for p in sources}
    directions={'legacy_points':ToFRaycaster().body_ray_dirs.reshape(8,8,1,3),
                'zone_centers':ZonalToF(ZonalConfig(samples_per_axis=1)).subray_directions}
    contracts={}
    for n in DENSITIES:
        sensor=ZonalToF(ZonalConfig(samples_per_axis=n))
        directions[str(n)]=sensor.subray_directions
        for rule in RULES:
            contracts[f'{rule}_n{n*n}']=ZonalToF(ZonalConfig(samples_per_axis=n,rule=rule)).contract()
    contracts.update(legacy_points=dict(sampling='FOV endpoints',directions=directions['legacy_points'].reshape(64,3).tolist()),
                     zone_centers=ZonalToF(ZonalConfig(samples_per_axis=1,rule='minimum')).contract())
    targets=scenes()
    plan=dict(version=VERSION,geometry_version=GEOMETRY_VERSION,created_at_utc=datetime.now(timezone.utc).isoformat(),
        research_only=True,training_ready=False,scope='static sensing only; no teacher or flight',
        frame='x forward, y left, z up; sensor origin (0,0,0)',max_range_m=MAX_RANGE,near_threshold_m=NEAR,
        background='No background surface; no hit returns range cap',
        variants=list(VARIANTS),sensor_contracts=contracts,case_count=len(targets),
        targets=[dict(specification=spec,surface_type=type(target).__name__,surface=asdict(target)) for spec,target in targets],
        code_sha256=hashes)
    with publication(args.out_dir) as stage:
        write_json(stage/'plan.json',plan)
        cases, arrays=[],[]
        for index,(spec,target) in enumerate(targets):
            matrices,metrics=evaluate(target,directions)
            arrays.append(matrices); cases.append(dict(index=index,specification=spec,metrics=metrics))
            if index % 200 == 0:
                print(f'{index+1}/{len(targets)} static scenes',flush=True)
        outputs=np.stack(arrays)
        summary=summarize(cases,outputs)
        np.savez_compressed(stage/'outputs.npz',output_ranges_m=outputs,variants=np.array(VARIANTS))
        plots(cases,outputs,summary,stage)
        if hashes != {p:file_sha256(root/p) for p in sources}:
            raise RuntimeError('Source changed during experiment')
        write_json(stage/'summary.json',summary)
        write_json(stage/'cases.json',cases)
        write_json(stage/'report.json',dict(version=VERSION,case_count=len(cases),variant_count=len(VARIANTS),
            observation_count=len(cases)*len(VARIANTS),research_only=True,training_ready=False,
            summary=summary,files=dict(plan='plan.json',cases='cases.json',outputs='outputs.npz'),
            limitations=['All scenes contain a known near surface point in the FOV; this set cannot measure false positives.',
                'No teacher, motion, collision test, materials, optical weights, noise, latency, or dropouts.',
                'A dense grid is another discretization, not ground-truth detection probability.',
                'Only convex spheres and flat rectangles; no arbitrary scanned mesh or concave surfaces.',
                'A range-based group is not necessarily a separate physical object.',
                'Aggregate rejection means losing a near output, not necessarily losing all evidence of an object.',
                'Physical thickness and collision geometry of patches are not implemented.']))
    print(f'Published {len(cases)} scenes x {len(VARIANTS)} variants to {args.out_dir}',flush=True)


if __name__ == '__main__':
    main()
