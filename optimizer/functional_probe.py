"""Paired open-loop stimuli and temporary state clamps in a research checkpoint.

No training, flight simulation, graph edits, or biological/flight admission.
The saved plan and arrays make this an exploratory, reproducible model experiment.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import platform

import numpy as np
import torch

from configs.flight_config import CONTROL_DT, TOF_RAYCASTER_MAX_RANGE_M
from artifacts.io import publication, write_json
from generator.source_contract import file_sha256
from artifacts.research import load_research_run

WARMUP, ACTIVE, RECOVERY = 50, 25, 100


def make_stimuli():
    """Nine paired cases: clear plus side x distance x width; flow=0, memory=1."""
    cases = [dict(name='clear', side=None, normalized_distance=1., width=0)]
    for distance in (.1, .2):
        for width in (1, 4):
            for side in ('left', 'right'):
                cases.append(dict(name=f'{side}_d{distance}_w{width}', side=side,
                                  normalized_distance=distance, width=width))
    obs = torch.zeros(len(cases), WARMUP+ACTIVE+RECOVERY, 74)
    obs[..., 2:] = 1
    for i, case in enumerate(cases[1:], 1):
        grid = obs[i, WARMUP:WARMUP+ACTIVE, 2:66].reshape(ACTIVE, 8, 8)
        width = case['width']
        columns = slice(0, width) if case['side'] == 'left' else slice(8-width, 8)
        grid[..., columns] = case['normalized_distance']
    return cases, obs


@torch.no_grad()
def trace_response(policy, obs, clamp_indices=()):
    """Clamp selected states to zero after EVERY integration step, before readout.

This silences recurrent and readout contributions, starting in warmup. It is
not an edge-specific intervention or a physiological silencing simulation.
Only a temporary hook is used; the model/checkpoint is never rewritten.
"""
    cell = policy.cfc_network.cell
    ids = list(clamp_indices)
    if (any(type(i) is not int or not 0 <= i < cell.hidden_size for i in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError('Clamp indices must be unique in-range integers')
    if (obs.ndim != 3 or obs.shape[0] < 1 or obs.shape[1] < 1
            or obs.shape[2] != 74 or not torch.isfinite(obs).all()):
        raise ValueError('Expected finite nonempty batch/time/74 observations')
    modes = [(module, module.training) for module in policy.modules()]
    states = []
    def capture(module, inputs, output):
        if not torch.isfinite(output).all():
            raise ValueError('Nonfinite neural state')
        if ids:
            output = output.clone()
            output[:, ids] = 0
        states.append(output.detach().cpu().clone())
        return output
    handle = cell.register_forward_hook(capture)
    try:
        policy.eval()
        parameter = next(policy.parameters())
        pwm, _ = policy(obs.to(parameter), hx=None, dt=CONTROL_DT)
        if len(states) != obs.shape[1] or not torch.isfinite(pwm).all():
            raise ValueError('Incomplete or nonfinite functional trace')
        return pwm.cpu().numpy(), torch.stack(states, dim=1).numpy()
    finally:
        handle.remove()
        for module, mode in modes:
            module.training = mode


def resolve_groups(meta, plan):
    """Exact annotation selectors; controls match side, sign and cluster size.

Controls exclude all selected groups and input/output interface clusters.
They are NOT matched for degree, weights or functional importance.
"""
    if plan.get('format_version') != 'functional-probe-plan-v1':
        raise ValueError('Unknown functional plan version')
    signatures = meta['provenance']['cluster_signatures']
    sizes = meta['provenance']['cluster_sizes']
    if len(signatures) != meta['k'] or len(sizes) != meta['k']:
        raise ValueError('Incomplete cluster annotations')
    groups = [dict(name='intact', indices=[], purpose='unmodified reference')]
    for spec in plan['groups']:
        field, value, name = spec['field'], spec['value'], spec['name']
        if field not in {'cell_type', 'sensor_roles', 'motor_roles'}:
            raise ValueError('Unsupported group selector')
        if not isinstance(name, str) or not name or any(g['name'] == name for g in groups):
            raise ValueError('Group names must be distinct and nonempty')
        ids = [i for i, sig in enumerate(signatures)
               if (sig[field] == value if field == 'cell_type' else value in sig[field])]
        if not ids:
            raise ValueError(f'Group {name} has no matching annotated clusters')
        groups.append(dict(name=name, indices=ids, purpose=spec['purpose'], selector=spec))
    target = next((g for g in groups if g['name'] == plan['matched_controls_for']), None)
    count = plan['matched_control_count']
    if (target is None or not target['indices'] or type(count) is not int or not 1 <= count <= 10
            or type(plan['control_seed']) is not int or plan['control_seed'] < 0):
        raise ValueError('Invalid matched control plan')
    excluded = {i for g in groups for i in g['indices']}
    def signature(i):
        s = signatures[i]
        return s['side'], s['outgoing_sign_profile'], sizes[i]
    required = Counter(signature(i) for i in target['indices'])
    pools = {key: [i for i, s in enumerate(signatures) if i not in excluded
                  and not s['sensor_roles'] and not s['motor_roles'] and signature(i) == key]
             for key in required}
    if any(len(pools[key]) < n for key, n in required.items()):
        raise ValueError('Insufficient eligible matched controls')
    rng, used = np.random.default_rng(plan['control_seed']), set()
    for number in range(count):
        for _ in range(100):
            ids = sorted(int(i) for key, n in required.items()
                         for i in rng.choice(pools[key], n, replace=False))
            if tuple(ids) not in used:
                break
        else:
            raise ValueError('Could not select distinct matched controls')
        used.add(tuple(ids))
        name = f'matched_control_{number+1}'
        if any(g['name'] == name for g in groups):
            raise ValueError('Reserved matched control name')
        groups.append(dict(name=name, indices=ids, purpose='matched intermediate control',
                           matched_to=target['name']))
    return groups


def response_metrics(pwm, tolerance=20.):
    """Subtract this intervention's clear case at the SAME timestamps.

Onset/settling use absolute evoked yaw, not desired direction or admission.
Time zero for samples is the end of the first control integration step.
"""
    if (pwm.ndim != 3 or pwm.shape[1:] != (WARMUP+ACTIVE+RECOVERY, 4)
            or not np.isfinite(pwm).all() or not np.isfinite(tolerance) or tolerance <= 0):
        raise ValueError('Invalid response trace or tolerance')
    rows = []
    for case in pwm:
        evoked = case[:, 3]-pwm[0, :, 3]
        active = evoked[WARMUP:WARMUP+ACTIVE]
        tail = evoked[WARMUP+ACTIVE:]
        hits = np.flatnonzero(abs(active) > tolerance)
        outside = np.flatnonzero(abs(tail) > tolerance)
        settle = int(outside[-1])+1 if len(outside) else 0
        rows.append(dict(mean_evoked_yaw_pwm=float(active.mean()),
            peak_abs_evoked_yaw_pwm=float(abs(active).max()),
            clear_mean_yaw_pwm=float(pwm[0, WARMUP:WARMUP+ACTIVE, 3].mean()),
            active_mean_yaw_pwm=float(case[WARMUP:WARMUP+ACTIVE, 3].mean()),
            onset_s=float(hits[0]*CONTROL_DT) if len(hits) else None,
            recovery_settle_s=settle*CONTROL_DT if settle < RECOVERY else None,
            recovery_peak_abs_evoked_yaw_pwm=float(abs(tail).max())))
    return rows


def plot_results(pwms, states, cases, groups, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout='constrained')
    times = (np.arange(pwms.shape[2])+1)*CONTROL_DT
    # Both broad near stimuli; side is explicit, never inferred from output sign.
    for ax, index in zip(axes[0], (3, 4)):
        for j, group in enumerate(groups):
            ax.plot(times, pwms[j, index, :, 3]-pwms[j, 0, :, 3],
                    label=group['name'], lw=1.4, alpha=.85)
        ax.axvspan(WARMUP*CONTROL_DT, (WARMUP+ACTIVE)*CONTROL_DT, color='grey', alpha=.12)
        ax.set(title=cases[index]['name'], xlabel='Time (s)', ylabel='Evoked yaw vs own clear (PWM)')
        ax.grid(alpha=.2)
    axes[0, 1].legend(fontsize=8)
    delta = states[0, 3]-states[0, 0]
    limit = max(float(abs(delta).max()), 1e-8)
    im = axes[1, 0].imshow(delta.T, aspect='auto', origin='lower',
                          extent=(0, times[-1], -.5, delta.shape[1]-.5),
                          cmap='RdBu_r', vmin=-limit, vmax=limit)
    axes[1, 0].set(title='Intact: left near broad stimulus minus clear',
                   xlabel='Time (s)', ylabel='Cluster index (see cluster map)')
    fig.colorbar(im, ax=axes[1, 0], label='Model state difference (not measured firing rate)')
    scores = [response_metrics(p)[3]['mean_evoked_yaw_pwm'] for p in pwms]
    axes[1, 1].barh([g['name'] for g in groups], scores)
    axes[1, 1].set(title='Left near broad: mean evoked yaw during stimulus', xlabel='PWM')
    fig.suptitle('Functional probe | trained checkpoint | exploratory open-loop model experiment')
    fig.savefig(path, dpi=140)
    plt.close(fig)


def run_probe(run, config, output, *, research_only=False):
    if not research_only:
        raise ValueError('Explicit research-only opt-in required')
    run, config, output = map(Path, (run, config, output))
    if output.exists():
        raise FileExistsError('Use a new output directory')
    policy, checkpoint = load_research_run(run)  # keep all existing contract checks
    meta_path, = (run/'graph').glob('meta_*.json')
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    plan = json.loads(config.read_text(encoding='utf-8'))
    groups = resolve_groups(meta, plan)
    cases, obs = make_stimuli()
    cmap = np.load(run/'graph'/meta['cmap_file'], allow_pickle=False)
    roots = meta['provenance']['source_root_ids']
    if (cmap.shape != (len(roots),) or not np.issubdtype(cmap.dtype, np.integer)
            or np.any(cmap < 0) or np.any(cmap >= meta['k'])
            or np.bincount(cmap, minlength=meta['k']).tolist() != meta['provenance']['cluster_sizes']):
        raise ValueError('Invalid source-root to cluster mapping')
    clusters = [dict(index=i, signature=s, root_ids=[str(roots[j]) for j in np.flatnonzero(cmap == i)])
                for i, s in enumerate(meta['provenance']['cluster_signatures'])]
    all_pwm, all_states, results = [], [], []
    for group in groups:
        pwm, states = trace_response(policy, obs, group['indices'])
        all_pwm.append(pwm)
        all_states.append(states)
        results.append(dict(intervention=group['name'], cases=response_metrics(pwm)))
        print(f"{group['name']}: {len(group['indices'])} clamped clusters; {len(cases)} stimuli", flush=True)
    # Verify no hook or persistent modification affected the unmodified run.
    repeated, repeated_states = trace_response(policy, obs)
    if not (np.array_equal(repeated, all_pwm[0]) and np.array_equal(repeated_states, all_states[0])):
        raise ValueError('Intact trace changed after interventions')
    for result in results:
        for row, base in zip(result['cases'], results[0]['cases']):
            row['change_from_intact_evoked_yaw_pwm'] = row['mean_evoked_yaw_pwm']-base['mean_evoked_yaw_pwm']
    root = Path(__file__).resolve().parents[1]
    code_files = ['optimizer/functional_probe.py', 'optimizer/benchmark_candidate.py',
                  'core/models.py', 'core/contracts.py', 'core/routing.py', 'simulation/policy.py',
                  'simulation/control_contract.py', 'simulation/altitude_control.py',
                  'configs/flight_config.py', 'generator/graph_reducer.py', 'artifacts/research.py', 'artifacts/io.py', 'core/invariants.py']
    report = dict(format_version='functional-probe-v1', research_only=True, training_ready=False,
        run=str(run.resolve()), candidate=checkpoint['candidate'], plan=plan,
        checkpoint_sha256=file_sha256(run/'checkpoint.pt'), config_sha256=file_sha256(config),
        graph_sha256={p.name:file_sha256(p) for p in (meta_path, run/'graph'/meta['w_file'], run/'graph'/meta['cmap_file'])},
        code_sha256={name:file_sha256(root/name) for name in code_files},
        runtime=dict(python=platform.python_version(), numpy=np.__version__, torch=str(torch.__version__),
                     device='cpu', threads=torch.get_num_threads()),
        dt=CONTROL_DT, warmup_steps=WARMUP, active_steps=ACTIVE, recovery_steps=RECOVERY,
        onset_tolerance_pwm=20., onset_time_definition='First post-step sample above threshold; index*dt within active phase.',
        normalized_distance_scale_m=TOF_RAYCASTER_MAX_RANGE_M,
        checkpoint_options={key:checkpoint.get(key) for key in
                            ('mode', 'preserve_signs', 'sensor_encoding', 'neutral_origin', 'tau_init')},
        cases=cases, interventions=groups, results=results, intact_repeat_exact=True,
        intervention_semantics='State clamped to zero after every step, including warmup, before recurrence/readout; no retraining.',
        arrays=dict(pwm='intervention,case,time,channel [throttle,roll,pitch,yaw]',
                    states='intervention,case,time,cluster', observations='case,time,sensor'),
        limitations=['Exploratory same-checkpoint contrasts, not independent validation or biological proof.',
                     'Synthetic static ToF pulses; not retinal looming and not closed-loop flight.',
                     'Signed model states are not measured biological firing rates.',
                     'Clamping is a computational intervention, not a faithful optogenetic model.',
                     'Controls match cluster size, side and sign only; not degree or connection strength. Three controls do not establish significance.',
                     'Input/output clamps are interface controls; they cannot prove a specific intermediate mechanism.',
                     'Source-quality, transmitter signs, graph completeness and sensor/decoder assumptions remain open.',
                     'Throttle is a placeholder; no altitude controller or flight admission in this experiment.'])
    pwms, states = np.stack(all_pwm), np.stack(all_states)
    with publication(output) as stage:
        write_json(stage/'clusters.json', clusters)
        np.savez_compressed(stage/'traces.npz', pwm=pwms, states=states, observations=obs.numpy(),
                            time_s=(np.arange(obs.shape[1])+1)*CONTROL_DT)
        plot_results(pwms, states, cases, groups, stage/'responses.png')
        report['artifact_sha256'] = {name:file_sha256(stage/name) for name in
                                    ('clusters.json', 'traces.npz', 'responses.png')}
        write_json(stage/'report.json', report)
    return report


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=root/'data/selection/registry_mean_spatial_v1/trial_00001')
    parser.add_argument('--config', type=Path, default=root/'configs/functional_probe_v1.json')
    parser.add_argument('--out-dir', type=Path)
    parser.add_argument('--research-only', action='store_true')
    args = parser.parse_args()
    output = args.out_dir or root/'data/diagnostics'/('functional_probe_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f'))
    torch.set_num_threads(1)
    result = run_probe(args.run, args.config, output, research_only=args.research_only)
    print('Left near broad: mean evoked yaw (PWM), relative to each variant\'s clear response')
    for row in result['results']:
        print(f"{row['intervention']:22} {row['cases'][3]['mean_evoked_yaw_pwm']:9.3f}")
    print(f'Results: {output.resolve()}')


if __name__ == '__main__':
    main()
