"""Finite, copy-only sensitivity grid on existing learned rate-model weights.

No teacher changes, training, time-step changes, external evaluation or flight
admission. Positive scaling preserves every nonzero recurrent edge and sign.
"""
import argparse
import copy
import itertools
import json
import math
from pathlib import Path

import torch

from core.models import TAU_FLOOR_S
from artifacts.io import publication, write_json
from generator.source_contract import file_sha256
from artifacts.research import load_research_run
from optimizer.search_pipeline import GATE
from simulation.policy_diagnostics import temporal_response_probe

TAU_SCALES = (.25, .5, .75, 1.)
RECURRENT_SCALES = (.5, .75, 1., 1.25)


def scaled_policy(policy, *, tau_scale, recurrent_scale):
    for value in (tau_scale, recurrent_scale):
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError('Scales must be positive and finite')
    result = copy.deepcopy(policy)
    result.reset_state()
    cell = result.cfc_network.cell
    with torch.no_grad():
        if tau_scale != 1.:
            value = cell.tau*tau_scale-TAU_FLOOR_S
            if (value <= 0).any() or not torch.isfinite(value).all():
                raise ValueError('Scaled tau must stay above the model floor')
            # Stable inverse softplus, not a change to the integration timestep.
            cell.tau_raw.copy_(value + torch.log(-torch.expm1(-value)))
        weight = cell.W_macro if cell.mode == 'fixed' else cell._w_macro_param
        weight.mul_(recurrent_scale)
        if not torch.isfinite(weight).all():
            raise ValueError('Scaled recurrent weights are nonfinite')
    before = policy.cfc_network.cell._effective_W().detach()
    after = cell._effective_W().detach()
    if not torch.equal(torch.sign(before), torch.sign(after)):
        raise ValueError('Scaling changed recurrent support/signs')
    result.eval()
    return result


def summarize_probe(probe, gate):
    counts = dict(cases=len(probe['cases']), direction_passed=0, amplitude_passed=0,
                  onset_passed=0, recovery_passed=0, joint_passed=0, broad_joint_passed=0)
    errors, settles = [], []
    for row in probe['cases']:
        error = abs(row['threat_yaw_pwm']-row['threat_target_yaw_pwm'])
        errors.append(error)
        def within(value, limit):
            return value is not None and math.isfinite(value) and 0 <= value <= limit
        passed = dict(direction=bool(row['direction_correct']),
            amplitude=within(error, gate['turn_final_error_max_pwm']),
            onset=within(row['onset_s'], gate['onset_max_s']),
            recovery=within(row['recovery_settle_s'], gate['recovery_max_s']))
        for name, value in passed.items():
            counts[name+'_passed'] += int(value)
        joint = all(passed.values()) and within(row['clear_abs_yaw_pwm'], gate['clear_yaw_max_pwm'])
        counts['joint_passed'] += int(joint)
        counts['broad_joint_passed'] += int(joint and row['active_columns'] == 4)
        if row['recovery_settle_s'] is not None:
            settles.append(row['recovery_settle_s'])
    return dict(**counts, mean_final_amplitude_error_pwm=sum(errors)/len(errors),
                max_final_amplitude_error_pwm=max(errors),
                max_recovery_s=max(settles) if len(settles) == len(errors) else None)


def run_grid(run, output):
    run = Path(run)
    parent = json.loads((run/'preflight.json').read_text(encoding='utf-8'))
    sources = ['optimizer/probe_recurrent_dynamics.py', 'optimizer/benchmark_candidate.py',
        'optimizer/search_pipeline.py', 'simulation/policy_diagnostics.py', 'simulation/policy.py',
        'core/models.py', 'core/routing.py', 'configs/flight_config.py',
        'generator/generate_reflex_dataset.py', 'generator/continuous_reflex_teacher.py',
        'generator/reflex_contract.py', 'artifacts/research.py', 'artifacts/io.py', 'core/invariants.py']
    root = Path(__file__).resolve().parents[1]
    hashes = {p:file_sha256(root/p) for p in sources}
    candidates = []
    with publication(output) as stage:
        for trial in sorted(run.glob('trial_*')):
            original, checkpoint = load_research_run(trial)
            parent_report = json.loads((trial/'report.json').read_text(encoding='utf-8'))
            if parent_report['experiment_fingerprint'] != parent['experiment_fingerprint']:
                raise ValueError('Mixed parent trials')
            digest = file_sha256(trial/'checkpoint.pt')
            rows = []
            for tau, gain in itertools.product(TAU_SCALES, RECURRENT_SCALES):
                policy = scaled_policy(original, tau_scale=tau, recurrent_scale=gain)
                spatial = temporal_response_probe(policy, coverage_columns=(1, 2, 4),
                    teacher_version=checkpoint['pretrain_info']['teacher_version'])
                cell = policy.cfc_network.cell
                summary = summarize_probe(spatial, GATE)
                rows.append(dict(tau_scale=tau, recurrent_scale=gain, summary=summary, probe=spatial,
                    tau_min_s=float(cell.tau.min()), tau_max_s=float(cell.tau.max())))
                print(checkpoint['candidate'], tau, gain, summary, flush=True)
            baseline = next(r for r in rows if r['tau_scale'] == r['recurrent_scale'] == 1.)
            # Identity transform must exactly replay the previously recorded probe.
            if baseline['probe'] != parent_report['spatial_response']:
                raise ValueError('Baseline no longer reproduces the parent spatial probe')
            for row in rows:
                s, b = row['summary'], baseline['summary']
                # Predeclared partial-screen rule, NOT full functional admission:
                # all four broad cases pass jointly, no loss of direction count
                # or worst amplitude accuracy over the full twelve cases.
                row['training_screen_passed'] = (s['broad_joint_passed'] == 4
                    and s['direction_passed'] >= b['direction_passed']
                    and s['max_final_amplitude_error_pwm'] <= b['max_final_amplitude_error_pwm'])
            if file_sha256(trial/'checkpoint.pt') != digest:
                raise ValueError('Original checkpoint changed')
            candidates.append(dict(candidate=checkpoint['candidate'], parent_checkpoint_sha256=digest,
                rows=rows, shortlisted=[dict(tau_scale=r['tau_scale'], recurrent_scale=r['recurrent_scale'])
                                      for r in rows if r['training_screen_passed']]))
        if {c['candidate'] for c in candidates} != set(parent['eligible_candidates']):
            raise ValueError('Missing eligible candidate')
        if hashes != {p:file_sha256(root/p) for p in hashes}:
            raise ValueError('Code changed during grid')
        report = dict(version='recurrent-sensitivity-v1', research_only=True, training_ready=False,
            parent_experiment_fingerprint=parent['experiment_fingerprint'], code_sha256=hashes,
            tau_scales=list(TAU_SCALES), recurrent_scales=list(RECURRENT_SCALES), gate=dict(GATE),
            training_screen='All four broad cases jointly pass; full-grid direction count and worst amplitude error no worse than baseline.',
            candidates=candidates, limitations=[
                'Learned weights perturbed without retraining; not evidence about all trainable parameter settings.',
                'Fixed synthetic sensor histories at the unchanged controller timestep; no physical flight admission.',
                'Known development probes, not independent held-out confirmation.',
                'Recurrent scaling is applied for the whole sequence, not only after threat removal.',
                'No external evaluation dataset read. No checkpoint modified or promoted.'])
        write_json(stage/'report.json', report)
        plot_grid(report, stage)
    return report


def plot_grid(report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for candidate in report['candidates']:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
        for ax, key, title in zip(axes, ('recovery_passed', 'joint_passed'),
                                  ('Recovery <= 0.8s (alone)', 'Direction + strength + onset + recovery')):
            grid = [[next(r['summary'][key] for r in candidate['rows']
                          if r['tau_scale'] == t and r['recurrent_scale'] == g)
                     for g in report['recurrent_scales']] for t in report['tau_scales']]
            im = ax.imshow(grid, vmin=0, vmax=12, cmap='viridis')
            ax.set(xticks=range(4), xticklabels=report['recurrent_scales'],
                   yticks=range(4), yticklabels=report['tau_scales'],
                   xlabel='Recurrent weight multiplier', ylabel='Tau multiplier', title=title)
            for i, row in enumerate(grid):
                for j, value in enumerate(row):
                    ax.text(j, i, f'{value}/12', ha='center', va='center', color='white' if value < 7 else 'black')
            fig.colorbar(im, ax=ax, label='Passing probe cases')
        fig.suptitle(candidate['candidate']+' | saved weights, no training')
        fig.savefig(Path(output)/(candidate['candidate']+'.png'), dpi=140)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--out-dir', required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run_grid(args.run, args.out_dir)


if __name__ == '__main__':
    main()
