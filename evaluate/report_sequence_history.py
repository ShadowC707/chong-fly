"""Compare complete histories and independent episode collections; no fitting."""
import argparse
import json
from pathlib import Path

from optimizer.continue_flow_reflex import ROOT, DATA, write, score_raw
from optimizer.probe_history_capacity import make_cases
from optimizer.history_objective import sequence_metrics
from optimizer.diagnose_candidate import heldout_predictions
from artifacts.research import load_research_run
from edge.runtime import NumpyPolicy, sha256
from edge.system import RunConfig, build_system
import numpy as np
import torch


def run(directory, previous):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    torch.set_num_threads(1)
    directory, previous = Path(directory), Path(previous)
    validation = json.loads((directory/'validation.json').read_text())
    if validation['completed_arms'] != 3:
        raise ValueError('Wait for all arms to finish')
    protocol = json.loads((directory/'protocol.json').read_text())
    previous_protocol = json.loads((previous.parent/'protocol.json').read_text())
    if previous_protocol['parent_sha256'] != protocol['parent_sha256']:
        raise ValueError('Previous objective must start from the same parent')
    for key in ('seed', 'updates', 'updates_per_stage', 'suffix_ticks', 'lr_interface', 'lr_W', 'lr_tau', 'anchors'):
        if previous_protocol['settings'][key] != protocol['settings'][key]:
            raise ValueError('Unmatched previous experiment setting: '+key)
    fixed = protocol['fixed_baseline_sha256']
    assert all(sha256(ROOT/p) == digest for p, digest in fixed.items())
    config = RunConfig.from_json(ROOT/'configs/mvp_runtime_v3.json')
    build_system(config)
    runs = dict(before=directory/'before', endpoint=previous,
                sequence_frozen=directory/'frozen', sequence_plastic=directory/'plastic')
    files = dict(old_external=DATA/'evaluation.pt',
        corrective=ROOT/'data/maneuver_corrections_v1/evaluation.pt',
        expert=ROOT/'data/maneuver_sequences_v1/evaluation.pt')
    datasets = {name: torch.load(path, weights_only=True, map_location='cpu') for name, path in files.items()}
    inputs = {str(path): sha256(path) for path in files.values()}
    inputs.update({str(path/'checkpoint.pt'): sha256(path/'checkpoint.pt') for path in runs.values()})
    data = torch.load(directory/'cases.pt', weights_only=True, map_location='cpu')
    prior_data = torch.load(previous.parent/'cases.pt', weights_only=True, map_location='cpu')
    for split in ('training', 'evaluation'):
        for n, case in prior_data[split].items():
            for field in ('X', 'Y'):
                if not torch.equal(case[field], data[split][n][field]):
                    raise ValueError('Unmatched history data')
    for field in ('anchors', 'anchor_target'):
        if not torch.equal(prior_data[field], data[field]):
            raise ValueError('Unmatched anchor data')
    overlaps=[]
    for i, x in enumerate(data['training'][150]['X']):
        for j, anchor in enumerate(data['anchors']):
            if torch.equal(x, anchor):
                target=data['training'][150]['Y'][i,:,2]
                reference=data['anchor_target'][j,:,2]
                overlaps.append(dict(history=data['training'][150]['records'][i],anchor_index=j,
                    target_mean_pitch_pwm=float(target.mean()),reference_mean_pitch_pwm=float(reference.mean()),
                    max_pitch_disagreement_pwm=float((target-reference).abs().max())))
    write(directory/'anchor_overlap.json',dict(exact_input_matches=overlaps,
        interpretation='Competing soft objectives, not two copies of one teacher dataset; no loss changed during this experiment'))
    inputs.update({str(path):sha256(path) for path in
        (directory/'cases.pt', previous.parent/'cases.pt')})
    draw = make_cases((.12,), 150)
    results, traces = {}, {}
    for name, path in runs.items():
        policy, _ = load_research_run(path)
        policy.eval()
        metrics = {}
        with torch.no_grad():
            for split in ('training', 'evaluation', 'duration_holdout'):
                metrics[split] = {str(n): sequence_metrics(policy(case['X'], dt=.02)[0], case['Y'])
                                  for n, case in data[split].items()}
            raw = {label: score_raw(heldout_predictions(policy, dataset, list(range(len(dataset['X'])))),
                                   dataset, list(range(len(dataset['X'])))) for label, dataset in datasets.items()}
            traces[name] = policy(draw['X'][:3], dt=.02)[0].numpy()
            long_case = data['duration_holdout'][400]['X']
            expected = policy(long_case, dt=.02)[0].numpy()
        long_targets=data['duration_holdout'][400]['Y'].numpy()
        long_details=[dict(**record, final_pitch_pwm=float(expected[i,-1,2]),
            target_final_pitch_pwm=float(long_targets[i,-1,2]),
            max_suffix_error_pwm=float(np.abs(expected[i,50:,2]-long_targets[i,50:,2]).max()))
            for i,record in enumerate(data['duration_holdout'][400]['records'])]
        if name=='sequence_plastic':
            np.savez_compressed(directory/'duration_holdout_traces.npz',prediction=expected,target=long_targets)
            fig,axes=plt.subplots(1,2,figsize=(13,4),sharey=True,layout='constrained')
            for ax,start,flow in zip(axes,(0,6),(.10,.16)):
                for i,(label,color) in enumerate(zip(('always clear','unsettled','settled'),('#267cba','#bc3a32','#33804a'))):
                    ax.plot(np.arange(450)*.02,expected[start+i,:,2],color=color,label=label)
                    ax.plot(np.arange(450)*.02,long_targets[start+i,:,2],color=color,ls='--',alpha=.5)
                ax.axvspan(1,9,color='.93',zorder=-1)
                ax.set(title=f'flow={flow:.2f}, left threat',xlabel='Time (s)',ylabel='Pitch PWM')
                ax.grid(alpha=.2)
            axes[0].legend(fontsize=8)
            fig.suptitle('Unseen flows / identical last 8s inputs: sequence_plastic\nSolid: network; dashed: target')
            fig.savefig(directory/'duration_holdout.png',dpi=150);plt.close(fig)
        runtime = NumpyPolicy(path/'numpy')
        worst = 0.
        for x, wanted in zip(long_case.numpy(), expected):
            runtime.reset_state()
            actual = np.stack([runtime.step_np(row[:2], row[2:66], row[66:], dt=.02) for row in x])
            worst = max(worst, float(np.abs(actual-wanted).max()))
        report = json.loads((path/'report.json').read_text())
        # Cruise eligibility is one snapshot condition, not full handover.
        deviation = max(abs(v-1500) for v in report['cruise']['final_pwm'][1:3])
        results[name] = dict(sequence=metrics, external=raw,
            duration_8s_cases=long_details,
            numpy_long_history_max_error_pwm=worst, numpy_long_history_within_002_pwm=worst <= .02,
            neutral_manual_attitude_condition=dict(deviation_pwm=deviation,
                threshold_pwm=config.bridge.max_attitude_difference_pwm,
                passed=deviation <= config.bridge.max_attitude_difference_pwm),
            scenes_passed=report['mvp_admission']['scenes_passed'],
            direction_passed=report['mvp_admission']['direction_passed'],
            recovery_passed=report['mvp_admission']['recovery_passed'],
            max_recovery_s=report['summary']['max_recovery_s'])
        write(directory/'sequence_comparison.json', results)
        print(name, '3s', metrics['training']['150'], 'scenes', results[name]['scenes_passed'], flush=True)
    np.savez_compressed(directory/'sequence_traces.npz', **traces, target=draw['Y'][:3].numpy())
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharey=True, layout='constrained')
    for ax, (name, pred) in zip(axes.flat, traces.items()):
        for i, (label, color) in enumerate(zip(('always clear', 'unsettled', 'settled'), ('#267cba', '#bc3a32', '#33804a'))):
            ax.plot(np.arange(200)*.02, pred[i, :, 2], color=color, label=label)
            ax.plot(np.arange(200)*.02, draw['Y'][i, :, 2], color=color, ls='--', alpha=.5)
        ax.axvspan(1, 4, color='.93', zorder=-1)
        ax.set(title=name, xlabel='Time (s)', ylabel='Pitch PWM')
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Full histories: flow=.12 / left threat / identical last 3s inputs\nSolid: network; dashed: target. Same parent; 300 updates per trained arm.')
    fig.savefig(directory/'sequence_comparison.png', dpi=150)
    plt.close(fig)
    lines = ['# Повна послідовність: порівняння', '',
        '| Модель | MAE спільних 3 с (train) | Максимум 3 с (train) | Максимум 3 с (eval) | Сцени /7 | Напрямок /12 | Згасання /12 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name, r in results.items():
        t, e = r['sequence']['training']['150'], r['sequence']['evaluation']['150']
        lines.append(f"| {name} | {t['suffix_mean_error_pwm']:.2f} | {t['suffix_max_error_pwm']:.2f} | {e['suffix_max_error_pwm']:.2f} | {r['scenes_passed']} | {r['direction_passed']} | {r['recovery_passed']} |")
    lines += ['', 'endpoint — попередній завершений запуск, інші гілки — поточна серія; один seed.',
        'Оцінювання на повних епізодах є open-loop: готові входи, не керування новою фізичною траєкторією.',
        'Жодна модель автоматично не замінює робочий пакет.']
    (directory/'summary.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    assert all(sha256(Path(p)) == digest for p, digest in inputs.items())
    assert all(sha256(ROOT/p) == digest for p, digest in fixed.items())
    write(directory/'post_validation.json', dict(input_sha256=inputs, inputs_unchanged=True,
        active_baseline_unchanged=True, pinned_system_build_passed=True,
        matched_settings_checked=True, matched_training_and_anchor_tensors=True, automatic_promotion=False,
        report_source_sha256=sha256(Path(__file__))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--previous', type=Path, default=ROOT/'output/plastic_history_v1/plastic')
    args = parser.parse_args()
    run(args.directory, args.previous)
