"""Read-only coverage audit of the exact dataset/split used by a saved trial.

Bins describe sensor observations, not physical object dimensions. Historical
teacher labels are replayed with their recorded version, never the new default.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from artifacts.io import publication, write_json
from generator.source_contract import file_sha256


def validate_split(train, validation, count):
    combined = train+validation
    if (not train or not validation or any(type(i) is not int for i in combined)
            or len(set(combined)) != len(combined) or set(combined) != set(range(count))):
        raise ValueError('Expected a disjoint complete episode split')


def summarize_mask(mask, valid, targets, ids, dt):
    selected = np.asarray(mask, dtype=bool)[ids] & valid[ids]
    lengths = []
    for row in selected:
        edges = np.diff(np.r_[False, row, False].astype(int))
        runs = np.flatnonzero(edges == -1)-np.flatnonzero(edges == 1)
        lengths.append(int(runs.max()) if len(runs) else 0)
    yaw = targets[ids, :, 3][selected]
    present = [ids[i] for i, row in enumerate(selected) if row.any()]
    return dict(frames=int(selected.sum()), episodes=len(present), episode_indices=present,
                duration_s=float(selected.sum()*dt), longest_run_s=max(lengths, default=0)*dt,
                episodes_with_0_5s_run=sum(n*dt >= .5-1e-9 for n in lengths),
                yaw_target_mean_pwm=float(yaw.mean()) if len(yaw) else None,
                yaw_target_min_pwm=float(yaw.min()) if len(yaw) else None,
                yaw_target_max_pwm=float(yaw.max()) if len(yaw) else None)


def condition_masks(x, *, max_range, threat_m):
    """Progressively stricter overlap with the static spatial probe.

Coarse: unilateral threat, exact occupied-column count, nearest range in band.
Full band: exact edge-anchored rectangle, ALL its rays in the distance band.
Neither definition demands all clear rays equal maximum range.
"""
    grid = x[..., 2:66].reshape(*x.shape[:2], 8, 8)*max_range
    threat = grid < threat_m
    columns = threat.any(axis=-2)
    nearest = grid.min(axis=(-2, -1))
    flow = np.linalg.norm(x[..., :2], axis=-1)
    masks = []
    for distance in (.35, .70):
        distance_band = (nearest >= distance-.10) & (nearest <= distance+.10)
        for width in (1, 2, 4):
            for side in ('left', 'right'):
                other = columns[..., 4:] if side == 'left' else columns[..., :4]
                coarse = distance_band & (columns.sum(-1) == width) & ~other.any(-1)
                expected = np.zeros((8, 8), dtype=bool)
                expected[:, :width] = side == 'left'
                if side == 'right':
                    expected[:, 8-width:] = True
                rectangular = (threat == expected).all(axis=(-2, -1))
                near_rays = grid[..., expected]
                full_band = rectangular & ((near_rays >= distance-.10)
                                             & (near_rays <= distance+.10)).all(-1)
                masks.append(dict(name=f'{side}_d{distance:.2f}_w{width}', side=side,
                    distance_m=distance, active_columns=width, coarse=coarse,
                    full_band=full_band, full_band_low_flow=full_band & (flow <= .05)))
    return masks


def run_audit(dataset, run, output):
    dataset, run, output = map(Path, (dataset, run, output))
    if output.exists():
        raise FileExistsError('Use a new audit output directory')
    trial = json.loads((run/'report.json').read_text(encoding='utf-8'))
    experiment = json.loads((run.parent/'preflight.json').read_text(encoding='utf-8'))['experiment']
    dataset_hash = file_sha256(dataset)
    if (dataset_hash != experiment['dataset_sha256']
            or file_sha256(run/'checkpoint.pt') != trial['checkpoint_sha256']):
        raise ValueError('Dataset/checkpoint hash does not match saved experiment')
    checkpoint = torch.load(run/'checkpoint.pt', weights_only=True, map_location='cpu')
    split = checkpoint['pretrain_info']
    if split != trial['pretrain']:
        raise ValueError('Trial split differs from checkpoint')
    data = torch.load(dataset, weights_only=True, map_location='cpu')
    meta = data['metadata']
    # Same content fingerprint as pretrain; file hash and tensor-content hash differ by design.
    digest = hashlib.sha256(json.dumps(meta, sort_keys=True, allow_nan=False).encode())
    for key in ('X', 'Y', 'applied_actions', 'valid', 'behavior'):
        tensor = data[key]
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    if digest.hexdigest() != split['dataset_sha256']:
        raise ValueError('Dataset content differs from checkpoint training data')
    x, y, valid = (data[key].numpy() for key in ('X', 'Y', 'valid'))
    dt = float(meta['dt'])
    if (x.ndim != 3 or x.shape[2] != 74 or y.shape != (*x.shape[:2], 4)
            or valid.shape != x.shape[:2] or valid.dtype != bool
            or not np.isfinite(x).all() or not np.isfinite(y).all()
            or np.any(np.abs(x[..., :2]) > 1) or np.any(x[..., 2:] < 0) or np.any(x[..., 2:] > 1)
            or not valid[:, 0].all() or np.any(valid[:, 1:] & ~valid[:, :-1])
            or not np.isfinite(dt) or dt <= 0):
        raise ValueError('Invalid dataset arrays')
    train, val = split['subset_indices'], split['validation_indices']
    validate_split(train, val, len(x))
    records = meta['episodes']
    if len(records) != len(x) or len({r['seed'] for r in records}) != len(x):
        raise ValueError('Expected distinct recorded episode seeds')
    from generator.generate_reflex_dataset import make_teacher
    if meta['teacher_parameters'].get('noise_std_pwm') != 0:
        raise ValueError('This exact label replay requires a deterministic teacher')
    replay = np.zeros_like(y)
    for i in range(len(x)):
        teacher = make_teacher(meta['teacher_version'], **meta['teacher_parameters'])
        for t in np.flatnonzero(valid[i]):
            replay[i, t] = teacher.step(x[i, t], dt=dt)
    error = np.abs(replay-y)[..., 1:4]
    coverage = condition_masks(x, max_range=meta['tof_max_range_m'], threat_m=meta['distance_threshold_m'])
    rows = []
    for condition in coverage:
        row = {k:v for k, v in condition.items() if not isinstance(v, np.ndarray)}
        for name, ids in (('train', train), ('validation', val)):
            row[name] = {level:summarize_mask(condition[level], valid, y, ids, dt)
                         for level in ('coarse', 'full_band', 'full_band_low_flow')}
        rows.append(row)
    grid = x[..., 2:66].reshape(*x.shape[:2], 8, 8)*meta['tof_max_range_m']
    threat = grid < meta['distance_threshold_m']
    summary = {}
    for name, ids in (('train', train), ('validation', val)):
        masks = dict(all=np.ones_like(valid), any_threat=threat.any(axis=(-2, -1)),
                     near_threat=grid.min(axis=(-2, -1)) <= .45,
                     low_flow=np.linalg.norm(x[..., :2], axis=-1) <= .05)
        summary[name] = {key:summarize_mask(mask, valid, y, ids, dt) for key, mask in masks.items()}
        summary[name]['teacher_replay_max_error_pwm'] = float(error[ids][valid[ids]].max())
        summary[name]['teacher_replay_mismatch_frames'] = int((error[ids].max(-1)[valid[ids]] > 1e-4).sum())
    widths = [r.get('scene_parameters', {}).get('obstacle_width_m') for r in records]
    report = dict(format_version='training-coverage-audit-v1', research_only=True, training_ready=False,
        dataset=str(dataset.resolve()), run=str(run.resolve()), candidate=checkpoint['candidate'],
        dataset_file_sha256=dataset_hash, dataset_content_sha256=digest.hexdigest(),
        checkpoint_sha256=trial['checkpoint_sha256'], split_verified_against_checkpoint=True,
        teacher_version=meta['teacher_version'], teacher_parameters=meta['teacher_parameters'], dt=dt,
        thresholds=dict(threat_m=meta['distance_threshold_m'], distance_tolerance_m=.10,
                        low_flow_norm=.05, sustained_match_s=.5),
        split=dict(train=train, validation=val), summary=summary, coverage=rows,
        episodes=[dict(index=i, split='train' if i in train else 'validation', seed=r['seed'],
                       scenario=r['scenario'], valid_frames=int(valid[i].sum()),
                       obstacle_width_m=widths[i]) for i, r in enumerate(records)],
        limitations=['Sensor coverage bins are descriptive, not scientific admission thresholds.',
                     'Full-band matches still differ in exact values/history from artificial probes.',
                     'Distinct episode seeds do not guarantee statistically independent environments.',
                     'No causal attribution of model error, no retraining, no claim that teacher actions are safe.',
                     'Exact label replay checks implementation consistency, not quality of behavior.'])
    root = Path(__file__).resolve().parents[1]
    report['code_sha256'] = {name:file_sha256(root/name) for name in
        ('scripts/audit_training_coverage.py', 'generator/generate_reflex_dataset.py',
         'generator/reflex_contract.py', 'configs/flight_config.py')}
    lines = ['# Покриття навчальних даних k306', '',
        f"Джерело та split звірено з checkpoint. Teacher: `{meta['teacher_version']}`.", '',
        'У таблиці — кількість кадрів / епізодів лише навчальної вибірки.',
        'Відстань ±0.10 м. Загроза: показник <0.8 м. Низький flow: норма ≤0.05.',
        'Coarse: однобічна загроза, задана кількість зайнятих стовпців, найближча точка у смузі відстаней.',
        'Full band: точна смуга від краю на всю висоту матриці; усі її відстані у заданому діапазоні.', '',
        '| Стимул | Coarse | Full band | Full band + low flow | Епізоди з останньою умовою ≥0.5 с |',
        '|---|---:|---:|---:|---:|']
    for row in rows:
        cells = [f"{row['train'][level]['frames']} / {row['train'][level]['episodes']}"
                 for level in ('coarse', 'full_band', 'full_band_low_flow')]
        lines.append('| '+row['name']+' | '+' | '.join(cells)+
                     f" | {row['train']['full_band_low_flow']['episodes_with_0_5s_run']} |")
    lines += ['', '## Відтворення міток', '']
    for name, s in summary.items():
        lines.append(f"- {name}: {s['all']['frames']} дійсних кадрів; "
                     f"максимальна помилка відтворення roll/pitch/yaw = {s['teacher_replay_max_error_pwm']:.8f} PWM; "
                     f"кадрів із похибкою >0.0001 PWM: {s['teacher_replay_mismatch_frames']}.")
    lines += ['', 'Покриття не доводить причину помилки. Збіг міток не доводить безпечності вчителя.',
              'Повний звіт містить validation, індекси/seed епізодів, тривалості та діапазони цільового yaw.']
    with publication(output) as stage:
        write_json(stage/'report.json', report)
        (stage/'README_UK.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('Condition              train coarse frames/episodes | full band | full band + low flow')
    for row in rows:
        cells = [f"{row['train'][level]['frames']}/{row['train'][level]['episodes']}"
                 for level in ('coarse', 'full_band', 'full_band_low_flow')]
        print(f"{row['name']:24} " + ' | '.join(cells))
    for name, s in summary.items():
        print(f"{name}: {s['all']['frames']} valid frames; teacher replay max error "
              f"{s['teacher_replay_max_error_pwm']:.8f} PWM")
    print(f'Results: {output.resolve()}')
    return report


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=root/'data/diagnostics/reflex_varied_train_20261007.pt')
    parser.add_argument('--run', type=Path, default=root/'data/selection/registry_mean_spatial_v1/trial_00001')
    parser.add_argument('--out-dir', type=Path)
    args = parser.parse_args()
    output = args.out_dir or root/'data/diagnostics'/('training_coverage_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f'))
    torch.set_num_threads(1)
    run_audit(args.dataset, args.run, output)


if __name__ == '__main__':
    main()
