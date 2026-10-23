"""Inspect the actual trained input path; never change encoding or checkpoint.

Captures SensorInputLayer and routed W_in in the real forward pass, then
checks them against the declared arithmetic. This is an exploratory audit.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from configs.flight_config import APF_DISTANCE_THRESHOLD_M, TOF_RAYCASTER_MAX_RANGE_M
from artifacts.io import publication, write_json
from generator.source_contract import file_sha256
from artifacts.research import load_research_run
from optimizer.functional_probe import make_stimuli, trace_response, WARMUP, ACTIVE
from simulation.policy import SensorInputLayer


def boundary_inputs():
    _, obs = make_stimuli()
    pair = obs[1, WARMUP].repeat(2, 1)
    # Last ray is remote from the eight near rays; only it crosses the threshold.
    pair[0, 65] = (APF_DISTANCE_THRESHOLD_M+.0001)/TOF_RAYCASTER_MAX_RANGE_M
    pair[1, 65] = (APF_DISTANCE_THRESHOLD_M-.0001)/TOF_RAYCASTER_MAX_RANGE_M
    return pair


@torch.no_grad()
def inspect_pipeline(policy, obs):
    cell = policy.cfc_network.cell
    if cell.connectivity != 'structured' or policy.sensor_layer.encoding != 'proximity-mean-v1':
        raise ValueError('This version audits structured proximity-mean-v1 checkpoints')
    encoded, projected = [], []
    def capture_encoded(module, inputs, output):
        encoded.append(output.detach().cpu().clone())
    def capture_projection(module, inputs, output):
        projected.append(output.detach().cpu().clone())
    handles = [policy.sensor_layer.register_forward_hook(capture_encoded),
               cell.W_in.register_forward_hook(capture_projection)]
    try:
        pwm, states = trace_response(policy, obs)
    finally:
        for handle in handles:
            handle.remove()
    if len(encoded) != 1 or len(projected) != obs.shape[1]:
        raise ValueError('Unexpected forward capture layout')
    actual_encoded = encoded[0].reshape_as(obs)
    actual_projection = torch.stack(projected, dim=1)
    base_layer = SensorInputLayer(sensor_dim=74, learnable_scale=False,
                                 encoding=policy.sensor_layer.encoding)
    pre_affine = base_layer(obs.cpu())
    expected_encoded = pre_affine*policy.sensor_layer.gain.detach().cpu()+policy.sensor_layer.bias.detach().cpu()
    weight = (cell.W_in.weight*cell.W_in.route_mask).detach().cpu()
    bias = (cell.W_in.bias*cell.W_in.bias_mask).detach().cpu()
    expected_projection = F.linear(expected_encoded, weight, bias)
    torch.testing.assert_close(actual_encoded, expected_encoded, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(actual_projection, expected_projection, rtol=1e-5, atol=1e-6)
    delta = actual_projection-actual_projection[0:1]
    allowed = cell.W_in.route_mask[:, 2:66].any(-1).cpu()
    leakage = float(delta[..., ~allowed].abs().max()) if (~allowed).any() else 0.
    # Actual old-state recurrence for the structured rate equation. This
    # reconstruction is for diagnosis, not a substitute inference path.
    previous = torch.cat([torch.zeros_like(torch.from_numpy(states[:, :1])),
                          torch.from_numpy(states[:, :-1])], dim=1)
    recurrent_weight = cell._effective_W().detach().cpu()
    if cell.mode == 'masked':
        recurrent_weight = recurrent_weight*cell._synapse_mask.cpu()
    recurrent = previous @ recurrent_weight
    drive = recurrent+actual_projection+cell.b.detach().cpu()
    arrays = dict(raw=obs.cpu().numpy(), pre_affine=pre_affine.numpy(),
        encoded=actual_encoded.numpy(), projection=actual_projection.numpy(),
        projection_delta=delta.numpy(), recurrent_drive=recurrent.numpy(),
        total_drive=drive.numpy(), states=states, pwm=pwm,
        input_weight=weight.numpy(), input_gain=policy.sensor_layer.gain.detach().cpu().numpy(),
        input_bias=policy.sensor_layer.bias.detach().cpu().numpy(), looming_indices=np.flatnonzero(allowed.numpy()))
    for value in arrays.values():
        if not np.isfinite(value).all():
            raise ValueError('Nonfinite encoding audit')
    report = dict(captured_pipeline_matches_forward=True,
        encoded_max_reconstruction_error=float((actual_encoded-expected_encoded).abs().max()),
        projection_max_reconstruction_error=float((actual_projection-expected_projection).abs().max()),
        direct_tof_leak_outside_looming_max_abs=leakage,
        units='Projection/drive/state units are arbitrary model units, not biological rates or PWM.')
    return report, arrays


def run_audit(run, output):
    run, output = Path(run), Path(output)
    if output.exists():
        raise FileExistsError('Use a new audit output directory')
    policy, checkpoint = load_research_run(run)
    cases, obs = make_stimuli()
    report, arrays = inspect_pipeline(policy, obs)
    ids = arrays['looming_indices']
    rows = []
    for i, case in enumerate(cases):
        t = WARMUP+ACTIVE-1
        pre = arrays['pre_affine'][i, WARMUP, 2:66]
        affine_delta = arrays['encoded'][i, WARMUP, 2:66]-arrays['encoded'][0, WARMUP, 2:66]
        projection_delta = arrays['projection_delta'][i, WARMUP, ids]
        rows.append(dict(**case, pre_affine_tof_sum=float(pre.sum()), pre_affine_tof_max=float(pre.max()),
            affine_tof_delta_sum=float(affine_delta.sum()), affine_tof_delta_l2=float(np.linalg.norm(affine_delta)),
            input_projection_delta_l2=float(np.linalg.norm(projection_delta)),
            looming_near_saturation_fraction=float((abs(np.tanh(arrays['total_drive'][i, t, ids])) > .95).mean()),
            final_yaw_pwm=float(arrays['pwm'][i, t, 3])))
    with torch.no_grad():
        boundary = boundary_inputs()
        base = SensorInputLayer(learnable_scale=False, encoding='proximity-mean-v1')(boundary)
        normalized = policy.sensor_layer(boundary)
        projected = policy.cfc_network.cell.W_in(normalized)
    crossing = dict(distance_change_m=.0002, threat_threshold_m=APF_DISTANCE_THRESHOLD_M,
        active_rays_before=int((base[0, 2:66] > 0).sum()), active_rays_after=int((base[1, 2:66] > 0).sum()),
        first_near_ray_before=float(base[0, 2]), first_near_ray_after=float(base[1, 2]),
        first_near_ray_relative_change=float(base[1, 2]/base[0, 2]-1),
        pre_affine_tof_sum_before=float(base[0, 2:66].sum()), pre_affine_tof_sum_after=float(base[1, 2:66].sum()),
        input_projection_change_l2=float(torch.linalg.vector_norm(projected[1]-projected[0])))
    arrays.update(boundary_raw=boundary.numpy(), boundary_pre_affine=base.numpy(),
                  boundary_encoded=normalized.numpy(), boundary_projection=projected.numpy())
    original = json.loads((run/'report.json').read_text(encoding='utf-8'))
    stored = {(r['obstacle_side'], r['normalized_distance'], r['active_columns']):r['threat_yaw_pwm']
              for r in original['spatial_response']['cases']}
    differences = [abs(row['final_yaw_pwm']-stored[(row['side'], row['normalized_distance'], row['width'])])
                   for row in rows if row['side']]
    root = Path(__file__).resolve().parents[1]
    report.update(format_version='sensor-encoding-audit-v1', research_only=True, training_ready=False,
        candidate=checkpoint['candidate'], checkpoint_sha256=file_sha256(run/'checkpoint.pt'),
        run=str(run.resolve()), encoding=policy.sensor_layer.encoding, cases=rows, boundary_probe=crossing,
        historical_spatial_probe_max_difference_pwm=max(differences),
        code_sha256={name:file_sha256(root/name) for name in
            ('scripts/audit_sensor_encoding.py', 'optimizer/functional_probe.py', 'optimizer/benchmark_candidate.py',
             'simulation/policy.py', 'core/models.py', 'core/routing.py', 'core/contracts.py', 'configs/flight_config.py', 'artifacts/research.py', 'artifacts/io.py', 'core/invariants.py')},
        limitations=['Exploratory audit of one trained checkpoint; no cause of behavioral failure proven.',
                     'Equal input sum does not imply equal per-neuron drive or behavior.',
                     'Hard support normalization is discontinuous at its threshold; no replacement tested here.',
                     'Routing is an engineering assumption, not validated retinotopy.',
                     'Saturation fraction is descriptive; no admission threshold.'])
    lines = ['# Аудит кодування сенсорів k306', '',
        'Вхід → proximity-mean-v1 → навчені gain/bias → маскована проєкція в клітини.',
        'Перевірено фактичний forward hooks; ваги та кодування не змінювалися.', '',
        '| Випадок | Сума до gain/bias | Максимум пікселя | Норма зміни входу в клітини | Yaw наприкінці стимулу |',
        '|---|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['name']} | {row['pre_affine_tof_sum']:.4f} | {row['pre_affine_tof_max']:.4f} | "
                     f"{row['input_projection_delta_l2']:.4f} | {row['final_yaw_pwm']:.2f} |")
    lines += ['', 'Норма входу — опис величини вектора, а не якості або потрібної сили повороту.', '',
        '## Перевірка порога', '',
        f"Один додатковий промінь: 0.8001 → 0.7999 м. Кількість активних променів: 8 → 9.",
        f"Значення вже активного близького пікселя: {crossing['first_near_ray_before']:.4f} → "
        f"{crossing['first_near_ray_after']:.4f} ({100*crossing['first_near_ray_relative_change']:.2f}%).",
        'Це властивість дискретного підрахунку активних пікселів, а не помилка множення матриць.',
        'Такий стрибок може робити кодування чутливим до шуму біля порога; вплив на політ ще не виміряний.', '',
        '## Межі висновку', '',
        'Кодування виконує записану формулу. Це не доводить, що формула вдала або біологічно коректна.',
        'Просторовий розподіл і навчена проєкція різні: однакова сума не гарантує однакової відповіді.',
        'Початкові навчальні дані погано покривають ці проби; причину помилки слід перевіряти контрольованим навчанням.',
        'Відтворення історичного yaw: максимальна розбіжність '
        f"{max(differences):.8f} PWM. Повні дані: report.json та encoding_traces.npz."]
    with publication(output) as stage:
        np.savez_compressed(stage/'encoding_traces.npz', **arrays)
        (stage/'README_UK.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
        report['trace_sha256'] = file_sha256(stage/'encoding_traces.npz')
        write_json(stage/'report.json', report)
    for row in rows:
        print(f"{row['name']:22} sum={row['pre_affine_tof_sum']:.4f} "
              f"projection_L2={row['input_projection_delta_l2']:.4f} yaw={row['final_yaw_pwm']:.2f}")
    print('Boundary probe:', crossing)
    print(f'Results: {output.resolve()}')
    return report


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=root/'data/selection/registry_mean_spatial_v1/trial_00001')
    parser.add_argument('--out-dir', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    output = args.out_dir or root/'data/diagnostics'/('encoding_audit_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f'))
    run_audit(args.run, output)


if __name__ == '__main__':
    main()
