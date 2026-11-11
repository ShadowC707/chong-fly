"""Fixed paired encoding experiment with an identical free-cruise objective."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = dict(seed=42, epochs=10, batch_size=8, updates=130, lr=.001,
    loss='channel-v4 + 10 * cruise navigation MSE', cruise_weight=10.,
    yaw_balance=.7, recovery_steps=50, history_auxiliary=False,
    frozen=['tau_raw', 'A', 'W'], device='cpu', gradient_clip=1.)


def cruise_cases():
    from optimizer.balanced_maneuvers import teacher_targets
    flow = torch.tensor([(0,0),(.01,0),(-.01,0),(.03,0),(.06,0),(.12,0),(-.06,0),(0,.03),(0,-.03)])
    x = torch.ones(len(flow), 200, 74)
    x[:, :, :2] = flow[:, None, :]
    return dict(X=x, Y=teacher_targets(x))


def cruise_loss(prediction, target):
    # PWM order is throttle, roll, pitch, yaw; altitude remains external.
    return ((prediction[..., 1:] - target[..., 1:]) / 500.).square().mean()


def motion_admission(quiet_pitch, clear):
    speed = clear['mean_forward_speed_mps']
    checks = dict(quiet_pitch=math.isfinite(quiet_pitch) and 1515 <= quiet_pitch <= 1535,
        full_horizon=clear['duration_s'] >= 10. - 1e-8,
        no_contact=not clear['crashed'], auto_exercised=clear['auto_s'] >= 8.,
        forward_motion=speed is not None and math.isfinite(speed) and speed >= .10)
    return dict(passed=all(checks.values()), checks=checks,
        reasons=[key for key, passed in checks.items() if not passed])


def fit(policy, data, guard, auxiliary, progress_callback, *, extra_objective=None, optimizer=None):
    from optimizer.navigation_loss import navigation_weights, navigation_loss
    x, y, valid, behavior = (data[k] for k in ('X', 'Y', 'valid', 'behavior'))
    weights = navigation_weights(y, valid, behavior, len(data['metadata']['behavior_names']),
        yaw_balance=.7, amplitude_bins=True, recovery_steps=50)
    normalizer_per_episode = float(valid.sum()) / len(x) * 3
    if optimizer is None:
        optimizer = torch.optim.Adam([p for p in policy.parameters() if p.requires_grad], lr=.001)
    parameters = [p for group in optimizer.param_groups for p in group['params']]
    if (len(set(map(id, parameters))) != len(parameters) or
        set(map(id, parameters)) != {id(p) for p in policy.parameters() if p.requires_grad}):
        raise ValueError('Optimizer must contain every trainable parameter exactly once')
    rng = np.random.default_rng(42)
    progress = []
    for epoch in range(10):
        order = rng.permutation(len(x))
        sums = np.zeros(2)
        extra_sum = 0.
        batches = 0
        for start in range(0, len(x), 8):
            ids = order[start:start+8]
            policy.train(); optimizer.zero_grad(set_to_none=True)
            prediction = policy(x[ids], dt=.02)[0]
            recorded = navigation_loss(prediction, y[ids], weights[ids],
                normalizer=normalizer_per_episode * len(ids))
            quiet = cruise_loss(policy(auxiliary['X'], dt=.02)[0], auxiliary['Y'])
            loss = recorded + 10. * quiet
            if extra_objective is not None:
                extra = extra_objective(prediction, y[ids], ids)
                loss = loss + extra
                extra_sum += float(extra.detach())
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite mixed objective')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1., error_if_nonfinite=True)
            optimizer.step(); policy.post_step(); guard(policy)
            sums += [float(recorded.detach()), float(quiet.detach())]
            batches += 1
        policy.eval()
        with torch.no_grad():
            final = policy(auxiliary['X'][:1], dt=.02)[0][0, -1].tolist()
        progress.append(dict(epoch=epoch+1, updates=guard.steps,
            recorded_loss=float(sums[0]/batches), cruise_loss=float(sums[1]/batches), quiet_final_pwm=final))
        if extra_objective is not None:
            progress[-1]['extra_loss'] = extra_sum / batches
        progress_callback(progress)
    assert guard.steps == 130
    return dict(**SETTINGS, progress=progress, source_episode_count=len(x),
        validation_indices=[], research_only=True,
        sampling='Each epoch has one independent permutation of all 104 complete episodes')


def clear_trial(run_dir, dest):
    from demo.runner import DemoRunner
    from optimizer.continue_flow_reflex import write
    r = DemoRunner(config=run_dir/'runtime_config.json', scene='clear', seed=20261021,
        start=(-.7, 0., 1.), yaw=0., collision_radius_m=.2, max_duration_s=10.,
        contact_model='sphere-sweep-v1', run_id='cruise-screen')
    try:
        r.resume(); r.event(dict(type='mode', mode='AUTO'))
        for _ in range(2500):
            r.advance()
            if r.terminal: break
        time = np.array([v['time_s'] for v in r.rows])
        dt = np.diff(time, append=r.time_s)
        auto = np.array([v['plant_source']=='auto' for v in r.rows])
        velocity = np.array([v['velocity_mps'] for v in r.rows])
        yaw = np.array([v['attitude_rad'][2] for v in r.rows])
        forward = velocity[:,0]*np.cos(yaw) + velocity[:,1]*np.sin(yaw)
        auto_s = float(dt[auto].sum())
        report = dict(duration_s=r.time_s, auto_s=auto_s,
            crashed=any(bool(v['collision']) for v in r.rows),
            mean_forward_speed_mps=float((forward*dt*auto).sum()/auto_s) if auto_s else None,
            final_horizontal_speed_mps=float(np.linalg.norm(velocity[-1,:2])),
            events=r.events)
        dest.mkdir()
        write(dest/'manifest.json', r.manifest())
        write(dest/'report.json', report)
        np.savez_compressed(dest/'trace.npz', time=time, velocity=velocity, yaw=yaw, auto=auto,
            pos=[v['position_m'] for v in r.rows])
        return report
    finally:
        r.env.close()


def run(out):
    from artifacts.research import load_research_run
    from edge.runtime import sha256
    from edge.validate_numpy import validate
    from optimizer.continue_flow_reflex import PARENT, DATA, write, export_and_verify
    from optimizer.finalize_mvp_candidate import evaluate_phase, save_phase, comparison_row
    from optimizer.mvp_constraints import RecurrentConstraints
    from optimizer.release_encoding import migrate_release_encoding
    from optimizer.probe_history_capacity import cruise
    from evaluate.teacher_encoding_contract import histories, teacher_targets
    from evaluate.practical_maneuver import trial, CASES

    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    dataset = ROOT/'output/balanced_maneuver_v1/train.pt'
    train = torch.load(dataset, weights_only=True, map_location='cpu')
    assert len(train['X']) == 104 and train['metadata']['teacher_parameters']['pwm_cruise_pitch'] == 1530.
    _, payload = load_research_run(PARENT)
    info = payload['pretrain_info']
    data = torch.load(DATA/'train.pt', weights_only=True, map_location='cpu')
    config = json.loads((PARENT/'report.json').read_text())['config']
    sample = data['X'][info['validation_indices'][:2]]
    active = ROOT/'data/edge/k306_tau025_integrated_v1'
    paths = {case: next((ROOT/'flight simulations/changed contract').glob('*'+case+'*.json')) for case in CASES}
    files = [dataset, DATA/'train.pt', PARENT/'checkpoint.pt', ROOT/'configs/mvp_runtime_v3.json',
        *paths.values(), *[p for p in active.iterdir() if p.is_file()]]
    hashes = {str(p): sha256(p) for p in files}
    auxiliary = cruise_cases()
    torch.save(auxiliary, out/'cruise.pt')
    write(out/'protocol.json', dict(version='release-cruise-paired-v1', training=SETTINGS,
        input_sha256=hashes, cruise_sha256=sha256(out/'cruise.pt'),
        source_sha256={str(p.relative_to(ROOT)):sha256(p) for folder in
            ('core','simulation','optimizer','generator','demo','edge','evaluate') for p in (ROOT/folder).glob('*.py')},
        arms=['old_encoding','release_encoding'], parent=str(PARENT),
        clear_screen=dict(duration_s=10., min_auto_s=8., min_mean_body_forward_speed_mps=.10,
            quiet_pitch_range=[1515.,1535.], no_contact=True, seed=20261021),
        room_trials=dict(cases=list(CASES), duration_s=45., only_if_motion_screen_passed=True),
        unchanged=['recorded dataset','teacher v5/1530','recurrent core','coordinator','hardware contract'],
        limitations='One seed, known rooms. Compared with preceding experiment, minibatch ordering also differs; no pure causal estimate of auxiliary loss.',
        automatic_promotion=False, no_parameter_search=True))
    results = {}; flights = []
    for arm in ('old_encoding','release_encoding'):
        p, ck = load_research_run(PARENT)
        if arm == 'release_encoding':
            p = migrate_release_encoding(p); ck['sensor_encoding'] = 'proximity-release-v1'
        guard = RecurrentConstraints(p, freeze_core=True)
        initial = evaluate_phase(p, data, info, config)
        with torch.no_grad(): initial['cruise'] = cruise(p)
        write(out/(arm+'_initial.json'), initial)
        torch.manual_seed(42)
        def progress(rows):
            write(out/(arm+'_training.json'), rows)
            print(arm, rows[-1], flush=True)
        training = fit(p, train, guard, auxiliary, progress)
        # Parent diagnostics describe another dataset, so retain them only in
        # initial_pretrain_info, never as current subset/validation statistics.
        p._pretrain_info = dict(training, dt=.02, dataset_sha256=hashes[str(dataset)],
            teacher_version=train['metadata']['teacher_version'],
            teacher_parameters=deepcopy(train['metadata']['teacher_parameters']),
            subset_size=len(train['X']), subset_indices=list(range(len(train['X']))),
            learned_channels=['roll','pitch','yaw'],
            loss_contract=dict(version='channel-v4-plus-cruise-v1', yaw_balance=.7,
                yaw_recovery_window_s=1., cruise_weight=10.,
                normalizer='expected-training-batch-mass'))
        p.eval()
        metrics = evaluate_phase(p, data, info, config)
        with torch.no_grad():
            quiet = cruise(p)
            predictions = p(torch.from_numpy(histories()), dt=.02)[0].numpy()
        witness = dict(final_pwm=predictions[:,-1].tolist(),
            teacher_final_pwm=teacher_targets(histories())[:,-1].tolist())
        ck['continuation'] = dict(experiment='release-cruise-paired-v1', arm=arm,
            parent_sha256=hashes[str(PARENT/'checkpoint.pt')], explicit_encoding_transfer=arm=='release_encoding')
        ck['initial_pretrain_info'] = deepcopy(ck['pretrain_info'])
        ck['pretrain_info'] = deepcopy(p._pretrain_info)
        report = dict(candidate=ck['candidate'], config=dict(config, sensor_encoding=p.sensor_layer.encoding),
            continuation_epochs=10, **metrics, witness=witness, cruise=quiet,
            pretrain=p._pretrain_info, research_only=True, training_ready=False)
        save_phase(out/arm, policy=p, checkpoint=ck, parent_directory=PARENT,
            report=report, constraints=guard, sample=sample)
        export_and_verify(out/arm, sample)
        write(out/arm/'bundled_parity.json', validate(out/arm, out/arm/'numpy', steps=400, execution='bundled'))
        clear = clear_trial(out/arm, out/arm/'clear')
        screen = motion_admission(quiet['final_pwm'][2], clear)
        results[arm] = dict(temporal=comparison_row(report), cruise=quiet, clear=clear,
            motion_admission=screen, witness=witness)
        write(out/'comparison.json', results)
        print(arm, 'motion screen', screen, clear, flush=True)
        if screen['passed']:
            dest = out/arm/'flights'; dest.mkdir()
            runs = dict(before=ROOT/'output/balanced_maneuver_v1/before', candidate=out/arm)
            for case in CASES:
                row = trial(case, paths[case], 'candidate', runs, dest/case)
                row['arm'] = arm; flights.append(row)
                write(out/'flights.json', flights)
                print(arm, case, row['metrics'], flush=True)
    assert all(sha256(Path(p)) == digest for p,digest in hashes.items())
    write(out/'validation.json', dict(completed_arms=len(results), completed_room_flights=len(flights),
        input_and_active_bundle_unchanged=True, automatic_promotion=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args().out)
