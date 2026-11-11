"""Paired, bounded tau-only adaptation of the existing connectome (research)."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT/'output/bilateral_turn_v1/magnitude'


class TauConstraints:
    """Adapt time constants, never recurrent topology, signed weights or amplitude."""
    def __init__(self, policy):
        from optimizer.mvp_constraints import RecurrentConstraints
        RecurrentConstraints(policy, freeze_core=True)
        self.cell = policy.cfc_network.cell
        self.reference = {name: getattr(self.cell, name).detach().clone()
            for name in ('_w_macro_param', 'A', '_synapse_mask', '_synapse_sign')}
        self.initial_tau = self.cell.tau.detach().clone()
        if not torch.isfinite(self.initial_tau).all() or not ((self.initial_tau >= .005) & (self.initial_tau <= 2.)).all():
            raise ValueError('Initial tau outside declared bounds')
        self.cell.tau_raw.requires_grad_(True)
        self.steps = self.clipped_elements = 0

    @torch.no_grad()
    def __call__(self, policy):
        from core.contracts import TAU_FLOOR_S
        from core.invariants import verify_recurrent_contract
        from optimizer.mvp_constraints import verify_clear_drive
        c = policy.cfc_network.cell
        if c is not self.cell: raise ValueError('Unexpected recurrent cell')
        for name, reference in self.reference.items():
            if not torch.equal(getattr(c, name), reference.to(c.A)):
                raise ValueError('Frozen W/A/topology changed: '+name)
        if c.A.requires_grad or c._w_macro_param.requires_grad or not c.tau_raw.requires_grad:
            raise ValueError('Unexpected core trainability')
        if not torch.isfinite(c.tau_raw).all(): raise ValueError('Nonfinite tau')
        bounds = c.tau_raw.new_tensor([.005, 2.]) - TAU_FLOOR_S
        raw = bounds + torch.log(-torch.expm1(-bounds))
        self.clipped_elements += int(((c.tau_raw < raw[0]) | (c.tau_raw > raw[1])).sum())
        c.tau_raw.clamp_(min=raw[0], max=raw[1])
        verify_recurrent_contract(c); verify_clear_drive(policy)
        self.steps += 1

    def audit(self):
        c = self.cell
        return dict(version='fixed-w-a-adaptive-tau-v1', optimizer_steps=self.steps,
            frozen_parameters=['_w_macro_param', 'A'], tau_bounds_s=[.005, 2.],
            clipped_elements=self.clipped_elements,
            W_unchanged=torch.equal(c._w_macro_param.detach(), self.reference['_w_macro_param']),
            A_unchanged=torch.equal(c.A.detach(), self.reference['A']),
            initial_tau_min_s=float(self.initial_tau.min()), initial_tau_max_s=float(self.initial_tau.max()),
            final_tau_min_s=float(c.tau.detach().min()), final_tau_max_s=float(c.tau.detach().max()),
            max_tau_change_s=float((c.tau.detach()-self.initial_tau).abs().max()))


def make_optimizer(policy, *, train_tau):
    tau = policy.cfc_network.cell.tau_raw
    if tau.requires_grad != train_tau: raise ValueError('Unexpected tau trainability')
    interface = [p for p in policy.parameters() if p.requires_grad and p is not tau]
    groups = [dict(params=interface, lr=.001)]
    if train_tau: groups.append(dict(params=[tau], lr=.01))
    return torch.optim.Adam(groups)


@torch.no_grad()
def history_probe(policy):
    # Same histories as prior capacity study, relabeled with current cruise1530.
    # Evaluation only: not added to fitting or fed teacher state.
    from optimizer.probe_history_capacity import make_cases, capacity_metrics
    from optimizer.balanced_maneuvers import teacher_targets
    result = {}
    for n in (5, 25, 50, 150):
        case = make_cases((.10, .16), n)
        case['Y'] = teacher_targets(case['X'])
        result[str(n)] = capacity_metrics(policy, case)
    return result


def run(out):
    from artifacts.research import load_research_run
    from edge.runtime import sha256
    from edge.validate_numpy import validate
    from optimizer.continue_flow_reflex import DATA, write, export_and_verify, PARENT as ORIGINAL
    from optimizer.release_cruise import fit, cruise_cases, clear_trial, motion_admission
    from optimizer.bilateral_turn import bilateral_mask, magnitude_loss, admission
    from optimizer.finalize_mvp_candidate import evaluate_phase, comparison_row, save_phase
    from optimizer.mvp_constraints import RecurrentConstraints
    from optimizer.probe_history_capacity import cruise
    from evaluate.practical_maneuver import trial, CASES

    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    dataset = ROOT/'output/balanced_maneuver_v1/train.pt'
    train = torch.load(dataset, weights_only=True, map_location='cpu')
    mask = bilateral_mask(train['X'], train['Y'], train['valid'])
    assert len(train['X']) == 104 and int(mask.sum()) == 428
    assert train['metadata']['teacher_parameters']['pwm_cruise_pitch'] == 1530.
    mass = float(mask.sum()) / len(train['X'])
    _, original = load_research_run(ORIGINAL); info = original['pretrain_info']
    data = torch.load(DATA/'train.pt', weights_only=True, map_location='cpu')
    sample = data['X'][info['validation_indices'][:2]]
    config = json.loads((PARENT/'report.json').read_text())['config']
    active = ROOT/'data/edge/k306_tau025_integrated_v1'
    paths = {case: next((ROOT/'flight simulations/changed contract').glob('*'+case+'*.json')) for case in CASES}
    files = [dataset, PARENT/'checkpoint.pt', ORIGINAL/'checkpoint.pt', DATA/'train.pt',
        ROOT/'configs/mvp_runtime_v3.json', *paths.values(), *[p for p in active.iterdir() if p.is_file()]]
    hashes = {str(p): sha256(p) for p in files}
    write(out/'protocol.json', dict(version='tau-only-paired-v1', parent=str(PARENT), input_sha256=hashes,
        sources={str(p.relative_to(ROOT)):sha256(p) for folder in ('optimizer','simulation','core','edge','evaluate','generator','demo') for p in (ROOT/folder).glob('*.py')},
        arms=['control', 'adaptive_tau'], epochs=10, updates=130, batch_size=8, seed=42,
        interface_lr=.001, tau_lr=.01, tau_bounds_s=[.005, 2.], initial_tau_scaling=False,
        loss='channel-v4 + 10 * cruise navigation MSE + bilateral magnitude MSE (coefficient 1)',
        rationale='Isolate tau plasticity from W plasticity; same lr_tau and bounds as earlier plastic-history study, no search.',
        history_probe='Evaluation only, current teacher1530, suffixes .1/.5/1/3 seconds; no new training data.',
        room_screen='Existing bilateral_turn admission, same four known rooms if passed; no changes to thresholds.',
        limitations='Finite local optimization, one seed, no guarantee tau reaches long-memory regime. Success is not a general memory or flight guarantee.',
        no_promotion=True))
    p, _ = load_research_run(PARENT)
    initial = evaluate_phase(p, data, info, config)
    initial['history'] = history_probe(p); initial['cruise'] = cruise(p)
    write(out/'initial.json', initial)
    auxiliary = cruise_cases(); results = {}; flights = []
    for arm in ('control', 'adaptive_tau'):
        p, ck = load_research_run(PARENT)
        guard = TauConstraints(p) if arm == 'adaptive_tau' else RecurrentConstraints(p, freeze_core=True)
        initial_tau = p.cfc_network.cell.tau.detach().numpy().copy()
        torch.manual_seed(42)
        def progress(rows):
            write(out/(arm+'_training.json'), rows); print(arm, rows[-1], flush=True)
        def extra(pred, target, ids):
            return magnitude_loss(pred, target, mask[ids], normalizer=mass*len(ids))
        training = fit(p, train, guard, auxiliary, progress, extra_objective=extra,
            optimizer=make_optimizer(p, train_tau=arm == 'adaptive_tau'))
        # fit's historical SETTINGS describe frozen-core training. Override all
        # changed declarations explicitly before persisting this experiment.
        training.update(frozen=['A', 'W'] if arm == 'adaptive_tau' else ['tau_raw','A','W'],
            lr_interface=.001, lr_tau=.01 if arm == 'adaptive_tau' else None,
            loss='channel-v4 + 10 * cruise navigation MSE + bilateral magnitude MSE',
            loss_contract=deepcopy(ck['pretrain_info']['loss_contract']))
        audit = guard.audit(); p.eval()
        metrics = evaluate_phase(p, data, info, config)
        quiet = cruise(p); history = history_probe(p)
        np.savez_compressed(out/(arm+'_tau.npz'), before=initial_tau, after=p.cfc_network.cell.tau.detach().numpy())
        ck['initial_pretrain_info'] = deepcopy(ck['pretrain_info'])
        ck['pretrain_info'] = dict(training, dt=.02, dataset_sha256=hashes[str(dataset)],
            teacher_version=train['metadata']['teacher_version'], teacher_parameters=train['metadata']['teacher_parameters'],
            subset_size=104, subset_indices=list(range(104)), learned_channels=['roll','pitch','yaw'])
        ck['continuation'] = dict(experiment='tau-only-paired-v1', arm=arm,
            parent_sha256=hashes[str(PARENT/'checkpoint.pt')], training_constraint_audit=audit,
            export_freezing='All core parameters frozen only AFTER training; see training audit.')
        report = dict(candidate=ck['candidate'], config=config, continuation_epochs=10, **metrics,
            cruise=quiet, history=history, training_constraint_audit=audit, research_only=True,
            training_ready=False, pretrain=ck['pretrain_info'])
        exported = RecurrentConstraints(p, freeze_core=True)
        save_phase(out/arm, policy=p, checkpoint=ck, parent_directory=PARENT, report=report, constraints=exported, sample=sample)
        export_and_verify(out/arm, sample)
        write(out/arm/'bundled_parity.json', validate(out/arm, out/arm/'numpy', steps=400, execution='bundled'))
        clear = clear_trial(out/arm, out/arm/'clear')
        motion = motion_admission(quiet['final_pwm'][2], clear); temporal = comparison_row(report)
        screen = admission(motion, temporal)
        results[arm] = dict(temporal=temporal, cruise=quiet, history=history, training_audit=audit,
            clear=clear, motion_admission=motion, room_screen=screen)
        write(out/'comparison.json', results); print(arm, 'screen', screen, flush=True)
        if screen['passed']:
            dest = out/arm/'flights'; dest.mkdir()
            for case in CASES:
                row = trial(case, paths[case], 'candidate', dict(before=ROOT/'output/balanced_maneuver_v1/before',candidate=out/arm), dest/case)
                row['arm'] = arm; flights.append(row); write(out/'flights.json', flights)
                print(arm, case, row['metrics'], flush=True)
    assert all(sha256(Path(p)) == h for p, h in hashes.items())
    write(out/'validation.json', dict(completed_arms=2, completed_room_flights=len(flights),
        input_and_active_bundle_unchanged=True, no_promotion=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args().out)
