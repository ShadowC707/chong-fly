"""Training-only teacher-state supervision, without new inference inputs or paths."""
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
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT/'output/tau_continuation_v1/adaptive_tau'
STAGES = (5, 25, 50, 100, 150)
STATE_WEIGHT = .02


def teacher_labels(x, valid):
    """Post-observation pending state: 0 none, 1 left, 2 right; -100 padding."""
    from generator.generate_reflex_dataset import make_teacher
    if (x.ndim != 3 or x.shape[-1] != 74 or valid.shape != x.shape[:2]
            or valid.dtype != torch.bool or x.device.type != 'cpu'):
        raise ValueError('Expected CPU episodes and a boolean valid prefix')
    if ((~valid[:, :-1]) & valid[:, 1:]).any() or not valid[:, 0].all():
        raise ValueError('Valid frames must be a nonempty prefix per episode')
    labels = torch.full(valid.shape, -100, dtype=torch.long)
    y = torch.zeros(*valid.shape, 4)
    for i in range(len(x)):
        teacher = make_teacher('geometry-reflex-v5', noise_std_pwm=0, pwm_cruise_pitch=1530.)
        for t in range(int(valid[i].sum())):
            y[i,t] = torch.from_numpy(teacher.step(x[i,t].numpy(), dt=.02))
            sign = teacher._escape_sign
            labels[i,t] = 0 if sign is None else (1 if sign < 0 else 2)
    return dict(state=labels, Y=y)


def forward_hidden(policy, x):
    """Observe the existing DN input; no reimplementation of policy dynamics."""
    if x.ndim != 3: raise ValueError('Expected batched sequences')
    captured = []
    handle = policy.dn_head.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
    try:
        pwm, _ = policy(x, dt=.02)
    finally:
        handle.remove()
    if len(captured) != 1: raise ValueError('Expected one sequence DN readout')
    return pwm, captured[0].reshape(*x.shape[:2], -1)


class StateReadout(torch.nn.Module):
    """Diagnostic classifier only. It is never connected to motor commands."""
    def __init__(self, amplitude):
        super().__init__()
        if amplitude.ndim != 1 or not torch.isfinite(amplitude).all():
            raise ValueError('Expected finite neuronal amplitudes')
        self.register_buffer('scale', amplitude.detach().abs().clamp_min(1e-6).clone())
        self.linear = torch.nn.Linear(len(amplitude), 3)
        torch.nn.init.normal_(self.linear.weight, std=.02)
        torch.nn.init.zeros_(self.linear.bias)

    def forward(self, hidden):
        return self.linear(hidden / self.scale)

    @torch.no_grad()
    def project(self):
        norm = self.linear.weight.norm()
        if not torch.isfinite(norm): raise ValueError('Nonfinite state readout')
        if norm > 4.:
            self.linear.weight.mul_((4./norm)*(1-4*torch.finfo(norm.dtype).eps))


def state_loss(logits, labels, valid, *, normalizer, class_weights):
    if (logits.shape != (*labels.shape,3) or valid.shape != labels.shape or valid.dtype != torch.bool
            or labels.dtype != torch.long or not math.isfinite(normalizer) or normalizer <= 0
            or class_weights.shape != (3,) or not torch.isfinite(class_weights).all()
            or (class_weights < 0).any() or ((labels[valid] < 0) | (labels[valid] > 2)).any()):
        raise ValueError('Invalid state supervision contract')
    target = labels.masked_fill(~valid, -100)
    values = F.cross_entropy(logits.reshape(-1,3), target.reshape(-1), weight=class_weights,
        ignore_index=-100, reduction='none').reshape(labels.shape)
    return values.sum() / (normalizer*math.log(3.))


def class_weights(labels, valid):
    counts = torch.bincount(labels[valid], minlength=3).float()
    if (counts == 0).any(): raise ValueError('All three states required in training source')
    return float(valid.sum())/(3.*counts)


def histories(flows, suffix):
    from optimizer.probe_history_capacity import make_cases
    result = make_cases(flows, suffix)
    result['valid'] = torch.ones(result['X'].shape[:2], dtype=torch.bool)
    result.update(teacher_labels(result['X'], result['valid']))
    return result


@torch.no_grad()
def classification(logits, labels, valid):
    pred = logits.argmax(-1)
    confusion = torch.bincount((3*labels[valid]+pred[valid]),minlength=9).reshape(3,3)
    counts = confusion.sum(1)
    recalls = [float(confusion[i,i]/counts[i]) if counts[i] else None for i in range(3)]
    present = [v for v in recalls if v is not None]
    return dict(frames=int(valid.sum()), confusion=confusion.tolist(), recall=recalls,
        balanced_accuracy=float(np.mean(present)) if present else None,
        accuracy=float((pred[valid] == labels[valid]).float().mean()) if valid.any() else None)


@torch.no_grad()
def evaluate_history(policy, readout, cases):
    result = {}
    for n, case in cases.items():
        pwm, h = forward_hidden(policy, case['X']); logits = readout(h)
        suffix = torch.zeros_like(case['valid']); suffix[:,50:] = True
        terminal = torch.zeros_like(suffix); terminal[:,-5:] = True
        error = (pwm[...,2:4]-case['Y'][...,2:4]).abs()
        groups = pwm[:,-1,2].reshape(-1,3)
        result[str(n)] = dict(state=classification(logits,case['state'],suffix),
            terminal_state=classification(logits,case['state'],terminal),
            suffix_pitch_mae_pwm=float(error[...,0][suffix].mean()),
            suffix_pitch_max_pwm=float(error[...,0][suffix].max()),
            suffix_yaw_mae_pwm=float(error[...,1][suffix].mean()),
            suffix_yaw_max_pwm=float(error[...,1][suffix].max()),
            terminal_pitch_span_pwm=float((groups.amax(1)-groups.amin(1)).max()))
    return result


def fit(policy, readout, data, labels, guard, auxiliary, history, *, supervise, progress):
    from optimizer.navigation_loss import navigation_weights, navigation_loss
    from optimizer.release_cruise import cruise_loss
    from optimizer.bilateral_turn import bilateral_mask, magnitude_loss
    from optimizer.tau_continuation import make_optimizer
    x, y, valid, behavior = (data[k] for k in ('X','Y','valid','behavior'))
    weights = navigation_weights(y,valid,behavior,len(data['metadata']['behavior_names']),
        yaw_balance=.7,amplitude_bins=True,recovery_steps=50)
    bilateral = bilateral_mask(x,y,valid); mass = float(bilateral.sum())/len(x)
    per_episode = float(valid.sum())/len(x)
    state_weights = class_weights(labels,valid)
    core_opt = make_optimizer(policy,train_tau=True)
    head_opt = torch.optim.Adam(readout.parameters(),lr=.01)
    rng = np.random.default_rng(42); rows = []
    for epoch in range(20):
        case = history[STAGES[epoch//4]]
        history_weights = class_weights(case['state'],case['valid'])
        order = rng.permutation(len(x)); sums = np.zeros(5); batches = 0
        for start in range(0,len(x),8):
            ids = order[start:start+8]
            policy.train(); core_opt.zero_grad(set_to_none=True); head_opt.zero_grad(set_to_none=True)
            pred, hidden = forward_hidden(policy,x[ids])
            hp, hh = forward_hidden(policy,case['X'])
            recorded = navigation_loss(pred,y[ids],weights[ids],normalizer=per_episode*len(ids)*3.)
            quiet = cruise_loss(policy(auxiliary['X'],dt=.02)[0],auxiliary['Y'])
            magnitude = magnitude_loss(pred,y[ids],bilateral[ids],normalizer=mass*len(ids))
            hist = ((hp[...,2:4]-case['Y'][...,2:4])/500.).square().mean()
            sh, hs = (hidden, hh) if supervise else (hidden.detach(), hh.detach())
            state = .5*(state_loss(readout(sh),labels[ids],valid[ids],
                normalizer=per_episode*len(ids),class_weights=state_weights)+
                state_loss(readout(hs),case['state'],case['valid'],
                normalizer=float(case['valid'].sum()),class_weights=history_weights))
            # The control trains an equally initialized diagnostic head but its
            # gradient cannot enter policy, including via global clipping.
            loss = recorded+10*quiet+magnitude+hist+STATE_WEIGHT*state
            if not torch.isfinite(loss): raise ValueError('Nonfinite mixed loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(),1.,error_if_nonfinite=True)
            torch.nn.utils.clip_grad_norm_(readout.parameters(),1.,error_if_nonfinite=True)
            core_opt.step(); policy.post_step(); guard(policy)
            head_opt.step(); readout.project()
            sums += [float(v.detach()) for v in (recorded,quiet,magnitude,hist,state)]; batches += 1
        rows.append(dict(epoch=epoch+1,updates=guard.steps,suffix_ticks=STAGES[epoch//4],
            losses=dict(zip(('recorded','cruise','bilateral','history_xy','state'),(sums/batches).tolist()))))
        progress(rows)
    assert guard.steps == 260
    policy.eval(); readout.eval()
    return rows


def run(out):
    from artifacts.research import load_research_run
    from edge.runtime import sha256
    from edge.validate_numpy import validate
    from optimizer.continue_flow_reflex import DATA, write, export_and_verify, PARENT as ORIGINAL
    from optimizer.release_cruise import cruise_cases, clear_trial, motion_admission
    from optimizer.tau_continuation import TauConstraints
    from optimizer.bilateral_turn import admission
    from optimizer.finalize_mvp_candidate import evaluate_phase, comparison_row, save_phase
    from optimizer.mvp_constraints import RecurrentConstraints
    from optimizer.probe_history_capacity import cruise
    from evaluate.practical_maneuver import trial, CASES
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    out = Path(out); out.mkdir(parents=True,exist_ok=False)
    dataset = ROOT/'output/balanced_maneuver_v1/train.pt'
    data = torch.load(dataset,weights_only=True,map_location='cpu')
    labels = teacher_labels(data['X'],data['valid'])
    delta = float((labels['Y'][data['valid']]-data['Y'][data['valid']]).abs().max())
    if delta > 1e-3: raise ValueError(f'Teacher replay differs from existing labels: {delta}')
    history = {n:histories((.06,.10,.14,.18),n) for n in STAGES}
    evaluation = {n:histories((.08,.12,.16,.20),n) for n in (25,50,150,225,400)}
    torch.save(dict(state=labels['state'],training_history=history,evaluation_history=evaluation,
        teacher_version='geometry-reflex-v5',cruise_pitch=1530.,labels_after_step=True),out/'state_labels.pt')
    active = ROOT/'data/edge/k306_tau025_integrated_v1'
    paths = {case:next((ROOT/'flight simulations/changed contract').glob('*'+case+'*.json')) for case in CASES}
    files = [dataset,PARENT/'checkpoint.pt',DATA/'train.pt',ORIGINAL/'checkpoint.pt',
        out/'state_labels.pt',ROOT/'configs/mvp_runtime_v3.json',*paths.values(),*[p for p in active.iterdir() if p.is_file()]]
    hashes = {str(p):sha256(p) for p in files}
    write(out/'protocol.json',dict(version='teacher-state-supervision-v1',parent=str(PARENT),input_sha256=hashes,
        source_sha256={str(p.relative_to(ROOT)):sha256(p) for folder in ('optimizer','simulation','core','edge','evaluate','generator','demo') for p in (ROOT/folder).glob('*.py')},
        teacher_pwm_max_replay_error=delta,label_counts=torch.bincount(labels['state'][data['valid']],minlength=3).tolist(),
        label_contract='After teacher.step: 0=no escape, 1=escape_sign<0, 2=escape_sign>0; never a policy input.',
        arms=['detached_control','state_supervised'],seed=42,epochs=20,updates=260,batch_size=8,
        history_curriculum_ticks=list(STAGES),epochs_per_stage=4,history_train_flows=[.06,.10,.14,.18],
        history_eval_flows=[.08,.12,.16,.20],history_eval_ticks=list(evaluation),
        loss='channel-v4 + 10*cruise MSE + bilateral magnitude MSE + full history Pitch/Yaw MSE + .02*balanced state CE/log(3)',
        state_weight=STATE_WEIGHT,state_weight_reason='At chance, weighted CE/log(3) is 1; .02 is comparable to existing action-loss scale. Fixed once, no search.',
        state_readout='Linear 306->3 on h/abs(A), Frobenius<=4, discarded for motor inference; equally trained detached control.',
        optimizer=dict(interface_lr=.001,tau_lr=.01,head_lr=.01,clip_policy=1.,clip_head_separately=1.),
        frozen=['W','A'],tau_bounds_s=[.005,2.],no_initial_tau_scaling=True,
        room_screen='Existing bilateral admission plus clear motion; history scores reported separately, no threshold tuning.',
        limits='Two known-parent continuations, one seed, known rooms. Additional history action loss shared by both arms, so comparison to parent is confounded; paired difference isolates auxiliary state gradients.',
        no_promotion=True))
    _, original = load_research_run(ORIGINAL); info = original['pretrain_info']
    old = torch.load(DATA/'train.pt',weights_only=True,map_location='cpu')
    sample = old['X'][info['validation_indices'][:2]]
    config = json.loads((PARENT/'report.json').read_text())['config']
    auxiliary = cruise_cases(); results = {}; flights = []
    for arm in ('detached_control','state_supervised'):
        p, ck = load_research_run(PARENT); guard = TauConstraints(p)
        torch.manual_seed(42); head = StateReadout(p.cfc_network.cell.A)
        def progress(rows):
            write(out/(arm+'_training.json'),rows); print(arm,rows[-1],flush=True)
        rows = fit(p,head,data,labels['state'],guard,auxiliary,history,
            supervise=arm=='state_supervised',progress=progress)
        audit = guard.audit(); metrics = evaluate_phase(p,old,info,config)
        quiet = cruise(p); measured = evaluate_history(p,head,evaluation)
        recorded_metrics = []
        with torch.no_grad():
            logits = torch.cat([head(forward_hidden(p,data['X'][i:i+8])[1]) for i in range(0,len(data['X']),8)])
            clear_pending = data['valid'] & (labels['state']>0) & (data['X'][...,2:66].amin(-1)*3.5>=1.)
            recorded_metrics = dict(all=classification(logits,labels['state'],data['valid']),
                clear_pending=classification(logits,labels['state'],clear_pending))
        ck['initial_pretrain_info'] = deepcopy(ck['pretrain_info'])
        ck['pretrain_info'] = dict(experiment='teacher-state-supervision-v1',arm=arm,epochs=20,updates=260,
            dt=.02,dataset_sha256=hashes[str(dataset)],state_labels_sha256=hashes[str(out/'state_labels.pt')],
            frozen=['W','A'],teacher_version=data['metadata']['teacher_version'],
            teacher_parameters=data['metadata']['teacher_parameters'],subset_size=104,subset_indices=list(range(104)),
            validation_indices=[],progress=rows,research_only=True,loss_contract=json.loads((out/'protocol.json').read_text())['loss'])
        ck['continuation'] = dict(experiment='teacher-state-supervision-v1',arm=arm,parent_sha256=hashes[str(PARENT/'checkpoint.pt')],
            training_constraint_audit=audit,auxiliary_head_in_inference=False,
            export_freezing='Core frozen only after fit; see training_constraint_audit.')
        report = dict(candidate=ck['candidate'],config=config,continuation_epochs=20,**metrics,cruise=quiet,
            history=measured,state_classification=recorded_metrics,training_constraint_audit=audit,
            research_only=True,training_ready=False,pretrain=ck['pretrain_info'])
        frozen = RecurrentConstraints(p,freeze_core=True)
        save_phase(out/arm,policy=p,checkpoint=ck,parent_directory=PARENT,report=report,constraints=frozen,sample=sample)
        torch.save(dict(state_dict=head.state_dict(),inference=False,reason='Training-only diagnostic readout'),out/arm/'state_readout.pt')
        export_and_verify(out/arm,sample)
        write(out/arm/'bundled_parity.json',validate(out/arm,out/arm/'numpy',steps=400,execution='bundled'))
        clear = clear_trial(out/arm,out/arm/'clear')
        motion = motion_admission(quiet['final_pwm'][2],clear); temporal = comparison_row(report)
        screen = admission(motion,temporal)
        results[arm] = dict(temporal=temporal,cruise=quiet,history=measured,state_classification=recorded_metrics,
            training_audit=audit,head_frobenius=float(head.linear.weight.detach().norm()),clear=clear,room_screen=screen)
        write(out/'comparison.json',results); print(arm,'screen',screen,flush=True)
        if screen['passed']:
            dest = out/arm/'flights'; dest.mkdir()
            for case in CASES:
                row = trial(case,paths[case],'candidate',dict(before=ROOT/'output/balanced_maneuver_v1/before',candidate=out/arm),dest/case)
                row['arm']=arm; flights.append(row); write(out/'flights.json',flights)
                print(arm,case,row['metrics'],flush=True)
    assert all(sha256(Path(p))==h for p,h in hashes.items())
    write(out/'validation.json',dict(completed_arms=2,completed_room_flights=len(flights),
        inputs_and_active_package_unchanged=True,auxiliary_head_not_exported=True,no_promotion=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    run(parser.parse_args().out)
