"""One fixed four-weight Pitch readout repair; frozen core, no architecture changes."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
from copy import deepcopy
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TRAIN_CASES = ('068fe108', '89d81a9f')
EVAL_CASES = ('9d5431c1', 'b0a9f212')


def fit_readout(groups, reference, *, regularization):
    """Minimize sum of group-mean logit MSE + lambda*||w-reference||²."""
    if regularization <= 0 or not groups:
        raise ValueError('Positive regularization and nonempty groups required')
    reference = np.asarray(reference, dtype=np.float64)
    lhs = regularization * np.eye(len(reference))
    rhs = regularization * reference
    for h, target in groups.values():
        h, target = np.asarray(h, dtype=np.float64), np.asarray(target, dtype=np.float64)
        if len(h) == 0 or target.shape != (len(h),) or h.shape[1] != len(reference):
            raise ValueError('Invalid readout group')
        if not np.isfinite(h).all() or not np.isfinite(target).all():
            raise ValueError('Nonfinite readout group')
        lhs += h.T @ h / len(h)
        rhs += h.T @ target / len(h)
    return np.linalg.solve(lhs, rhs)


def run(out):
    import torch
    from edge.runtime import NumpyPolicy, sha256
    from artifacts.research import load_research_run
    from optimizer.continue_flow_reflex import PARENT, DATA, write, export_and_verify
    from optimizer.finalize_mvp_candidate import evaluate_phase, save_phase, comparison_row
    from optimizer.mvp_constraints import RecurrentConstraints
    from evaluate.practical_maneuver import trial, CASES

    torch.set_num_threads(1)
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    parent = ROOT / 'output/balanced_maneuver_v1/plastic'
    source = ROOT / 'output/practical_maneuver_v1'
    p, payload = load_research_run(parent)
    snapshot = deepcopy(p.state_dict())
    mask = p.dn_head.proj.route_mask[2].cpu().numpy()
    ids = np.flatnonzero(mask)
    assert len(ids) == 4
    reference = p.dn_head.proj.weight[2, ids].detach().numpy().astype(np.float64)
    bias = float(p.dn_head.proj.bias[2].detach())
    hashes = {str(path): sha256(path) for path in [parent/'checkpoint.pt', parent/'report.json',
        *source.glob('*/decisions.npz')]}
    fixed = json.loads((parent.parent/'protocol.json').read_text())['fixed_baseline_sha256']
    protocol = dict(version='recovery-pitch-readout-v1', parent=str(parent), input_sha256=hashes,
        source_sha256=sha256(__file__), training_rooms=TRAIN_CASES, evaluation_rooms=EVAL_CASES,
        trainable=f'Only Pitch decoder weights at {ids.tolist()}; bias and every other parameter frozen',
        objective='Equal group-mean logit MSE: recovery teacher, threat parent, other parent, clear-flow parent; ridge to parent',
        regularization=.001, fit_count=1, recovery_mask='AUTO and min_ToF>=0.95m and flow_x<-0.02',
        teacher='Already saved passive v5/1530; unchanged',
        evaluation='All four existing rooms, same starts/seed/runtime; two excluded from fit but previously inspected',
        promotion=False, limitations='Four-weight feasibility intervention; training rooms are resubstitution, evaluation rooms are not a blind test')
    write(out/'protocol.json', protocol)
    recorded = {}
    for case in CASES:
        for driver in ('before', 'candidate'):
            key = case+'_'+driver
            with np.load(source/key/'decisions.npz') as a:
                model = NumpyPolicy(parent/'numpy'); features=[]; predicted=[]
                for i, x in enumerate(a['X']):
                    if i == 0 or a['segment'][i] != a['segment'][i-1]: model.reset_state()
                    predicted.append(model.step_np(x[:2], x[2:66], x[66:], dt=.02))
                    features.append(model.h[ids].copy())
                np.testing.assert_array_equal(predicted, a['candidate'])
                x = a['X']; distance=x[:, 2:66].min(axis=1)*3.5
                masks=dict(recovery=a['auto'] & (distance>=.95) & (x[:, 0]<-.02),
                           threat=a['auto'] & (distance<.8))
                masks['other']=a['auto'] & ~masks['recovery'] & ~masks['threat']
                recorded[key]=dict(h=np.asarray(features), original=a['candidate'].copy(), teacher=a['teacher'].copy(), masks=masks)
    groups={}
    for group in ('recovery', 'threat', 'other'):
        hs=[];targets=[]
        for key, data in recorded.items():
            if key.split('_')[0] not in TRAIN_CASES: continue
            sel=data['masks'][group]; hs.append(data['h'][sel])
            target=data['teacher'][sel, 2] if group=='recovery' else data['original'][sel, 2]
            targets.append(np.arctanh(np.clip((target-1500.)/500., -.9999, .9999))-bias)
        groups[group]=(np.concatenate(hs), np.concatenate(targets))
    # Preserve clear-flight responses over fixed signed flow values; no new labels here.
    hs=[]
    for fx in (-.2, -.12, -.06, -.03, 0., .03, .06, .12, .2):
        model=NumpyPolicy(parent/'numpy')
        for _ in range(100):
            model.step_np(np.array([fx, 0.], np.float32), np.ones(64, np.float32), dt=.02)
            hs.append(model.h[ids].copy())
    h=np.asarray(hs); groups['clear']=(h, h@reference)
    weights=fit_readout(groups, reference, regularization=.001)
    write(out/'fit.json', dict(reference_weights=reference.tolist(), fitted_weights=weights.tolist(), bias=bias,
        counts={g:len(h) for g,(h,_) in groups.items()},
        loss={g:dict(before=float(np.mean((h@reference-y)**2)), after=float(np.mean((h@weights-y)**2))) for g,(h,y) in groups.items()}))
    np.savez_compressed(out/'fit_data.npz', **{g+'_h':h for g,(h,_) in groups.items()}, **{g+'_target':y for g,(_,y) in groups.items()})
    for param in p.parameters(): param.requires_grad_(False)
    with torch.no_grad(): p.dn_head.proj.weight[2, ids]=torch.from_numpy(weights).to(p.dn_head.proj.weight)
    for key, value in p.state_dict().items():
        if key=='dn_head.proj.weight':
            expected=snapshot[key].clone(); expected[2, ids]=value[2, ids]
            assert torch.equal(value, expected)
        elif torch.is_tensor(value): assert torch.equal(value, snapshot[key]), key
        else: assert value==snapshot[key], key
    raw_report={}
    for key, data in recorded.items():
        pred=1500+500*np.tanh(data['h']@weights+bias)
        raw_report[key]={}
        for group, sel in data['masks'].items():
            raw_report[key][group]=dict(frames=int(sel.sum()),
                before_teacher_mae=float(np.abs(data['original'][sel,2]-data['teacher'][sel,2]).mean()) if sel.any() else None,
                after_teacher_mae=float(np.abs(pred[sel]-data['teacher'][sel,2]).mean()) if sel.any() else None,
                before_reverse_pitch=int((data['original'][sel,2]<1495).sum()),
                after_reverse_pitch=int((pred[sel]<1495).sum()))
    write(out/'recorded_scores.json', raw_report)
    original, old_payload=load_research_run(PARENT)
    config=json.loads((PARENT/'report.json').read_text())['config']
    data=torch.load(DATA/'train.pt', weights_only=True, map_location='cpu')
    print('Four-weight fit done; checking temporal responses and seven scenes', flush=True)
    metrics=evaluate_phase(p, data, old_payload['pretrain_info'], config)
    payload['continuation']=dict(experiment=protocol['version'], parent_sha256=hashes[str(parent/'checkpoint.pt')],
        training_scope=protocol['trainable'], method='One deterministic regularized least-squares solve; no gradient updates')
    payload['pretrain_info']=dict(experiment=protocol['version'], parent_pretrain=payload['pretrain_info'],
                                training_rooms=list(TRAIN_CASES), fitting_steps=1)
    guard=RecurrentConstraints(p, freeze_core=True)
    sample=data['X'][old_payload['pretrain_info']['validation_indices'][:2]]
    report=dict(candidate=payload['candidate'], config=config, continuation_epochs=0, **metrics,
                research_only=True, training_ready=False, continuation=payload['continuation'])
    save_phase(out/'candidate', policy=p, checkpoint=payload, parent_directory=parent, report=report, constraints=guard, sample=sample)
    export_and_verify(out/'candidate', sample)
    write(out/'temporal_summary.json', comparison_row(report))
    print('Temporal', comparison_row(report), flush=True)
    # Verify other command channels exactly in NumPy on recorded inputs.
    new=NumpyPolicy(out/'candidate/numpy')
    for key in recorded:
        with np.load(source/key/'decisions.npz') as a:
            from evaluate.recovery_attribution import replay_inputs
            predicted=replay_inputs(new, a['X'], a['segment'])
            np.testing.assert_array_equal(predicted[:, [0,1,3]], a['candidate'][:, [0,1,3]])
    flights=out/'flights'; flights.mkdir(); results=[]
    runs=dict(before=ROOT/'output/balanced_maneuver_v1/before', candidate=out/'candidate')
    for case in CASES:
        path=next((ROOT/'flight simulations/changed contract').glob('*'+case+'*.json'))
        result=trial(case, path, 'candidate', runs, flights/(case+'_candidate'))
        results.append(result); write(flights/'report.json', results)
        print(case, result['metrics'], flush=True)
    assert all(sha256(Path(path))==h for path,h in hashes.items())
    assert all(sha256(ROOT/path)==h for path,h in fixed.items())
    write(out/'validation.json', dict(completed_flights=4, only_four_pitch_weights_changed=True,
        other_channels_exact_on_all_eight_recordings=True, input_and_active_baseline_unchanged=True, automatic_promotion=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args().out)
