"""Matched-state, open-loop interventions on saved closed-loop observations."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def branch_step(policy, x, *, reset=False, flow_x=None, steps=1):
    original = policy.h.copy()
    row = np.array(x, copy=True)
    if flow_x is not None:
        row[0] = flow_x
    try:
        if reset:
            policy.reset_state()
        for _ in range(steps):
            y = policy.step_np(row[:2], row[2:66], row[66:], dt=.02)
        return y.copy()
    finally:
        policy.h = original


def replay_inputs(policy, x, segments):
    predicted = []
    for i, row in enumerate(x):
        if i == 0 or segments[i] != segments[i - 1]:
            policy.reset_state()
        predicted.append(policy.step_np(row[:2], row[2:66], row[66:], dt=.02))
    return np.asarray(predicted)


def run(out):
    from edge.runtime import NumpyPolicy, sha256
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    source = ROOT / 'output/practical_maneuver_v1'
    runs = dict(before=ROOT / 'output/balanced_maneuver_v1/before/numpy',
                candidate=ROOT / 'output/balanced_maneuver_v1/plastic/numpy')
    reports = json.loads((source / 'report.json').read_text())
    fixed = json.loads((ROOT / 'output/balanced_maneuver_v1/protocol.json').read_text())['fixed_baseline_sha256']
    files = [source / 'report.json', *source.glob('*/decisions.npz')]
    files += [p for r in runs.values() for p in r.iterdir() if p.is_file()]
    hashes = {str(p): sha256(p) for p in files}
    protocol = dict(version='recovery-attribution-v1', input_sha256=hashes, source_sha256=sha256(__file__),
                    scope='Last 1s of each recorded trial; paired models on exactly the same inputs and reset events',
                    interventions=['unchanged history', 'reset before current frame', 'same state, current flow_x=0',
                                   'same state, current flow_x negated', 'last input repeated for 1s, with/without prior state'],
                    limitations='Counterfactual sensor streams are imposed, not physically rolled out. Reset is diagnostic, never a runtime proposal.',
                    no_training=True, promotion=False)
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    results = {}
    for r in reports:
        key = r['case'] + '_' + r['driver']
        with np.load(source / key / 'decisions.npz') as a:
            x, segments, t = a['X'], a['segment'], a['time']
            selected = t >= t[-1] - 1. + 1e-8
            rows = {}
            for name, path in runs.items():
                p = NumpyPolicy(path)
                actual = []; branches = []; endpoint = None
                for i, row in enumerate(x):
                    if i == 0 or segments[i] != segments[i - 1]: p.reset_state()
                    if selected[i]:
                        branches.append([branch_step(p, row, reset=True)[2],
                                         branch_step(p, row, flow_x=0)[2],
                                         branch_step(p, row, flow_x=-float(row[0]))[2]])
                    if i == len(x) - 1:
                        endpoint = dict(actual=branch_step(p, row)[2].item(),
                            reset=branch_step(p, row, reset=True)[2].item(),
                            hold_1s=branch_step(p, row, steps=50)[2].item(),
                            reset_hold_1s=branch_step(p, row, reset=True, steps=50)[2].item())
                    actual.append(p.step_np(row[:2], row[2:66], row[66:], dt=.02))
                actual = np.asarray(actual)
                np.testing.assert_array_equal(actual, a[name])
                branch = np.asarray(branches)
                rows[name] = dict(endpoint=endpoint, exact_replay=True,
                    last_second_mean_pitch=dict(actual=float(actual[selected, 2].mean()),
                        reset=float(branch[:, 0].mean()), zero_flow_x=float(branch[:, 1].mean()),
                        flipped_flow_x=float(branch[:, 2].mean())))
                np.savez_compressed(out / (key + '_' + name + '.npz'), time=t[selected], X=x[selected],
                                    actual=actual[selected], reset_pitch=branch[:, 0], zero_fx_pitch=branch[:, 1],
                                    flipped_fx_pitch=branch[:, 2])
            results[key] = dict(models=rows, endpoint_input=dict(time=float(t[-1]), flow=x[-1, :2].tolist(),
                min_tof_m=float(x[-1, 2:66].min()*3.5)), collision=r['metrics']['collision'])
    sweep = {}
    for name, path in runs.items():
        p = NumpyPolicy(path); sweep[name] = []
        for fx in (-.2, -.12, -.06, -.03, -.01, 0, .01, .03, .06, .12, .2):
            row = np.ones(74, np.float32); row[:2] = [fx, 0]
            sweep[name].append(dict(flow_x=fx, pitch_1s=float(branch_step(p, row, reset=True, steps=50)[2])))
    assert all(sha256(Path(p)) == h for p, h in hashes.items())
    assert all(sha256(ROOT / p) == h for p, h in fixed.items())
    report = dict(results=results, clear_flow_sweep=sweep, exact_replays=16,
                  input_and_active_baseline_unchanged=True)
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    for key, r in results.items():
        if key.endswith('candidate'):
            print(key, r['models']['candidate']['endpoint'], flush=True)
    print('clear flow', sweep, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args().out)
