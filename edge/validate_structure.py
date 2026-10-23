"""Bitwise isolated before/after replay for structural changes (host-side tool)."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from edge.bundle_validation import replay_bundle
from edge.validate_numpy import _replay_inputs


def compare(before, after, *, steps=2000):
    cases = [dict(coordinator_mode=mode, coordinator_preset=preset,
                  levy_explore=behavior, stop_hover=behavior, seed=20261021)
             for mode, preset, behavior in [('baseline', 'original_v1', False),
                 ('both', 'original_v1', False), ('both', 'near_zero_v1', False),
                 ('both', 'original_v1', True)]]
    inputs = _replay_inputs(steps)
    stop = np.zeros((len(cases), steps), dtype=bool)
    stop[-1, max(0, steps-250):] = True
    old = replay_bundle(before, inputs, cases, stop_requests=stop)
    new = replay_bundle(after, inputs, cases, stop_requests=stop, assembly=True)
    rows = []
    for index, case in enumerate(cases):
        stages = {}
        for name in ('network_pwm', 'supervised_pwm', 'hidden'):
            a, b = old[name][index], new[name][index]
            stages[name] = dict(bitwise_equal=a.dtype == b.dtype and a.shape == b.shape
                               and a.tobytes() == b.tobytes(), dtype=str(a.dtype),
                               max_abs_error=float(np.max(np.abs(a-b))))
        reasons_equal = old['report']['reasons'][index] == new['report']['reasons'][index]
        rows.append(dict(case=case, stages=stages, reasons_equal=reasons_equal))
    same_weights = (Path(before)/'model.npz').read_bytes() == (Path(after)/'model.npz').read_bytes()
    # These control laws must be untouched in the structural phase.
    same_control_code = {name: old['report']['executed_modules'][name]['sha256'] ==
                         new['report']['executed_modules'][name]['sha256']
                         for name in ('altitude_control.py', 'xy_coordinator.py', 'flight_bridge.py')}
    def evidence(result):
        return {k: v for k, v in result['report'].items() if k != 'reasons'}
    return dict(version='structural-parity-v1', passed=same_weights and all(same_control_code.values())
                and all(r['reasons_equal'] and all(s['bitwise_equal'] for s in r['stages'].values()) for r in rows),
                steps_per_case=steps, total_steps=steps*len(cases), weights_unchanged=same_weights,
                control_code_unchanged=same_control_code, cases=rows,
                input_arrays={k: dict(shape=list(v.shape), dtype=str(v.dtype),
                    sha256=hashlib.sha256(v.tobytes()).hexdigest()) for k, v in (inputs | {'stop': stop}).items()},
                before=evidence(old), after=evidence(new),
                limitation='Same-machine bitwise replay; not cross-platform equality or flight admission. '
                            'Builder is executed; Bridge/Gate ticks and transport are not replayed here.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Use a new report')
    report = compare(args.before, args.after)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf-8') as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write('\n')
    print(json.dumps(dict(passed=report['passed'], total_steps=report['total_steps'], report=str(args.out))))
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
