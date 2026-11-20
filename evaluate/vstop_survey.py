"""Plan step 3: distribution of filtered forward flow while the drone is truly standing still.

Reads recorded baseline traces only (no new flights). A tick counts as standing still when the
true horizontal speed over +-0.1 s is below 0.02 m/s. The result motivates stage-1 v_stop.
"""
import argparse
import json
from pathlib import Path

import numpy as np

PCT = (50, 90, 95, 99, 99.9)


def samples(directory, still_mps=.02):
    decisions = json.loads((directory/'decisions.json').read_text(encoding='utf-8'))
    flow = {round(d['time_s'], 3): d['decision'].get('filtered_flow_x') for d in decisions}
    with np.load(directory/'trace.npz') as z:
        t, xy, control_t, auto = z['time'], z['xy'], z['control_time'], z['auto']
    out = []
    for tick, is_auto in zip(control_t, auto):
        value = flow.get(round(float(tick), 3))
        if not is_auto or value is None or tick < .1:
            continue
        i0, i1 = np.searchsorted(t, tick-.1), np.searchsorted(t, tick+.1)
        if i1 >= len(t):
            continue
        if np.linalg.norm(xy[i1]-xy[i0])/(t[i1]-t[i0]) < still_mps:
            out.append(value)
    return np.array(out)


def main(root, out):
    per, pooled = {}, []
    for d in sorted(root.glob('*/*/')):
        if (d/'decisions.json').is_file():
            s = samples(d)
            per[f'{d.parent.name}/{d.name}'] = dict(ticks=len(s))
            pooled.append(s)
    a = np.concatenate(pooled)
    result = dict(still_ticks=len(a), signed_percentiles={str(p): float(np.percentile(a, p)) for p in (1, 5, 50, 95, 99, 99.9)},
        abs_percentiles={str(p): float(np.percentile(abs(a), p)) for p in PCT}, abs_max=float(abs(a).max()),
        per_trace=per, source=str(root))
    out.mkdir(parents=True, exist_ok=False)
    (out/'vstop_survey.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('output/reiteration_baseline_v1'))
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    main(a.root, a.out)
