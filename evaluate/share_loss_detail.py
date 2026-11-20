"""Detailed network_share loss of one run: owner stage, its reason, channel, and time profile.

Uses the same exclusive attribution as simulation.navigation_metrics.share_loss and adds, for the
side veto, the remembered gap that triggered it, so the effect of the memory error model is visible.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

COLS = {1: 'roll', 2: 'pitch', 3: 'yaw'}


def detail(run_dir, tolerance=10.):
    run_dir = Path(run_dir)
    with np.load(run_dir/'trace.npz') as z:
        network, coordinated, times = z['network'], z['coordinated'], z['control_time']
        analyzed = z['auto'] & ~z['stop_requested']
    decisions = {round(d['time_s'], 3): d['decision'] for d in
                 json.loads((run_dir/'decisions.json').read_text(encoding='utf-8'))}
    n = int(analyzed.sum())
    owners, reasons, channels, gaps, per5s = Counter(), Counter(), Counter(), [], Counter()
    for net, out, t, keep in zip(network, coordinated, times, analyzed):
        if not keep or np.all(np.abs(out[1:]-net[1:]) <= tolerance):
            continue
        d = decisions.get(round(float(t), 3)) or {}
        owner, why, chans = 'xy_base', ','.join(d.get('reasons', [])) or 'none', []
        for r in d.get('stages') or []:
            dev = [c for c in COLS if abs(r['after'][c]-net[c]) > tolerance]
            if dev:
                owner, why, chans = r['stage'], r['reason'], dev
                if r['stage'] == 'side_veto':
                    gaps.append(min(v for v in (r.get('left_gap_m'), r.get('right_gap_m')) if v is not None))
                break
        if owner == 'xy_base':
            chans = [c for c in COLS if abs(out[c]-net[c]) > tolerance]
        owners[owner] += 1
        reasons[(owner, why)] += 1
        for c in chans:
            channels[(owner, COLS[c])] += 1
        per5s[int(t//5)*5] += 1
    gaps = np.array(gaps)
    return dict(analyzed_ticks=n, network_share_loss=sum(owners.values())/n,
        by_stage={k: v/n for k, v in owners.most_common()},
        by_reason={f'{k[0]}:{k[1]}': v/n for k, v in reasons.most_common(8)},
        by_channel={f'{k[0]}:{k[1]}': v/n for k, v in channels.most_common()},
        veto_gap_m=dict(count=len(gaps), p10=float(np.quantile(gaps, .1)) if len(gaps) else None,
                        median=float(np.median(gaps)) if len(gaps) else None,
                        share_gap_0_15_to_0_30=float(np.mean((gaps >= .15) & (gaps < .30))) if len(gaps) else None),
        lost_ticks_per_5s={k: v for k, v in sorted(per5s.items())})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run_dirs', nargs='+')
    a = p.parse_args()
    for d in a.run_dirs:
        print(d)
        print(json.dumps(detail(d), indent=1, ensure_ascii=False))
