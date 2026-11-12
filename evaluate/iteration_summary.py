"""Iteration summary: safety / network-share trade-off with 95 % Wilson intervals, and the holdout
candidate by the author's rule (journal, after the spread and after round 6b).

Inputs are the perturbed-start summaries (5 perturbations x 5 scenarios per profile). The acceptance
set excludes mvp_b0a9f212_native (sensor-limited, author decision), so each profile has 20 runs.
Rule: fewest runs with a contact; on a tie, more runs with network_share >= 0.80. A profile dominated
by another on both axes cannot be the candidate. If the difference to the best profile is
statistically indistinguishable in BOTH groups (overlapping 95 % Wilson intervals), the profile with
fewer contacts is chosen, and on equal contacts the simpler profile (phased_side_guard, the baseline).
"""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = 'mvp_b0a9f212_native'
# simplest first: the behavioural baseline, then the pipeline by the number of stages it adds
SIMPLICITY = ['phased_side_guard', 'pipeline_v1', 'pipeline_v1_r5', 'pipeline_v1_r6', 'pipeline_v1_r6b', 'pipeline_v1_r7']


def wilson(k, n, z=1.96):
    if n == 0:
        return (math.nan, math.nan)
    p = k/n
    den = 1+z*z/n
    centre = (p+z*z/(2*n))/den
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return (max(0., centre-half), min(1., centre+half))


def overlap(a, b):
    return a[0] <= b[1] and b[0] <= a[1]


def load(sources):
    runs = []
    for path, profiles in sources:
        for r in json.loads((ROOT/path).read_text(encoding='utf-8')):
            if 'error' not in r and r['profile'] in profiles and r['base_scenario'] != EXCLUDED:
                runs.append(r)
    return runs


def stats(runs, profiles):
    out = {}
    for p in profiles:
        R = [r for r in runs if r['profile'] == p]
        n = len(R)
        c = sum(r['contacts'] > 0 for r in R)
        s = sum((r.get('network_share') or 0) >= .80 for r in R)
        out[p] = dict(n=n, contact_runs=c, contact_ci=wilson(c, n), share_runs=s, share_ci=wilson(s, n),
                      safety_runs=sum(r['verdict']['safety'] for r in R),
                      low_clearance_runs=sum(r['min_clearance_m'] < .10 for r in R))
    return out


def candidate(st):
    names = list(st)
    dominated = {p for p in names for q in names if q != p
                 and st[q]['contact_runs'] < st[p]['contact_runs'] and st[q]['share_runs'] > st[p]['share_runs']}
    eligible = [p for p in names if p not in dominated]
    ranked = sorted(eligible, key=lambda p: (st[p]['contact_runs'], -st[p]['share_runs']))
    best = ranked[0]
    # statistically indistinguishable from the best in both groups -> fewer contacts, then simpler
    tied = [p for p in eligible if overlap(st[p]['contact_ci'], st[best]['contact_ci'])
            and overlap(st[p]['share_ci'], st[best]['share_ci'])]
    rank = lambda p: (st[p]['contact_runs'], SIMPLICITY.index(p) if p in SIMPLICITY else len(SIMPLICITY))
    chosen = sorted(tied, key=rank)[0]
    return dict(dominated=sorted(dominated), by_rule=best, indistinguishable_from_best=sorted(tied), chosen=chosen)


def table(st):
    f = lambda ci: f'{ci[0]:.2f}…{ci[1]:.2f}'
    lines = ['| Профіль | прогонів із контактом | 95% CI | net ≥ 0,80 | 95% CI | безпека ✓ | зазор < 0,10 |',
             '|---|---:|---|---:|---|---:|---:|']
    for p, s in st.items():
        lines.append(f"| {p} | {s['contact_runs']}/{s['n']} | {f(s['contact_ci'])} | {s['share_runs']}/{s['n']} | "
                     f"{f(s['share_ci'])} | {s['safety_runs']}/{s['n']} | {s['low_clearance_runs']}/{s['n']} |")
    return '\n'.join(lines)


if __name__ == '__main__':
    sources = [('output/pipeline_v1_spread/runs/summary.json', {'pipeline_v1', 'phased_side_guard', 'pipeline_v1_r5'}),
               ('output/pipeline_v1_round6/spread_r6/summary.json', {'pipeline_v1_r6'}),
               ('output/pipeline_v1_round6b/spread_r6b/summary.json', {'pipeline_v1_r6b'}),
               ('output/pipeline_v1_round7/spread_r7/summary.json', {'pipeline_v1_r7'})]
    profiles = ['phased_side_guard', 'pipeline_v1', 'pipeline_v1_r5', 'pipeline_v1_r6', 'pipeline_v1_r6b', 'pipeline_v1_r7']
    st = stats(load(sources), profiles)
    print(table(st))
    print(json.dumps(candidate(st), ensure_ascii=False))
