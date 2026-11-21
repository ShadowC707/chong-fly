"""Step 9 table: phased_failed_exit against phased_side_guard on the seed set and the 20-run spread."""
import json
from pathlib import Path
import sys

import numpy as np

from evaluate.iteration_summary import wilson

STEP9 = Path('output/pipeline_v1_step9')
CONTROL_SPREAD = Path('output/pipeline_v1_spread/runs/summary.json')
P0, P1 = 'phased_side_guard', 'phased_failed_exit'


def load(path, profile, drop_b0a9=False):
    return [r for r in json.loads(Path(path).read_text(encoding='utf-8'))
            if 'error' not in r and r['profile'] == profile
            and not (drop_b0a9 and r.get('base_scenario') == 'mvp_b0a9f212_native')]


def share(R, phases):
    total = sum(sum(r['phase_counts'].values()) for r in R)
    return sum(r['phase_counts'].get(p, 0) for r in R for p in phases)/total


def row(name, R):
    n = len(R)
    ci = lambda k: '%.2f…%.2f' % wilson(k, n)
    c = sum(r['contacts'] > 0 for r in R)
    s = sum(r['verdict']['safety'] for r in R)
    low = sum(r['min_clearance_m'] < .1 for r in R)
    stopped = sum(r['phase_counts'].get('STOP', 0) > 0 for r in R)
    long180 = [r for r in R if r['completed'] and r['duration_s'] >= 179.9]
    prog = sum(bool(r['verdict']['progress']) for r in long180)
    nn = np.median([r['longest_no_new_cell']['duration_s'] for r in long180]) if long180 else float('nan')
    nc = np.median([r['new_cells_last_60s'] for r in long180]) if long180 else float('nan')
    net = np.median([r['network_share'] for r in R if r['network_share'] is not None])
    return (f"| {name} | {n} | {c}/{n} ({ci(c)}) | **{100*share(R, ('FAILED',)):.1f} %** | {100*share(R, ('STOP',)):.1f} % | "
            f"{100*share(R, ('FAILED', 'STOP')):.1f} % | {stopped}/{n} | {s}/{n} ({ci(s)}) | {low}/{n} | {net:.2f} | "
            f"{prog}/{len(long180)} | {nn:.1f} | {nc:.0f} | {np.median([r['visited_xy_cells_25cm'] for r in R]):.0f} |")


HEAD = ('| Набір · профіль | прогонів | контакт (95% CI) | FAILED, частка тактів | STOP після бюджету | застрягання (FAILED+STOP) | '
        'прогонів зі STOP | безпека ✓ (95% CI) | зазор < 0,10 | частка мережі, медіана | поступ ✓ (180 с) | '
        'без нової клітини, с, медіана (180 с) | нових за 60 с, медіана (180 с) | клітин 25 см, медіана |\n'
        '|---|---:|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|')


def spread_summary(name):
    d = STEP9/name
    return d/'runs/summary.json' if (d/'runs/summary.json').exists() else d/'summary.json'


def main():
    v1, v2 = STEP9/'seeds/summary.json', STEP9/'seeds_v2/summary.json'
    lines = [HEAD,
             row(f'seed · `{P0}`', load(v1, P0)), row(f'seed · `{P1}` v1 (STOP)', load(v1, P1)),
             row(f'seed · `{P1}` v2 (утримання)', load(v2, P1)),
             row(f'розкид · `{P0}`', load(CONTROL_SPREAD, P0, True)),
             row(f'розкид · `{P1}` v1', load(spread_summary('spread'), P1, True)),
             row(f'розкид · `{P1}` v2', load(spread_summary('spread_v2'), P1, True)), '',
             '| Сцена | мін. зазор, м: контроль / v1 / v2 | v2 ≥ контроль − 0,01 | FAILED, с: контроль / v2 | STOP, с: v1 / v2 | '
             'подія «вичерпано» v2, с | контакт: контроль / v2 | клітин: контроль / v2 |',
             '|---|---|:---:|---|---|---:|---|---|']
    A, B1, B2 = ({r['case']: r for r in load(f, p)} for f, p in ((v1, P0), (v1, P1), (v2, P1)))
    f = lambda r, ph: .02*r['phase_counts'].get(ph, 0)
    fails = 0
    for case in sorted(A, key=lambda k: (k.split('_')[0] != 'author', k.split('_')[0] != 'demo', k)):
        a, b1, b2 = A[case], B1[case], B2.get(case)
        if b2 is None:
            lines.append(f'| {case} | — | — | | | | | |')
            continue
        ok = b2['min_clearance_m'] >= a['min_clearance_m']-.01
        fails += not ok
        ex = exhausted_s(case)
        lines.append(f"| {case} | {a['min_clearance_m']:.3f} / {b1['min_clearance_m']:.3f} / {b2['min_clearance_m']:.3f} | "
                     f"{'✓' if ok else '**✗**'} | {f(a, 'FAILED'):.1f} / {f(b2, 'FAILED'):.1f} | {f(b1, 'STOP'):.1f} / {f(b2, 'STOP'):.1f} | "
                     f"{ex if ex is not None else '—'} | {a['contacts'] > 0:d} / {b2['contacts'] > 0:d} | "
                     f"{a['visited_xy_cells_25cm']} / {b2['visited_xy_cells_25cm']} |")
    lines.append('')
    lines.append(f'Сцен, де v2 нижче контролю більш ніж на 1 см: {fails} з {len(A)}.')
    text = '\n'.join(lines)
    (STEP9/'table_v2.md').write_text(text+'\n', encoding='utf-8')
    print(text)


def exhausted_s(case):
    path = STEP9/'seeds_v2'/case/P1/'decisions.json'
    if not path.exists():
        return None
    for d in json.loads(path.read_text(encoding='utf-8')):
        if d['decision'].get('recovery_event') == 'failed_retry_budget_exhausted':
            return round(d['time_s'], 1)
    return None


if __name__ == '__main__':
    sys.exit(main())
