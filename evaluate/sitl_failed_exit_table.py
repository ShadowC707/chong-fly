"""Step 9 item 4: phased_failed_exit in Betaflight SITL against phased_side_guard (step 7), same protocol.

Reads output/pipeline_v1_step7/step7_metrics.json (old profile) and
output/pipeline_v1_step9/sitl_step7/step7_metrics.json (written by
`python -m evaluate.sitl_step7 table --profile phased_failed_exit --out output/pipeline_v1_step9/sitl_step7`).
Also checks whether each SITL flight is bit-identical to its step-7 counterpart (state trace).
"""
import json
from pathlib import Path
import sys

import numpy as np

from evaluate.reiteration_baseline import LIMITS

OLD = Path('output/pipeline_v1_step7')
NEW = Path('output/pipeline_v1_step9/sitl_step7')


def safety(m):
    return (m['contacts'] == 0 and m['min_clearance_m'] >= LIMITS['min_clearance_m']
            and m['max_height_error_m'] <= LIMITS['max_height_error_m']
            and m.get('reverse_outside_corridor_m', m.get('reverse_distance_body_m', 0)) <= LIMITS['reverse_distance_body_m'])


def progress(m):
    if 'longest_no_new_cell' not in m or not m.get('completed'):
        return None
    return (m['longest_no_new_cell']['duration_s'] <= LIMITS['max_no_new_cell_s']
            and m['new_cells_last_60s'] >= LIMITS['min_new_cells_last_60s'])


def identical(name, k):
    a, b = OLD/'sitl'/f'{name}_p{k}'/'trace.npz', NEW/'sitl'/f'{name}_p{k}'/'trace.npz'
    if not (a.is_file() and b.is_file()):
        return None
    with np.load(a) as x, np.load(b) as y:
        if x['state'].shape != y['state'].shape:
            return False
        return bool(np.array_equal(x['state'], y['state']) and np.array_equal(x['coordinated_pwm'], y['coordinated_pwm']))


def cell(m, key, f='%.3f'):
    v = m.get(key)
    return '—' if v is None else f % v


def mark(v):
    return '—' if v is None else ('✓' if v else '✗')


def main():
    old = {(r['scene'], r['k']): r for r in json.loads((OLD/'step7_metrics.json').read_text(encoding='utf-8'))}
    new = {(r['scene'], r['k']): r for r in json.loads((NEW/'step7_metrics.json').read_text(encoding='utf-8'))}
    lines = ['| Сцена | Збурення | SITL-політ = крок 7 побітово | контакт: psg / fe | мін. зазор, м: psg / fe | net: psg / fe | '
             'FAILED: psg / fe | безпека ✓: psg / fe | поступ ✓: psg / fe | клітин 25 см: psg / fe | Standalone fe: зазор / FAILED |',
             '|---|---|:---:|---|---|---|---|---|---|---|---|']
    summary = {}
    for name in ('stop45', 'room180'):
        for k in range(1, 6):
            a, b = old[(name, k)]['sitl'], new[(name, k)]['sitl']
            st = new[(name, k)]['standalone']
            same = identical(name, k)
            lines.append(f"| {name} | p{k} | {mark(same)} | {a['contacts']} / {b['contacts']} | {cell(a, 'min_clearance_m')} / {cell(b, 'min_clearance_m')} | "
                         f"{cell(a, 'network_share')} / {cell(b, 'network_share')} | {cell(a, 'failed_share')} / {cell(b, 'failed_share')} | "
                         f"{mark(safety(a))} / {mark(safety(b))} | {mark(progress(a))} / {mark(progress(b))} | "
                         f"{a.get('visited_xy_cells_25cm', '—')} / {b.get('visited_xy_cells_25cm', '—')} | "
                         f"{cell(st, 'min_clearance_m')} / {cell(st, 'failed_share')} |")
            s = summary.setdefault(name, dict(n=0, same=0, c=[0, 0], safe=[0, 0], prog=[0, 0], clr=[[], []], net=[[], []], failed=[[], []]))
            s['n'] += 1
            s['same'] += bool(same)
            for i, m in enumerate((a, b)):
                s['c'][i] += m['contacts'] > 0
                s['safe'][i] += safety(m)
                s['prog'][i] += bool(progress(m))
                s['clr'][i].append(m['min_clearance_m'])
                s['net'][i].append(m['network_share'])
                s['failed'][i].append(m['failed_share'])
    lines += ['', '| Сцена | профіль | контакти | мін. зазор, медіана / найгірший | net, медіана / найгірший | FAILED, медіана / найгірший | безпека ✓ | поступ ✓ | побітово = крок 7 |',
              '|---|---|---:|---|---|---|---:|---:|---:|']
    for name, s in summary.items():
        for i, p in enumerate(('phased_side_guard (крок 7)', 'phased_failed_exit')):
            lines.append(f"| {name} | {p} | {s['c'][i]}/{s['n']} | {np.median(s['clr'][i]):.3f} / {min(s['clr'][i]):.3f} | "
                         f"{np.median(s['net'][i]):.3f} / {min(s['net'][i]):.3f} | {np.median(s['failed'][i]):.3f} / {max(s['failed'][i]):.3f} | "
                         f"{s['safe'][i]}/{s['n']} | {s['prog'][i] if name == 'room180' else '—'}{'/'+str(s['n']) if name == 'room180' else ''} | "
                         f"{s['same'] if i else '—'}{'/'+str(s['n']) if i else ''} |")
    text = '\n'.join(lines)
    (NEW/'table.md').write_text(text+'\n', encoding='utf-8')
    sys.stdout.reconfigure(encoding='utf-8')
    print(text)


if __name__ == '__main__':
    sys.exit(main())
