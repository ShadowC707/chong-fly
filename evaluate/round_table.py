"""Section 2 table of one protocol-6a round next to the baseline (and the previous round)."""
import argparse
import json
from pathlib import Path

COLUMNS = [('network_share', 'net', '{:.3f}'), ('generated_share', 'gen', '{:.3f}'),
           ('generated_active_share', 'утрим. сире', '{:.3f}'), ('contacts', 'конт.', '{}'),
           ('min_clearance_m', 'зазор, м', '{:.3f}'), ('max_height_error_m', 'висота, м', '{:.3f}'),
           ('reverse_distance_body_m', 'відкат (старий), м', '{:.3f}'),
           ('reverse_outside_corridor_m', 'відкат поза коридором, м', '{:.3f}')]


def fmt(m, key, pattern):
    v = m.get(key)
    return '—' if v is None else pattern.format(v)


def loss(run_dir):
    """Exclusive network_share loss by stage, recomputed from the saved trace and decision log."""
    import numpy as np
    from simulation.navigation_metrics import share_loss
    trace, decisions = run_dir/'trace.npz', run_dir/'decisions.json'
    if not trace.exists() or not decisions.exists():
        return None
    with np.load(trace) as z:
        network, coordinated, times = z['network'], z['coordinated'], z['control_time']
        analyzed = z['auto'] & ~z['stop_requested']
    by_time = {round(d['time_s'], 3): d['decision'].get('stages') for d in
               json.loads(decisions.read_text(encoding='utf-8'))}
    if not any(by_time.values()):
        return None
    log = [by_time.get(round(float(t), 3)) or [] for t in times]
    return share_loss(network, coordinated, log, analyzed)


def recount(run_dir):
    """generated_share under the current rule, recomputed from the saved stage log (the rule changed
    after round 4: own contribution of the generating stage > 10 PWM)."""
    import numpy as np
    from simulation.navigation_metrics import generated_active_share, generated_share
    trace, decisions = run_dir/'trace.npz', run_dir/'decisions.json'
    if not trace.exists() or not decisions.exists():
        return None
    with np.load(trace) as z:
        times, analyzed = z['control_time'], z['auto'] & ~z['stop_requested']
    by_time = {round(d['time_s'], 3): d['decision'].get('stages') for d in
               json.loads(decisions.read_text(encoding='utf-8'))}
    if not any(by_time.values()):
        return None
    log = [by_time.get(round(float(t), 3)) or [] for t in times]
    return dict(generated_share=generated_share(log, analyzed), generated_active_share=generated_active_share(log, analyzed))


def rows(summary):
    out = {(m['scenario'], m['profile']): m for m in json.loads(Path(summary).read_text(encoding='utf-8'))}
    for (scenario, profile), m in out.items():
        m['share_loss'] = loss(Path(summary).parent/scenario/profile)
        m.update(recount(Path(summary).parent/scenario/profile) or {})
    return out


def loss_text(m):
    parts = m.get('share_loss')
    if not parts:
        return '—'
    names = dict(brake_no_rollback='гальмо', side_veto='вето', velocity_hold='утрим.', xy_base='XY')
    return ' / '.join(f"{names.get(k, k)} {v:.3f}" for k, v in parts.items())


def progress(m):
    if m['scenario'] != 'live_room':
        return '—'
    if not m.get('completed'):
        return 'перервано'
    return f"{m['longest_no_new_cell']['duration_s']:.1f} с / {m['new_cells_last_60s']}"


def verdict(m):
    v = m.get('verdict') or {}
    parts = [('частка', v.get('share')), ('безпека', v.get('safety')), ('поступ', v.get('progress')),
             ('доставка', v.get('delivery'))]
    return ', '.join(f"{n} {'✓' if x else '✗'}" for n, x in parts if x is not None)


def table(sources, profiles):
    data = [(label, rows(path)) for label, path in sources]
    scenarios = sorted({s for _, d in data for s, _ in d})
    head = ['Сценарій', 'Профіль', *[c[1] for c in COLUMNS], 'втрати net', 'тривалість, с', 'поступ (без нової / за 60 с)', 'вердикт']
    out = ['| ' + ' | '.join(head) + ' |', '|' + '---|'*len(head)]
    for s in scenarios:
        for label, d in data:
            for p in profiles:
                m = d.get((s, p))
                if m is None:
                    continue
                out.append('| ' + ' | '.join([s, f'{p} ({label})' if label else p,
                    *[fmt(m, k, f) for k, _, f in COLUMNS], loss_text(m), f"{m['duration_s']:.1f}", progress(m), verdict(m)]) + ' |')
    return '\n'.join(out)


if __name__ == '__main__':
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--source', nargs=2, action='append', metavar=('LABEL', 'SUMMARY'), required=True)
    a.add_argument('--profiles', nargs='+', required=True)
    args = a.parse_args()
    print(table(args.source, args.profiles))
