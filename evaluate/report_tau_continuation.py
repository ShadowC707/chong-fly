"""Read-only figures and tables for the fixed tau-continuation experiment."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle


def report(out):
    from optimizer.finalize_mvp_candidate import comparison_row
    out = Path(out)
    initial = json.loads((out/'initial.json').read_text())
    results = json.loads((out/'comparison.json').read_text())
    rows = {'parent': dict(temporal=comparison_row(dict(initial, continuation_epochs=0)),
        history=initial['history'], cruise=initial['cruise']), **results}
    lines = ['# Навчання часових сталих без зміни W', '',
        'Однакові дані, loss, 10 епох та seed; tau навчається лише в adaptive_tau. Активний пакет не замінено.', '',
        '| Варіант | Напрямок /12 | Згасання /12 | Макс. час, с | Вузькі /8 | Широкі /4 | Сцени /7 | Крейсерський Pitch |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm, r in rows.items():
        t = r['temporal']
        lines.append(f'| {arm} | {t["direction"]} | {t["recovery"]} | {t["max_recovery_s"]} | '
            f'{t["narrow_amplitude"]} | {t["broad_joint"]} | {t["scenes"]} | {r["cruise"]["final_pwm"][2]:.2f} |')
    lines += ['', '| Варіант | Спільний вхід, с | Макс. різниця Pitch між історіями, PWM | Макс. помилка, PWM |',
        '|---|---:|---:|---:|']
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    for arm, r in rows.items():
        points = []
        for n, h in r['history'].items():
            span = max(g['pitch_span_pwm'] for g in h['groups'])
            lines.append(f'| {arm} | {int(n)*.02:.1f} | {span:.6g} | {h["max_abs_pitch_error_pwm"]:.2f} |')
            points.append((int(n)*.02, span))
        axes[0, 0].plot(*np.array(points).T, 'o-', label=arm)
    for arm in results:
        with np.load(out/(arm+'_tau.npz')) as a:
            axes[0, 1].plot(np.sort(a['after']), label=arm)
        progress = json.loads((out/(arm+'_training.json')).read_text())
        axes[1, 0].plot([r['epoch'] for r in progress], [r['recorded_loss'] for r in progress], label=arm)
        with np.load(out/arm/'clear/trace.npz') as a:
            speed = a['velocity'][:, 0]*np.cos(a['yaw']) + a['velocity'][:, 1]*np.sin(a['yaw'])
            axes[1, 1].plot(a['time'], speed, label=arm)
    axes[0, 0].set(title='Identical-input suffix: history separation', xlabel='Suffix (s)', ylabel='Max Pitch span (PWM)')
    axes[0, 1].set(title='Time constants (sorted independently)', xlabel='Sorted index', ylabel='Tau (s)')
    axes[1, 0].set(title='Recorded-sequence training loss', xlabel='Additional epoch', ylabel='Loss')
    axes[1, 1].set(title='Integrated clear start', xlabel='Time (s)', ylabel='Body-forward speed (m/s)')
    for ax in axes.flat: ax.grid(alpha=.2); ax.legend()
    fig.savefig(out/'comparison.png', dpi=150); plt.close(fig)
    flights = json.loads((out/'flights.json').read_text()) if (out/'flights.json').exists() else []
    lines += ['', '| Кімната | Варіант | Час, с | Контакт | Зазор, см | Шлях, м |', '|---|---|---:|---|---:|---:|']
    minima = []
    for r in flights:
        m = r['metrics']
        lines.append(f'| {r["case"]} | {r["arm"]} | {m["duration_s"]:.3f} | {m["collision"] or "немає"} | '
            f'{100*m["min_clearance_m"]:.2f} | {m["distance_m"]:.2f} |')
        with np.load(out/r['arm']/'flights'/r['case']/'physical.npz') as a:
            i = int(a['clearance'].argmin())
            minima.append(dict(arm=r['arm'], case=r['case'], time_s=float(a['time'][i]),
                position_m=a['pos'][i].tolist(), clearance_m=float(a['clearance'][i]),
                horizontal_speed_mps=float(np.linalg.norm(a['velocity'][i, :2]))))
    (out/'minima.json').write_text(json.dumps(minima, indent=2)+'\n', encoding='utf-8')
    if flights:
        parent = Path(json.loads((out/'protocol.json').read_text())['parent'])
        cases = list(dict.fromkeys(r['case'] for r in flights))
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), layout='constrained')
        for ax, case in zip(axes.flat, cases):
            geo = json.loads((parent/'flights'/case/'manifest.json').read_text())['geometry']
            for b in geo['boxes']:
                ax.add_patch(Rectangle((b['x_min'], b['y_min']), b['x_max']-b['x_min'], b['y_max']-b['y_min'], color='.86'))
            for c in geo['cylinders']:
                ax.add_patch(Circle((c['center_x'], c['center_y']), c['radius'], color='.86'))
            paths = [('parent', parent/'flights'/case/'physical.npz')]
            paths += [(r['arm'], out/r['arm']/'flights'/case/'physical.npz') for r in flights if r['case'] == case]
            for name, path in paths:
                with np.load(path) as a:
                    line, = ax.plot(*a['pos'][:, :2].T, label=name)
                    ax.scatter(*a['pos'][-1, :2], s=20, color=line.get_color())
            ax.set(xlim=(-4,4), ylim=(-4,4), title=case, xlabel='X (m)', ylabel='Y (m)')
            ax.set_aspect('equal'); ax.legend(); ax.grid(alpha=.2)
        fig.savefig(out/'trajectories.png', dpi=150); plt.close(fig)
        fig, ax = plt.subplots(figsize=(10, 4), layout='constrained')
        case = '9d5431c1'
        for name, path in [('parent', parent/'flights'/case/'physical.npz')]+[
                (r['arm'], out/r['arm']/'flights'/case/'physical.npz') for r in flights if r['case'] == case]:
            with np.load(path) as a:
                ax.plot(a['time'], 100*a['clearance'], label=name)
        ax.axhline(10, color='.4', ls=':', label='10 cm reference')
        ax.set(xlim=(15, 45), ylim=(0, 40), xlabel='Time (s)', ylabel='Body clearance (cm)',
            title='Known difficult room: completing the run is not a clearance guarantee')
        ax.legend(); ax.grid(alpha=.2)
        fig.savefig(out/'difficult_clearance.png', dpi=150); plt.close(fig)
    for arm, r in results.items():
        if not r['room_screen']['passed']: lines += ['', f'{arm}: кімнати пропущено; {r["room_screen"]["reasons"]}.']
    lines += ['', 'Різниця між історіями не означає правильність команди; дивіться також помилку. '
        'Кімнати вже відомі з діагностики. Це обмежений дослід, не допуск до фізичного польоту.']
    (out/'summary.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    report(parser.parse_args().out)
