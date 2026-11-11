"""Read-only audit of learned state, command behavior, and paired room results."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle


@torch.no_grad()
def confidence_audit(out):
    from artifacts.research import load_research_run
    from optimizer.maneuver_state import StateReadout, forward_hidden
    torch.set_num_threads(1)
    cases = torch.load(out/'state_labels.pt',weights_only=True)['evaluation_history']
    all_metrics = {}; traces = {}
    for arm in ('detached_control','state_supervised'):
        p, _ = load_research_run(out/arm)
        head = StateReadout(p.cfc_network.cell.A)
        head.load_state_dict(torch.load(out/arm/'state_readout.pt',weights_only=True)['state_dict'])
        all_metrics[arm] = {}
        for n, case in cases.items():
            pwm, h = forward_hidden(p,case['X']); logits = head(h)
            logp = logits.log_softmax(-1)
            y = case['state']; correct = logits.argmax(-1) == y
            truth = logp.gather(-1,y.unsqueeze(-1)).squeeze(-1)
            per_class_ce = [-float(truth[:,-5:][y[:,-5:] == c].mean()) for c in range(3)]
            all_metrics[arm][str(n)] = dict(terminal_macro_ce_over_log3=float(np.mean(per_class_ce)/np.log(3)),
                terminal_mean_true_probability=float(truth[:,-5:].exp().mean()),
                terminal_confident_correct_fraction=float((correct[:,-5:] & (truth[:,-5:].exp() >= .6)).float().mean()))
            if n == 150:
                traces[arm+'_pwm'] = pwm[:3].numpy()
                traces[arm+'_pending_probability'] = (1-logp[:3,:,0].exp()).numpy()
                traces['teacher_pwm'] = case['Y'][:3].numpy()
                traces['teacher_pending'] = (y[:3] != 0).numpy()
    (out/'history_confidence.json').write_text(json.dumps(all_metrics,indent=2)+'\n',encoding='utf-8')
    np.savez_compressed(out/'history_examples.npz',**traces)
    return all_metrics, traces


def report(out):
    out = Path(out)
    rows = json.loads((out/'comparison.json').read_text())
    confidence, traces = confidence_audit(out)
    lines = ['# Навчання стану маневру', '',
        'Парний контроль теж навчає класифікатор, але відрізає його градієнт від мережі. '
        'Обидві гілки мають однакові допоміжні історії та loss команд.', '',
        '| Варіант | Напрямок /12 | Відновлення /12 | Макс. час, с | Вузькі /8 | Широкі /4 | Сцени /7 | Крейсерський Pitch |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm, r in rows.items():
        t = r['temporal']
        lines.append(f'| {arm} | {t["direction"]} | {t["recovery"]} | {t["max_recovery_s"]} | '
            f'{t["narrow_amplitude"]} | {t["broad_joint"]} | {t["scenes"]} | {r["cruise"]["final_pwm"][2]:.2f} |')
    lines += ['', '| Варіант | Спільні входи, с | Кінцева збалансована точність стану | CE / log(3) | Макс. помилка Pitch, PWM | Різниця кінцевих Pitch, PWM |',
        '|---|---:|---:|---:|---:|---:|']
    fig, axes = plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for arm, r in rows.items():
        durations = []; acc = []; errors = []
        for n, h in r['history'].items():
            durations.append(int(n)*.02); acc.append(h['terminal_state']['balanced_accuracy']); errors.append(h['suffix_pitch_max_pwm'])
            ce = confidence[arm][n]['terminal_macro_ce_over_log3']
            lines.append(f'| {arm} | {int(n)*.02:.1f} | {100*acc[-1]:.2f}% | {ce:.4f} | '
                f'{h["suffix_pitch_max_pwm"]:.2f} | {h["terminal_pitch_span_pwm"]:.6g} |')
        axes[0,0].plot(durations,acc,'o-',label=arm)
        axes[0,1].plot(durations,errors,'o-',label=arm)
        progress = json.loads((out/(arm+'_training.json')).read_text())
        axes[1,0].plot([r['epoch'] for r in progress],[r['losses']['state'] for r in progress],label=arm)
        axes[1,1].plot([r['epoch'] for r in progress],[r['losses']['history_xy'] for r in progress],label=arm)
    axes[0,0].axhline(1/3,color='.5',ls=':'); axes[0,0].set(xlabel='Shared-input suffix (s)',ylabel='Balanced terminal accuracy',ylim=(0,1.02))
    axes[0,1].set(xlabel='Shared-input suffix (s)',ylabel='Worst suffix Pitch error (PWM)')
    axes[1,0].set(xlabel='Epoch',ylabel='Training state CE / log(3)')
    axes[1,1].set(xlabel='Epoch',ylabel='History Pitch/Yaw MSE')
    for ax in axes.flat: ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.savefig(out/'comparison.png',dpi=150); plt.close(fig)
    fig, axes = plt.subplots(2,3,figsize=(15,7),layout='constrained')
    time = np.arange(traces['teacher_pwm'].shape[1])*.02
    for i, name in enumerate(('Always clear','Unfinished braking','Completed braking')):
        axes[0,i].plot(time,traces['teacher_pwm'][i,:,2],color='black',ls='--',label='teacher')
        axes[1,i].plot(time,traces['teacher_pending'][i],color='black',ls='--',label='teacher state')
        for arm in rows:
            axes[0,i].plot(time,traces[arm+'_pwm'][i,:,2],label=arm)
            axes[1,i].plot(time,traces[arm+'_pending_probability'][i],label=arm)
        axes[0,i].set(title=name,ylabel='Pitch (PWM)')
        axes[1,i].set(xlabel='Time (s)',ylabel='Pending probability',ylim=(-.02,1.02))
        for ax in axes[:,i]:
            ax.axvspan(1.,time[-1],color='.9',alpha=.5); ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.savefig(out/'history_examples.png',dpi=150); plt.close(fig)
    flights = json.loads((out/'flights.json').read_text()) if (out/'flights.json').exists() else []
    lines += ['', '| Кімната | Варіант | Час, с | Контакт | Мін. зазор, см |', '|---|---|---:|---|---:|']
    for r in flights:
        m = r['metrics']
        lines.append(f'| {r["case"]} | {r["arm"]} | {m["duration_s"]:.3f} | {m["collision"] or "немає"} | {100*m["min_clearance_m"]:.2f} |')
    if flights:
        parent = Path(json.loads((out/'protocol.json').read_text())['parent'])
        cases = list(dict.fromkeys(r['case'] for r in flights))
        fig, axes = plt.subplots(2,2,figsize=(12,10),layout='constrained')
        for ax, case in zip(axes.flat,cases):
            geo = json.loads((parent/'flights'/case/'manifest.json').read_text())['geometry']
            for b in geo['boxes']:
                ax.add_patch(Rectangle((b['x_min'],b['y_min']),b['x_max']-b['x_min'],b['y_max']-b['y_min'],color='.86'))
            for c in geo['cylinders']:
                ax.add_patch(Circle((c['center_x'],c['center_y']),c['radius'],color='.86'))
            paths = [('parent',parent/'flights'/case/'physical.npz')]+[(r['arm'],out/r['arm']/'flights'/case/'physical.npz') for r in flights if r['case']==case]
            for name, path in paths:
                with np.load(path) as a:
                    line, = ax.plot(*a['pos'][:,:2].T,label=name)
                    ax.scatter(*a['pos'][-1,:2],s=20,color=line.get_color())
            ax.set(xlim=(-4,4),ylim=(-4,4),title=case); ax.set_aspect('equal'); ax.legend(fontsize=8)
        fig.savefig(out/'trajectories.png',dpi=150); plt.close(fig)
    for arm,r in rows.items():
        if not r['room_screen']['passed']: lines += ['',f'{arm}: кімнати не запускалися — {r["room_screen"]["reasons"]}.']
    lines += ['', 'Класифікатор не входить у керування. Висока точність класифікації не замінює правильних команд або фізичної перевірки. '
        'CE і ймовірності додані, щоб не прийняти майже рівні логіти за корисну пам’ять. Активний пакет незмінний.']
    (out/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out',type=Path)
    report(parser.parse_args().out)
