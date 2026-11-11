"""Fixed local heading audit: retained neural candidate vs native teacher, no fitting."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import numpy as np
from evaluate.practical_maneuver import ObservedPolicy

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'output/tau_continuation_v1/adaptive_tau'
CASE = '9d5431c1'
OFFSETS = (0., -5., 5.)
EARLY_WINDOW = (6.,10.)
REFERENCE = ROOT/'output/maneuver_robustness_v1'


class AuditDriver(ObservedPolicy):
    def __init__(self, policy, teacher, selected):
        if selected not in ('candidate','teacher'): raise ValueError('Unknown driver')
        self.selected = selected
        super().__init__(dict(candidate=policy),teacher,'candidate')

    def step_np(self,*args,**kwargs):
        super().step_np(*args,**kwargs)
        return self.records[-1]['outputs'][self.selected].copy()


class EarlyIntervention(AuditDriver):
    """Teacher observes continuously; selected channels substituted only at 6<=t<10."""
    def __init__(self,policy,teacher,channels):
        if channels not in ((3,),(2,3)): raise ValueError('Only Yaw or Pitch/Yaw supported')
        self.channels=channels; self.time_s=0.
        super().__init__(policy,teacher,'candidate')

    def step_np(self,*args,**kwargs):
        pwm=super().step_np(*args,**kwargs)
        active=EARLY_WINDOW[0]<=self.time_s<EARLY_WINDOW[1]
        if active: pwm[list(self.channels)]=self.records[-1]['outputs']['teacher'][list(self.channels)]
        self.records[-1].update(driver_pwm=pwm.copy(),intervention=active)
        return pwm


def start_spec(manifest, offset_deg):
    if not math.isfinite(offset_deg): raise ValueError('Nonfinite heading offset')
    return dict(scene=manifest['scene'],seed=manifest['seed'],
        start=deepcopy(manifest['start']['position_m']),
        yaw=float(manifest['start']['yaw_rad'])+math.radians(offset_deg),
        scene_options=deepcopy(manifest.get('scene_options')))


def report(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle
    out = Path(out)
    rows = json.loads((out/'report.json').read_text())
    early=json.loads((out/'protocol.json').read_text()).get('early_turn',False)
    locations={r['id']:out for r in rows}
    if early:
        references=json.loads((REFERENCE/'report.json').read_text())
        rows=references+rows
        locations.update({r['id']:REFERENCE for r in references})
    manifest = json.loads((SOURCE/'flights'/CASE/'manifest.json').read_text())
    lines = ['# '+('Коротка корекція раннього маневру' if early else 'Стійкість до малого зсуву початкового курсу'), '',
        ('Teacher Yaw або Pitch/Yaw лише на [6,10) с, далі керує мережа зі своїм неперервним станом. '
         'Контроль candidate/teacher — незмінні збережені прогони попередньої серії.' if early else
         'Одна відома кімната, початкові курси 0 / -5 / +5 градусів відносно записаного. '
         'Обидва драйвери працюють від першого кадру, включно з SHADOW; спільний flight stack.'), '',
        '| Курс | Драйвер | Час, с | Контакт | AUTO, с | Мін. зазор, см | Нижче 5 см, с | Нижче 10 см, с | Шлях, м |',
        '|---:|---|---:|---|---:|---:|---:|---:|---:|']
    fig, axes = plt.subplots(2,3,figsize=(15,9),layout='constrained')
    for col, offset in enumerate(OFFSETS):
        ax = axes[0,col]
        for b in manifest['geometry']['boxes']:
            ax.add_patch(Rectangle((b['x_min'],b['y_min']),b['x_max']-b['x_min'],b['y_max']-b['y_min'],color='.86'))
        for c in manifest['geometry']['cylinders']:
            ax.add_patch(Circle((c['center_x'],c['center_y']),c['radius'],color='.86'))
        for row in [r for r in rows if r['offset_deg']==offset]:
            m = row['metrics']
            lines.append(f'| {offset:+.0f} | {row["driver"]} | {m["duration_s"]:.3f} | {m["collision"] or "немає"} | '
                f'{m["auto_s"]:.2f} | {100*m["min_clearance_m"]:.3f} | {m["below_5cm_s"]:.3f} | '
                f'{m["below_10cm_s"]:.3f} | {m["distance_m"]:.2f} |')
            with np.load(locations[row['id']]/row['id']/'physical.npz') as a:
                line, = ax.plot(*a['pos'][:,:2].T,label=row['driver'])
                ax.scatter(*a['pos'][-1,:2],color=line.get_color(),s=25)
                axes[1,col].plot(a['time'],100*a['clearance'],label=row['driver'])
        ax.set(xlim=(-4,4),ylim=(-4,4),title=f'Initial heading offset {offset:+.0f} deg',xlabel='X (m)',ylabel='Y (m)')
        ax.set_aspect('equal'); ax.legend(fontsize=8)
        axes[1,col].axhline(10,color='.5',ls=':'); axes[1,col].axhline(5,color='.6',ls='--')
        axes[1,col].set(xlim=(0,45),ylim=(0,50),xlabel='Time (s)',ylabel='Body clearance (cm)')
        axes[1,col].grid(alpha=.2); axes[1,col].legend(fontsize=8)
    fig.savefig(out/'comparison.png',dpi=150); plt.close(fig)
    lines += ['', 'Зазор уже враховує радіус корпусу 0.2 м. Лінії 5/10 см — описові орієнтири, не сертифікат безпеки.',
        'Статичні стартові зсуви не оцінюють раптові рухомі перешкоди, огляд, покриття простору чи sim2real. '
        'Teacher тут евристичний контроль, не нейромережа. Жодні ваги або пороги не підлаштовувалися.']
    (out/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def run(out, *, early_turn=False):
    from demo.runner import DemoRunner
    from demo.recording import plain
    from edge.system import FlightSystem
    from edge.runtime import sha256
    from generator.generate_reflex_dataset import make_teacher
    from optimizer.continue_flow_reflex import write
    from evaluate.compare_maneuver_repair import summarize
    out = Path(out); out.mkdir(parents=True,exist_ok=False)
    reference_dir = SOURCE/'flights'/CASE
    m = json.loads((reference_dir/'manifest.json').read_text())
    drivers=('early_yaw','early_xy') if early_turn else ('candidate','teacher')
    tasks = [dict(id=f'{driver}_{int(offset):+d}',driver=driver,offset_deg=offset,
        start_spec=start_spec(m,offset)) for offset in OFFSETS for driver in drivers]
    active = ROOT/'data/edge/k306_tau025_integrated_v1'
    files = [SOURCE/'checkpoint.pt',SOURCE/'runtime_config.json',reference_dir/'manifest.json',reference_dir/'physical.npz',
        ROOT/'configs/mvp_runtime_v3.json',*[p for d in (SOURCE/'numpy',active) for p in d.iterdir() if p.is_file()]]
    if early_turn:
        files += [REFERENCE/'report.json',REFERENCE/'validation.json',
            *[REFERENCE/f'{driver}_{int(offset):+d}'/'physical.npz' for driver in ('candidate','teacher') for offset in OFFSETS]]
    hashes = {str(p):sha256(p) for p in files}
    protocol = dict(version='early-turn-intervention-v1' if early_turn else 'local-heading-robustness-v1',early_turn=early_turn,
        intervention_window_s=list(EARLY_WINDOW) if early_turn else None,
        channels=dict(early_yaw=[3],early_xy=[2,3]) if early_turn else None,
        interpretation=('Diagnostic causal intervention, not a neural repair. Teacher history runs from reset; no oracle geometry. '
            'Yaw-only changes timing/sign/amplitude; XY also changes braking. Window selected once from earlier 6/8s snapshots, no time search.' if early_turn else None),
        source=str(SOURCE),case=CASE,tasks=tasks,
        horizon_s=45.,radius_m=.2,input_sha256=hashes,
        source_sha256={str(p.relative_to(ROOT)):sha256(p) for folder in ('evaluate','simulation','core','edge','generator','demo') for p in (ROOT/folder).glob('*.py')},
        teacher='geometry-reflex-v5, cruise_pitch=1530, noise_std_pwm=0; observes from first tick; '+
            ('substitutes selected channels at 6<=t<10 only' if early_turn else 'native AUTO when teacher selected'),
        frozen='All model weights, teacher rules, coordinator, sensor rates, geometry and seed; explicit driver and initial yaw vary.',
        metrics='Contact, horizon, AUTO exposure, body clearance, times below 5/10cm, actual motion; not teacher-state agreement.',
        selection='Offsets fixed before running. Known problematic room; local neighborhood audit, not an independent environment holdout.',
        decisions=['If candidate fails but teacher succeeds, investigate demonstrated imitation/control gap; memory architecture not thereby proven necessary.',
            'If both fail, teacher and behavioral strategy require review before more imitation training.',
            'No contact with millimeter clearance is not robust safety. No tuning or automatic promotion from these six trials.'],
        no_training=True,no_promotion=True)
    write(out/'protocol.json',protocol)
    results = []
    for task in tasks:
        r = DemoRunner(config=SOURCE/'runtime_config.json',**task['start_spec'],collision_radius_m=.2,
            max_duration_s=45.,contact_model='sphere-sweep-v1',run_id=task['id'])
        try:
            for key in ('room','boxes','cylinders'): assert r.manifest()['geometry'][key] == m['geometry'][key]
            teacher=make_teacher('geometry-reflex-v5',noise_std_pwm=0,pwm_cruise_pitch=1530.)
            observer = (EarlyIntervention(r.system.runtime.policy,teacher,(3,) if task['driver']=='early_yaw' else (2,3))
                if early_turn else AuditDriver(r.system.runtime.policy,teacher,task['driver']))
            r.system.runtime.policy = observer
            manifest = deepcopy(r.system.manifest)
            manifest['experiment'] = dict(version=protocol['version'],**task,
                early_window_s=list(EARLY_WINDOW) if early_turn else None,
                substituted_channels=list(observer.channels) if early_turn else [],
                stage_note='network_pwm is selected raw driver; neural and teacher commands saved separately; Throttle owned by AltitudeHold')
            r.system = FlightSystem(r.system.config,r.system.runtime,manifest)
            r._manifest = plain(r._build_manifest())
            dest = out/task['id']; dest.mkdir(); write(dest/'manifest.json',r.manifest())
            r.resume(); r.event(dict(type='mode',mode='AUTO'))
            for _ in range(11250):
                n = len(observer.records); now = r.time_s
                if early_turn: observer.time_s=now
                r.advance()
                if len(observer.records)>n: observer.records[-1]['time_s'] = now
                if r.terminal: break
            t = np.array([v['time_s'] for v in r.rows]); dt = np.diff(t,append=r.time_s)
            pos = np.array([v['position_m'] for v in r.rows]); velocity = np.array([v['velocity_mps'] for v in r.rows])
            gap = np.array([v['clearance_m'] for v in r.rows]); auto = np.array([v['plant_source']=='auto' for v in r.rows])
            yaw = np.array([v['attitude_rad'][2] for v in r.rows]); speed = np.linalg.norm(velocity[:,:2],axis=1)
            exact = None
            if early_turn:
                with np.load(REFERENCE/f'candidate_{int(task["offset_deg"]):+d}'/'physical.npz') as old:
                    prefix=t<=EARLY_WINDOW[0]; old_prefix=old['time']<=EARLY_WINDOW[0]
                    for key,value in dict(time=t,pos=pos,velocity=velocity,yaw=yaw,clearance=gap,auto=auto).items():
                        np.testing.assert_array_equal(value[prefix],old[key][old_prefix])
                exact=True
            if task['driver']=='candidate' and task['offset_deg']==0:
                with np.load(reference_dir/'physical.npz') as old:
                    for key, value in dict(time=t,pos=pos,velocity=velocity,yaw=yaw,clearance=gap,auto=auto).items():
                        np.testing.assert_array_equal(value,old[key])
                exact = True
            np.savez_compressed(dest/'physical.npz',time=t,pos=pos,velocity=velocity,yaw=yaw,clearance=gap,auto=auto)
            records = observer.records
            np.savez_compressed(dest/'decisions.npz',time=[v['time_s'] for v in records],X=[v['X'] for v in records],
                segment=[v['segment'] for v in records],phase=[v['phase'] for v in records],
                network=[v['outputs']['candidate'] for v in records],teacher=[v['outputs']['teacher'] for v in records],
                **(dict(driver=[v['driver_pwm'] for v in records],intervention=[v['intervention'] for v in records]) if early_turn else {}))
            control = r.rows[::5]
            np.savez_compressed(dest/'trace.npz',time=[v['time_s'] for v in control],
                **{name:np.array([v['trace'].get(key) or [np.nan]*4 for v in control]) for name,key in
                    [('selected_raw','network_pwm'),('coordinated','coordinated_pwm'),('supervised','supervised_pwm'),
                     ('issued','issued_pwm'),('accepted','accepted_pwm')]},
                applied=[v.get('plant_input_pwm') or [np.nan]*4 for v in control],min_tof=[v['min_tof_m'] for v in control])
            metrics = summarize(r.rows,r.time_s); i = int(gap.argmin())
            metrics.update(below_5cm_s=float(dt[auto&(gap<.05)].sum()),moving_auto_s=float(dt[auto&(speed>.03)].sum()),
                mean_auto_speed_mps=float((dt[auto]*speed[auto]).sum()/dt[auto].sum()) if auto.any() else None,
                minimum=dict(time_s=float(t[i]),position_m=pos[i].tolist(),speed_mps=float(speed[i])))
            row = dict(id=task['id'],driver=task['driver'],offset_deg=task['offset_deg'],metrics=metrics,
                events=r.events,reference_exact=exact)
            write(dest/'report.json',row); results.append(row); write(out/'report.json',results)
            print(task['id'],metrics,flush=True)
        finally: r.env.close()
    assert all(sha256(Path(p)) == h for p,h in hashes.items())
    write(out/'validation.json',dict(completed_trials=6,reference_reproduced_exactly=True,
        reference_scope='Physical prefix through 6s' if early_turn else 'Full candidate 0deg trajectory',
        input_and_active_bundle_unchanged=True,no_training=True,no_promotion=True))
    report(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--report-only',action='store_true')
    parser.add_argument('--early-turn',action='store_true')
    args = parser.parse_args()
    report(args.out) if args.report_only else run(args.out,early_turn=args.early_turn)
