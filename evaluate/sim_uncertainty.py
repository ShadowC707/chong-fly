"""Finite exploratory sensitivity study; frozen model/controller, no tuning.

This is not a calibrated hardware model. All perturbations are opt-in here;
the interactive demo, production runtime and its defaults are untouched.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import gc
import json
import math
from pathlib import Path
import shutil

import numpy as np

from demo.controllers import install
from demo.recording import plain
from demo.runner import DemoRunner
from edge.runtime import sha256
from evaluate.compare_maneuver_repair import summarize
from evaluate.phased_holdout import cases as holdout_cases, longest
from simulation.drone_env import BoxObstacle, CylinderObstacle

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'output/tau_continuation_v1/adaptive_tau'
REFERENCE = ROOT / 'output/phased_v2_holdout_v1'
HORIZON = 45.


@dataclass(frozen=True)
class Perturbation:
    yaw_scale: float = 1.
    angle_scale: float = 1.
    motor_tau_scale: float = 1.
    mass_scale: float = 1.
    flow_scale: float = 1.
    tof_bias_m: float = 0.

    def __post_init__(self):
        for key, value in asdict(self).items():
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError('Perturbations must be finite real numbers')
            if key.endswith('_scale') and value <= 0:
                raise ValueError('Scales must be positive')


def profiles():
    return {
        'nominal': Perturbation(),
        'yaw_70pct': Perturbation(yaw_scale=.7),
        'angle_60deg': Perturbation(angle_scale=60/35),
        'motor_lag_2x': Perturbation(motor_tau_scale=2.),
        'mass_125pct': Perturbation(mass_scale=1.25),
        'flow_70pct': Perturbation(flow_scale=.7),
        'flow_130pct': Perturbation(flow_scale=1.3),
        'tof_plus_10cm': Perturbation(tof_bias_m=.1),
        'combined': Perturbation(yaw_scale=.7, motor_tau_scale=2., flow_scale=.7, tof_bias_m=.1),
    }


def scenarios():
    selected = ('wall_glancing', 'box_front', 'gap_1p3', 'dead_end')
    indexed = {c['name']: c for c in holdout_cases()}
    return [indexed[name] for name in selected] + [dict(
        name='stop_room', seed=20261107, start=[0., 0., 1.], yaw=math.pi/6,
        boxes=[], cylinders=[], stop_at_s=35.)]


class PerturbedSchedule:
    """Distort measurements before capture/hold, never rewrite held payloads.

Flow scale includes the pre-existing sensor noise. No extra RNG calls, latency
or validity errors are introduced in this first series. ToF bias is in metres;
the controller still uses the original 3.5 m normalization contract.
"""
    def __init__(self, base, profile):
        self.base, self.profile = base, profile

    def frame(self, i, flow, tof, altitude, **kwargs):
        p = self.profile
        if p.flow_scale != 1.:
            flow = np.clip(np.asarray(flow) * p.flow_scale, -1., 1.)
        if p.tof_bias_m != 0.:
            tof = np.clip(np.asarray(tof) + p.tof_bias_m / 3.5, 0., 1.)
        return self.base.frame(i, flow, tof, altitude, **kwargs)


def apply_plant(env, profile):
    p = env.params
    p.max_yaw_rate_rads *= profile.yaw_scale
    p.max_angle_rad *= profile.angle_scale
    p.motor_tau *= profile.motor_tau_scale
    if profile.mass_scale != 1.:
        # Same shape/material distribution, heavier vehicle, not a point payload.
        p.mass *= profile.mass_scale
        for name in ('ixx', 'iyy', 'izz'):
            setattr(p, name, getattr(p, name) * profile.mass_scale)
        env.physics._update_inertias()
    # Keep initial position, velocity and actual rotor RPMs unchanged.
    # Updating the plant's hover cache must NOT retune AltitudeHold or pre-spin
    # the modified craft to its new equilibrium using privileged information.


def write(path, value):
    path.write_text(json.dumps(plain(value), indent=2, allow_nan=False) + '\n', encoding='utf-8')


def build_runner(case, profile, run_id, *, turn_envelope=None):
    r = DemoRunner(config=SOURCE/'runtime_config.json', scene='room', seed=case['seed'],
        start=case['start'], yaw=case['yaw'], collision_radius_m=.2,
        max_duration_s=HORIZON, contact_model='sphere-sweep-v1', run_id=run_id)
    try:
        r.env.boxes = [BoxObstacle(**b) for b in case['boxes']]
        r.env.cylinders = [CylinderObstacle(**c) for c in case['cylinders']]
        hit, gap = r.contact(r.start, r.start, r.env.room, r.env.boxes, r.env.cylinders, .2)
        if hit or gap < .2:
            raise ValueError('Invalid scenario starting clearance')
        r.obs = r.env._sample_observation(0.)  # same RNG history as geometric holdout
        r.env.demo_geometry_spec = dict(version='explicit-holdout-v1', scene_spec=case)
        r.system = install(r.system, turn_envelope=turn_envelope)
        nominal = asdict(r.env.params)
        initial_rpms = r.env.physics.motor_rpms.copy()
        apply_plant(r.env, profile)
        r.schedule = PerturbedSchedule(r.schedule, profile)
        r._manifest = plain(r._build_manifest())
        r._manifest['uncertainty_study'] = dict(
            version='exploratory-sensitivity-v1', calibrated=False,
            perturbation=asdict(profile), nominal_dynamics=nominal,
            resolved_dynamics=asdict(r.env.params), initial_rpms=initial_rpms.tolist(),
            fc=dict(angle_p_gain=r.env.pid.angle_p_gain,
                max_rate_rads=r.env.pid.max_rate_rads,
                **{axis:asdict(getattr(r.env.pid, 'pid_'+axis).constants) for axis in ('roll','pitch','yaw')}),
            controller_hover_pwm=r.system.runtime.altitude.config.hover_pwm,
            plant_hover_pwm=1000+1000*r.env.physics.hover_throttle,
            measurement_transform='before sample-and-hold; flow gain includes existing noise',
            initial_condition='Same nominal rotor RPMs and zero velocity; no retuning of hover trim',
            stop_at_s=case.get('stop_at_s'),
            limitations=['No transport/compute delay perturbation in this series',
                'No sensor dropout or residual rotational flow model',
                'ANGLE 60 deg is a linear mapping sensitivity, not Betaflight ACTUAL emulation'])
        return r
    except Exception:
        r.env.close()
        raise


def metrics_for(rows, end_s, decisions, events, case):
    m = summarize(rows, end_s)
    t = np.array([v['time_s'] for v in rows])
    dt = np.diff(t, append=end_s)
    speed = np.linalg.norm([v['velocity_mps'][:2] for v in rows], axis=1)
    auto = np.array([v['plant_source']=='auto' for v in rows])
    gap = np.array([v['clearance_m'] for v in rows])
    stop = case.get('stop_at_s', HORIZON)
    before = t < stop
    pos = np.array([v['position_m'] for v in rows])
    last = t >= HORIZON-1.-1e-8
    stop_events = [v for v in events if v['kind']=='stop' and v.get('active')]
    ix = int(gap.argmin())
    m.update(
        below_5cm_s=float(dt[auto & (gap<.05)].sum()),
        near_10cm_longest_s=longest(auto & (gap<.1), dt),
        stationary_before_stop_longest_s=longest(auto & before & (speed<=.03), dt),
        distance_before_stop_m=float(np.linalg.norm(np.diff(pos[before, :2], axis=0), axis=1).sum()),
        timeout_events=sum(d.get('phased_event') in ('brake_timeout','turn_timeout','episode_timeout') for d in decisions),
        completed_maneuvers=sum(d.get('phased_event')=='completed' for d in decisions),
        invalid_events=sum(e['kind']=='simulation_fault' or bool(e.get('invalid')) for e in events),
        minimum=dict(time_s=float(t[ix]), position_m=pos[ix].tolist()),
        stop_requested=bool(stop_events),
        stop_request_time_s=stop_events[0]['time_s'] if stop_events else None,
        stop_active=any(d.get('phased_phase')=='STOP' for d in decisions),
        stop_last_second_complete=bool(end_s>=HORIZON-1e-8 and dt[last].sum()>=1.-1e-8),
        stop_last_second_all_auto=bool(last.any() and auto[last].all()),
        stop_last_second_max_speed_mps=float(speed[last].max()) if last.any() else None,
    )
    return m


def assess(m, case):
    checks = dict(no_contact=m['collision'] is None, horizon=m['completed'],
        valid_physics=m['invalid_events']==0, clearance_5cm=m['min_clearance_m']>=.05,
        near_10cm_episode=m['near_10cm_longest_s']<=1., near_10cm_total=m['below_10cm_s']<=2.,
        no_timeouts=m['timeout_events']==0, no_stall_before_stop=m['stationary_before_stop_longest_s']<=5.,
        motion_before_stop=m['distance_before_stop_m']>=4., auto_exposure=m['auto_s']>=40.,
        height=m['max_height_error_m']<=.1)
    if case.get('stop_at_s') is not None:
        checks.update(stop_requested=m['stop_requested'], stop_active=m['stop_active'],
            stop_full_last_second=m['stop_last_second_complete'],
            stop_last_second_auto=m['stop_last_second_all_auto'],
            stop_speed=m['stop_last_second_max_speed_mps'] is not None
                       and m['stop_last_second_max_speed_mps']<.03)
    return dict(passed=all(checks.values()), checks=checks,
        failed=[key for key, passed in checks.items() if not passed])


def trial(out, case, name, profile, *, turn_envelope=None):
    run_id = case['name']+'__'+name
    dest = out/run_id
    dest.mkdir()
    r = build_runner(case, profile, run_id, turn_envelope=turn_envelope)
    try:
        write(dest/'manifest.json', r.manifest())
        r.resume()
        r.event(dict(type='mode', mode='AUTO'))
        decisions = []
        for step in range(round(HORIZON/.004)):
            if 'stop_at_s' in case and step==round(case['stop_at_s']/.004):
                r.event(dict(type='stop'))
            r.advance()
            if step % 5 == 0 and r.rows:
                decisions.append(dict(time_s=r.rows[-1]['time_s'],
                    **plain(r.system.runtime.coordinator.last_decision)))
            if r.terminal:
                break
        rows = r.rows
        physical = dict(time=np.array([v['time_s'] for v in rows]),
            pos=np.array([v['position_m'] for v in rows]),
            velocity=np.array([v['velocity_mps'] for v in rows]),
            yaw=np.array([v['attitude_rad'][2] for v in rows]),
            clearance=np.array([v['clearance_m'] for v in rows]),
            auto=np.array([v['plant_source']=='auto' for v in rows]))
        np.savez_compressed(dest/'physical.npz', **physical)
        exact = None
        if name=='nominal' and 'stop_at_s' not in case:
            with np.load(REFERENCE/(case['name']+'_phased')/'physical.npz') as old:
                for key, values in physical.items():
                    np.testing.assert_array_equal(values, old[key], err_msg=run_id+':'+key)
            exact = True
        controls = rows[::5]
        np.savez_compressed(dest/'trace.npz', time=[v['time_s'] for v in controls],
            **{name:np.array([v['trace'].get(key) or [np.nan]*4 for v in controls]) for name,key in
               [('network','network_pwm'),('coordinated','coordinated_pwm'),('supervised','supervised_pwm'),
                ('issued','issued_pwm'),('accepted','accepted_pwm')]},
            applied=[v['plant_input_pwm'] or [np.nan]*4 for v in controls],
            flow=[v['flow_xy'] for v in controls], min_tof=[v['min_tof_m'] for v in controls],
            tof=[v['sensors']['tof']['value'] for v in controls])
        write(dest/'decisions.json', decisions)
        m = metrics_for(rows, r.time_s, decisions, r.events, case)
        result = dict(id=run_id, case=case['name'], profile=name, metrics=m,
            admission=assess(m,case), nominal_exact=exact, events=r.events)
        write(dest/'report.json', result)
        print(json.dumps(dict(id=run_id, contact=m['collision'], clearance_m=m['min_clearance_m'],
            failed=result['admission']['failed'])), flush=True)
        return result
    finally:
        r.env.close()
        del r
        gc.collect()


def report(out, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle
    lines = ['# Exploratory sensitivity v1', '',
        'Frozen candidate A / phased v2. Not hardware calibration or flight admission.', '',
        '| Case | Profile | Contact | Clearance cm | Height max cm | AUTO s | STOP max m/s, last 1s | Failed criteria |',
        '|---|---|---|---:|---:|---:|---:|---|']
    for row in rows:
        m = row['metrics']
        stop = m['stop_last_second_max_speed_mps'] if row['case']=='stop_room' else None
        lines.append(f'| {row["case"]} | {row["profile"]} | {m["collision"] or "—"} | '
            f'{100*m["min_clearance_m"]:.2f} | {100*m["max_height_error_m"]:.2f} | {m["auto_s"]:.2f} | '
            f'{stop if stop is not None else "—"} | {", ".join(row["admission"]["failed"]) or "passed"} |')
    (out/'summary.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    fig, axes = plt.subplots(2,3, figsize=(15,10), layout='constrained')
    for ax, case in zip(axes.flat, scenarios()):
        for b in case['boxes']:
            ax.add_patch(Rectangle((b['x_min'],b['y_min']),b['x_max']-b['x_min'],b['y_max']-b['y_min'],color='.85'))
        for b in case['cylinders']:
            ax.add_patch(Circle((b['center_x'],b['center_y']),b['radius'],color='.85'))
        for name in ('nominal','combined','angle_60deg'):
            with np.load(out/(case['name']+'__'+name)/'physical.npz') as data:
                line, = ax.plot(*data['pos'][:,:2].T, label=name)
                ax.scatter(*data['pos'][int(data['clearance'].argmin()),:2],marker='x',color=line.get_color())
        ax.set(xlim=(-4,4),ylim=(-4,4),title=case['name'],xlabel='X, m',ylabel='Y, m')
        ax.set_aspect('equal');ax.legend(fontsize=8)
    axes.flat[-1].axis('off')
    axes.flat[-1].text(0,.8,'Frozen controller; exploratory bounds\nX = minimum body clearance\nSTOP requested at 35 s in stop_room\nFull 9-profile results in summary.md',fontsize=12)
    fig.savefig(out/'comparison.png',dpi=140);plt.close(fig)
    fig, ax = plt.subplots(figsize=(10,8), layout='constrained')
    grid = np.array([[next(r['metrics']['min_clearance_m']*100 for r in rows
        if r['case']==c['name'] and r['profile']==name) for c in scenarios()] for name in profiles()])
    im = ax.imshow(grid, cmap='RdYlGn', vmin=0, vmax=30)
    for i, name in enumerate(profiles()):
        for j, c in enumerate(scenarios()):
            row = next(r for r in rows if r['case']==c['name'] and r['profile']==name)
            ax.text(j,i,f'{grid[i,j]:.1f}'+(' *' if not row['admission']['passed'] else ''),ha='center',va='center')
    ax.set_xticks(range(5),[c['name'] for c in scenarios()],rotation=25,ha='right')
    ax.set_yticks(range(len(profiles())),list(profiles()))
    ax.set_title('Minimum body clearance, cm; * = at least one failed criterion')
    fig.colorbar(im,ax=ax,label='cm');fig.savefig(out/'clearance.png',dpi=150);plt.close(fig)


def run(out, workers=3):
    out.mkdir(parents=True, exist_ok=False)
    inputs = [SOURCE/'checkpoint.pt', SOURCE/'runtime_config.json',
        *[p for p in (SOURCE/'numpy').iterdir() if p.is_file()],
        *[p for p in (ROOT/'data/edge/k306_tau025_integrated_v1').iterdir() if p.is_file()],
        *[REFERENCE/(c['name']+'_phased')/'physical.npz' for c in scenarios() if 'stop_at_s' not in c]]
    sources = [p for folder in ('simulation','edge','demo','evaluate','configs','core')
               for p in (ROOT/folder).rglob('*.py')]
    hashes = {str(p.relative_to(ROOT)):sha256(p) for p in inputs+sources}
    frozen = out/'frozen_inputs'
    for relative in hashes:
        target = frozen/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/relative,target)
    tasks = [(c,name,p) for c in scenarios() for name,p in profiles().items()]
    write(out/'protocol.json', dict(version='sim-uncertainty-v1', scenario_count=5, trials=len(tasks),
        scenarios=scenarios(), profiles={k:asdict(v) for k,v in profiles().items()}, input_sha256=hashes,
        horizon_s=HORIZON, seed=20261107, collision_radius_m=.2,
        frozen_model=str(SOURCE.relative_to(ROOT)), coordinator='phased-v2',
        criteria=dict(no_contact=True,no_invalid_physics=True,min_clearance_m=.05,
            near_10cm_longest_s=1.,near_10cm_total_s=2.,max_height_error_m=.1,min_auto_s=40.,
            min_distance_before_stop_m=4.,max_stationary_before_stop_s=5.,no_timeouts=True,
            stop='explicit request at 35 s; full final second in AUTO at speed <0.03 m/s'),
        parameter_search=False, hardware_promotion=False, calibrated=False,
        initial_condition='Nominal initial rotor RPM retained under all perturbations; hover controller unchanged',
        scope='Static geometry, plant gains/lag/mass and measurement scale/bias; not delay/dropout or full sim2real',
        comparison='Four nominal trajectories must exactly match prior geometric holdout; STOP is new',
        interpretation='Pass only at these sampled conditions; no statistical success rate or continuous robustness bound'))
    rows=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending=[pool.submit(trial,out,c,n,p) for c,n,p in tasks]
        for future in as_completed(pending):
            rows.append(future.result())
            rows.sort(key=lambda r:next(i for i,(c,n,p) in enumerate(tasks) if r['id']==c['name']+'__'+n))
            write(out/'report.json',rows)
    for relative,h in hashes.items():
        if sha256(ROOT/relative)!=h or sha256(frozen/relative)!=h:
            raise RuntimeError('Input changed during experiment: '+relative)
    write(out/'validation.json',dict(completed_trials=len(rows),inputs_unchanged=True,
        nominal_exact=sum(r['nominal_exact'] is True for r in rows),
        passed=sum(r['admission']['passed'] for r in rows),
        profiles={name:dict(passed=sum(r['admission']['passed'] for r in rows if r['profile']==name),
            contacts=sum(r['metrics']['collision'] is not None for r in rows if r['profile']==name)) for name in profiles()},
        calibrated=False, hardware_admission=False))
    report(out,rows)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--workers',type=int,choices=(1,2,3),default=3)
    args=parser.parse_args()
    run(args.out,args.workers)
