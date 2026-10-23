"""One small experiment: move -> brake -> scan -> resume, with measured traces.

Run from the repository root: python -m scripts.scan_probe
This scripted reference maneuver uses flow feedback and the existing altitude
controller. It does not load/train a connectome or choose exploration targets.
"""
import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from configs.flight_config import CONTROL_DT, DEFAULT_DT
from artifacts.io import publication, write_json
from generator.source_contract import file_sha256
from optimizer.rollout import simulate_policy_rollout
from simulation.altitude_control import AltitudeHold
from simulation.drone_env import DroneSimulationEnv, RoomBoundaries

PHASES = (('move', 2.), ('brake', 3.), ('scan', 2.), ('resume', 3.))


@dataclass(frozen=True)
class ProbeConfig:
    flow_gain_pwm: float = 400.
    scan_yaw_offset_pwm: float = 80.
    initial_speed_mps: float = .6
    settle_flow: float = .05
    settle_s: float = .2

    def __post_init__(self):
        if (not all(np.isfinite(v) for v in asdict(self).values())
                or not 0 <= self.flow_gain_pwm <= 1000
                or not 0 <= self.scan_yaw_offset_pwm <= 200
                or not 0 <= self.initial_speed_mps <= 2
                or not 0 < self.settle_flow <= .1 or not 0 < self.settle_s <= 3):
            raise ValueError('Invalid scan probe configuration')


class ScanProbePolicy:
    """Prescribed phases, sensor-only navigation feedback; no position access."""
    def __init__(self, config):
        self.config = config
        self.reset_state()

    def reset_state(self):
        self.step_index, self.settled_s = 0, 0.
        self.phase, self.blocked_reason = 'move', None

    def step_np(self, flow, tof, memory_ring=None, dt=CONTROL_DT):
        if dt != CONTROL_DT:
            raise ValueError('This experiment uses the standard 20 ms control period')
        flow = np.asarray(flow, dtype=float)
        if flow.shape != (2,) or not np.isfinite(flow).all() or np.any(abs(flow) > 1):
            raise ValueError('Expected finite normalized flow[2]')
        elapsed, boundary = self.step_index*dt, 0.
        for name, duration in PHASES:
            boundary += duration
            if elapsed < boundary:
                self.phase = name
                break
        if self.phase == 'brake':
            self.settled_s = self.settled_s+dt if np.linalg.norm(flow) <= self.config.settle_flow else 0.
        scan_start_step = round(sum(d for _,d in PHASES[:2])/dt)
        if self.step_index == scan_start_step and self.settled_s+1e-9 < self.config.settle_s:
            self.blocked_reason = 'Motion did not settle before the scheduled scan'
        if self.blocked_reason:
            self.phase = 'blocked'
        cruise = 100. if self.phase in ('move','resume') else 0.
        yaw = self.config.scan_yaw_offset_pwm if self.phase == 'scan' else 0.
        self.step_index += 1
        return np.clip([1213., 1500.+self.config.flow_gain_pwm*flow[1],
                        1500.+cruise-self.config.flow_gain_pwm*flow[0], 1500.+yaw], 1000., 2000.)


class RecordingEnv(DroneSimulationEnv):
    """Records ground truth for evaluation only; policy receives sensor inputs."""
    def __init__(self, policy):
        super().__init__(dt=DEFAULT_DT, engine='standalone', headless=True,
                         room=RoomBoundaries(-4,4,-4,4,0,4))
        self.boxes, self.cylinders = [], []
        self.policy, self.rows = policy, []

    def reset(self, seed=None):
        self.rows.clear()
        super().reset(initial_pos=np.array([0.,0.,1.]), seed=seed)
        self.physics.quat[:] = [1.,0.,0.,0.]
        self.physics.vel[:] = [self.policy.config.initial_speed_mps,0.,0.]
        return self._sample_observation(elapsed_dt=0.)

    def record(self, action, phase=None):
        x,y,z = map(float,self.physics.pos)
        yaw = float(self.physics.quaternion_to_euler(self.physics.quat)[2])
        obs = self._get_observation()  # cached snapshot, not a new noisy sample
        self.rows.append(dict(time_s=self.step_count*self.dt, phase=phase or self.policy.phase,
            x_m=x, y_m=y, z_m=z, yaw_rad=yaw,
            horizontal_speed_mps=float(np.linalg.norm(self.physics.vel[:2])),
            flow_norm=float(np.linalg.norm(obs[:2])),
            throttle_pwm=float(action[0]), roll_pwm=float(action[1]),
            pitch_pwm=float(action[2]), yaw_pwm=float(action[3])))

    def step(self, action, action_type='auto'):
        self.record(action)  # applied command, including altitude-owned throttle
        return super().step(action, action_type=action_type)


def phase_metrics(rows):
    """Use phase-entry position; include the boundary endpoint; unwrap yaw."""
    if not rows:
        return {}
    yaw = np.unwrap([r['yaw_rad'] for r in rows])
    metrics = {}
    for phase in dict.fromkeys(r['phase'] for r in rows):
        indices = [i for i,r in enumerate(rows) if r['phase'] == phase]
        first, end = indices[0], min(indices[-1]+1, len(rows)-1)
        window = rows[first:end+1]
        xy = np.array([[r['x_m'],r['y_m']] for r in window])
        metrics[phase] = dict(duration_s=window[-1]['time_s']-window[0]['time_s'],
            max_drift_m=float(np.linalg.norm(xy-xy[0],axis=1).max()),
            horizontal_path_m=float(np.linalg.norm(np.diff(xy,axis=0),axis=1).sum()),
            max_height_error_m=max(abs(r['z_m']-1.) for r in window),
            yaw_change_deg=-float(np.rad2deg(yaw[end]-yaw[first])),
            mean_speed_mps=float(np.mean([r['horizontal_speed_mps'] for r in window])))
    return metrics


def run_probe(config, seed=42):
    policy = ScanProbePolicy(config)
    env = RecordingEnv(policy)
    try:
        _, rollout = simulate_policy_rollout(policy, env=env, seed=seed,
            eval_steps=round(sum(d for _,d in PHASES)/CONTROL_DT), altitude_hold=AltitudeHold())
        if env.rows:
            env.record([env.rows[-1][k] for k in ('throttle_pwm','roll_pwm','pitch_pwm','yaw_pwm')],
                       phase=env.rows[-1]['phase'])
        rows = env.rows
        phases = phase_metrics(rows)
        # Heading sectors are an observation-direction proxy, not mapped area
        # or information gain. Repeated visits do not earn additional sectors.
        before, during = set(), set()
        yaw0 = rows[0]['yaw_rad'] if rows else 0.
        for r in rows:
            sector = int(np.floor((r['yaw_rad']-yaw0+np.pi/8)/(np.pi/4))) % 8
            if r['phase'] in ('move','brake'):
                before.add(sector)
            elif r['phase'] == 'scan':
                during.add(sector)
        report = dict(format_version='scan-probe-v1', seed=seed, config=asdict(config),
            schedule=[dict(phase=p,duration_s=d) for p,d in PHASES],
            research_only=True, training_ready=False, rollout=rollout, phases=phases,
            blocked_reason=policy.blocked_reason,
            scan_new_heading_sectors=len(during-before) if 'scan' in phases else None,
            reported_yaw_convention='right-positive; trace yaw_rad is native FLU/left-positive',
            limitations=['Scripted maneuver in an empty room; no connectome policy or exploration decision.',
                        'No moving obstacles, takeover, or memory-based planning in this probe.',
                        'Heading sectors are not information gain; reported drift uses simulator ground truth.',
                        'No acceptance thresholds agreed: completing the horizon is not hover admission.'])
        return report, rows
    finally:
        env.close()


def plot_trace(rows, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3,2,figsize=(12,9),layout='constrained')
    t = np.array([r['time_s'] for r in rows])
    if not len(t):
        for ax in axes.flat:
            ax.text(.1,.5,'No executed trajectory', transform=ax.transAxes)
    else:
        values = lambda key: np.array([r[key] for r in rows])
        axes[0,0].plot(values('x_m'),values('y_m'))
        axes[0,0].scatter(values('x_m')[0],values('y_m')[0],label='Start',color='green')
        axes[0,0].set(xlabel='x (m)',ylabel='y (m)',title='Horizontal trajectory',aspect='equal')
        axes[0,0].legend()
        series = [('horizontal_speed_mps','Horizontal speed (m/s)'),
                  ('z_m','Height (m)'), ('yaw_rad','Yaw (degrees, right positive)'),
                  ('pitch_pwm','Pitch command (PWM)'), ('yaw_pwm','Yaw command (PWM)')]
        colors = dict(move='#c8e6c9',brake='#ffe0b2',scan='#bbdefb',resume='#d1c4e9',blocked='#ffcdd2')
        for ax,(key,label) in zip(list(axes.flat)[1:],series):
            y = -np.rad2deg(np.unwrap(values(key))) if key == 'yaw_rad' else values(key)
            ax.plot(t,y,color='#263238',lw=1.3)
            start = 0
            for i in range(1,len(rows)+1):
                if i == len(rows) or rows[i]['phase'] != rows[start]['phase']:
                    end = t[min(i,len(t)-1)]
                    phase = rows[start]['phase']
                    ax.axvspan(t[start],end,color=colors[phase],alpha=.5)
                    ax.text((t[start]+end)/2,1.01,phase,ha='center',fontsize=8,
                            transform=ax.get_xaxis_transform())
                    start = i
            ax.set(xlabel='Time (s)',ylabel=label)
            ax.grid(alpha=.2)
    fig.suptitle('Move -> brake -> scan -> resume | scripted reference maneuver',fontsize=14)
    fig.savefig(output,dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--flow-gain',type=float,default=400.)
    parser.add_argument('--yaw-offset',type=float,default=80.)
    parser.add_argument('--initial-speed',type=float,default=.6)
    parser.add_argument('--out-dir',type=Path)
    args = parser.parse_args()
    config = ProbeConfig(flow_gain_pwm=args.flow_gain,scan_yaw_offset_pwm=args.yaw_offset,
                         initial_speed_mps=args.initial_speed)
    report, rows = run_probe(config,args.seed)
    import platform
    import matplotlib
    import torch
    report['runtime_versions'] = dict(python=platform.python_version(),numpy=np.__version__,
                                     torch=torch.__version__,matplotlib=matplotlib.__version__)
    root = Path(__file__).resolve().parents[1]
    files = ['scripts/scan_probe.py','optimizer/rollout.py','simulation/drone_env.py',
             'simulation/drone_interface.py','simulation/altitude_control.py',
             'simulation/avionics_filter.py','simulation/pmw3901_emulator.py',
             'simulation/memory.py','simulation/metrics.py','simulation/collisions.py',
             'simulation/control_contract.py','configs/flight_config.py']
    report['code_sha256'] = {name:file_sha256(root/name) for name in files}
    output = args.out_dir or root/'data'/'diagnostics'/('scan_probe_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f'))
    with publication(output) as stage:
        write_json(stage/'report.json',report)
        with (stage/'trace.csv').open('w',newline='',encoding='utf-8') as stream:
            if rows:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
        plot_trace(rows,stage/'trace.png')
    print('Phase      duration(s)  max drift(m)  height error(m)  yaw change(deg)')
    for phase,m in report['phases'].items():
        print(f"{phase:10} {m['duration_s']:10.2f} {m['max_drift_m']:13.3f} "
              f"{m['max_height_error_m']:16.3f} {m['yaw_change_deg']:16.1f}")
    print(f"Horizon completed: {report['rollout']['horizon_completed']}; "
          f"scan blocked: {report['blocked_reason']}; new heading sectors: {report['scan_new_heading_sectors']}")
    if report['rollout']['failure_reason'] or report['rollout']['crashed']:
        print(f"Failure: {report['rollout']['failure_reason'] or report['rollout']['collision_kind']}")
    print(f'Results: {output.resolve()}')


if __name__ == '__main__':
    main()
