"""Fixed Step-4 physical protocol: receiver, two-hop link, expiry and reset.

No socket, FC connection, weight change or controller tuning. Simulated clocks
have distinct origins with an explicitly known mapping for sensor timestamps.
"""
import argparse
from collections import Counter, deque
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path

import numpy as np

from edge.runtime import AltitudeSample, sha256
from edge.sensors import SensorSnapshot
from edge.session_receiver import CommandPacket, SessionReceiver
from edge.system import RunConfig, build_system, observe_receiver
from edge.transport import LinkConfig, SimulatedTransport
from simulation.sensor_schedule import SampleSchedule
from simulation.altitude_control import sample_from_observation
from simulation.benchmark_scenes import NavigationScene
from simulation.receiver_fallback import ReceiverFallback

DT = .004
PI_ORIGIN = 10.
RX_ORIGIN = 1000.


def within_roundtrip_budget(seconds):
    # The strict 20 ms requirement must not depend on subtraction roundoff
    # between the distinct simulated clock origins.
    return math.isfinite(seconds) and 0 <= seconds < .02-1e-9


class PiComputeQueue:
    """One FIFO worker, polled on physics ticks; no overlapping inference.

    The control tick is evaluated at completion (including age validation).
    This is a fixed service-time model, not a measured OS/CPU scheduler.
    """
    def __init__(self, delay_s):
        if not math.isfinite(delay_s) or delay_s < 0:
            raise ValueError('Pi compute delay must be finite and nonnegative')
        self.delay_s = delay_s
        self.waiting = deque()
        self.active = None

    def _start(self, now_s):
        if self.active is None and self.waiting:
            payload, received = self.waiting.popleft()
            self.active = dict(payload=payload, received_s=received,
                               started_s=now_s, due_s=now_s+self.delay_s)

    def submit(self, payload, *, now_s):
        self.waiting.append((payload, now_s))
        self._start(now_s)

    def drain(self, now_s):
        ready = []
        while self.active is not None and self.active['due_s'] <= now_s+1e-10:
            ready.append(dict(self.active, completed_s=now_s))
            self.active = None
            self._start(now_s)
        return ready


@dataclass(frozen=True)
class Case:
    name: str
    fault: str = ''
    fault_at_s: float = 4.
    profile: str = 'multi'
    jitter_s: float = 0.
    reentry: bool = False
    scene: str = 'clear'
    maneuver: bool = False
    altitude_delivery_hold_s: float = 0.
    uplink_delay_s: float = .004
    horizon_s: float = 10.
    pi_compute_delay_s: float = 0.


CASES = (
    Case('normal'), Case('bounded_jitter', jitter_s=.002),
    Case('response_loss', fault='response_loss'),
    Case('response_expiry', fault='response_delay'),
    Case('sensor_loss', fault='sensor_loss'),
    Case('manual_reentry', reentry=True),
    Case('watchdog_reentry', fault='response_loss', reentry=True),
    Case('old_session_reentry', fault='old_duplicate', reentry=True),
    Case('maneuver_sync', profile='sync', scene='obstacle_left', maneuver=True),
    Case('maneuver_multi', scene='obstacle_left', maneuver=True),
    Case('maneuver_latency', scene='obstacle_left', maneuver=True, uplink_delay_s=.012),
    Case('maneuver_held', scene='obstacle_left', maneuver=True,
         altitude_delivery_hold_s=.02, uplink_delay_s=.012),
    Case('maneuver_loss', scene='obstacle_left', maneuver=True, fault='response_loss', fault_at_s=6.24),
    Case('maneuver_manual_reentry', scene='obstacle_left', maneuver=True, reentry=True),
)


def authority(case, t):
    if case.reentry:
        revoke = 5. if case.fault == 'response_loss' else 4.
        reset = revoke+.5
        if revoke <= t < reset: return 'MANUAL'
        if reset <= t < reset+1.5: return 'SHADOW'
    return 'SHADOW' if t < 2. else 'AUTO'


def reset_time(case):
    return (5.5 if case.fault == 'response_loss' else 4.5) if case.reentry else None


def map_frame(frame, sequence):
    """Known simulation clock offset, not a hardware synchronization algorithm."""
    if isinstance(frame, SensorSnapshot):
        channels = {ch: replace(getattr(frame, ch),
                    timestamp_s=getattr(frame, ch).timestamp_s+PI_ORIGIN)
                    for ch in ('flow', 'tof', 'altitude', 'attitude')}
        return replace(frame, sequence=sequence, timestamp_s=frame.timestamp_s+PI_ORIGIN, **channels)
    return replace(frame, sequence=sequence, timestamp_s=frame.timestamp_s+PI_ORIGIN,
                   altitude=replace(frame.altitude, timestamp_s=frame.altitude.timestamp_s+PI_ORIGIN))


def physical_case(config, case):
    compute = PiComputeQueue(case.pi_compute_delay_s)
    receiver = SessionReceiver()
    link_config = LinkConfig(uplink_delay_s=case.uplink_delay_s,
                            jitter_s=case.jitter_s, seed=20261023)
    link = SimulatedTransport(link_config)
    fallback = ReceiverFallback()
    schedule = SampleSchedule(case.profile)
    local_schedule = SampleSchedule('multi')
    session = case.name+'-0'
    receiver.reset(session_id=session, now_s=RX_ORIGIN)
    def assemble(session):
        return build_system(replace(config, session_id=session,
            sensor_mode='multi_rate_v1' if case.profile == 'multi' else 'strict_synchronous_v1'))
    system = assemble(session)
    manifests = [system.manifest]
    receiver.set_mode('SHADOW', now_s=RX_ORIGIN)
    env = NavigationScene(case.scene)
    traces, extras, rows, events, resets = {}, {}, [], [], []
    previous_pwm = (1213., 1500., 1500., 1500.)
    fallback_pwm = previous_pwm
    sequence = 0
    altitude_queue = []
    held_altitude = None
    last_captured = None
    done = False
    info = {}
    try:
        env.reset(initial_pos=[-.7 if case.maneuver else 0., 0., 1.], seed=20261021)
        env.physics.vel[:] = 0.
        obs = env._sample_observation(0.)
        for step in range(round(case.horizon_s/DT)):
            t = step*DT
            rx_now = RX_ORIGIN+t
            # Local revocation precedes link delivery. Reset is explicit in MANUAL;
            # old datagrams intentionally remain queued across the new session.
            if reset_time(case) is not None and abs(t-reset_time(case)) < 1e-9:
                receiver.set_mode('MANUAL', now_s=rx_now)
                session = case.name+'-1'
                receiver.reset(session_id=session, now_s=rx_now)
                system = assemble(session)
                manifests.append(system.manifest)
                sequence = 0
                resets.append(dict(time_s=t, session_id=session,
                    hidden_max_abs=float(np.max(abs(system.runtime.policy.h))),
                    altitude_integral=system.runtime.altitude._integral,
                    bridge_state=system.bridge.state))
            requested = authority(case, t)
            receiver.set_mode(requested, now_s=rx_now)
            before = receiver.output(now_s=rx_now, fallback_pwm=fallback_pwm)
            if step % 5 == 0:
                sample = sample_from_observation(env.get_isaac_obs())
                local = local_schedule.frame(step//5, obs[:2], obs[2:], sample)
                local_sample = AltitudeSample(*local.altitude.value, local.altitude.timestamp_s, True)
                local_flow = local.flow.value
            fallback_pwm = fallback.step(local_flow, local_sample, now_s=t,
                active=before['source']=='fallback', previous_pwm=previous_pwm)
            if step % 5 == 0:
                frame = schedule.frame(step//5, obs[:2], obs[2:], sample)
                if case.altitude_delivery_hold_s:
                    # Add an explicit sensor-delivery delay, retaining its capture
                    # timestamp AND capture angles. Bootstrap the first sample.
                    if held_altitude is None: held_altitude = frame.altitude
                    if last_captured != frame.altitude.sample_id:
                        altitude_queue.append((t+case.altitude_delivery_hold_s, frame.altitude))
                        last_captured = frame.altitude.sample_id
                    while altitude_queue and altitude_queue[0][0] <= t+1e-9:
                        _, held_altitude = altitude_queue.pop(0)
                    frame = replace(frame, altitude=held_altitude)
                frame = map_frame(frame, sequence)
                receiver.announce_sensor(sequence, now_s=rx_now)
                sender_request = 'SHADOW' if receiver.gate.tripped and requested=='AUTO' else requested
                packet = dict(session_id=session, frame=frame, request=sender_request,
                    manual_pwm=fallback_pwm, stop_requested=not case.maneuver or t < 2.5,
                    sensor_sent_global_s=t)
                link.send('uplink', packet, now_s=t,
                    drop=case.fault=='sensor_loss' and abs(t-case.fault_at_s)<1e-9)
                sequence += 1
            for delivery in link.drain(t):
                if delivery.direction == 'uplink':
                    compute.submit(delivery.payload, now_s=t)
                else:
                    packet = delivery.payload
                    event = receiver.receive(packet, now_s=rx_now)
                    events.append(dict(asdict(event), time_s=t))
                    key = (packet.session_id, packet.sequence)
                    trace = traces.get(key)
                    if trace is not None and trace.issued_pwm is not None and trace.receiver_status=='unobserved':
                        traces[key] = observe_receiver(trace, session_id=packet.session_id,
                            sequence=packet.sequence, accepted_pwm=event.accepted_pwm,
                            receiver_time_s=rx_now, reason=event.reason)
            for job in compute.drain(t):
                packet = job['payload']
                if packet['session_id'] != session:
                    events.append(dict(time_s=t, reason='old_sensor_session_ignored'))
                    continue
                result = system.tick(packet['frame'], now_s=PI_ORIGIN+t,
                    request=packet['request'], manual_pwm=packet['manual_pwm'],
                    stop_requested=packet['stop_requested'])
                trace = result['trace']
                key = (trace.session_id, trace.sequence)
                traces[key] = trace
                extras[key] = dict(manual_pwm=packet['manual_pwm'],
                    sensor_sent_global_s=packet['sensor_sent_global_s'],
                    issue_global_s=t,
                    pi_received_global_s=job['received_s'],
                    pi_started_global_s=job['started_s'],
                    pi_due_global_s=job['due_s'],
                    range_capture_age_s=PI_ORIGIN+t-(packet['frame'].altitude.timestamp_s),
                    range_capture_angles_rad=(list(packet['frame'].altitude.value[1:])
                        if isinstance(packet['frame'],SensorSnapshot) else
                        [packet['frame'].altitude.roll_rad,packet['frame'].altitude.pitch_rad]),
                    current_angles_rad=np.asarray(env.get_isaac_obs().attitude_euler[:2]).tolist(),
                    estimated_height_m=(None if system.runtime.altitude.last_command is None
                                        else system.runtime.altitude.last_command.height_m),
                    actual_height_m=float(env.physics.pos[2]),
                    manual_to_first_issue_pwm=None if result['lease'] is None
                    or result['reason']!='engaged' else float(np.max(abs(
                        np.asarray(result['lease'].pwm)-packet['manual_pwm']))))
                response = CommandPacket(session, trace.sequence, result['lease'])
                fault_frame = abs(packet['sensor_sent_global_s']-case.fault_at_s)<1e-9
                link.send('downlink', response, now_s=t,
                    drop=case.fault=='response_loss' and fault_frame,
                    delay_s=.064 if case.fault=='response_delay' and fault_frame else None)
                if case.fault=='old_duplicate' and abs(packet['sensor_sent_global_s']-3.98)<1e-9:
                    link.send('downlink', response, now_s=t, delay_s=2.4)
            selected = receiver.output(now_s=rx_now, fallback_pwm=fallback_pwm)
            applied = selected['applied_pwm']
            angles = env.get_isaac_obs().attitude_euler
            rows.append(dict(time_s=t, **selected, position_m=env.physics.pos.tolist(),
                velocity_mps=env.physics.vel.tolist(), attitude_rad=np.asarray(angles).tolist(),
                local_mode=requested, fallback_integral=fallback.altitude._integral,
                deadline_receiver_s=receiver.gate.deadline,
                transition_jump_pwm=float(np.max(abs(np.asarray(applied)-previous_pwm)))))
            previous_pwm = applied
            obs, _, done, info = env.step(applied)
            rows[-1]['next_time_s']=t+DT
            rows[-1]['next_clearance_m']=float(info['clearance_m'])
            if done: break
    finally:
        env.close()
    trace_rows = [dict(trace.to_dict(), **extras[key]) for key, trace in traces.items()]
    auto = [r for r in rows if r['source']=='auto']
    maneuver_rows = [r for r in auto if max(abs(x) for x in r['attitude_rad'][:2]) >= np.deg2rad(5)]
    active_intervals = []
    for i, row in enumerate(rows):
        if row['source']=='auto' and (i==0 or rows[i-1]['source']!='auto'):
            active_intervals.append(row['time_s'])
    trips = [r for i,r in enumerate(rows) if r['tripped'] and (i==0 or not rows[i-1]['tripped'])]
    accepted = [r for r in trace_rows if r['receiver_status']=='accepted']
    ages = [r['range_capture_age_s'] for r in trace_rows if r['authority']=='AUTO']
    maneuver_traces = [r for r in trace_rows if r['receiver_status']=='accepted'
                      and max(abs(x) for x in r['current_angles_rad'])>=np.deg2rad(5)]
    latencies = [r['receiver_time_s']-RX_ORIGIN-r['sensor_sent_global_s'] for r in accepted]
    reentry = reset_time(case)
    stopped = [r for r in rows if r['time_s'] >= case.horizon_s-1]
    metric = dict(completed_s=len(rows)*DT, crashed=bool(info.get('crashed', False)), terminated=bool(done),
        auto_entries_s=active_intervals, auto_duration_s=len(auto)*DT,
        trips=[dict(time_s=r['time_s'], reason=r['reason'], accepted_pwm=r['accepted_pwm']) for r in trips],
        accepted_count=len(accepted), issued_count=sum(r['issued_pwm'] is not None for r in trace_rows),
        unobserved_issued_count=sum(r['issued_pwm'] is not None and r['receiver_status']=='unobserved' for r in trace_rows),
        max_accepted_roundtrip_s=max(latencies, default=None),
        max_pi_queue_wait_s=max((r['pi_started_global_s']-r['pi_received_global_s']
                                for r in trace_rows), default=None),
        max_pi_service_elapsed_s=max((r['issue_global_s']-r['pi_started_global_s']
                                     for r in trace_rows), default=None),
        max_altitude_age_auto_s=max(ages, default=None),
        max_altitude_age_maneuver_s=max((r['range_capture_age_s'] for r in maneuver_traces),default=None),
        max_capture_current_angle_difference_deg=max((float(np.rad2deg(np.max(abs(
            np.asarray(r['current_angles_rad'])-r['range_capture_angles_rad'])))) for r in maneuver_traces),default=None),
        max_estimated_height_error_maneuver_m=max((abs(r['estimated_height_m']-r['actual_height_m'])
            for r in maneuver_traces if r['estimated_height_m'] is not None),default=None),
        max_height_error_auto_m=max((abs(r['position_m'][2]-1) for r in auto),default=None),
        max_height_error_maneuver_m=max((abs(r['position_m'][2]-1) for r in maneuver_rows),default=None),
        maneuver_duration_s=len(maneuver_rows)*DT,
        max_tilt_deg=max(float(np.rad2deg(max(abs(x) for x in r['attitude_rad'][:2]))) for r in rows),
        final_height_m=float(env.physics.pos[2]),
        final_speed_mps=float(np.linalg.norm(env.physics.vel[:2])),
        last_second_max_speed_mps=max(float(np.linalg.norm(r['velocity_mps'][:2])) for r in stopped),
        receiver_reasons=dict(Counter(e['reason'] for e in events)),
        bridge_faults=dict(Counter(r['fault'] for r in trace_rows if r['fault'])),
        entry_issue_manual_jumps_pwm=[r['manual_to_first_issue_pwm'] for r in trace_rows
                                     if r['manual_to_first_issue_pwm'] is not None],
        shadow_integral_max_abs=max((abs(r['altitude_integral']) for r in trace_rows
            if r['authority']=='SHADOW' and (reentry is None or r['issue_global_s']>=reentry)), default=0.),
        min_clearance_m=min(r['next_clearance_m'] for r in rows),
        entry_actual_throttle_jump_pwm=[abs(r['applied_pwm'][0]-rows[i-1]['applied_pwm'][0])
            for i,r in enumerate(rows) if i>0 and r['source']=='auto' and rows[i-1]['source']!='auto'],
        resets=resets)
    expected_trip = bool(case.fault in ('response_loss','response_delay','sensor_loss') or case.reentry)
    checks = dict(full_horizon=not done and abs(metric['completed_s']-case.horizon_s)<1e-8,
        entered_auto=bool(auto), bounded_accepted_latency=bool(latencies) and within_roundtrip_budget(max(latencies)),
        expected_trip=bool(trips)==expected_trip,
        trip_has_no_accepted=all(r['accepted_pwm'] is None and r['source']=='fallback' for r in rows if r['tripped']),
        reentry_after_reset=(len(active_intervals)==2 and active_intervals[-1]>=reentry+1.5) if case.reentry
                            else len(active_intervals)==1,
        first_issue_tracks_manual=all(x==0. for x in metric['entry_issue_manual_jumps_pwm']),
        reset_shadow_integral_zero=metric['shadow_integral_max_abs']==0. if not expected_trip or case.reentry else True,
        old_session_ignored=(metric['receiver_reasons'].get('session_mismatch',0)==1
                             and len(trips)==1) if case.fault=='old_duplicate' else True,
        watchdog_timing=(bool(trips) and abs(trips[0]['time_s']-(case.fault_at_s+.02))<=DT+1e-9)
            if case.fault in ('response_loss','response_delay','sensor_loss') else True,
        revocation_immediate=(bool(trips) and trips[0]['time_s']==4.)
            if case.reentry and case.fault!='response_loss' else True,
        reset_clears_hidden_and_integral=all(r['hidden_max_abs']==0. and r['altitude_integral']==0.
            and r['bridge_state']=='MANUAL' for r in resets),
        maneuver_exercised=bool(maneuver_rows) if case.maneuver else True,
        held_age_exercised=(metric['max_altitude_age_maneuver_s'] is not None
            and .04<=metric['max_altitude_age_maneuver_s']<=.06+1e-8)
            if case.altitude_delivery_hold_s else True)
    return dict(case=asdict(case), link=asdict(link_config), metrics=metric, checks=checks,
                passed=all(checks.values()), manifests=manifests, traces=trace_rows,
                physics=rows, receiver_events=events, link_events=link.events)


def plot(output, reports):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(reports), 3, figsize=(15, 2.3*len(reports)), squeeze=False)
    for axes_row, report in zip(axes, reports):
        rows = report['physics']
        t = [r['time_s'] for r in rows]
        axes_row[0].plot(t, [r['position_m'][2] for r in rows])
        axes_row[0].axhline(1., color='gray', ls='--', lw=.7)
        axes_row[0].set_ylabel(report['case']['name']+'\nheight (m)')
        axes_row[1].plot(t, [r['applied_pwm'][0] for r in rows], label='plant throttle')
        axes_row[1].plot(t, [r['accepted_pwm'][0] if r['accepted_pwm'] else np.nan for r in rows],
                         ls='--', label='active accepted')
        axes_row[2].plot(t, [r['source']=='auto' for r in rows], label='AUTO applied')
        axes_row[2].plot(t, [r['tripped'] for r in rows], label='receiver tripped', alpha=.7)
        for ax in axes_row: ax.grid(alpha=.2); ax.set_xlabel('time (s)')
    axes[0,1].legend(); axes[0,2].legend()
    fig.suptitle('Step 4: physical simulation with receiver and transport (not a flight test)')
    fig.tight_layout(rect=(0,0,1,.985)); fig.savefig(output/'transport_summary.png', dpi=120)
    plt.close(fig)
    from matplotlib.patches import Rectangle
    fig, axes = plt.subplots(2,2,figsize=(12,8))
    for report in reports:
        if report['case']['name'] not in ('maneuver_sync','maneuver_multi','maneuver_latency','maneuver_held'):
            continue
        rows, traces = report['physics'], report['traces']
        label=report['case']['name'].removeprefix('maneuver_')
        t=np.array([r['time_s'] for r in rows])
        axes[0,0].plot([r['position_m'][0] for r in rows],[r['position_m'][1] for r in rows],label=label)
        axes[0,1].plot(t,[100*(r['position_m'][2]-1.) for r in rows],label=label)
        axes[1,0].plot(t,[np.rad2deg(max(abs(x) for x in r['attitude_rad'][:2])) for r in rows],label=label)
        axes[1,1].plot([r['issue_global_s'] for r in traces],
                      [1000*r['range_capture_age_s'] for r in traces],label=label,alpha=.8)
    axes[0,0].add_patch(Rectangle((1.1,.02),.15,.48,color='gray',alpha=.5))
    axes[0,0].set(xlabel='x (m)',ylabel='y (m)',title='Trajectory and physical obstacle',aspect='equal')
    axes[0,1].set(xlabel='time (s)',ylabel='height error (cm)',title='Height relative to 1 m')
    axes[1,0].set(xlabel='time (s)',ylabel='max |roll/pitch| (degrees)',title='Actual attitude')
    axes[1,1].set(xlabel='time (s)',ylabel='range age at Pi (ms)',title='Capture-to-processing age')
    for ax in axes.flat: ax.grid(alpha=.2); ax.legend()
    fig.suptitle('Held range keeps capture angles; gains and weights unchanged')
    fig.tight_layout(); fig.savefig(output/'maneuver_comparison.png',dpi=150); plt.close(fig)


def run(config_path, output, *, pi_compute_delay_s=0.):
    PiComputeQueue(pi_compute_delay_s)  # reject invalid settings before creating output
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    config = RunConfig.from_json(config_path)
    cases = tuple(replace(c, pi_compute_delay_s=pi_compute_delay_s) for c in CASES)
    protocol = dict(version='transport-loop-v4', cases=[asdict(c) for c in cases],
        config=config.to_dict(), physics_dt_s=DT, control_period_s=.02,
        clock_origins=dict(pi=PI_ORIGIN,receiver=RX_ORIGIN),
        clock_mapping='Known constant offsets only; no estimated drift or hardware clock sync.',
        acceptance='All explicit checks in report; height/maneuver errors descriptive, no flight qualification.',
        ordering='local revoke/reset, watchdog poll, sensor send, link delivery, Pi completion, receiver selection, physics',
        link='Two seeded datagram queues; service every 4 ms; no retries.',
        pi_compute=dict(delay_s=pi_compute_delay_s, workers=1, queue='FIFO',
            evaluation='Control tick and sensor age validation at completion; polled every 4 ms.',
            limitation='Fixed service-time assumption, not measured Pi latency; no serialization or OS jitter.'),
        fallback='Sensor-only local altitude hold plus flow damping; simulation reference, not G4/ESP firmware.',
        source_sha256={str(p):sha256(p) for p in map(Path, [__file__, 'edge/session_receiver.py',
            'edge/transport.py','simulation/receiver_fallback.py','simulation/sensor_schedule.py',
            'simulation/benchmark_scenes.py','simulation/drone_env.py'])})
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n',encoding='utf-8')
    results=[]
    for case in cases:
        r=physical_case(config,case)
        for kind in ('traces','physics','receiver_events','link_events'):
            with (output/(case.name+'_'+kind+'.jsonl')).open('x',encoding='utf-8') as f:
                for row in r[kind]: f.write(json.dumps(row,allow_nan=False)+'\n')
        results.append(r)
        print(json.dumps(dict(case=case.name,passed=r['passed'],metrics=r['metrics'])),flush=True)
    plot(output,results)
    summaries=[{k:v for k,v in r.items() if k not in ('traces','physics','receiver_events','link_events')} for r in results]
    report=dict(protocol=protocol,passed=all(r['passed'] for r in results),runs=summaries)
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path('configs/mvp_runtime_v3.json'))
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--pi-compute-delay-s',type=float,default=0.,
                   help='Assumed single-worker service time; completion is polled every 4 ms')
    args=p.parse_args()
    report=run(args.config,args.out_dir,pi_compute_delay_s=args.pi_compute_delay_s)
    if not report['passed']: raise SystemExit(1)


if __name__=='__main__': main()
