"""Closed-loop simulation trial of explicit empirical cruise permissions."""
import argparse
from collections import Counter, deque
from contextlib import closing
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np

from demo.runner import DemoRunner
from edge.empirical_corridor import EmpiricalCorridor
from edge.free_space_swath import FreeSpaceSwath, motion_endpoint
from edge.runtime import sha256
from evaluate.passive_corridor import RevisitProbe, USER_ROOM
from evaluate.compare_maneuver_repair import summarize
from simulation import side_clearance
from simulation.navigation_metrics import progress_metrics, share_metrics
from simulation.phased_yaw import YawRamp
from simulation.survey_heading_sensor import SimulatedHeadingSensor


def run(case, out, duration=180., mode='legacy'):
    out.mkdir(parents=True, exist_ok=False)
    frozen = out / 'source'
    frozen.mkdir()
    source_hashes = {}
    for source in (Path('edge/empirical_corridor.py'), Path('edge/free_space_swath.py'), Path(__file__)):
        target = frozen / source.name
        target.write_bytes(source.read_bytes())
        source_hashes[str(source)] = sha256(target)
    fixture = json.loads(USER_ROOM.read_text(encoding='utf-8'))
    live = case != 'clear'
    settings = dict(scene=fixture['scene'], start=fixture['start'], yaw=fixture['yaw'],
                    scene_options=fixture['scene_options']) if live else dict(scene='clear')
    profile = 'phased_side_guard' if case in ('box', 'phase') or (mode == 'swath' and case == 'clear') else 'candidate_a_brake'
    if mode == 'swath' and case == 'cylinder':
        raise ValueError('Swath recovery requires a phased controller; use phase or box')
    original_class = side_clearance.SideClearanceCoordinator
    if case == 'box':
        side_clearance.SideClearanceCoordinator = RevisitProbe
    try:
        runner = DemoRunner(controller_mode=profile, seed=fixture['seed'], max_duration_s=duration, **settings)
    finally:
        side_clearance.SideClearanceCoordinator = original_class
    with closing(runner):
        observer = FreeSpaceSwath(case) if mode == 'swath' else EmpiricalCorridor(case)
        if mode == 'swath':
            x, y = runner.start[:2]
            distances = [np.hypot(max(b.x_min-x, 0., x-b.x_max), max(b.y_min-y, 0., y-b.y_max))
                         for b in runner.env.boxes]
            distances.extend(np.hypot(c.center_x-x, c.center_y-y)-c.radius for c in runner.env.cylinders)
            room = runner.env.room
            distances.extend((x-room.x_min, room.x_max-x, y-room.y_min, room.y_max-y))
            if min(distances) < observer.swath_config.bootstrap_radius_m:
                raise ValueError('Preflight disk unsupported by this fixture')
        # Qualified for this synthetic trial only. Relative heading has bounded
        # noise and zero imposed drift; airborne heading is a separate contract.
        sensor = SimulatedHeadingSensor(case, drift_rad_s=0., timestamp_error_s=0.)
        original_tick = runner.system.tick
        coordinator = runner.system.runtime.coordinator
        original_apply = coordinator.apply
        decision = {}
        attitude = (0., 0.)
        history, trace = deque(), []
        yaw = YawRamp()
        intervention = 0.
        clear_s = 0.
        direction = 0
        blocked = False
        needs_coverage = False
        resume_covered = False
        visited_local = set()
        if mode == 'swath' and hasattr(coordinator, '_clear_dwell_s'):
            # A full observation sweep can require more than the native short
            # avoidance turn. Translation remains damped throughout this phase.
            coordinator.phased_config = replace(coordinator.phased_config, turn_timeout_s=12., episode_timeout_s=16.)
            native_dwell = coordinator._clear_dwell_s
            coordinator._clear_dwell_s = lambda: 1000. if not resume_covered else native_dwell()

        def tick(frame, *, now_s, **kwargs):
            nonlocal decision, attitude
            attitude = frame.attitude.value
            heading = sensor.capture(runner.env.physics.quaternion_to_euler(runner.env.physics.quat)[2], timestamp_s=now_s)
            decision = observer.update(frame, heading, now_s=now_s)
            visited_local.add(tuple(np.floor(observer.position[:2] / .5).astype(int)))
            # Physical position is exclusively used to check the motion error model.
            truth = runner.env.physics.pos.copy()
            while history and now_s - history[0][0] > observer.config.history_s + 1e-8:
                history.popleft()
            errors = []
            for t, estimate, actual, distance in history:
                error = np.linalg.norm((observer.position - estimate)[:2] - (truth - actual)[:2])
                budget = (observer.error(now_s-t, observer.distance-distance) if mode == 'swath'
                          else .01 + .02 * (now_s - t) + .10 * (observer.distance - distance))
                errors.append(float(error - budget))
            history.append((now_s, observer.position.copy(), truth, observer.distance))
            result = original_tick(frame, now_s=now_s, **kwargs)
            if mode == 'swath' and abs(now_s - 20.) < 1e-8:
                np.savez_compressed(out / 'observer_20s.npz',
                    position=observer.position, distance=observer.distance,
                    bootstrap_time=observer.bootstrap_time_s,
                    heading=observer.last_report['heading_rad'],
                    captures=[f[0] for f in observer.sectors], origins=[f[1] for f in observer.sectors],
                    distances=[f[2] for f in observer.sectors], angles=[f[3] for f in observer.sectors],
                    ranges=[f[4] for f in observer.sectors], edges=[f[5] for f in observer.sectors],
                    heading_errors=[f[6] for f in observer.sectors])
            _, clearance = runner.contact(truth, truth, runner.env.room, runner.env.boxes,
                                          runner.env.cylinders, runner.collision_radius_m)
            trace.append(dict(time_s=now_s, **decision, intervention_s=intervention,
                              blocked=blocked or getattr(coordinator, 'navigation_blocked', False),
                              motion_budget_excess_m=max(errors, default=0.),
                              truth_clearance_m=clearance,
                              native_phase=getattr(coordinator, 'phase', None)))
            return result

        def apply(network_pwm, *, flow_xy, tof_m, dt, stop_requested=False):
            nonlocal intervention, clear_s, direction, blocked, needs_coverage, resume_covered
            invalid_swath = mode == 'swath' and decision.get('state') == 'INVALID'
            if mode == 'swath' and not invalid_swath:
                resume = observer.assess([.34, 0.], now_s=runner.time_s)
                resume_covered = resume['covered']
                decision['resume_swath'] = resume
                angle = observer.last_report['heading_rad']
                ahead = observer.position[:2] + np.array([.5, 1.])[:, None] * np.array([np.cos(angle), np.sin(angle)])
                new_targets = sum(tuple(cell) not in visited_local for cell in np.floor(ahead / .5).astype(int))
                decision['exploration_new_targets'] = new_targets
                # Preference only: local flow odometry never grants clearance.
                # Release the preference near the bounded survey deadline.
                if not new_targets and getattr(coordinator, 'turn_elapsed', 0.) < 9.:
                    resume_covered = False
                if not resume_covered and getattr(coordinator, 'phase', None) == 'TURN':
                    # Count continuous coverage, not time during which only the
                    # current frontal ToF happened to be clear.
                    coordinator.open_elapsed = 0.
            action = original_apply(network_pwm, flow_xy=flow_xy, tof_m=tof_m, dt=dt,
                                    stop_requested=stop_requested or blocked or invalid_swath)
            if invalid_swath:
                return action
            if stop_requested or blocked or getattr(coordinator, 'navigation_blocked', False):
                direction = 0
                intervention = clear_s = 0.
                yaw.synchronize(float(action[3]))
                return action
            state = decision.get('state', 'INVALID')
            if mode == 'native':
                return action
            if mode == 'speed_only' and state != 'SLOW':
                return action
            if mode == 'obstacle_only' and state == 'SLOW':
                state = 'PROCEED_LIMITED'
            if mode == 'swath':
                endpoint = motion_endpoint(observer.velocity, action, attitude)
                check = observer.assess(endpoint, now_s=runner.time_s)
                needs_coverage = not check['covered']
                decision.update(swath=check, observer_state=state, reason=check['reason'],
                                state='PROCEED_LIMITED' if check['covered'] else 'OUTSIDE_SWATH')
                if needs_coverage:
                    forward = observer.assess([max(0., endpoint[0]), 0.], now_s=runner.time_s)
                    damping = 1500. + np.clip(400.*np.array([coordinator.filtered_flow_y, -coordinator.filtered_flow]), -100., 100.)
                    action[1] = damping[0]
                    if not forward['covered']:
                        action[2] = damping[1]
                    phase = getattr(coordinator, 'phase', None)
                    if not forward['covered'] and phase in ('CRUISE', 'RECOVER'):
                        if phase == 'CRUISE':
                            coordinator.direction = coordinator._choose_direction(np.asarray(tof_m).reshape(8, 8))
                        coordinator.episode_elapsed = coordinator.turn_elapsed = 0.
                        coordinator.episode_active = True
                        coordinator.armed = False
                        coordinator._enter('BRAKE')
                        action[3] = 1500.
                        coordinator.synchronize_issued_yaw(1500.)
                    coordinator.previous_pr = action[1:3].copy()
                return action
            if state == 'RESTRICT' and not direction:
                image = np.asarray(tof_m).reshape(8, 8)
                direction = -1 if image[:, :4].mean() >= image[:, 4:].mean() else 1
            active = direction != 0
            if active:
                intervention += dt
                clear_s = clear_s + dt if state == 'PROCEED_LIMITED' else 0.
                if clear_s >= .3:
                    direction = 0
                    intervention = clear_s = 0.
                elif intervention >= 4.:
                    blocked = True
            if active or state in ('SLOW', 'UNKNOWN', 'INVALID'):
                flow = np.array([coordinator.filtered_flow_y, -coordinator.filtered_flow])
                action[1:3] = 1500. + np.clip(400. * flow, -100., 100.)
                quiet = np.hypot(coordinator.filtered_flow, coordinator.filtered_flow_y) <= .015
                turning = quiet and active and not blocked and state not in ('INVALID', 'UNKNOWN')
                target = 1500. + 140. * direction if turning else 1500.
                action[3] = yaw.step(target, target, dt)
                coordinator.previous_pr = action[1:3].copy()
            else:
                yaw.synchronize(float(action[3]))
            return action

        runner.system.tick = tick
        coordinator.apply = apply
        runner.resume()
        runner.event(dict(type='mode', mode='AUTO'))
        while not runner.terminal:
            runner.advance(5)
        rows = runner.rows
        metrics = summarize(rows, runner.time_s)
        times = np.array([r['time_s'] for r in rows])
        xy = np.array([r['position_m'][:2] for r in rows])
        samples = np.r_[np.arange(0, len(rows)-1, 5), len(rows)-1]
        metrics.update(progress_metrics(times, xy, samples, runner.time_s))
        metrics.update(case=case, duration_s=runner.time_s, pause_reason=runner.pause_reason,
                       completed=runner.pause_reason == 'duration_limit', path_m=metrics['distance_m'],
                       states=dict(Counter(r['state'] for r in trace)),
                       navigation_blocked=blocked or getattr(coordinator, 'navigation_blocked', False),
                       max_motion_budget_excess_m=max(r['motion_budget_excess_m'] for r in trace),
                       invalid_observation_s=.02 * sum(r['state'] == 'INVALID' for r in trace),
                       max_retained_frames=max(r.get('frames', 0) for r in trace),
                       wait_or_failed_s=.02 * sum(r['blocked'] or r['native_phase'] in ('WAIT_CLEAR', 'FAILED')
                                                for r in trace),
                       bridge_faults=[e for e in runner.events if e['kind']=='bridge_fault'])
        auto = [r['plant_source'] for r in rows if r['plant_input_pwm'] is not None]
        auto_start = next((i for i, s in enumerate(auto) if s == 'auto'), None)
        metrics['continuous_auto'] = auto_start is not None and all(s == 'auto' for s in auto[auto_start:])
        accepted = (runner.pause_reason == 'duration_limit' and metrics['min_clearance_m'] >= .1
                    and not metrics['navigation_blocked'] and metrics['max_motion_budget_excess_m'] <= 0.
                    and metrics['longest_no_new_cell']['duration_s'] <= 25.
                    and (duration < 180. or metrics['new_cells_last_60s'] >= 10)
                    and not metrics['repeat_motion_windows']
                    and metrics['reverse_distance_body_m'] <= .25
                    and metrics['wait_or_failed_s'] <= 5.
                    and metrics['invalid_observation_s'] == 0.
                    and metrics['continuous_auto'] and not metrics['bridge_faults']
                    and metrics['max_height_error_m'] <= .1)
        metrics['checks_passed'] = bool(accepted)
        metrics['accepted'] = bool(accepted and duration >= 180.)
        protocol = dict(case=case, mode=mode, empirical_policy=True, parameters=asdict(observer.config),
                        swath_parameters=asdict(observer.swath_config) if mode == 'swath' else None,
                        swath_control=(dict(
                            phased_config=asdict(coordinator.phased_config),
                            resume_forward_m=.34, continuous_coverage_dwell=True,
                            exploration_cell_m=.5, exploration_targets_m=[.5, 1.],
                            exploration_preference_release_turn_s=9.,
                            coverage_scope='incremental expansion; known-hit veto covers full tube',
                            motion_reserve='1.2 * velocity + outward tilt/command reserve',
                            native_stop_priority=True) if mode == 'swath' else None),
                        heading_noise_deg=.25, heading_drift_deg_s=0., manifest=runner.manifest(),
                        source_sha256=source_hashes, controller_profile=profile,
                        speed_cap_applied=mode in ('legacy', 'speed_only'))
        controls = [r for r in rows if r['plant_input_pwm'] is not None
                    and abs(r['time_s']/.02-round(r['time_s']/.02)) < 1e-7]
        nan4 = lambda r, key: r['trace'][key] or [np.nan]*4
        network = [nan4(r, 'network_pwm') for r in controls]
        coordinated = [nan4(r, 'coordinated_pwm') for r in controls]
        control_auto = [r['plant_source'] == 'auto' for r in controls]
        control_stop = [r['stop_requested'] for r in controls]
        np.savez_compressed(out / 'trace.npz', time=times, xy=xy,
                            clearance=[r['clearance_m'] for r in rows],
                            control_time=[r['time_s'] for r in controls], auto=control_auto,
                            stop_requested=control_stop, network=network, coordinated=coordinated,
                            accepted=[nan4(r, 'accepted_pwm') for r in controls])
        try:  # no stage log in this stand: generated_share stays null
            metrics.update(share_metrics(network, coordinated, control_auto, control_stop))
        except ValueError as error:
            metrics.update(network_share=None, generated_share=None, share_error=str(error))
        for name, value in (('protocol',protocol),('report',metrics),('decisions',trace)):
            (out / (name + '.json')).write_text(json.dumps(value, indent=2), encoding='utf-8')
        print(json.dumps(metrics, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', choices=('clear', 'box', 'phase', 'cylinder'))
    parser.add_argument('--mode', choices=('legacy', 'swath', 'native', 'speed_only', 'obstacle_only'), default='legacy')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--duration', type=float, default=180.)
    args = parser.parse_args()
    run(args.case, args.out, args.duration, args.mode)
