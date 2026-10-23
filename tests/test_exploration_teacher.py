import numpy as np
import pytest

from configs.flight_config import CONTROL_DT


def observation(distance=None, flow=0.):
    x = np.ones(74, dtype=np.float32)
    x[:2] = [flow, 0.]
    if distance is not None:
        x[2:66].reshape(8, 8)[:, :4] = distance/3.5
    return x


def teacher(**kwargs):
    from generator.exploration_teacher import ExplorationTeacher, ScanConfig
    return ExplorationTeacher(ScanConfig(interval_s=.6, duration_s=.3, **kwargs))


def advance(policy, x, steps):
    return [policy.step(x, dt=CONTROL_DT) for _ in range(steps)]


def test_periodic_scan_requires_braking_then_resumes_with_ramp():
    p = teacher()
    modes, actions = [], []
    for _ in range(90):
        actions.append(p.step(observation()))
        modes.append(p.mode)
    assert modes[:20] == ['move']*20
    assert modes.index('brake') < modes.index('scan') < modes.index('recover')
    scan = np.array([a for a, m in zip(actions, modes) if m == 'scan'])
    assert np.all(scan[:, 2] == 1500)
    assert np.all(scan[:, 3] == 1580)
    ramp = [a[2] for a, m in zip(actions, modes) if m == 'recover']
    assert 1500 < ramp[0] < ramp[-1] <= 1600
    assert p.completed_scans >= 1
    scan_event = next(e for e in p.events if e['to'] == 'scan')
    assert scan_event['time_s'] == pytest.approx(modes.index('scan')*CONTROL_DT)


def test_due_timer_cannot_start_scan_while_flow_is_high():
    p = teacher()
    actions = advance(p, observation(flow=.2), 80)
    assert p.mode == 'brake'
    assert all(a[3] == 1500 for a in actions)
    assert actions[-1][2] < 1500


def test_new_threat_immediately_preempts_scan():
    p = teacher()
    for _ in range(60):
        p.step(observation())
        if p.mode == 'scan':
            break
    assert p.mode == 'scan'
    action = p.step(observation(.35, flow=.2))
    assert p.mode == 'avoid'
    assert action[3] == 1500  # existing v4 suppresses yaw during fast translation
    assert action[2] < 1500
    assert p.interrupted_scans == 1


def test_avoidance_requests_scan_without_waiting_for_period():
    from generator.exploration_teacher import ExplorationTeacher, ScanConfig
    p = ExplorationTeacher(ScanConfig(interval_s=20.))
    p.step(observation(.35))
    assert p.mode == 'avoid'
    for _ in range(100):
        p.step(observation())
        if p.mode == 'scan':
            break
    assert p.mode == 'scan'
    assert p.scan_reason == 'after_avoidance'
    assert p.elapsed_s < 3


def test_obstacle_has_priority_over_periodic_scan_for_entire_episode():
    p = teacher()
    advance(p, observation(.35), 120)
    assert p.mode == 'avoid'
    assert p.completed_scans == 0
    assert not any(e['to'] == 'scan' for e in p.events)


def test_brake_timeout_is_explicit_instead_of_forced_scan():
    from generator.exploration_teacher import ScanBlocked
    p = teacher(brake_timeout_s=.5)
    with pytest.raises(ScanBlocked, match='settle'):
        advance(p, observation(flow=.2), 100)
    assert p.mode == 'blocked'
    with pytest.raises(ScanBlocked):
        p.step(observation())


def test_drift_during_scan_returns_to_braking_without_yaw():
    p = teacher()
    for _ in range(60):
        p.step(observation())
        if p.mode == 'scan':
            break
    action = p.step(observation(flow=.1))
    assert p.mode == 'brake'
    assert action[3] == 1500 and action[2] < 1500


def test_reset_replays_same_labels_and_events():
    p = teacher()
    first = advance(p, observation(), 90)
    events = list(p.events)
    p.reset()
    np.testing.assert_array_equal(advance(p, observation(), 90), first)
    assert p.events == events


@pytest.mark.parametrize('kwargs', [{'interval_s':0}, {'duration_s':float('nan')},
                                   {'yaw_offset_pwm':500}, {'abort_flow':.01}])
def test_invalid_configuration_fails(kwargs):
    from generator.exploration_teacher import ScanConfig
    with pytest.raises(ValueError):
        ScanConfig(**kwargs)


def test_roughness_is_diagnostic_not_an_independent_scan_trigger():
    from generator.exploration_teacher import ExplorationTeacher
    p = ExplorationTeacher()
    x = observation()
    x[2:66:2] = 1.1/3.5
    p.step(x)
    assert p.mode == 'move'
    assert p.diagnostics['tof_spread_m'] > 1


def test_invalid_observation_does_not_advance_clock():
    p = teacher()
    with pytest.raises(ValueError):
        p.step(np.full(74, np.nan))
    assert p.elapsed_s == 0


def test_old_teacher_factory_default_is_not_replaced():
    from generator.reflex_contract import TEACHER_VERSION
    from generator.generate_reflex_dataset import make_teacher
    p = make_teacher(TEACHER_VERSION, noise_std_pwm=0)
    assert advance(p, observation(), 500)[-1][2] == 1600


def test_teacher_uses_shared_rollout_interface_and_altitude_controller():
    from optimizer.rollout import simulate_policy_rollout
    from simulation.altitude_control import AltitudeHold
    from simulation.benchmark_scenes import NavigationScene
    p = teacher()
    env = NavigationScene('clear')
    try:
        _, result = simulate_policy_rollout(p, env=env, eval_steps=3,
                                            altitude_hold=AltitudeHold())
        assert result['horizon_completed']
        assert not result['fatal_failure']
        assert p.elapsed_s == pytest.approx(3*CONTROL_DT)
    finally:
        env.close()


def test_repeated_scans_are_measured_separately_with_endpoints_and_yaw_unwrap():
    from scripts.probe_exploration_teacher import scan_segments
    def row(t, mode, x, yaw):
        return dict(time_s=t, mode=mode, x_m=x, y_m=0., z_m=1.,
                    yaw_rad=np.deg2rad(yaw), horizontal_speed_mps=.02)
    rows = [row(0, 'scan', 0, 179), row(1, 'scan', .01, -179),
            row(2, 'recover', .02, -177), row(3, 'move', 2, -177),
            row(4, 'scan', 4, 0), row(5, 'end', 4.01, -10)]
    segments = scan_segments(rows)
    assert len(segments) == 2
    assert segments[0]['duration_s'] == 2
    assert segments[0]['max_drift_m'] == pytest.approx(.02)
    assert segments[0]['yaw_change_right_deg'] == pytest.approx(-4)
    assert segments[1]['max_drift_m'] == pytest.approx(.01)
    assert segments[1]['yaw_change_right_deg'] == pytest.approx(10)
