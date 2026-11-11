import math
import numpy as np
import pytest

from simulation.phased_coordinator import PhasedCoordinator, PhasedConfig
from simulation.phased_yaw import SmoothPhasedCoordinator, YawRamp


def tick(c, yaw=1500., distance=.7, flow=(0., 0.), stop=False):
    return c.apply([1213., 1500., 1530., yaw], flow_xy=flow,
                   tof_m=np.full(64, distance), dt=.02, stop_requested=stop)


def test_constant_front_and_interrupted_front_are_anchored_at_last_output():
    r = YawRamp()
    assert r.step(1640., 'turn', .02) == 1500.
    for _ in range(6): value = r.step(1640., 'turn', .02)
    assert value == pytest.approx(1570.)
    assert r.step(1500., 'brake', .02) == value
    assert r.anchor == value
    for _ in range(12): value = r.step(1500., 'brake', .02)
    assert value == pytest.approx(1500.)


def test_moving_target_remains_bounded_after_ramp_finishes():
    r = YawRamp()
    for _ in range(30): r.step(1640., 'recover', .02)
    previous = r.output
    for i in range(60):
        value = r.step(1360. if i % 2 else 1640., 'recover', .02)
        assert abs(value-previous) <= r.rate*.02+1e-9
        previous = value


def test_cruise_history_and_brake_to_turn_interruption():
    c = SmoothPhasedCoordinator()
    for _ in range(30): out = tick(c, yaw=1600., distance=2.)
    assert c.previous_yaw == out[3] == 1600.
    first = tick(c, yaw=1600.)
    assert first[3] == 1600.
    values = [first[3]]
    for _ in range(20): values.append(tick(c, yaw=1600.)[3])
    assert c.phase == 'TURN'
    assert max(abs(np.diff(values))) <= c.yaw.rate*.02+1e-9


def test_recover_first_decays_to_neutral_before_following_live_base():
    c = SmoothPhasedCoordinator()
    for _ in range(30): tick(c)
    while c.phase != 'RECOVER': tick(c, yaw=1400., distance=2.)
    assert c.previous_yaw == 1640.
    values = []
    for _ in range(12): values.append(tick(c, yaw=1400., distance=2.)[3])
    assert values[-1] == pytest.approx(1500.)
    assert min(values) >= 1500.-1e-9
    for _ in range(45): tick(c, yaw=1400., distance=2.)
    assert c.phase == 'CRUISE' and c.previous_yaw == 1400.


def test_failed_decay_is_reserved_inside_original_turn_budget():
    c = SmoothPhasedCoordinator(phased_config=PhasedConfig(turn_timeout_s=1.))
    turn_start = failed_at = None
    values = []
    for i in range(150):
        out = tick(c)
        if c.phase == 'TURN' and turn_start is None: turn_start = i
        if c.phase == 'FAILED' and failed_at is None: failed_at = i
        values.append(out[3])
        if failed_at is not None and i >= turn_start+50:
            assert out[3] == pytest.approx(1500.)
    assert failed_at < turn_start+50
    assert c.failures == 1
    assert max(abs(np.diff(values))) <= c.yaw.rate*.02+1e-9


def test_episode_budget_includes_failed_tail_after_repeated_threats():
    c = SmoothPhasedCoordinator(phased_config=PhasedConfig(episode_timeout_s=1.2))
    for i in range(80):
        out = tick(c, distance=2. if i % 25 >= 10 else .7)
        if i >= 60: assert out[3] == pytest.approx(1500.)
    assert c.failures == 1


def test_stop_and_idle_critical_bypass_smoothing_and_keep_evasion():
    for distance, stop in ((1.2, True), (.3, True), (.3, False)):
        a = PhasedCoordinator(stop_hover=True)
        b = SmoothPhasedCoordinator(stop_hover=True)
        np.testing.assert_array_equal(tick(a, yaw=1640., distance=distance, stop=stop),
                                      tick(b, yaw=1640., distance=distance, stop=stop))
        assert b.last_decision['yaw_smoothing']['bypass']


def test_invalid_reset_and_disabled_behavior():
    a = PhasedCoordinator(enabled=False)
    b = SmoothPhasedCoordinator(enabled=False)
    for yaw in (1600., 1300., 1500.):
        np.testing.assert_array_equal(tick(a, yaw=yaw), tick(b, yaw=yaw))
        assert b.previous_yaw == yaw
    with pytest.raises(ValueError): tick(b, flow=(math.nan, 0.))
    assert b.previous_yaw == 1500.


def test_handover_uses_issued_yaw_without_fabricating_acceptance():
    from edge.system import RunConfig
    from edge.pinned_system import build_pinned_system
    from demo.yaw_experiment import install
    from tests.test_flight_bridge import frame
    system = install(build_pinned_system(RunConfig.from_json('configs/demo_candidate_a.json')))
    manual = [1213., 1500., 1500., 1515.]
    engaged = None
    for i in range(80):
        result = system.tick(frame(i), now_s=i*.02, request='AUTO', manual_pwm=manual)
        if result['reason'] == 'engaged':
            engaged = result
            break
    assert engaged is not None
    assert result['trace'].issued_pwm[3] == 1515.
    assert result['trace'].accepted_pwm is None
    c = system.runtime.coordinator
    assert c.previous_yaw == 1515.
    assert result['yaw_handover']['source'] == 'issued'
    result = system.tick(frame(i+1), now_s=(i+1)*.02, request='AUTO', manual_pwm=manual)
    assert abs(result['trace'].issued_pwm[3]-1515.) <= c.yaw.rate*.02+1e-9
    result = system.tick(None, now_s=(i+2)*.02, request='MANUAL', manual_pwm=manual)
    assert result['lease'] is None
    # A fresh shadow interval and re-entry must not reuse the old ramp anchor.
    for j in range(i+3, i+70):
        result = system.tick(frame(j), now_s=j*.02, request='AUTO', manual_pwm=manual)
        if result['reason'] == 'engaged': break
    assert result['reason'] == 'engaged'
    assert c.previous_yaw == 1515.
    result = system.tick(None, now_s=(j+1)*.02, request='AUTO', manual_pwm=manual)
    assert result['state'] == 'FAULT' and result['lease'] is None


@pytest.mark.parametrize('sign', [-1, 1])
def test_neutral_recovery_within_point_eight_seconds_both_directions(sign):
    c = SmoothPhasedCoordinator()
    c.next_tie = sign
    for _ in range(30): tick(c)
    assert c.previous_yaw == 1500+sign*140
    while c.phase != 'RECOVER': tick(c, distance=2.)
    values = [tick(c, distance=2.)[3] for _ in range(40)]
    assert all(abs(v-1500.)<=20 for v in values[12:])


def test_watchdog_bypasses_active_yaw_front():
    from edge.session_receiver import CommandPacket, SessionReceiver
    from edge.flight_bridge import CommandLease
    c = SmoothPhasedCoordinator()
    for _ in range(20): tick(c)
    assert c.previous_yaw > 1500
    r = SessionReceiver()
    r.reset(session_id='test', now_s=0.)
    r.announce_sensor(0, now_s=0.)
    r.set_mode('AUTO', now_s=0.)
    r.receive(CommandPacket('test', 0, CommandLease(0, (1213.,1500.,1500.,c.previous_yaw), .04)), now_s=0.)
    out = r.output(now_s=.041, fallback_pwm=(1213.,1500.,1500.,1500.))
    assert out['tripped'] and out['accepted_pwm'] is None
    assert out['applied_pwm'][3] == 1500.


def test_recovery_metric_does_not_pass_censored_or_last_sample_settling():
    from evaluate.sitl_phased_yaw import recovery_metrics
    t = np.arange(100)*.02
    signal = np.zeros(100)
    assert recovery_metrics(t, signal, 0., .5, 20.)['censored']
    signal[:-2] = 140.
    assert recovery_metrics(t, signal, 0., 2., 20.)['settle_s'] is None


def test_full_smoothing_decision_is_json_serializable():
    import json
    c = SmoothPhasedCoordinator(stop_hover=True)
    for i in range(100):
        tick(c, distance=.7 if i<30 else 2., stop=i>80)
        json.dumps(c.last_decision)
