import numpy as np

from simulation.failed_exit import FailedExitCoordinator
from simulation.side_clearance import SideClearanceCoordinator


def step(c, tof=.7, flow=(0, 0), stop=False):
    return c.apply([1213, 1500, 1530, 1500], flow_xy=list(flow), tof_m=np.full(64, tof), dt=.02, stop_requested=stop)


def failed(c, direction=1):
    c.direction = direction
    c.armed = False
    c.episode_active = False
    c._enter('FAILED')


def run_until_retry(c, limit=200):
    for _ in range(limit):
        step(c)
        if c.phase != 'FAILED' or c.navigation_blocked:
            return
    raise AssertionError('no retry')


def test_stationary_failed_retries_continue_then_alternate_then_hold_in_failed_with_operator_event():
    c = FailedExitCoordinator(stop_hover=True)
    failed(c, direction=1)
    directions = []
    for attempt in range(1, 4):
        run_until_retry(c)
        assert c.phase == 'BRAKE' and c.episode_active and c.failed_retry_count == attempt
        directions.append(c.direction)
        failed(c, direction=c.direction)
    assert directions == [1, -1, 1]
    for _ in range(60):
        step(c)
    assert c.failed_exhausted and c.operator_event == 'failed_retry_budget_exhausted'
    assert not c.navigation_blocked
    for _ in range(300):  # hold: FAILED flow damping, no native STOP, no further retries
        out = step(c, flow=(.01, -.01))
        assert c.phase == 'FAILED' and c.failed_retry_count == 3
    np.testing.assert_allclose(out[1:4], [1500+400*c.filtered_flow_y, 1500-400*c.filtered_flow, 1500])
    assert c.last_decision['operator_event'] == 'failed_retry_budget_exhausted'
    for _ in range(40):  # a clear front still ends the hold, as in phased_side_guard
        step(c, tof=2.)
    assert c.phase in ('RECOVER', 'CRUISE')
    c.reset()
    assert not c.failed_exhausted and c.failed_retry_count == 0 and c.operator_event is None


def test_budget_restored_by_sustained_clear_motion_clears_the_operator_event():
    c = FailedExitCoordinator(stop_hover=True)
    c.failed_retry_count, c.failed_exhausted, c.operator_event = 3, True, 'failed_retry_budget_exhausted'
    for _ in range(260):
        step(c, tof=2., flow=(.05, 0))
    assert c.failed_retry_count == 0 and not c.failed_exhausted and c.operator_event is None


def test_retry_waits_one_second_of_quiet_flow():
    c = FailedExitCoordinator(stop_hover=True)
    failed(c)
    for _ in range(100):
        step(c, flow=(.1, 0))
    assert c.phase == 'FAILED' and c.failed_retry_count == 0
    for _ in range(49):
        step(c)
    assert c.phase == 'FAILED'
    for _ in range(10):
        step(c)
    assert c.phase == 'BRAKE' and c.failed_retry_count == 1


def test_requested_stop_never_retries_failed():
    c = FailedExitCoordinator(stop_hover=True)
    failed(c)
    for _ in range(200):
        step(c, stop=True)
    assert c.failed_retry_count == 0 and c.phase == 'STOP'


def test_clear_front_still_leaves_failed_without_a_retry():
    c = FailedExitCoordinator(stop_hover=True)
    failed(c)
    for _ in range(40):
        step(c, tof=2.)
    assert c.phase in ('RECOVER', 'CRUISE') and c.failed_retry_count == 0


def test_identical_to_phased_side_guard_outside_failed():
    a, b = SideClearanceCoordinator(stop_hover=True), FailedExitCoordinator(stop_hover=True)
    rng = np.random.default_rng(3)
    for _ in range(300):
        args = dict(flow_xy=list(rng.normal(0, .03, 2)), tof_m=rng.uniform(1.1, 3., 64), dt=.02)
        np.testing.assert_array_equal(a.apply([1213, 1500, 1510, 1490], **args), b.apply([1213, 1500, 1510, 1490], **args))
        assert a.phase == b.phase != 'FAILED'


def test_live_view_shows_operator_hold_after_exhausted_budget():
    from demo.live_view import controller_of
    frame = dict(plant_source='auto', stop_requested=False,
                 trace=dict(network_pwm=[1213, 1500, 1530, 1520], coordinated_pwm=[1213, 1500, 1500, 1500]))
    view = controller_of(frame, dict(failed_exhausted=True, phased_phase='FAILED',
                                     operator_event='failed_retry_budget_exhausted'))
    assert view['kind'] == 'phase' and view['reason'] == 'failed_retry_budget_exhausted'
    assert 'оператор' in view['label']
