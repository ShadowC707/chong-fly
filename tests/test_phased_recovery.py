import numpy as np
from simulation.phased_recovery import RecoveryPhasedCoordinator


def step(c, stop=False):
    return c.apply([1213,1500,1530,1500],flow_xy=[0,0],tof_m=np.full(64,.7),dt=.02,stop_requested=stop)


def wait(c):
    c.armed=False;c.episode_active=False;c._enter('WAIT_CLEAR')
    for _ in range(50):step(c)


def test_wait_clear_retries_with_a_continuous_bounded_attempt_budget():
    c=RecoveryPhasedCoordinator(stop_hover=True)
    for attempt in range(1,4):
        wait(c)
        assert c.retry_count==attempt and c.phase=='BRAKE' and c.episode_active
    wait(c)
    assert c.navigation_blocked
    for _ in range(200):assert step(c)[3]==1500
    assert c.retry_count==3
    c.reset();assert not c.navigation_blocked and c.retry_count==0


def test_requested_stop_never_starts_recovery():
    c=RecoveryPhasedCoordinator(stop_hover=True)
    c._enter('WAIT_CLEAR');c.armed=False
    for _ in range(200):step(c,stop=True)
    assert c.retry_count==0 and c.phase=='STOP'


def test_normal_cruise_is_identical_to_smooth_controller():
    from simulation.phased_yaw import SmoothPhasedCoordinator
    a=SmoothPhasedCoordinator(stop_hover=True);b=RecoveryPhasedCoordinator(stop_hover=True)
    for _ in range(100):
        args=dict(flow_xy=[.05,0],tof_m=np.full(64,2.),dt=.02)
        np.testing.assert_array_equal(a.apply([1213,1500,1510,1500],**args),b.apply([1213,1500,1510,1500],**args))


def test_wait_retry_requires_quiet_flow_and_does_not_retry_failed_timeout():
    c=RecoveryPhasedCoordinator(stop_hover=True)
    c.armed=False;c._enter('WAIT_CLEAR')
    for _ in range(100):
        c.apply([1213,1500,1530,1500],flow_xy=[.1,0],tof_m=np.full(64,.7),dt=.02)
    assert c.retry_count==0
    c._enter('FAILED')
    for _ in range(200):step(c)
    assert c.retry_count==0 and c.phase=='FAILED'


def test_retry_budget_reset_requires_sustained_clear_motion_not_completed_count():
    c=RecoveryPhasedCoordinator(stop_hover=True)
    c.retry_count=3;c.completed=100
    for _ in range(250):
        c.apply([1213,1500,1530,1500],flow_xy=[0,0],tof_m=np.full(64,2.),dt=.02)
    assert c.retry_count==3
    for _ in range(249):
        c.apply([1213,1500,1530,1500],flow_xy=[.05,0],tof_m=np.full(64,2.),dt=.02)
    assert c.retry_count==3
    # The low-pass filter needs additional ticks to cross quiet_flow.
    for _ in range(3):
        c.apply([1213,1500,1530,1500],flow_xy=[.05,0],tof_m=np.full(64,2.),dt=.02)
    assert c.retry_count==0


def test_fast_repeat_and_retry_preserve_completed_direction_and_extend_dwell():
    from simulation.phased_recovery import PocketEscapeCoordinator

    image = np.tile([2., 2., 2., 2., .7, .7, .7, .7], (8, 1))
    for retry in (False, True):
        c = PocketEscapeCoordinator(stop_hover=True)
        c.completed_direction = 1
        c.since_completed_s = 2.2
        if retry:
            c.armed = False
            c._enter('WAIT_CLEAR')
        for _ in range(50 if retry else 1):
            c.apply([1213,1500,1530,1500], flow_xy=[0.,0.], tof_m=image, dt=.02)
        assert c.direction == 1
        assert np.isclose(c._clear_dwell_s(), .35)
        assert 'consistent_escape' in c.last_decision['reasons']

    c.reset()
    c.completed_direction = 1
    c.since_completed_s = 3.48
    c.apply([1213,1500,1530,1500], flow_xy=[0.,0.], tof_m=image, dt=.02)
    assert c.direction == -1
    assert c._clear_dwell_s() == .20


def test_edge_restraint_requires_loss_then_expires():
    from simulation.side_clearance import SideClearanceCoordinator

    c = SideClearanceCoordinator(stop_hover=True)
    edge = np.full((8, 8), 2.)
    edge[:, -1] = .92
    opposite = np.fliplr(edge)
    c.filtered_flow = .03
    c.last_decision = {'reasons': []}
    assert c._cruise_yaw(1440., [1213, 1500, 1530, 1440], edge, .02) == 1440.
    for _ in range(24):
        assert c._cruise_yaw(1525., [1213, 1500, 1530, 1525], opposite, .02) == 1525.
    assert c._cruise_yaw(1519., [1213, 1500, 1530, 1519], opposite, .02) == 1519.
    assert c._cruise_yaw(1525., [1213, 1500, 1530, 1525], opposite, .02) == 1500.
    assert c._cruise_yaw(1525., [1213, 1500, 1530, 1525], edge, .02) == 1525.
    for _ in range(50):
        result = c._cruise_yaw(1525., [1213, 1500, 1530, 1525], opposite, .02)
    assert result == 1525. and c.edge_side == 0
