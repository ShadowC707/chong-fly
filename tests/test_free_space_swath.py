import numpy as np
import pytest

from edge.free_space_swath import FreeSpaceSwath, SwathConfig, motion_endpoint
from edge.sensors import Measurement, SensorSnapshot
from edge.survey_heading import HeadingSample


def feed(observer, i, *, flow=(.055, 0.), yaw=0., tof=None):
    t = i * .02
    sample = SensorSnapshot(i, t, Measurement(i, t, flow),
                            tof or Measurement(i, t, (1.,)*64),
                            Measurement(i, t, (1., 0., 0.)), Measurement(i, t, (0., 0.)))
    return observer.update(sample, HeadingSample('test', i, t, yaw), now_s=t)


@pytest.mark.parametrize('flow_x', [.055, .1])
def test_straight_observed_route_stays_open_and_side_exit_is_blocked(flow_x):
    observer = FreeSpaceSwath('test')
    tof = None
    for i in range(501):
        t = i * .02
        epoch = int(t*15 + 1e-8)
        if tof is None or tof.sample_id != epoch:
            tof = Measurement(epoch, t, (1.,)*64)
        feed(observer, i, tof=tof, flow=(flow_x, 0.))
        assert observer.assess([1.2 * 4. * flow_x + .06, 0.], now_s=t)['covered']
    # A small side step is covered by the widening cone captured earlier.
    # This larger step takes the airframe beyond that observed strip.
    assert not observer.assess([0., 1.5], now_s=10.)['covered']
    assert not observer.assess([0., -1.5], now_s=10.)['covered']
    assert len(observer.sectors) == 121


def test_new_heading_does_not_repaint_held_frame():
    observer = FreeSpaceSwath('test')
    tof = Measurement(0, 0., (1.,)*64)
    feed(observer, 0, tof=tof)
    origin = observer.sectors[0][1].copy()
    angle = observer.sectors[0][3]
    feed(observer, 1, tof=tof, yaw=.03)
    assert len(observer.sectors) == 1
    np.testing.assert_array_equal(observer.sectors[0][1], origin)
    assert observer.sectors[0][3] == angle


def test_bootstrap_is_explicit_and_close_returns_do_not_paint_free_space():
    observer = FreeSpaceSwath('test', SwathConfig(bootstrap_radius_m=0.))
    feed(observer, 0, tof=Measurement(0, 0., (.05,)*64))
    assert not observer.assess([.2, 0.], now_s=0.)['covered']


def test_invalid_capture_clears_free_sectors():
    observer = FreeSpaceSwath('test')
    feed(observer, 0)
    assert feed(observer, 1, tof=Measurement(0, 0., (.2,)*64))['state'] == 'INVALID'
    assert not observer.sectors
    assert not observer.assess([.2, 0.], now_s=.02)['covered']


def test_stale_query_cannot_reuse_start_clearance():
    observer = FreeSpaceSwath('test')
    feed(observer, 0)
    assert not observer.assess([.2, 0.], now_s=.2)['covered']


def test_acceleration_is_screened_before_flow_reports_drift():
    right = motion_endpoint([0., 0.], [1500., 1600., 1500., 1500.], [0., 0.])
    left = motion_endpoint([0., 0.], [1500., 1400., 1500., 1500.], [0., 0.])
    assert right[1] < 0. < left[1]
    braking = motion_endpoint([.2, 0.], [1500., 1500., 1400., 1500.], [0., 0.])
    assert braking[0] >= .24  # Braking cannot erase the existing momentum.


def test_known_near_obstacle_vetoes_even_zero_expansion():
    observer = FreeSpaceSwath('test')
    feed(observer, 0, tof=Measurement(0, 0., (.06,) * 64))
    check = observer.assess([0., 0.], now_s=0.)
    assert not check['covered']
    assert check['reason'] == 'remembered_obstacle'


def test_missing_heading_fails_closed_without_exception():
    observer = FreeSpaceSwath('test')
    frame = SensorSnapshot(0, 0., Measurement(0, 0., (0., 0.)),
        Measurement(0, 0., (1.,) * 64), Measurement(0, 0., (1., 0., 0.)),
        Measurement(0, 0., (0., 0.)))
    assert observer.update(frame, None, now_s=0.)['state'] == 'INVALID'
    assert not observer.assess([.2, 0.], now_s=0.)['covered']


def test_zero_obstacle_hits_do_not_authorize_unseen_side_sector():
    observer = FreeSpaceSwath('test', SwathConfig(bootstrap_radius_m=0.))
    feed(observer, 0, flow=(0., 0.))
    assert len(observer.hit_points) == 0
    check = observer.assess([0., .2], now_s=0.)
    assert not check['covered']
    assert check['reason'] == 'outside_observed_swath'
