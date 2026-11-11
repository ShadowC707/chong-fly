from dataclasses import replace

import pytest

from edge.empirical_corridor import CorridorConfig, EmpiricalCorridor
from edge.sensors import Measurement, SensorSnapshot
from edge.survey_heading import HeadingSample


def sample(i, *, ranges=None, tof=None, flow=(0., 0.)):
    t = i * .02
    return SensorSnapshot(i, t, Measurement(i, t, flow),
                          tof or Measurement(i, t, ranges or (1.,) * 64),
                          Measurement(i, t, (1., 0., 0.)), Measurement(i, t, (0., 0.)))


def update(observer, frame, angle=0.):
    heading = HeadingSample(observer.reference_id, frame.sequence, frame.timestamp_s, angle)
    return observer.update(frame, heading, now_s=frame.timestamp_s)


def test_free_cruise_has_explicit_permission_and_obstacle_restricts():
    observer = EmpiricalCorridor('test')
    assert update(observer, sample(0))['state'] == 'PROCEED_LIMITED'
    assert update(observer, sample(1, ranges=(.07,) * 64))['state'] == 'RESTRICT'


def test_held_frame_retains_original_position_and_heading():
    observer = EmpiricalCorridor('test')
    frame = sample(0, ranges=(.15,) * 64)
    update(observer, frame)
    original_hits = observer.frames[0][-1].copy()
    result = update(observer, sample(1, tof=frame.tof, flow=(.05, 0.)), angle=.03)
    assert result['tof_state'] == 'held'
    assert result['frames'] == 1
    assert (observer.frames[0][-1] == original_hits).all()
    assert observer.position[0] > 0.


def test_scan_keeps_four_seconds_and_expires_by_capture_time():
    observer = EmpiricalCorridor('test')
    tof = None
    epoch = -1
    for i in range(251):
        t = i * .02
        current = int(t * 15 + 1e-8)
        if current != epoch:
            tof = Measurement(current, t, (1.,) * 64)
            epoch = current
        result = update(observer, sample(i, tof=tof))
    assert result['frames'] == 61
    assert observer.frames[0][0] == pytest.approx(1.)


def test_corrupt_held_frame_latches_fault_until_reset():
    observer = EmpiricalCorridor('test')
    frame = sample(0)
    update(observer, frame)
    corrupt = replace(frame.tof, value=(.2,) * 64)
    assert update(observer, sample(1, tof=corrupt))['state'] == 'INVALID'
    assert update(observer, sample(2))['state'] == 'INVALID'
    with pytest.raises(ValueError):
        observer.reset('test')
    observer.reset('new-session')
    assert update(observer, sample(0))['state'] == 'PROCEED_LIMITED'


def test_gap_and_saturated_flow_invalidate_motion():
    observer = EmpiricalCorridor('test')
    update(observer, sample(0))
    assert update(observer, sample(3))['state'] == 'INVALID'
    observer.reset('new-session')
    assert update(observer, sample(0, flow=(1., 0.)))['state'] == 'INVALID'
    with pytest.raises(ValueError):
        CorridorConfig(history_s=-1.)
