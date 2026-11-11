import inspect

import numpy as np
import pytest

from evaluate.sitl_neural import trial, validate_duration
from evaluate.sitl_long_flight import long_scene, trace_metrics


def test_duration_defaults_preserve_previous_protocol():
    args = inspect.signature(trial).parameters
    assert args['horizon_s'].default == 20.
    assert args['stop_at_s'].default == 17.
    validate_duration(45., 40.)


@pytest.mark.parametrize('horizon,stop', [(0, 0), (45, 45), (45, -1),
    (float('nan'), 40), (45, float('inf')), (45.001, 40), (45, 40.001), (45, 44.5)])
def test_duration_rejects_invalid_or_incomplete_stop_window(horizon, stop):
    with pytest.raises(ValueError):
        validate_duration(horizon, stop)


def test_long_scene_has_clear_start_and_mixed_geometry():
    from evaluate.sitl_neural import SceneSensors
    from demo.contacts import contact_and_clearance
    spec = long_scene()
    sensors = SceneSensors('long_arena', spec=spec)
    hit, gap = contact_and_clearance(spec['start'], spec['start'], sensors.room,
        sensors.boxes, sensors.cylinders, spec['radius_m'])
    assert hit is None and gap > .1
    assert len(spec['boxes']) == len(spec['cylinders']) == 2


def test_trace_metrics_use_physics_resolution_and_do_not_hide_short_flight():
    t = np.arange(5)*.00025
    s = np.zeros((5, 12)); s[:, 2] = 1.; s[:, 3] = [-1, 1, 1, 0, 0]
    gaps = np.array([.2, .04, .04, .08])
    ack = np.column_stack([np.arange(4), np.arange(4)*250, np.full(4, 2)])
    m = trace_metrics(t, s, gaps, ack)
    assert m['seconds_below_5cm'] == pytest.approx(.0005)
    assert m['seconds_below_10cm'] == pytest.approx(.00075)
    assert m['min_clearance_time_s'] == .0005
    assert m['episodes_below_10cm'] == 1
    assert m['clock_passed']
    assert m['reverse_distance_m'] == pytest.approx(.00025)
    ack[2, 1] += 1
    assert not trace_metrics(t, s, gaps, ack)['clock_passed']
