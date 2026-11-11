import numpy as np
import pytest

from evaluate.sitl_resolution import compare_traces


def trace(dt):
    t = np.arange(round(4/dt)+1)*dt
    s = np.zeros((len(t), 12))
    s[:, 0] = t
    s[:, 2] = 1
    return {'time_s': t, 'state': s}


def test_resolution_comparison_aligns_physical_time_not_row_number():
    result = compare_traces(trace(.004), trace(.001))
    assert result['passed']
    assert result['max_position_difference_m'] == 0
    shifted = trace(.001)
    shifted['state'][:, 0] += .03
    assert not compare_traces(trace(.004), shifted)['passed']


def test_resolution_comparison_rejects_partial_trajectory():
    short = trace(.001)
    short = {k: v[:-100] for k, v in short.items()}
    with pytest.raises(ValueError, match='horizon'):
        compare_traces(trace(.004), short)


def test_resolution_comparison_wraps_angles():
    a, b = trace(.004), trace(.001)
    a['state'][:, 8] = np.pi-.001
    b['state'][:, 8] = -np.pi+.001
    assert compare_traces(a, b)['max_angle_difference_deg'] == pytest.approx(np.degrees(.002))
