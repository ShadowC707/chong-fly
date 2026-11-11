import numpy as np
import pytest

from evaluate.sitl_yaw_profiles import yaw_offset, settled_delay


@pytest.mark.parametrize('sign', [-1, 1])
def test_profiles_match_peak_and_discrete_commanded_angle(sign):
    outputs = {}
    for name in ('step', 'cosine'):
        y = np.array([yaw_offset(name, i, sign) for i in range(300)])
        outputs[name] = y
        assert np.max(abs(y)) == 140
        assert np.all(sign*y >= 0)
        assert .02*y.sum()*.4 == pytest.approx(sign*16.8, abs=1e-10)
        assert np.all(y[:50] == 0) and np.all(y[77:] == 0)
    assert np.max(abs(np.diff(outputs['step']))) == 140
    assert np.max(abs(np.diff(outputs['cosine']))) < 19
    assert outputs['cosine'][50] == outputs['cosine'][77] == 0


def test_settling_does_not_count_first_crossing_before_rebound():
    t = np.arange(11)*.1
    signal = np.array([0, 9, 9, 2, 1, 8, 9, 2, 1, 0, 0])
    assert settled_delay(t, signal, end_s=.2, band=3) == pytest.approx(.5)
    signal[-1] = 4
    assert settled_delay(t, signal, end_s=.2, band=3) is None
    assert settled_delay(t, signal, end_s=2., band=3) is None


def test_profile_invalid_arguments_fail_closed():
    for args in [('typo', 10, 1), ('step', -1, 1), ('step', 2.2, 1), ('step', 1, 0)]:
        with pytest.raises(ValueError): yaw_offset(*args)


def test_motor_limit_count_uses_pinned_idle_floor_not_zero():
    from evaluate.sitl_yaw_profiles import motor_limit_counts
    m = np.array([[.055000003, .2, .2, .3], [.2,.3,.3,.3], [.4,.2,1.,.3]])
    assert motor_limit_counts(m) == dict(lower_limit_ticks=1, upper_limit_ticks=1, saturated_motor_ticks=2)


def test_twr_profile_requires_both_target_and_measured_map(tmp_path):
    from evaluate.sitl_yaw_profiles import trial
    for kwargs in (dict(target_twr=5.),dict(calibration_rows=[])):
        with pytest.raises(ValueError,match='together'):
            trial(tmp_path/'not_created',tmp_path,'cosine',1,**kwargs)
    assert not (tmp_path/'not_created').exists()
