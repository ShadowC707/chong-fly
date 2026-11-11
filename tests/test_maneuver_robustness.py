from copy import deepcopy
import numpy as np
import pytest

from evaluate.maneuver_robustness import AuditDriver, start_spec
from tests.test_practical_maneuver import Policy, Teacher


@pytest.mark.parametrize('driver', ['candidate','teacher'])
def test_driver_selection_is_from_first_tick_and_keeps_separate_records(driver):
    p = AuditDriver(Policy(10), Teacher(), driver)
    x = (np.zeros(2),np.ones(64))
    output = p.step_np(*x)
    np.testing.assert_array_equal(output,p.records[-1]['outputs'][driver])
    output[3] = 1000
    assert p.records[-1]['outputs'][driver][3] != 1000
    assert p.teacher.n == 1 and p.policies['candidate'].n == 1
    p.reset_state(); p.step_np(*x)
    assert p.teacher.n == 1 and p.policies['candidate'].n == 1


def test_unknown_driver_is_rejected():
    with pytest.raises(ValueError): AuditDriver(Policy(10),Teacher(),'unknown')


def test_heading_perturbation_changes_only_yaw_without_mutating_source():
    m = dict(start=dict(position_m=[-.7,0,1],yaw_rad=.2),seed=4,scene='random_room',scene_options={'a':1})
    original = deepcopy(m)
    spec = start_spec(m,-5.)
    assert spec['yaw'] == pytest.approx(.2-np.deg2rad(5))
    assert spec['start'] == [-.7,0,1] and spec['seed'] == 4
    spec['start'][0] = 12; spec['scene_options']['a'] = 2
    assert m == original
    with pytest.raises(ValueError): start_spec(m,float('nan'))


@pytest.mark.parametrize('channels', [(3,), (2,3)])
def test_early_intervention_has_exact_half_open_window_and_no_state_reset(channels):
    from evaluate.maneuver_robustness import EarlyIntervention
    p = EarlyIntervention(Policy(10), Teacher(), channels)
    args = (np.zeros(2),np.ones(64))
    for t in (5.98,6.,9.98,10.):
        p.time_s = t
        out = p.step_np(*args)
        row = p.records[-1]
        expected = row['outputs']['candidate'].copy()
        if 6 <= t < 10: expected[list(channels)] = row['outputs']['teacher'][list(channels)]
        np.testing.assert_array_equal(out,expected)
        np.testing.assert_array_equal(row['driver_pwm'],out)
    assert p.teacher.n == 4 and p.policies['candidate'].n == 4
    assert p.records[-1]['outputs']['candidate'][3] != 1500
    with pytest.raises(ValueError): EarlyIntervention(Policy(0),Teacher(),(0,))
