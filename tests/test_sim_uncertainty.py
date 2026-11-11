from dataclasses import asdict
import math

import numpy as np
import pytest

from evaluate.sim_uncertainty import (
    Perturbation, PerturbedSchedule, apply_plant, profiles, scenarios,
    assess, metrics_for,
)
from simulation.altitude_control import AltitudeSample
from simulation.sensor_schedule import SampleSchedule
from simulation.drone_env import DroneDynamicsParams, QuadcopterDynamics


def sample():
    return AltitudeSample(1., 0., 0., 0., True)


def test_zero_perturbation_preserves_samples_exactly():
    base = SampleSchedule('multi')
    wrapped = PerturbedSchedule(SampleSchedule('multi'), Perturbation())
    for i in range(20):
        flow = np.array([.0123 * i, -.003 * i], dtype=np.float32)
        tof = np.linspace(0., 1., 64, dtype=np.float32)
        assert asdict(base.frame(i, flow, tof, sample())) == asdict(wrapped.frame(i, flow, tof, sample()))


def test_distortion_occurs_at_capture_not_on_held_samples():
    schedule = PerturbedSchedule(SampleSchedule('multi'), Perturbation(flow_scale=.7, tof_bias_m=.1))
    a = schedule.frame(0, np.array([.2, -.3]), np.full(64, .2), sample())
    b = schedule.frame(1, np.array([.1, -.1]), np.full(64, .8), sample())
    np.testing.assert_allclose(a.flow.value, [.14, -.21])
    np.testing.assert_allclose(a.tof.value, .2 + .1/3.5)
    assert a.tof == b.tof
    assert b.flow.sample_id != a.flow.sample_id
    saturated = schedule.frame(4, [2., -2.], np.ones(64), sample())
    assert saturated.flow.value == (1., -1.)
    assert max(saturated.tof.value) == 1.


def test_mass_changes_inertia_and_preserves_initial_state_and_rpms():
    from types import SimpleNamespace
    params = DroneDynamicsParams()
    physics = QuadcopterDynamics(params)
    physics.motor_rpms[:] = 1234
    physics.pos[:] = [1, 2, 3]
    old_j = physics.J.copy()
    env = SimpleNamespace(params=params, physics=physics)
    apply_plant(env, Perturbation(mass_scale=1.25))
    assert params.mass == pytest.approx(.1625)
    np.testing.assert_allclose(physics.J, old_j * 1.25)
    np.testing.assert_array_equal(physics.motor_rpms, [1234]*4)
    np.testing.assert_array_equal(physics.pos, [1, 2, 3])


@pytest.mark.parametrize('kwargs', [dict(flow_scale=0), dict(mass_scale=-1),
    dict(yaw_scale=math.nan), dict(tof_bias_m=math.inf), dict(motor_tau_scale=True)])
def test_invalid_perturbations_rejected(kwargs):
    with pytest.raises(ValueError): Perturbation(**kwargs)


def test_protocol_is_finite_and_combination_is_explicit():
    assert len(profiles()) == 9 and len(scenarios()) == 5
    assert profiles() == profiles() and scenarios() == scenarios()
    assert sum(c.get('stop_at_s') is not None for c in scenarios()) == 1
    assert profiles()['combined'].yaw_scale == .7
    assert profiles()['combined'].flow_scale == .7
    assert profiles()['angle_60deg'].angle_scale == pytest.approx(60/35)


def good_metrics():
    return dict(collision=None, completed=True, min_clearance_m=.2,
        near_10cm_longest_s=0., below_10cm_s=0., timeout_events=0,
        stationary_before_stop_longest_s=1., distance_before_stop_m=6.,
        auto_s=44., max_height_error_m=.02, invalid_events=0,
        stop_requested=True, stop_active=True, stop_last_second_complete=True,
        stop_last_second_all_auto=True, stop_last_second_max_speed_mps=.02)


def test_stop_uses_whole_last_second_and_requires_authority_and_actual_request():
    case = {'stop_at_s':35.}
    assert assess(good_metrics(), case)['passed']
    for key, value in [('stop_last_second_max_speed_mps', .031),
                       ('stop_requested', False), ('stop_active', False),
                       ('stop_last_second_complete', False),
                       ('stop_last_second_all_auto', False), ('auto_s', 0.),
                       ('invalid_events', 1), ('completed', False)]:
        assert not assess(good_metrics() | {key:value}, case)['passed']


def test_metrics_dont_count_intentional_stop_as_stall_and_detect_transient_speed():
    rows = []
    for i in range(450):
        t = i * .1
        rows.append(dict(time_s=t, position_m=[t*.1, 0, 1],
            velocity_mps=[.02 if t>=35 else .1, 0, 0], attitude_rad=[0,0,0],
            plant_source='auto', clearance_m=.2, collision=None, trace={},
            stop_requested=t>=35))
    rows[442]['velocity_mps'] = [.04, 0, 0]
    m = metrics_for(rows, 45., [], [], {'stop_at_s':35.})
    assert m['stationary_before_stop_longest_s'] == 0
    assert m['stop_last_second_max_speed_mps'] == .04
    assert m['stop_last_second_complete']


def test_no_contact_is_not_success_without_motion_or_auto():
    for key, value in [('distance_before_stop_m', 0.), ('auto_s', 0.),
                       ('stationary_before_stop_longest_s', 6.),
                       ('min_clearance_m', .049), ('timeout_events', 1)]:
        assert not assess(good_metrics() | {key:value}, {})['passed']
