import numpy as np
import pytest

from generator.generate_reflex_dataset import ExpertReflexPolicy
from tests.test_teacher_contract import observation


def test_fast_approach_reduces_yaw_and_strengthens_initial_braking():
    from generator.generate_reflex_dataset import make_teacher
    x = observation('right', .4); x[0] = .2
    previous = make_teacher('geometry-reflex-v3').step(x)
    current = ExpertReflexPolicy().step(x)
    assert current[2] < previous[2] - 80
    assert current[3] == 1500


def test_slowing_down_restores_escape_turn_without_changing_its_direction():
    teacher = ExpertReflexPolicy()
    x = observation('left', .4); x[0] = .2
    assert teacher.step(x)[3] == 1500
    x[0] = .08
    intermediate = teacher.step(x)[3]
    x[0] = .02
    assert 1500 < intermediate < teacher.step(x)[3]
    assert teacher.latched_turn == 1


@pytest.mark.parametrize('settings', [dict(turn_full_flow=.15, turn_stop_flow=.12),
    dict(turn_stop_flow=float('inf')), dict(turn_full_flow=-1)])
def test_turn_gating_requires_ordered_finite_flow_thresholds(settings):
    with pytest.raises(ValueError, match='turn'):
        ExpertReflexPolicy(**settings)


def test_last_recorded_collision_now_completes_and_recovers_motion():
    from optimizer.compare_teachers import _rollout
    from generator.reflex_contract import TEACHER_VERSION
    result = _rollout({'scenario':'obstacle_right', 'seed':210110646},
                      'varied-v1', TEACHER_VERSION, {}, 500)
    assert result['completed'] and not result['crashed']
    assert result['resumed_cruise_after_turn'] and result['travel_m'] > 4
