import numpy as np
import pytest

from generator.generate_reflex_dataset import ExpertReflexPolicy
from tests.test_teacher_contract import observation


def test_losing_obstacle_does_not_release_brake_while_motion_continues():
    teacher = ExpertReflexPolicy()
    threat = observation('left', .4); threat[:2] = [.15, 0]
    teacher.step(threat)
    clear = observation(); clear[:2] = [.12, .04]
    for _ in range(75):
        action = teacher.step(clear)
        assert action[2] < 1500 and teacher.latched_turn == 1
        assert action[3] >= 1500  # v4 delays yaw while planar motion is fast


def test_brake_release_requires_sustained_low_motion_and_recovers_cruise():
    teacher = ExpertReflexPolicy()
    teacher.step(observation('right', .4))
    still = observation()
    for _ in range(5):
        assert teacher.step(still)[3] < 1500
    moving = observation(); moving[1] = .2
    teacher.step(moving)
    assert teacher.latched_turn == -1  # lateral drift resets settling too
    assert teacher.step(still)[3] < 1500
    actions = [teacher.step(still) for _ in range(40)]
    assert actions[-1][3] == 1500 and teacher.latched_turn is None
    assert actions[-1][2] == teacher.pwm_cruise_pitch
    released = [a[2] for a in actions if a[3] == 1500]
    assert max(np.diff([1500., *released])) <= 11


def test_braking_does_not_push_backwards_after_forward_motion_stops():
    teacher = ExpertReflexPolicy()
    teacher.step(observation('left', .35))
    backward = observation(); backward[:2] = [-.15, 0]
    assert teacher.step(backward)[2] > 1500
    teacher.reset()
    fresh = teacher.step(observation())
    assert fresh[2] == teacher.pwm_cruise_pitch and fresh[3] == 1500


@pytest.mark.parametrize('dt', [.004, .02, .04])
def test_brake_settling_uses_elapsed_time(dt):
    teacher = ExpertReflexPolicy()
    teacher.step(observation('left', .4), dt=dt)
    elapsed = 0.
    while elapsed < 1.:
        action = teacher.step(observation(), dt=dt); elapsed += dt
        if action[3] == 1500:
            break
    assert .20-1e-8 <= elapsed <= .20+dt+1e-8


def test_versioned_factory_retains_historical_teacher_for_exact_replay():
    from generator.generate_reflex_dataset import make_teacher
    from generator.reflex_contract import TEACHER_VERSION, LEGACY_TEACHER_VERSION
    assert TEACHER_VERSION != LEGACY_TEACHER_VERSION
    old = make_teacher(LEGACY_TEACHER_VERSION)
    new = make_teacher(TEACHER_VERSION)
    threat = observation('left', .4); threat[0] = .15
    old.step(threat); new.step(threat)
    clear = observation(); clear[0] = .12
    for _ in range(10):
        old_action, new_action = old.step(clear), new.step(clear)
    assert old_action[2] > 1500 and new_action[2] < 1500
    with pytest.raises(ValueError, match='version'):
        make_teacher('unknown')


@pytest.mark.parametrize('settings', [dict(brake_release_flow=0), dict(brake_settle_s=float('nan')),
    dict(brake_flow_scale=.02), dict(cruise_recovery_s=-1)])
def test_invalid_braking_parameters_are_rejected(settings):
    with pytest.raises(ValueError, match='braking'):
        ExpertReflexPolicy(**settings)


def test_historical_demonstrations_cannot_silently_train_current_teacher_contract():
    import torch
    from generator.generate_reflex_dataset import generate_reflex_dataset
    from generator.reflex_contract import LEGACY_TEACHER_VERSION
    from optimizer.pretrain import pretrain_policy
    data = generate_reflex_dataset(num_episodes=1, seq_len=1)
    data['metadata']['teacher_version'] = LEGACY_TEACHER_VERSION
    with pytest.raises(ValueError, match='teacher version'):
        pretrain_policy(torch.nn.Linear(74,4), data, epochs=1)
