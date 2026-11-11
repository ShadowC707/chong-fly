import numpy as np
import pytest

from generator.maneuver_sequences import make_plan, make_environment, collect
from generator.generate_reflex_dataset import make_teacher


def test_plan_is_finite_reproducible_and_separates_geometry_draws():
    train, evaluation = make_plan('train'), make_plan('evaluation')
    assert len(train) == 32 and len(evaluation) == 16
    assert train == make_plan('train')
    assert not ({r['seed'] for r in train} & {r['seed'] for r in evaluation})
    assert not ({r['pair_id'] for r in train} & {r['pair_id'] for r in evaluation})
    with pytest.raises(ValueError):
        make_plan('invalid')


@pytest.mark.parametrize('index', [0, 8, 16, 24])
def test_mirrored_initial_states_and_obstacles(index):
    plan = make_plan('train')
    a, _ = make_environment(plan[index]); b, _ = make_environment(plan[index+1])
    try:
        np.testing.assert_allclose(b.physics.vel, a.physics.vel*[1,-1,1], atol=1e-7)
        for left, right in zip(a.boxes, b.boxes):
            assert (left.x_min,left.x_max) == (right.x_min,right.x_max)
            assert (left.y_min,left.y_max) == (-right.y_max,-right.y_min)
        for left, right in zip(a.cylinders, b.cylinders):
            assert left.center_x == right.center_x
            assert left.center_y == -right.center_y
        assert a.collision_radius_m == b.collision_radius_m == .2
    finally:
        a.close(); b.close()


def test_collection_labels_replay_and_episode_state_does_not_leak():
    data = collect(make_plan('train')[:2], steps=8)
    assert data['X'].shape == (2,8,74)
    assert not (data['valid'][:,1:] & ~data['valid'][:,:-1]).any()
    m = data['metadata']
    for i in range(2):
        teacher = make_teacher(m['teacher_version'], **m['teacher_parameters'])
        for t in data['valid'][i].nonzero().flatten().tolist():
            np.testing.assert_array_equal(teacher.step(data['X'][i,t].numpy()),data['Y'][i,t].numpy())
    assert not data['metadata']['training_admitted']


def test_pair_cannot_cross_explicit_train_evaluation_split():
    plan = make_plan('train')[:2]
    assert plan[0]['pair_id'] == plan[1]['pair_id']
    assert plan[0]['side'] == -plan[1]['side']
    with pytest.raises(ValueError):
        collect(plan, steps=0)


def test_corrective_labels_remain_teacher_targets_not_executed_actor_actions():
    class Actor:
        def reset_state(self):
            self.calls = 0
        def step_np(self, flow, tof, memory, dt):
            self.calls += 1
            return np.array([1213.,1500.,1500.,1500.],np.float32)
    actor = Actor()
    data = collect(make_plan('train')[:2], steps=3, actor=actor)
    assert actor.calls == 3  # reset at the second episode
    np.testing.assert_array_equal(data['applied_actions'][...,2],1500.)
    assert not np.array_equal(data['Y'][...,2],data['applied_actions'][...,2])
    assert data['metadata']['collection_contract']['execution_actor'] == 'provided_policy'
    assert data['metadata']['collection_contract']['labels'] == 'v5 teacher on actor observation history'
