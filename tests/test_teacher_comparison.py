import pytest

from generator.generate_reflex_dataset import generate_reflex_dataset


def test_comparison_reuses_recorded_scene_seeds_and_runs_both_versions():
    from optimizer.compare_teachers import compare_teachers
    data = generate_reflex_dataset(num_episodes=2, seq_len=2, seed=19, scene_profile='varied-v1')
    result = compare_teachers(data, steps=3)
    assert len(result['episodes']) == 2
    for row, original in zip(result['episodes'], data['metadata']['episodes']):
        assert row['seed'] == original['seed'] and row['scenario'] == original['scenario']
        for version in ('legacy', 'braking'):
            assert row[version]['duration_s'] == pytest.approx(.06)
            assert row[version]['completed']
            assert len(row[version]['final_position_m']) == 3


@pytest.mark.parametrize('steps', [0, -1, True, 1.5])
def test_comparison_rejects_invalid_horizon(steps):
    from optimizer.compare_teachers import compare_teachers
    with pytest.raises(ValueError, match='steps'):
        compare_teachers({}, steps=steps)


def test_subset_comparison_retains_original_episode_indices_and_previous_version():
    from optimizer.compare_teachers import compare_teachers
    data = generate_reflex_dataset(num_episodes=3, seq_len=1, seed=29)
    progress = []
    report = compare_teachers(data, steps=2, episode_indices=[2,0],
        versions=('previous','braking'), progress=lambda done,total:progress.append((done,total)))
    assert [r['episode'] for r in report['episodes']] == [2,0]
    assert progress == [(1,2),(2,2)]
    assert report['episodes'][0]['previous']['teacher_version'] == 'geometry-reflex-v3'
    assert report['episodes'][0]['braking']['teacher_version'] == 'geometry-reflex-v4'


@pytest.mark.parametrize('indices', [[], [True], [4], [0,0]])
def test_invalid_comparison_subsets_are_rejected(indices):
    from optimizer.compare_teachers import compare_teachers
    data = generate_reflex_dataset(num_episodes=1, seq_len=1)
    with pytest.raises(ValueError, match='episode_indices'):
        compare_teachers(data, episode_indices=indices)


@pytest.mark.parametrize('scenario,seed', [('obstacle_left',2032569604),
    ('obstacle_right',289463710), ('obstacle_left',233837588), ('obstacle_right',2065484514)])
def test_recorded_collision_is_prevented_after_obstacle_leaves_view(scenario, seed):
    from optimizer.compare_teachers import _rollout
    from generator.reflex_contract import TEACHER_VERSION, LEGACY_TEACHER_VERSION
    scene = dict(scenario=scenario, seed=seed)
    before = _rollout(scene, 'varied-v1', LEGACY_TEACHER_VERSION, {}, 100)
    after = _rollout(scene, 'varied-v1', TEACHER_VERSION, {}, 500)
    assert before['crashed'] and before['collision_kind'] == 'box'
    assert after['completed'] and not after['crashed']
    # A safe pause is expected during braking; judge recovery over 10 seconds.
    assert after['resumed_cruise_after_turn'] and after['travel_m'] > 4.
