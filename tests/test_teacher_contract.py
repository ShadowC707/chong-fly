import numpy as np
import pytest
import torch

from generator.generate_reflex_dataset import ExpertReflexPolicy, generate_reflex_dataset
from configs.flight_config import CONTROL_DT, TOF_RAYCASTER_MAX_RANGE_M


def observation(side=None, distance=.5):
    x = np.ones(74, dtype=np.float32)
    x[:2] = 0
    if side:
        grid = x[2:66].reshape(8, 8)
        grid[:, :4 if side == 'left' else 8] = distance/TOF_RAYCASTER_MAX_RANGE_M
        if side == 'right': grid[:, :4] = 1
    return x


def test_teacher_default_range_matches_simulator():
    assert ExpertReflexPolicy().max_sensor_range_m == TOF_RAYCASTER_MAX_RANGE_M


def test_symmetric_obstacle_produces_repeatable_escape_without_random_labels():
    x = observation('center')
    a, b = ExpertReflexPolicy(seed=1), ExpertReflexPolicy(seed=99)
    left, right = a.step(x), b.step(x)
    np.testing.assert_array_equal(left, right)
    assert left[3] > 1500 and left[2] < 1500
    assert a.latched_turn == 1


def test_latched_turn_survives_a_single_opposite_or_clear_frame_then_releases():
    teacher = ExpertReflexPolicy()
    assert teacher.step(observation('left'))[3] > 1500
    assert teacher.step(observation('right'))[3] > 1500
    assert teacher.step(observation())[3] > 1500
    for _ in range(30): action = teacher.step(observation())
    assert action[3] == 1500 and teacher.latched_turn is None
    teacher.reset()
    assert teacher.step(observation('right'))[3] < 1500


def test_flow_feedback_opposes_forward_and_lateral_drift():
    teacher = ExpertReflexPolicy()
    still = teacher.step(observation())
    drift = observation(); drift[:2] = [.5, .5]
    action = teacher.step(drift)
    assert action[2] < still[2] and action[1] > still[1]


@pytest.mark.parametrize('x', [np.ones(5), np.full(74, np.nan), np.full(74, 2.)])
def test_invalid_observations_are_not_replaced_with_clear_space(x):
    with pytest.raises(ValueError): ExpertReflexPolicy().step(x)


def test_dataset_has_balanced_geometric_scenarios_clean_labels_and_timebase():
    data = generate_reflex_dataset(num_episodes=12, seq_len=8, seed=9)
    metadata = data['metadata']
    assert metadata['dt'] == CONTROL_DT
    assert metadata['physics_dt'] == .004
    assert metadata['dataset_version'] == 'reflex-v3'
    assert metadata['observation_source'] == 'geometric_raycast'
    counts = metadata['scenario_counts']
    assert len(counts) >= 6 and max(counts.values()) - min(counts.values()) <= 1
    assert data['valid'].dtype == torch.bool
    assert data['valid'].shape == (12, 8)
    assert (data['Y'][:,:,3] < 1450).any() and (data['Y'][:,:,3] > 1550).any()
    first = data['X'][metadata['episode_scenarios'].index('obstacle_left'), 0, 2:66].reshape(8, 8)
    assert first[:, :4].min() < first[:, 4:].min()


def test_generator_reproducible_and_rejects_nonintegral_control_period():
    import torch
    first = generate_reflex_dataset(num_episodes=6, seq_len=4, seed=5)
    second = generate_reflex_dataset(num_episodes=6, seq_len=4, seed=5)
    for key in ('X', 'Y', 'valid'): assert torch.equal(first[key], second[key])
    with pytest.raises(ValueError, match='dt|period'):
        generate_reflex_dataset(num_episodes=1, seq_len=2, dt=.005)


def test_generator_stops_at_termination_instead_of_resetting_inside_sequence(monkeypatch):
    from generator import generate_reflex_dataset as generator
    from types import SimpleNamespace
    class Env:
        dt = .004
        tof = SimpleNamespace(max_range=3.5)
        physics = SimpleNamespace(quat=np.array([1, 0, 0, 0]),
                                  quaternion_to_euler=lambda q: [0, 0, 0])
        def step(self, *args, **kwargs):
            return observation()[:66], 0., True, {'crashed': True}
        def get_isaac_obs(self):
            return SimpleNamespace(laser_distance=.982, laser_valid=True,
                                   attitude_euler=[0, 0, 0], sim_time=0.)
        def close(self): pass
    monkeypatch.setattr(generator, '_make_scenario', lambda *a: (Env(), observation()[:66]))
    data = generator.generate_reflex_dataset(num_episodes=1, seq_len=8)
    assert data['valid'].tolist() == [[True] + [False]*7]
    assert data['metadata']['episodes'][0]['physics_ticks'] == 1
    assert data['metadata']['episodes'][0]['crashed']


def test_execution_noise_does_not_contaminate_teacher_labels():
    clean = generate_reflex_dataset(num_episodes=6, seq_len=1, seed=3, noise_std_pwm=0)
    noisy = generate_reflex_dataset(num_episodes=6, seq_len=1, seed=3, noise_std_pwm=20)
    torch.testing.assert_close(clean['X'], noisy['X'], rtol=0, atol=0)
    torch.testing.assert_close(clean['Y'], noisy['Y'], rtol=0, atol=0)
