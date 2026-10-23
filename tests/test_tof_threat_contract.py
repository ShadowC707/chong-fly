"""Sensor/teacher arithmetic contracts, not physical flight admission."""
import numpy as np
import pytest
import torch

from configs.flight_config import APF_DISTANCE_THRESHOLD_M, TOF_RAYCASTER_MAX_RANGE_M
from generator.generate_reflex_dataset import ExpertReflexPolicy
from simulation.policy import SensorInputLayer


def observation(depth=.7, side='left', width=1):
    x = np.ones(74, dtype=np.float32)
    x[:2] = 0
    grid = x[2:66].reshape(8, 8)
    grid[:, :width] = depth/TOF_RAYCASTER_MAX_RANGE_M
    if side == 'right':
        grid[:] = grid[:, ::-1].copy()
    return x


def crossing_pair(depth=.7, epsilon=.0001):
    pair = np.stack([observation(depth)]*2)
    pair[:, 65] = (APF_DISTANCE_THRESHOLD_M+np.array([epsilon, -epsilon]))/TOF_RAYCASTER_MAX_RANGE_M
    return pair


def current_action(x):
    return ExpertReflexPolicy(noise_std_pwm=0).step(x)


def continuous_action(x):
    from generator.continuous_reflex_teacher import ContinuousReflexTeacher
    return ContinuousReflexTeacher(noise_std_pwm=0).step(x)


@pytest.mark.parametrize('depth', [.35, .7])
def test_weak_extra_ray_does_not_weaken_braking(depth):
    pair = crossing_pair(depth)
    assert current_action(pair[1])[2] > current_action(pair[0])[2]+1
    assert continuous_action(pair[1])[2] == continuous_action(pair[0])[2]


def test_force_summary_is_continuous_when_support_changes():
    from generator.continuous_reflex_teacher import ContinuousReflexTeacher
    p = ContinuousReflexTeacher(noise_std_pwm=0)
    near = np.zeros(64); near[:8] = 20
    near_active = near > 0
    baseline = p._summarize_repulsion(near, near_active)
    changes = []
    for epsilon in (1e-2, 1e-3, 1e-4):
        perturbed = near.copy(); perturbed[-1] = epsilon
        actual = p._summarize_repulsion(perturbed, perturbed > 0)
        assert actual[0] == baseline[0]
        changes.append(abs(actual[1]-baseline[1]))
    assert changes[1] < changes[0]/9
    assert changes[2] < changes[1]/9
    assert p._summarize_repulsion(np.zeros(64), np.zeros(64, dtype=bool)) == (0., 0.)


@pytest.mark.parametrize('width', [1, 2, 4])
@pytest.mark.parametrize('depth', [.35, .7])
def test_uniform_single_obstacle_retains_existing_teacher_behavior(width, depth):
    x = observation(depth, width=width)
    np.testing.assert_allclose(continuous_action(x), current_action(x), rtol=0, atol=1e-4)
    mirrored = continuous_action(observation(depth, side='right', width=width))
    left = continuous_action(x)
    assert mirrored[2] == left[2]
    assert mirrored[3]-1500 == -(left[3]-1500)


def test_adding_closer_obstacle_cannot_reduce_initial_braking():
    x = observation(.7)
    before = continuous_action(x)
    x[65] = .4/TOF_RAYCASTER_MAX_RANGE_M
    assert continuous_action(x)[2] < before[2]


def test_new_teacher_preserves_high_flow_turn_suppression_and_reset():
    from generator.continuous_reflex_teacher import ContinuousReflexTeacher
    p = ContinuousReflexTeacher(noise_std_pwm=0)
    x = observation(.35); x[0] = .2
    first = p.step(x)
    assert first[3] == 1500 and first[2] < 1500
    x[0] = 0
    assert p.step(x)[3] > 1500
    p.reset(); x[0] = .2
    np.testing.assert_array_equal(p.step(x), first)


def test_local_encoding_does_not_rescale_existing_pixels_at_threshold():
    x = torch.from_numpy(crossing_pair(.35))
    local = SensorInputLayer(learnable_scale=False, encoding='proximity-v1')(x)
    legacy = SensorInputLayer(learnable_scale=False, encoding='proximity-mean-v1')(x)
    torch.testing.assert_close(local[0, :65], local[1, :65], rtol=0, atol=0)
    assert legacy[1, 2]/legacy[0, 2] == pytest.approx(8/9)
    assert local[1, 65] > 0
    assert local[1, 65] < .0002


def test_local_encoding_is_monotone_bounded_and_keeps_spatial_coverage():
    distances = torch.linspace(0, 3.5, 1001)
    x = torch.ones(len(distances), 74); x[:, :2] = 0
    x[:, 2] = distances/3.5
    y = SensorInputLayer(learnable_scale=False, encoding='proximity-v1')(x)
    assert (y[:, 2].diff() <= 0).all()
    assert y.min() >= 0 and y.max() <= 1
    pair = torch.from_numpy(np.stack([observation(.35, width=w) for w in (1, 4)]))
    result = SensorInputLayer(learnable_scale=False, encoding='proximity-v1')(pair)
    assert result[0, 2] == result[1, 2]
    assert result[1, 2:66].sum() == 4*result[0, 2:66].sum()


def test_local_encoder_cannot_silently_load_mean_checkpoint(tiny_model_meta):
    from simulation.policy import ChongFlyMSPPolicy
    old = ChongFlyMSPPolicy.from_meta(str(tiny_model_meta), sensor_encoding='proximity-mean-v1', neutral_origin=True)
    local = ChongFlyMSPPolicy.from_meta(str(tiny_model_meta), sensor_encoding='proximity-v1', neutral_origin=True)
    with pytest.raises(RuntimeError, match='contract'):
        local.load_state_dict(old.state_dict(), strict=False)


def test_old_teacher_is_numerically_unchanged_at_audited_boundary():
    pair = crossing_pair(.7)
    np.testing.assert_allclose([current_action(x)[2] for x in pair],
                              [1469.863037109375, 1479.739990234375], rtol=0, atol=1e-4)
