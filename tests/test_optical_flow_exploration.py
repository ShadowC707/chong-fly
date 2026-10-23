from types import SimpleNamespace

import numpy as np
import pytest

from simulation.drone_env import OpticalFlowOUWrapper


class SensorEnv:
    dt = .004
    tof = SimpleNamespace(max_range=3.5)

    def reset(self, **kwargs):
        return observation(3.)

    def step(self, action):
        return observation(3.), 0., False, {}


def observation(distance):
    return np.r_[.1, -.2, np.full(64, distance / 3.5)].astype(np.float32)


def test_distance_ramp_uses_metres_and_preserves_real_sensor_input():
    wrapper = OpticalFlowOUWrapper(SensorEnv(), sigma=0., theta=0.)
    wrapper.state[:] = 1.
    obs = observation(1.05)
    original = obs.copy()
    out = wrapper._inject_noise(obs)
    expected_gain = .5 * (1 - np.exp(-wrapper.dt / wrapper.gate_tau_s))
    assert wrapper.gain == pytest.approx(expected_gain, abs=1e-7)
    np.testing.assert_array_equal(obs, original)
    np.testing.assert_array_equal(out[2:], original[2:])
    np.testing.assert_allclose(out[:2] - original[:2], wrapper.state * wrapper.gain, atol=1e-7)


def test_occlusion_jump_does_not_restore_stale_noise_in_one_tick():
    wrapper = OpticalFlowOUWrapper(SensorEnv(), sigma=0., theta=0.)
    wrapper.state[:] = 1.
    wrapper.gain = 1.
    near = observation(.6)
    first = wrapper._inject_noise(near)[:2] - near[:2]
    # Fade rather than an instantaneous 1 -> 0 relay.
    assert np.all((first > .8) & (first < 1.))
    for _ in range(249):
        wrapper._inject_noise(near)
    assert np.linalg.norm(wrapper.state) < 1e-3
    far = observation(3.)
    returned = wrapper._inject_noise(far)[:2] - far[:2]
    assert np.linalg.norm(returned) < 1e-4
    # Even a large latent state cannot switch the gain 0 -> 1 on a ToF jump.
    wrapper.state[:] = 1.
    wrapper.gain = 0.
    returned = wrapper._inject_noise(far)[:2] - far[:2]
    assert np.all((returned > 0.) & (returned < .04))


def test_reset_replays_seed_and_does_not_advance_noise_clock():
    wrapper = OpticalFlowOUWrapper(SensorEnv(), seed=9)
    def episode(seed):
        initial = wrapper.reset(seed=seed)
        np.testing.assert_array_equal(initial, observation(3.))
        return np.array([wrapper.step(None)[0][:2] for _ in range(50)])
    np.testing.assert_array_equal(episode(12), episode(12))
    assert not np.array_equal(episode(12), episode(13))


def test_tuple_and_flat_observations_have_identical_perturbations():
    flat = OpticalFlowOUWrapper(SensorEnv(), seed=4)
    pair = OpticalFlowOUWrapper(SensorEnv(), seed=4)
    for distance in [3., 1.05, .5, 3.]:
        obs = observation(distance)
        a = flat._inject_noise(obs)
        b = pair._inject_noise((obs[:2], obs[2:]))
        np.testing.assert_array_equal(a, np.concatenate(b))


@pytest.mark.parametrize('kwargs', [dict(gate_tau_s=0.), dict(state_decay_tau_s=-1.),
    dict(clearance_start_m=1.3, clearance_full_m=1.2), dict(sigma=float('nan'))])
def test_invalid_noise_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        OpticalFlowOUWrapper(SensorEnv(), **kwargs)
