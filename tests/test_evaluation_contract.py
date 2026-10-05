"""Fixed trajectories make benchmark semantics independent of neural training luck."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from optimizer.evaluate import simulate_policy_rollout, objective
from optimizer.rollout import BENCHMARK_VERSION
from simulation.control_contract import control_contract


class ScriptedEnv:
    dt = .004
    def __init__(self, crash_at=None, sideways=False):
        self.crash_at = crash_at
        self.sideways = sideways
        self.physics = SimpleNamespace(pos=np.zeros(3), vel=np.zeros(3),
                                       total_mass=.13, quat=np.array([1., 0, 0, 0]),
                                       quaternion_to_euler=lambda q: np.zeros(3))
        self.tof = SimpleNamespace(max_range=3.5)
        self.steps = 0
    def reset(self, seed=None):
        self.steps = 0
        self.physics.pos[:] = [0, 0, 1]
        self.physics.vel[:] = [0, 1, 0] if self.sideways else [1, 0, 0]
        return np.r_[.25, -.25, np.ones(64)].astype(np.float32)
    def step(self, action):
        self.steps += 1
        self.physics.pos += self.physics.vel * self.dt
        crash = self.steps == self.crash_at
        return self.reset_obs(), 0., crash, {"crashed": crash, "clearance_m": 1.,
                                             "power_w": 10.}
    def reset_obs(self):
        return np.r_[.25, -.25, np.ones(64)].astype(np.float32)
    def get_chong_fly_obs(self):
        raise AssertionError("Evaluator must consume returned observations, not resample sensors")


class RecordingPolicy:
    default_dt = .004
    def __init__(self, invalid=False):
        self.calls = []
        self.invalid = invalid
    def reset_state(self):
        self.calls.clear()
    def step_np(self, flow, tof, memory_ring=None, dt=None):
        self.calls.append((dt, flow.copy(), tof.copy()))
        return np.array([np.nan if self.invalid else 1213., 1500., 1500., 1500.])


def test_control_period_holds_action_for_five_physics_steps():
    env, policy = ScriptedEnv(), RecordingPolicy()
    _, metrics = simulate_policy_rollout(policy, env=env, eval_steps=1500, dt=.02)
    assert env.steps == 7500
    assert len(policy.calls) == 1500
    assert all(call[0] == .02 for call in policy.calls)
    assert metrics["survival_time_s"] == pytest.approx(30.)
    assert metrics["total_energy_j"] == pytest.approx(300.)
    assert metrics["feasible"]
    np.testing.assert_array_equal(policy.calls[0][1], [.25, -.25])


def test_crash_in_middle_of_control_period_stops_immediately():
    env = ScriptedEnv(crash_at=7)
    _, metrics = simulate_policy_rollout(RecordingPolicy(), env=env, eval_steps=10, dt=.02)
    assert env.steps == 7
    assert metrics["survival_time_s"] == pytest.approx(.028)
    assert metrics["crashed"] and not metrics["feasible"]
    assert metrics["jitter_pr_l2"] == 0  # smooth crash must still fail the safety gate


def test_hover_is_not_falsely_marked_as_a_collision():
    env = ScriptedEnv()
    original_reset = env.reset
    def reset(seed=None):
        obs = original_reset(seed)
        env.physics.vel[:] = 0
        return obs
    env.reset = reset
    _, metrics = simulate_policy_rollout(RecordingPolicy(), env=env, eval_steps=100, dt=.02)
    assert env.steps == 500
    assert not metrics["crashed"]
    assert metrics["coverage_count"] == 1


def test_sideways_flight_is_reported_truthfully():
    _, metrics = simulate_policy_rollout(RecordingPolicy(), env=ScriptedEnv(sideways=True), eval_steps=5, dt=.02)
    assert metrics["is_crab_flight"]


@pytest.mark.parametrize("dt", [0, -.02, float("nan"), .006])
def test_invalid_or_nonintegral_control_period_is_rejected(dt):
    with pytest.raises(ValueError):
        simulate_policy_rollout(RecordingPolicy(), env=ScriptedEnv(), eval_steps=5, dt=dt)


def test_nan_policy_never_reaches_physics():
    env = ScriptedEnv()
    _, metrics = simulate_policy_rollout(RecordingPolicy(invalid=True), env=env, eval_steps=5, dt=.02)
    assert env.steps == 0
    assert metrics["fatal_failure"] and not metrics["feasible"]


def test_actual_model_storage_includes_buffers_and_active_heads():
    class Policy(RecordingPolicy, torch.nn.Module):
        def __init__(self):
            torch.nn.Module.__init__(self)
            RecordingPolicy.__init__(self)
            self.cfc_network = torch.nn.Linear(2, 2, bias=False)
            self.head = torch.nn.Linear(2, 4, bias=False)
            self.register_buffer("fixed_connectome", torch.zeros(2, 2))
    policy = Policy()
    scores, metrics = simulate_policy_rollout(policy, env=ScriptedEnv(), eval_steps=2, dt=.02)
    assert scores[0] == 64  # 16 float32 elements, including stored zeros
    assert metrics["model_storage_bytes"] == 64


def test_objective_aggregates_all_seeds_not_just_last(monkeypatch):
    import optimizer.evaluate as evaluator
    monkeypatch.setattr(evaluator, "create_model", lambda *a, **kw: torch.nn.Linear(1, 1))
    seeds = []
    def rollout(**kwargs):
        seeds.append(kwargs["seed"])
        failed = len(seeds) == 1
        return (8., 0., 3.), {"crashed": failed, "feasible": not failed,
            "benchmark_version": BENCHMARK_VERSION, "control_contract": control_contract(),
            "fatal_failure": False, "survival_time_s": .02 if failed else .04,
            "min_clearance_m": .0 if failed else 1., "mean_clearance": 1.,
            "is_crab_flight": False}
    monkeypatch.setattr(evaluator, "simulate_policy_rollout", rollout)
    class Trial:
        number = 9
        def __init__(self): self.attrs = {}
        def set_user_attr(self, key, value): self.attrs[key] = value
    trial = Trial()
    objective(trial, pretrain=False, eval_steps=2, seed=42, device="cpu")
    assert seeds == [42, 142, 242]
    assert trial.attrs["crashed"] and not trial.attrs["feasible"]
    assert trial.attrs["crash_rate"] == pytest.approx(1/3)
    assert len(trial.attrs["rollouts"]) == 3
    assert trial.attrs["constraints"][0] > 0


def test_external_termination_on_last_tick_is_not_a_success():
    env = ScriptedEnv()
    step = env.step
    def interrupted(action):
        obs, reward, _, info = step(action)
        return obs, reward, env.steps == 5, info
    env.step = interrupted
    _, metrics = simulate_policy_rollout(RecordingPolicy(), env=env, eval_steps=1, dt=.02)
    assert not metrics["crashed"]  # interruption and collision are distinct
    assert not metrics["feasible"]


def test_policy_programming_error_is_not_retried_as_another_call_signature():
    class BrokenPolicy(RecordingPolicy):
        def step_np(self, *args, **kwargs):
            self.calls.append("called")
            raise TypeError("internal policy bug")
    policy = BrokenPolicy()
    with pytest.raises(TypeError, match="internal policy bug"):
        simulate_policy_rollout(policy, env=ScriptedEnv(), eval_steps=1, dt=.02)
    assert policy.calls == ["called"]
