import json
import numpy as np
import pytest


def test_obstacle_scenes_are_mirrored_and_survive_benchmark_reset():
    from simulation.benchmark_scenes import NavigationScene
    left, right = NavigationScene('obstacle_left'), NavigationScene('obstacle_right')
    try:
        left_obs, right_obs = left.reset(seed=20261007), right.reset(seed=20261007)
        np.testing.assert_allclose(left_obs[2:].reshape(8, 8),
                                   right_obs[2:].reshape(8, 8)[:, ::-1], atol=1e-5)
        assert left_obs[2:].min() < 1
        np.testing.assert_allclose(left.physics.vel, [.3, 0, 0])
        assert len(left.boxes) == len(right.boxes) == 1
        # A second reset is exactly the operation performed by rollout.
        np.testing.assert_array_equal(left.reset(seed=20261007), left_obs)
    finally:
        left.close(); right.close()


def test_drift_and_fast_approach_keep_distinct_initial_conditions():
    from simulation.benchmark_scenes import NavigationScene
    for name, velocity in [('drift_left', [.3, .55, 0]), ('drift_right', [.3, -.55, 0]),
                           ('fast_center', [1.5, 0, 0])]:
        env = NavigationScene(name)
        try:
            env.reset(seed=1)
            np.testing.assert_allclose(env.physics.vel, velocity)
            assert bool(env.boxes) == (name == 'fast_center')
        finally:
            env.close()


def test_checkpoint_benchmark_refuses_modified_checkpoint_before_loading(tmp_path):
    from optimizer.benchmark_candidate import load_research_run
    (tmp_path/'report.json').write_text(json.dumps({'checkpoint_sha256':'0'*64}))
    (tmp_path/'checkpoint.pt').write_bytes(b'not a checkpoint')
    with pytest.raises(ValueError, match='hash'):
        load_research_run(tmp_path)


def test_tof_ablation_preserves_real_flow_and_memory_and_resets_state():
    from optimizer.benchmark_candidate import ToFAblation
    import torch
    class Recorder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.reset_count = 0
        def reset_state(self):
            self.reset_count += 1
        def step_np(self, flow_xy, tof_8x8, memory_ring=None, dt=None):
            self.seen = (flow_xy.copy(), tof_8x8.copy(), memory_ring.copy(), dt)
            return np.full(4, 1500.)
    policy = Recorder()
    ablated = ToFAblation(policy)
    flow, tof, memory = np.array([.2, -.3]), np.full(64, .1), np.arange(8)/8
    ablated.step_np(flow, tof, memory_ring=memory, dt=.02)
    np.testing.assert_array_equal(policy.seen[0], flow)
    np.testing.assert_array_equal(policy.seen[1], np.ones(64))
    np.testing.assert_array_equal(policy.seen[2], memory)
    np.testing.assert_array_equal(tof, np.full(64, .1))
    ablated.reset_state()
    assert policy.reset_count == 1 and policy.seen[3] == .02
