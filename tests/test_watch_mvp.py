import json

import numpy as np
import pytest

from optimizer.rollout import simulate_policy_rollout
from tests.test_evaluation_contract import RecordingPolicy, ScriptedEnv


def test_trace_preserves_rollout_results_and_both_timebases():
    _, ordinary = simulate_policy_rollout(RecordingPolicy(), env=ScriptedEnv(), eval_steps=4)
    policy = RecordingPolicy()
    _, traced = simulate_policy_rollout(policy, env=ScriptedEnv(), eval_steps=4, record_trace=True)
    trace = traced.pop('trace')
    assert traced == ordinary
    assert len(trace['control']) == 4 and len(trace['physics']) == 21
    assert all(call[0] == .02 for call in policy.calls)
    np.testing.assert_allclose([r['time_s'] for r in trace['control']], [0, .02, .04, .06])
    np.testing.assert_allclose([r['time_s'] for r in trace['physics']], np.arange(21)*.004)
    np.testing.assert_allclose(trace['physics'][0]['position_m'], [0, 0, 1])
    np.testing.assert_allclose(trace['physics'][-1]['position_m'], [.08, 0, 1])
    assert trace['control'][0]['min_tof_m'] == 3.5


def test_trace_stops_at_actual_collision_tick():
    _, report = simulate_policy_rollout(RecordingPolicy(), env=ScriptedEnv(crash_at=7),
                                        eval_steps=100, record_trace=True)
    assert len(report['trace']['control']) == 2
    assert len(report['trace']['physics']) == 8
    assert report['trace']['physics'][-1]['time_s'] == pytest.approx(.028)
    assert report['crashed']


def test_trace_records_altitude_applied_throttle_at_control_time():
    from simulation.altitude_control import AltitudeHold
    from simulation.benchmark_scenes import NavigationScene
    class RecordingHold(AltitudeHold):
        def __init__(self):
            super().__init__()
            self.applied = []
        def apply(self, pwm, sample, *, now_s):
            result = super().apply(pwm, sample, now_s=now_s)
            self.applied.append((now_s, result.copy()))
            return result
    controller = RecordingHold()
    env = NavigationScene('clear')
    try:
        _, metrics = simulate_policy_rollout(RecordingPolicy(), env=env, eval_steps=4,
                                             altitude_hold=controller, record_trace=True)
    finally:
        env.close()
    for row, (time, action) in zip(metrics['trace']['control'], controller.applied):
        assert row['time_s'] == time
        np.testing.assert_array_equal(row['rc_pwm'], action)
    assert len(controller.applied) == 4
    assert metrics['physics_steps'] == 20


def test_viewer_environment_is_standalone_and_ou_window_uses_metres():
    from evaluate.watch_mvp import make_environment
    for scene in ('clear', 'arena'):
        env = make_environment(scene, explore=True, seed=42)
        try:
            assert env.engine == 'standalone' and env.headless
            assert env.dt == .004
            assert env.clearance_start_m == .85
            assert env.clearance_full_m == 1.25
            assert env.tof_range_m == env.tof.max_range
            assert env.sigma == .06
            obs = np.r_[0., 0., np.ones(64)].astype(np.float32)
            changed = env._inject_noise(obs)
            assert np.any(changed[:2] != 0)
            np.testing.assert_array_equal(changed[2:], obs[2:])
        finally:
            env.close()


def test_burst_marker_uses_physical_event_time_not_pwm_jump():
    from evaluate.watch_mvp import burst_positions
    trace = dict(physics=[dict(time_s=t, position_m=[t, 2*t, 1]) for t in (0, .004, .008)])
    result = burst_positions(trace, dict(events=[dict(start_s=.006)]))
    np.testing.assert_allclose(result, [[.006, .012]])


def test_viewer_exports_exact_loaded_model_without_training(tmp_path, monkeypatch):
    import torch
    from evaluate import watch_mvp
    from simulation.policy import ChongFlyMSPPolicy
    from tests.synthetic_graphs import write_tiny_model
    meta = write_tiny_model(tmp_path/'graph')
    policy = ChongFlyMSPPolicy.from_meta(str(meta), dt=.02)
    checkpoint = dict(candidate='synthetic_test_only', continuation=dict(selected_tau_scale=.25))
    source = tmp_path/'source'; source.mkdir()
    (source/'checkpoint.pt').write_bytes(b'test-fixture')
    (source/'report.json').write_text(json.dumps(dict(mvp_admission=dict(passed=False))))
    loaded = []
    def loader(path):
        loaded.append(path)
        return policy, checkpoint
    monkeypatch.setattr(watch_mvp, 'load_research_run', loader)
    state = {k:v.clone() for k,v in policy.state_dict().items() if isinstance(v, torch.Tensor)}
    png = tmp_path/'flight.png'
    report = watch_mvp.run_view(source, scene='clear', steps=5, plot_out=png)
    assert loaded == [source]
    assert report['candidate'] == 'synthetic_test_only'
    assert report['checkpoint_mvp_admission']['passed'] is False
    assert report['metrics']['control_steps'] == 5
    assert png.read_bytes().startswith(b'\x89PNG')
    assert json.loads(png.with_suffix('.json').read_text())['candidate'] == 'synthetic_test_only'
    for k,v in state.items():
        torch.testing.assert_close(policy.state_dict()[k], v, rtol=0, atol=0)
    with pytest.raises(FileExistsError):
        watch_mvp.run_view(source, scene='clear', steps=5, plot_out=png)
