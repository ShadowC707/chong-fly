"""Behavior priorities and deterministic exploration, independent of learned weights."""
import numpy as np
import pytest

from simulation.xy_coordinator import XYCoordinator, BehaviorConfig


def step(c, distance=3.5, yaw=1500., flow=(0., 0.), stop=False, tof=None):
    return c.apply([1213., 1500., 1527., yaw], flow_xy=flow,
                   tof_m=np.full(64, distance) if tof is None else tof,
                   dt=.02, stop_requested=stop)


def active_search(c):
    for _ in range(310):
        out = step(c)
        if abs(out[3]-1500) > 10:
            return out
    pytest.fail('No search pulse in an uninterrupted clear interval')


def test_levy_is_seeded_bounded_and_reset_replays():
    c = XYCoordinator(levy_explore=True, seed=7)
    intervals = [c._sample_interval() for _ in range(2000)]
    assert min(intervals) >= 1.5 and max(intervals) <= 6.
    assert np.median(intervals) < 3.
    c.reset()
    a = np.array([step(c) for _ in range(700)])
    c.reset()
    b = np.array([step(c) for _ in range(700)])
    np.testing.assert_array_equal(a, b)
    assert np.max(abs(a[:, 3]-1500)) > 90
    assert np.max(abs(a[:, 3]-1500)) <= 110
    np.testing.assert_array_equal(a[:, :3], np.tile([1213,1500,1527], (700,1)))
    assert np.max(abs(np.diff(a[:, 3]))) < 30  # smooth pulse, no threats


@pytest.mark.parametrize('distance,yaw', [(.94,1500.), (3.5,1525.), (3.5,1475.)])
def test_search_is_preempted_in_same_step_by_range_or_network(distance, yaw):
    c = XYCoordinator(anti_rollback=True, turn_latch=True, levy_explore=True)
    active_search(c)
    out = step(c, distance=distance, yaw=yaw)
    assert out[3] == yaw  # remove search contribution, NOT the reflex
    assert c.last_decision['search_pwm'] == 0
    assert c.last_decision['search_event'] == 'preempted'


def test_search_waits_for_clear_space_and_chooses_correct_image_side():
    c = XYCoordinator(levy_explore=True)
    for _ in range(400):
        assert step(c, distance=1.1)[3] == 1500
    tof = np.full((8,8), 2.); tof[:, :4] = 3.
    outputs = [step(c, tof=tof)[3] for _ in range(310)]
    assert min(outputs) < 1450 and max(outputs) == 1500  # left = negative RC yaw


def test_stop_cancels_search_damps_both_axes_and_yields_to_critical_threat():
    c = XYCoordinator(anti_rollback=True, turn_latch=True, levy_explore=True, stop_hover=True)
    active_search(c)
    for _ in range(80):
        out = step(c, flow=(.1,.1), stop=True)
        assert out[3] == 1500 and out[0] == 1213
    assert out[2] < 1500 and out[1] > 1500
    assert c.last_decision['state'] == 'STOP'
    out = step(c, distance=.49, yaw=1400, stop=True)
    assert out[3] == 1400 and out[2] == 1527
    assert c.last_decision['state'] == 'STOP_EVASION'
    for _ in range(150):
        out = step(c, stop=True)
    assert c.last_decision['state'] == 'HOVER'
    np.testing.assert_allclose(out[1:], [1500]*3, atol=.01)


def test_stop_is_opt_in_and_invalid_request_does_not_advance_state():
    c = XYCoordinator()
    with pytest.raises(ValueError):
        step(c, stop=True)
    assert c.filtered_flow is None
    with pytest.raises(ValueError):
        BehaviorConfig(levy_min_s=7, levy_max_s=6)


def test_latch_blocks_search_until_its_clear_dwell_expires():
    c = XYCoordinator(turn_latch=True, levy_explore=True)
    step(c, distance=.6, yaw=1400)
    for _ in range(39):
        assert step(c)[3] == 1500
        assert c.turn_sign == -1 and c.wait_elapsed == 0
    step(c)
    assert c.turn_sign == 0 and c.wait_elapsed > 0


def test_search_diagnostics_distinguish_wait_reset_from_pulse_preemption():
    c = XYCoordinator(levy_explore=True)
    step(c)  # begin waiting, no pulse yet
    tof = np.full((8,8),3.); tof[:,-1]=1.05
    out=step(c,tof=tof)
    assert out[3] == 1500
    d=c.last_decision
    assert d['search_event']=='wait_reset'
    assert d['search_cancelled_phase']=='waiting'
    assert d['search_blockers']==['range_not_clear']
    assert d['tof_center_min_m']==3. and d['tof_min_m']==1.05
    active_search(c)
    step(c,distance=.9)
    assert c.last_decision['search_cancelled_phase']=='pulse'
    assert c.last_decision['search_blockers']==['range_preempt']


def test_search_diagnostics_report_simultaneous_latch_range_and_reflex_blocks():
    c=XYCoordinator(levy_explore=True,turn_latch=True)
    step(c,distance=.6,yaw=1400)
    assert set(c.last_decision['search_blockers'])=={'range_preempt','network_reflex','turn_latch'}


def test_stop_metrics_distinguish_path_from_displacement_and_initial_acceleration():
    from evaluate.compare_xy_behaviors import summarize
    report=dict(trace=dict(control=[],physics=[
        dict(time_s=t,position_m=[x,0,1],body_velocity_mps=[v,0,0])
        for t,x,v in [(0,0,.1),(35,0,.1),(35.3,.15,.2),(36,.2,.02),(39,.19,.001),(40,.18,.001)]]),
        metrics=dict(horizon_completed=True,feasible=True,crashed=False,failure_reason=None,
                     survival_time_s=40.,min_clearance_m=.2),
        stop_at_s=35.,target_height_m=1.,scene='test',levy_explore=False,checkpoint_sha256='test')
    s=summarize(report)['stop']
    assert s['speed_at_request_mps']==.1 and s['peak_speed_after_request_mps']==.2
    assert s['path_length_after_request_m']==pytest.approx(.22)
    assert s['final_drift_from_stop_m']==pytest.approx(.18)


def test_stop_schedule_uses_control_time_and_keeps_five_physics_substeps():
    from optimizer.rollout import simulate_policy_rollout
    from tests.test_evaluation_contract import ScriptedEnv, RecordingPolicy
    c = XYCoordinator(stop_hover=True)
    _, metrics = simulate_policy_rollout(RecordingPolicy(), env=ScriptedEnv(),eval_steps=5,
        coordinator=c,stop_at_s=.04,record_trace=True)
    rows = metrics['trace']['control']
    assert [r['coordinator']['stop_requested'] for r in rows] == [False,False,True,True,True]
    assert metrics['physics_steps'] == 25
    with pytest.raises(ValueError):
        simulate_policy_rollout(RecordingPolicy(),env=ScriptedEnv(),stop_at_s=0)


def test_stop_metric_cannot_pass_on_a_lucky_final_sample_or_early_termination():
    from evaluate.compare_xy_behaviors import summarize
    metrics = dict(horizon_completed=True, feasible=True, crashed=False, failure_reason=None,
                   survival_time_s=40., min_clearance_m=.2)
    report = dict(trace=dict(control=[],physics=[
        dict(time_s=t,position_m=[1,1,1],body_velocity_mps=[v,0,0])
        for t,v in [(0,.2),(35,.2),(39,.1),(40,0)]]),metrics=metrics,
        stop_at_s=35.,target_height_m=1.,scene='test',levy_explore=False,checkpoint_sha256='test')
    assert not summarize(report)['stop']['speed_target_passed']
    report['metrics']['horizon_completed'] = False
    report['trace']['physics'] = report['trace']['physics'][:1]
    assert summarize(report)['stop'] is None


@pytest.mark.parametrize('name,count', [('arena_sparse',2),('arena_medium',5),('arena_dense',8)])
def test_density_scenes_have_measured_clear_passages_and_clear_start(name,count):
    from simulation.xy_scenes import DensityArena
    env = DensityArena(name)
    try:
        assert len(env.boxes)+len(env.cylinders) == count
        spec = env.specification()
        assert spec['minimum_obstacle_gap_m'] >= 1.3
        assert spec['minimum_wall_gap_m'] >= 1.3
        env.reset(seed=42)
        np.testing.assert_allclose(env.physics.pos, [0,0,1])
        assert env.room.x_min == env.room.y_min == -4
        assert env.room.x_max == env.room.y_max == 4
    finally:
        env.close()
