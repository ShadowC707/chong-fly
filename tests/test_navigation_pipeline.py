"""Pipeline stages 1-2 are 'limits': per channel the result lies between 1500 and the network command."""
import json
from contextlib import closing
from pathlib import Path

import numpy as np
import pytest

from simulation.navigation_metrics import generated_share, share_metrics
from simulation.navigation_pipeline import (NavigationPipeline, PipelineConfig, ThreatMemory, VetoConfig,
                                            brake_stage, veto_stage, within_limits)

ROOT = Path(__file__).resolve().parents[1]
RECORDED = sorted((ROOT/'output/reiteration_baseline_v1').glob('*/*/trace.npz'))


def random_commands(n, seed):
    rng = np.random.default_rng(seed)
    commands = rng.uniform(1000, 2000, size=(n, 4))
    commands[::7, 1:] = 1500.  # exact neutral channels
    commands[::11, 1:] = rng.choice([1000., 2000.], size=(len(commands[::11]), 3))  # extremes
    return commands


def recorded_commands(step=20):
    rows = []
    for path in RECORDED:
        with np.load(path) as z:
            network = z['network'][~np.isnan(z['network']).any(axis=1)]
        rows.append(network[::step])
    return np.concatenate(rows)


def test_within_limits_definition():
    network = np.array([1500., 1600., 1400., 1700.])
    assert within_limits(network, network)
    assert within_limits(network, [1500., 1500., 1500., 1500.])
    assert not within_limits(network, [1500., 1400., 1400., 1700.])  # opposite side
    assert not within_limits(network, [1500., 1650., 1400., 1700.])  # larger modulus
    assert not within_limits(network, [1501., 1600., 1400., 1700.])  # throttle is not ours


@pytest.mark.parametrize('commands', [random_commands(5000, 1),
    pytest.param(None, marks=pytest.mark.skipif(not RECORDED, reason='no recorded baseline traces'))])
def test_stages_limit_every_tick(commands):
    if commands is None:
        commands = recorded_commands()
    rng = np.random.default_rng(2)
    assert len(commands) > 500
    for network in commands:
        flow = rng.uniform(-.05, .12)
        stage1, info = brake_stage(network, flow, .0105)
        assert within_limits(network, stage1)
        left, right = rng.choice([np.inf, .1, .25, .31, .6]), rng.choice([np.inf, .1, .25, .31, .6])
        stage2, info2 = veto_stage(stage1, left, right, .30)
        assert within_limits(network, stage2)
        assert within_limits(stage1, stage2)
        assert stage2[0] == network[0] and stage2[PITCH] == stage1[PITCH]
        assert info2['active'] == bool(info2['vetoed'])


PITCH = 2


def test_brake_passes_while_moving_and_fades_at_rest():
    network = np.array([1500., 1500., 1300., 1500.])
    moving, info = brake_stage(network, .0105, .0105)
    assert moving[2] == 1300. and not info['active']
    assert brake_stage(network, .05, .0105)[0][2] == 1300.
    rest, info = brake_stage(network, 0., .0105)
    assert rest[2] == 1500. and info['active']
    assert brake_stage(network, -.02, .0105)[0][2] == 1500.  # rolling back: no brake
    halfway = brake_stage(network, .00525, .0105)[0][2]
    assert 1300. < halfway < 1500.
    forward = np.array([1500., 1500., 1700., 1500.])
    assert brake_stage(forward, 0., .0105)[0][2] == 1700.  # forward command untouched


def test_veto_only_for_commands_toward_the_near_side():
    network = np.array([1500., 1560., 1500., 1450.])  # right roll, left yaw (below 1500)
    right_blocked, info = veto_stage(network, np.inf, .25, .30)
    assert right_blocked[1] == 1500. and right_blocked[3] == 1450. and info['vetoed'] == ['roll']
    left_blocked, info = veto_stage(network, .25, np.inf, .30)
    assert left_blocked[1] == 1560. and left_blocked[3] == 1500. and info['vetoed'] == ['yaw']
    far, info = veto_stage(network, .31, .5, .30)
    np.testing.assert_array_equal(far, network)
    assert not info['active'] and info['reason'] == 'no_remembered_side_obstacle'
    unknown, info = veto_stage(network, np.inf, np.inf, .30)  # unproven space never vetoes
    np.testing.assert_array_equal(unknown, network)
    assert not info['active']


def memory_with_hits(hits, *, heading=0.):
    memory = ThreatMemory('test-ref', VetoConfig())
    memory.observer.frames.append((1., np.zeros(3), 0., heading, np.asarray(hits, float)))
    memory.observer.last_s = 1.
    memory.report = dict(state='PROCEED_LIMITED', heading_rad=heading)
    return memory


def test_memory_side_gaps_use_side_points_only():
    # FLU: +y is left. Front, rear and far points must not veto.
    memory = memory_with_hits([[.28, .0, 0.], [-.28, 0., 0.], [.0, 2., 0.], [.1, .25, 0.]])
    left, right = memory.side_gaps(1.)
    assert .20 < left < .26 and right == np.inf   # the (0.1, 0.25) point, minus ~0.02 error
    memory = memory_with_hits([[.0, -.27, 0.]])
    left, right = memory.side_gaps(1.)
    assert left == np.inf and .22 < right < .27
    # A world point straight ahead of the start sits on the right after a 90 degree turn left,
    # and a point that was on the left is then straight ahead (front: no veto).
    turned = memory_with_hits([[.27, .0, 0.]], heading=np.pi/2)
    left, right = turned.side_gaps(1.)
    assert left == np.inf and .22 < right < .27
    assert memory_with_hits([[.0, .27, 0.]], heading=np.pi/2).side_gaps(1.) == (np.inf, np.inf)


def pipeline(**kw):
    options = dict(anti_rollback=False, turn_latch=True, stop_hover=True)
    return NavigationPipeline(reference_id='test-pipe', **options | kw)


def test_pipeline_apply_logs_both_stages_and_limits_on_random_input():
    pipe = pipeline()
    rng = np.random.default_rng(3)
    tof = np.full(64, 3.)
    for i, network in enumerate(random_commands(1500, 4)):
        stop = i % 50 == 49
        pipe.apply(network, flow_xy=rng.uniform(-.08, .08, 2), tof_m=tof, dt=.02, stop_requested=stop)
        d = pipe.last_decision
        assert [r['stage'] for r in d['stages']] == ['brake_no_rollback', 'side_veto']
        assert all(r['kind'] == 'limits' for r in d['stages'])
        assert not d['stage_limit_violation']
        if stop:
            assert not any(r['active'] for r in d['stages'])
        for r in d['stages']:
            assert set(r) >= {'stage', 'kind', 'active', 'reason', 'before', 'after'}
            assert within_limits(network, r['after'])
        json.dumps(d, allow_nan=False)


def test_pipeline_remembered_side_obstacle_vetoes_and_logs_reason():
    pipe = pipeline()
    pipe.memory = memory_with_hits([[.0, -.25, 0.]])
    pipe.now_s = 1.
    pipe.apply(np.array([1500., 1560., 1500., 1500.]), flow_xy=np.array([.03, 0.]), tof_m=np.full(64, 3.), dt=.02)
    veto = pipe.last_decision['stages'][1]
    assert veto['active'] and veto['after'][1] == 1500. and 'side_veto' in pipe.last_decision['reasons']


def test_generated_share_is_zero_for_limit_stages():
    log = [[dict(stage='a', kind='limits', active=True), dict(stage='b', kind='limits', active=True)]]*10
    assert generated_share(log) == 0.
    metrics = share_metrics(np.full((10, 4), 1500.), np.full((10, 4), 1500.), [True]*10, [False]*10, stage_log=log)
    assert metrics['generated_share'] == 0.


def test_config_file_matches_defaults_and_keeps_turn_latch():
    config = PipelineConfig.from_json(ROOT/'configs/pipeline_v1.json')
    assert config == PipelineConfig()
    assert config.anti_rollback is False and config.turn_latch is True
    assert config.brake.v_stop == .0105 and config.veto.r_veto_m == .30


def test_profile_flight_logs_stages_and_never_exceeds_limits():
    from demo.runner import DemoRunner
    with closing(DemoRunner(controller_mode='pipeline_v1', scene='room', seed=3, start=(-.7, 0., 1.),
                            max_duration_s=8, collision_radius_m=.2)) as runner:
        assert runner.manifest()['controller_mode'] == 'pipeline_v1'
        assert runner.system.runtime.coordinator.anti_rollback is False
        assert runner.system.runtime.coordinator.turn_latch is True
        runner.resume()
        runner.event(dict(type='mode', mode='AUTO'))
        seen = 0
        while not runner.terminal:
            runner.advance(5)
            decision = runner.system.runtime.coordinator.last_decision
            if 'stages' in decision:
                seen += 1
                assert not decision['stage_limit_violation']
                assert all(r['kind'] == 'limits' for r in decision['stages'])
        assert seen > 300
        trace = runner.rows[-2]['trace']
        assert trace['network_pwm'] is not None
