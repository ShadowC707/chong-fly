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


# Round 1 (category b): hazard memory aged by transport error, not by time or a frame FIFO.
from simulation.navigation_pipeline import MemoryConfig, TransportMemory


def transport_memory(**kw):
    memory = TransportMemory('test-ref', VetoConfig(), MemoryConfig(aging='transport', **kw))
    memory.report = dict(state='PROCEED_LIMITED', heading_rad=0.)
    return memory


def test_hovering_never_forgets_a_side_obstacle_by_time_or_frame_count():
    memory = transport_memory()
    memory._insert(np.array([[0., .25, 0.]]), 1.)  # box on the left, then 30 s of hover frames
    rng = np.random.default_rng(5)
    wall = np.c_[np.full(64, 2.), rng.uniform(-1, 1, 64), rng.uniform(-.3, .3, 64)]
    for k in range(450):  # 15 Hz for 30 s: far beyond 4 s and 64 frames
        memory._insert(wall+rng.normal(0, .01, wall.shape), 1.+k/15)
        memory._prune()
    left, right = memory.side_gaps(31.)
    assert .20 < left < .25 and right > 1.9  # the right gap is the hover wall at 2 m
    assert memory.removed_by_capacity == 0 and memory.removed_by_error == 0


def test_points_age_by_path_and_turn_and_go_when_error_exceeds_r_veto():
    memory = transport_memory()
    memory._insert(np.array([[0., .25, 0.], [3., 0., 0.]]), 0.)
    memory.turn = np.pi/2  # quarter turn on the spot: both stay, the far point is less certain
    memory._prune()
    assert len(memory.points) == 2
    error = memory.transport_error(np.linalg.norm(memory.points, axis=1))
    assert error[0] < .04 and .15 < error[1] < .30
    memory.turn = 2*np.pi  # full turn: far point 0.02 + (0.03*2pi + 0.005)*3 > 0.30 goes, near ~0.07 stays
    memory._prune()
    np.testing.assert_allclose(memory.points, [[0., .25, 0.]])
    assert memory.transport_error(np.array([.25]))[0] < .08
    memory.observer.distance = 9.  # 0.02 + 0.03*9 = 0.29 + turn term at 0.25 m > 0.30
    memory._prune()
    assert len(memory.points) == 0 and memory.removed_by_error == 2
    assert memory.side_gaps(100.) == (np.inf, np.inf)


def test_voxel_keeps_the_newest_point_and_capacity_drops_the_least_certain():
    memory = transport_memory()
    memory._insert(np.array([[1.01, 0., 0.]]), 1.)
    memory._insert(np.array([[1.02, .01, .01]]), 2.)  # same 5 cm voxel
    assert len(memory.points) == 1 and memory.captured[0] == 2.
    memory = transport_memory(max_points=64)
    memory._insert(np.array([[0., .25, 0.]]), 0.)  # oldest, but near: the most certain
    memory.observer.distance = .1
    far = np.c_[np.linspace(-3, 3, 100), np.full(100, 3.), np.zeros(100)]
    memory._insert(far, 1.)  # newer, but at 3 m the heading lever makes them less certain
    memory.observer.distance = .2
    memory._prune()
    assert len(memory.points) == 64 and memory.removed_by_capacity == 37
    assert any(np.allclose(p, [0., .25, 0.]) for p in memory.points)


def test_round1_config_changes_only_memory_aging():
    v1 = PipelineConfig.from_json(ROOT/'configs/pipeline_v1.json')
    r1 = PipelineConfig.from_json(ROOT/'configs/pipeline_v1_r1.json')
    assert v1.memory.aging == 'time' and r1.memory.aging == 'transport'
    assert (r1.brake, r1.veto, r1.anti_rollback, r1.turn_latch) == (v1.brake, v1.veto, v1.anti_rollback, v1.turn_latch)
    assert isinstance(pipeline(pipeline_config=r1).memory, TransportMemory)
    assert type(pipeline(pipeline_config=v1).memory) is ThreatMemory


def test_round1_profile_flight_keeps_memory_and_limits():
    from demo.runner import DemoRunner
    with closing(DemoRunner(controller_mode='pipeline_v1_r1', scene='room', seed=3, start=(-.7, 0., 1.),
                            max_duration_s=8, collision_radius_m=.2)) as runner:
        coordinator = runner.system.runtime.coordinator
        assert isinstance(coordinator.memory, TransportMemory)
        runner.resume()
        runner.event(dict(type='mode', mode='AUTO'))
        sizes = []
        while not runner.terminal:
            runner.advance(5)
            d = coordinator.last_decision
            if 'stages' in d:
                assert not d['stage_limit_violation']
                assert all(r['kind'] == 'limits' for r in d['stages'])
                sizes.append(d['memory_points'])
        assert len(sizes) > 300 and max(sizes) > 64 and coordinator.memory.resets == 0
        assert coordinator.contract()['pipeline']['memory']['aging'] == 'transport'


# Round 2 (stage 1, rollback): speed-proportional brake limit, and the share-loss attribution.
from simulation.navigation_metrics import network_share, share_loss
from simulation.navigation_pipeline import BrakeConfig, proportional_brake_stage


@pytest.mark.parametrize('commands', [random_commands(5000, 7),
    pytest.param(None, marks=pytest.mark.skipif(not RECORDED, reason='no recorded baseline traces'))])
def test_proportional_brake_limits_every_tick(commands):
    if commands is None:
        commands = recorded_commands()
    rng = np.random.default_rng(8)
    for network in commands:
        after, info = proportional_brake_stage(network, rng.uniform(-.05, .12), 500.)
        assert within_limits(network, after)
        assert after[0] == network[0] and np.array_equal(after[[1, 3]], network[[1, 3]])


def test_proportional_brake_shrinks_with_speed_and_is_zero_at_rest():
    network = np.array([1500., 1500., 1300., 1500.])
    assert proportional_brake_stage(network, .5, 500.)[0][2] == 1300.     # fast: full network brake
    mid, info = proportional_brake_stage(network, .1, 500.)
    assert mid[2] == pytest.approx(1450.) and info['active'] and info['reason'] == 'brake_above_speed'
    assert proportional_brake_stage(network, 0., 500.)[0][2] == 1500.     # at rest: no brake
    assert proportional_brake_stage(network, -.02, 500.)[0][2] == 1500.   # moving back: no brake
    gentle, info = proportional_brake_stage(np.array([1500., 1500., 1480., 1500.]), .1, 500.)
    assert gentle[2] == 1480. and not info['active']                      # within the cap: untouched
    forward = np.array([1500., 1500., 1700., 1500.])
    assert proportional_brake_stage(forward, 0., 500.)[0][2] == 1700.
    with pytest.raises(ValueError):
        BrakeConfig(law='bang')


def test_round2_config_changes_only_the_brake_law():
    r1 = PipelineConfig.from_json(ROOT/'configs/pipeline_v1_r1.json')
    r2 = PipelineConfig.from_json(ROOT/'configs/pipeline_v1_r2.json')
    assert r1.brake.law == 'smoothstep' and r2.brake.law == 'proportional' and r2.brake.gain_pwm_per_flow == 500.
    assert (r2.veto, r2.memory, r2.anti_rollback, r2.turn_latch) == (r1.veto, r1.memory, r1.anti_rollback, r1.turn_latch)
    pipe = pipeline(pipeline_config=r2)
    pipe.apply(np.array([1500., 1500., 1300., 1500.]), flow_xy=np.array([.0, 0.]), tof_m=np.full(64, 3.), dt=.02)
    brake = pipe.last_decision['stages'][0]
    assert brake['after'][2] == 1500. and brake['reason'] == 'brake_above_speed'


def test_share_loss_attributes_each_lost_tick_once():
    net = np.full((5, 4), 1500.)
    out = net.copy()
    out[1, 2] = 1450.   # brake stage
    out[2, 3] = 1500.   # veto
    net[2, 3] = 1450.
    out[3, 3] = 1600.   # nothing in the stages: XY base (turn latch)
    log = [[dict(stage='brake', after=list(net[i])), dict(stage='veto', after=list(net[i]))] for i in range(5)]
    log[1][0]['after'] = list(out[1]); log[1][1]['after'] = list(out[1])
    log[2][1]['after'] = list(out[2])
    loss = share_loss(net, out, log)
    assert loss == {'brake': .2, 'veto': .2, 'xy_base': .2}
    assert sum(loss.values()) == pytest.approx(1-network_share(net, out))


# Round 3: retreat along the own corridor, and the corridor rollback metric.
from simulation.navigation_metrics import reverse_outside_corridor
from simulation.navigation_pipeline import RetreatConfig, retreat_check

RETREAT = RetreatConfig(enabled=True)


def straight_trail(length=1., lateral=0.):
    xs = np.arange(-length, 0.001, .005)
    return np.c_[xs, np.full(len(xs), lateral), np.zeros(len(xs))]


def test_retreat_needs_the_own_corridor_and_no_threat_behind():
    ok, info = retreat_check(straight_trail(), None, 0., RETREAT, .30)
    assert ok and info['corridor_m'] >= .05
    # Empty memory behind is never a permission by itself (box9: never seen).
    ok, info = retreat_check(None, None, 0., RETREAT, .30)
    assert not ok and info['retreat_block'] == 'no_corridor'
    ok, info = retreat_check(straight_trail(lateral=.08), None, 0., RETREAT, .30)
    assert not ok and info['retreat_block'] == 'outside_corridor'
    # The check length grows with the back speed: 0.05 + 4.1*0.2 = 0.87 m > 0.5 m of corridor.
    ok, info = retreat_check(straight_trail(.5), None, .2, RETREAT, .30)
    assert not ok and info['retreat_block'] == 'outside_corridor' and .45 < info['corridor_m'] <= .55  # tolerance 0.05
    assert retreat_check(straight_trail(1.), None, .2, RETREAT, .30)[0]
    threat = (np.array([[-.40, .0, 0.]]), np.array([.05]), np.array([0.]))
    assert retreat_check(straight_trail(), threat, 0., RETREAT, .30)[0]  # at rest L = 0.05: gap 0.30
    ok, info = retreat_check(straight_trail(), threat, .05, RETREAT, .30)  # L = 0.255: gap 0.095
    assert not ok and info['retreat_block'] == 'remembered_threat'
    ahead = (np.array([[.25, .0, 0.]]), np.array([.05]), np.array([0.]))  # retreat moves away from it
    assert retreat_check(straight_trail(), ahead, 0., RETREAT, .30)[0]


def test_round3_stage_still_limits_and_passes_retreat_only_in_corridor():
    r3 = PipelineConfig.from_json(ROOT/'configs/pipeline_v1_r3.json')
    r2 = PipelineConfig.from_json(ROOT/'configs/pipeline_v1_r2.json')
    assert r3.retreat.enabled and not r2.retreat.enabled
    assert (r3.brake, r3.veto, r3.memory) == (r2.brake, r2.veto, r2.memory)
    pipe = pipeline(pipeline_config=r3)
    pipe.memory = transport_memory()
    pipe.now_s = 10.
    back = np.array([1500., 1500., 1300., 1500.])
    pipe.apply(back, flow_xy=np.array([0., 0.]), tof_m=np.full(64, 3.), dt=.02)
    assert pipe.last_decision['stages'][0]['after'][2] == 1500.  # no corridor yet: r2 behaviour
    pipe.trail.extend((10.-i*.02, np.array([-i*.005, 0., 0.])) for i in range(200))
    pipe.apply(back, flow_xy=np.array([0., 0.]), tof_m=np.full(64, 3.), dt=.02)
    stage = pipe.last_decision['stages'][0]
    assert stage['after'][2] == 1300. and stage['reason'] == 'retreat_along_corridor' and not stage['active']
    assert not pipe.last_decision['stage_limit_violation']
    rng = np.random.default_rng(9)
    for network in random_commands(1000, 10):
        pipe.apply(network, flow_xy=rng.uniform(-.05, .08, 2), tof_m=np.full(64, 3.), dt=.02)
        assert all(within_limits(network, r['after']) for r in pipe.last_decision['stages'])


def test_reverse_outside_corridor_counts_only_new_space():
    t = np.arange(0, 20, .004)
    x = np.where(t < 10, .1*t, 1.-.05*(t-10))           # forward 1 m, then back 0.5 m along the path
    pos = np.c_[x, np.zeros_like(t), np.ones_like(t)]
    vel = np.c_[np.gradient(x, t), np.zeros_like(t), np.zeros_like(t)]
    outside, total = reverse_outside_corridor(t, pos, vel, np.zeros_like(t), np.ones_like(t, bool))
    assert total == pytest.approx(.5, abs=.01) and outside == pytest.approx(0., abs=.01)
    x = np.where(t < 2, .1*t, .2-.05*(t-2))              # 0.2 m forward, then 0.9 m back into new space
    pos[:, 0] = x
    vel[:, 0] = np.gradient(x, t)
    outside, total = reverse_outside_corridor(t, pos, vel, np.zeros_like(t), np.ones_like(t, bool))
    assert total == pytest.approx(.9, abs=.01) and outside == pytest.approx(.65, abs=.02)
