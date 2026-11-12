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


# Round 3b: the backward command is limited so the coast predicted from tilt fits the corridor.
from simulation.navigation_pipeline import retreat_room


def test_coast_prediction_matches_the_brake_model_and_limits_the_backward_command():
    r = PipelineConfig.from_json(ROOT/'configs/pipeline_v1_r3b.json').retreat
    assert r.prediction == 'tilt'
    steady_tilt = r.gamma_per_s*.1/r.beta_mps2_per_deg      # tilt that holds 0.1 m/s against drag
    assert r.coast_m(.1, steady_tilt) == pytest.approx(.1/r.gamma_per_s+.1*r.tilt_tau_s, rel=.02)
    assert r.coast_m(0., 14.5) > 3.  # a full backward command (1300) coasts over 3 m: round-3 failure
    assert r.max_back_pwm(.05, 0., 0.) == 0.                 # no room beyond the margin: nothing passes
    previous = 0.
    for room in (.1, .3, .6, 1., 1.6):
        allowed = r.max_back_pwm(room, 0., 0.)
        assert allowed > previous
        previous = allowed
        tilt_next = r.tilt_gain_deg_per_pwm_tick*allowed
        assert r.coast_m(0., tilt_next) <= room-r.margin_m+1e-6
    assert r.max_back_pwm(.6, .1, steady_tilt) < r.max_back_pwm(.6, 0., 0.)  # already moving back: less


def test_retreat_room_stops_at_the_corridor_end_and_at_a_threat_behind():
    r = PipelineConfig.from_json(ROOT/'configs/pipeline_v1_r3b.json').retreat
    room, info = retreat_room(straight_trail(.8), None, r, .30)
    assert .8 <= room <= .86 and info['room_limit'] == 'outside_corridor'
    assert retreat_room(None, None, r, .30)[0] == 0.          # empty memory, no corridor: no room
    threat = (np.array([[-.6, 0., 0.]]), np.array([.05]), np.array([0.]))
    room, info = retreat_room(straight_trail(1.5), threat, r, .30)
    assert .2 < room < .27 and info['room_limit'] == 'remembered_threat'


def test_round3b_stage_limits_and_passes_only_what_the_coast_allows():
    r3 = PipelineConfig.from_json(ROOT/'configs/pipeline_v1_r3.json')
    r3b = PipelineConfig.from_json(ROOT/'configs/pipeline_v1_r3b.json')
    assert r3.retreat.prediction == 'speed'
    assert (r3b.brake, r3b.veto, r3b.memory) == (r3.brake, r3.veto, r3.memory)
    pipe = pipeline(pipeline_config=r3b)
    pipe.memory = transport_memory()
    pipe.now_s, pipe.tilt_deg = 10., 0.
    pipe.trail.extend((10.-i*.02, np.array([-i*.005, 0., 0.])) for i in range(200))  # 1 m corridor
    back = np.array([1500., 1500., 1300., 1500.])
    pipe.apply(back, flow_xy=np.array([0., 0.]), tof_m=np.full(64, 3.), dt=.02)
    stage = pipe.last_decision['stages'][0]
    assert stage['reason'] == 'retreat_along_corridor' and stage['after'][2] == 1300.  # 200 fits 1 m at rest
    pipe.tilt_deg = -8.  # already tilted back 8 deg: the predicted coast no longer fits, command is limited
    pipe.apply(back, flow_xy=np.array([-.02, 0.]), tof_m=np.full(64, 3.), dt=.02)
    stage = pipe.last_decision['stages'][0]
    assert 1300. < stage['after'][2] <= 1500. and stage['active']
    rng = np.random.default_rng(11)
    for network in random_commands(1000, 12):
        pipe.tilt_deg = rng.uniform(-10, 10)
        pipe.apply(network, flow_xy=rng.uniform(-.05, .08, 2), tof_m=np.full(64, 3.), dt=.02)
        assert all(within_limits(network, r['after']) for r in pipe.last_decision['stages'])
        assert not pipe.last_decision['stage_limit_violation']


def test_signed_coast_credits_forward_motion_and_matches_a_simulation():
    r = PipelineConfig.from_json(ROOT/'configs/pipeline_v1_r3b2.json').retreat
    assert r.signed and r.prediction == 'tilt'
    # Braking from 0.15 m/s forward with 2.2 deg back tilt: the net motion back is small, not 0.38 m.
    assert r.signed_coast_m(-.15, 2.2) < .1 < r.coast_m(0., 2.2)
    # Exact for the linear model: simulate tilt decay and drag after the command is removed.
    for v0, tilt0 in ((-.15, 2.2), (.1, 1.), (.0, 5.), (.2, -1.)):
        a = np.exp(-.02/r.tilt_tau_s)
        v, tilt, x, far = v0, tilt0, 0., 0.
        for _ in range(3000):
            v += .02*(r.beta_mps2_per_deg*tilt-r.gamma_per_s*v)
            tilt *= a
            x += v*.02
            far = max(far, x)
        predicted = (v0+r.beta_mps2_per_deg*tilt0*r.tilt_tau_s)/r.gamma_per_s
        # continuous tau vs the discrete sum dt/(1-a) (0.270 s vs 0.260 s): within 5 %
        assert far == pytest.approx(max(predicted, far if predicted < 0 else predicted), rel=.05, abs=.005)
    room = .6
    allowed = r.signed_max_back_pwm(room, -.1, 1.)
    tilt_next = np.exp(-.02/r.tilt_tau_s)*1.+r.tilt_gain_deg_per_pwm_tick*allowed
    assert r.signed_coast_m(-.1, 1., tilt_next) == pytest.approx(room-r.margin_m, abs=1e-6)
    assert r.signed_max_back_pwm(.05, .1, 3.) == 0.


# Round 4: velocity hold (kind generates) and the author's generated_share rule.
from simulation.navigation_metrics import generated_active_share
from simulation.navigation_pipeline import HoldConfig, hold_stage


def test_hold_corrects_only_neutral_axes_and_never_beyond_the_cap():
    h = HoldConfig(enabled=True)
    neutral = np.array([1500., 1503., 1497., 1500.])
    after, info = hold_stage(neutral, np.array([-.005, .002]), h, trigger='near_obstacle')
    assert after[2] == pytest.approx(1497.+2.) and after[1] == pytest.approx(1503.+.8)  # STOP-law signs
    assert info['active'] and info['reason'] == 'hold_near_obstacle'
    big, _ = hold_stage(neutral, np.array([-.2, .2]), h, trigger='near_obstacle')
    assert np.allclose(np.abs(big[1:3]-neutral[1:3]), 10.)                              # cap
    moving = np.array([1500., 1560., 1520., 1500.])
    unchanged, info = hold_stage(moving, np.array([-.01, .01]), h, trigger='near_obstacle')
    assert np.array_equal(unchanged, moving) and not info['active']                      # network acts
    assert np.array_equal(hold_stage(neutral, np.array([-.01, 0.]), h, trigger=None)[0], neutral)
    assert not hold_stage(neutral, None, h, trigger='near_obstacle')[1]['active']        # window filling


def test_generated_share_counts_only_commands_beyond_the_tolerance():
    def tick(network, hold_before, hold_after):
        return [dict(stage='brake', kind='limits', active=False, before=network, after=network),
                dict(stage='velocity_hold', kind='generates', active=hold_before != hold_after,
                     before=hold_before, after=hold_after)]
    n = [1500., 1500., 1500., 1500.]
    small = tick(n, n, [1500., 1500., 1506., 1500.])                     # own contribution 6 PWM
    blocked = tick([1500., 1500., 1300., 1500.], n, [1500., 1500., 1504., 1500.])  # own 4, 204 from network
    big = tick(n, n, [1500., 1500., 1515., 1500.])                       # own contribution 15 PWM
    log = [small, blocked, big, tick(n, n, n)]
    # Author's rule after round 4: only the stage's own contribution above 10 PWM counts.
    assert generated_share(log) == .25 and generated_active_share(log) == .75
    assert generated_share([[dict(stage='x', kind='generates', active=True)]]) == 1.  # old logs


def test_round4_profile_adds_only_the_hold_stage():
    r3b2 = PipelineConfig.from_json(ROOT/'configs/pipeline_v1_r3b2.json')
    r4 = PipelineConfig.from_json(ROOT/'configs/pipeline_v1_r4.json')
    assert r4.hold.enabled and not r3b2.hold.enabled and r4.hold.window_s == 1. and r4.hold.cap_pwm == 10.
    assert (r4.brake, r4.veto, r4.memory, r4.retreat) == (r3b2.brake, r3b2.veto, r3b2.memory, r3b2.retreat)
    pipe = pipeline(pipeline_config=r4)
    pipe.memory = transport_memory()
    pipe.memory._insert(np.array([[-.35, 0., 0.]]), 0.)
    pipe.now_s, pipe.tilt_deg = 1., 0.
    for _ in range(60):  # fill the 1 s window with a slow backward drift
        pipe.apply(np.array([1500., 1500., 1500., 1500.]), flow_xy=np.array([-.005, 0.]), tof_m=np.full(64, 3.), dt=.02)
    hold = pipe.last_decision['stages'][-1]
    assert hold['stage'] == 'velocity_hold' and hold['kind'] == 'generates' and hold['active']
    assert hold['trigger'] == 'near_obstacle' and 1500. < hold['after'][2] <= 1510.
    assert pipe.contract()['stage_kinds']['velocity_hold'] == 'generates'


# Round 5: calibrated memory error model, hardware turn term switched off.
def test_round5_changes_only_the_memory_error_model():
    r4 = PipelineConfig.from_json(ROOT/'configs/pipeline_v1_r4.json')
    r5 = PipelineConfig.from_json(ROOT/'configs/pipeline_v1_r5.json')
    assert r4.memory.turn_term and not r5.memory.turn_term
    assert (r5.veto.base_error_m, r5.veto.distance_error_fraction) == (.035, .020)
    assert (r5.brake, r5.retreat, r5.hold) == (r4.brake, r4.retreat, r4.hold)
    assert r5.veto.r_veto_m == r4.veto.r_veto_m and r5.memory.voxel_m == r4.memory.voxel_m
    memory = TransportMemory('test-ref', r5.veto, r5.memory)
    memory.report = dict(state='PROCEED_LIMITED', heading_rad=0.)
    memory._insert(np.array([[0., .5, 0.]]), 0.)
    memory.turn = 50.  # many turns: no effect without the hardware term
    memory.observer.distance = 2.
    assert memory.transport_error(np.array([.5]))[0] == pytest.approx(.035+.020*2+.005*.5)
    memory.observer.distance = 13.5   # 0.035 + 0.27 + 0.0025 > 0.30: removed
    memory._prune()
    assert len(memory.points) == 0


# Round 6: integral term of the hold (synthetic checks before runs).
def test_hold_integral_removes_a_steady_drift_and_respects_the_cap():
    h = HoldConfig(enabled=True, integral_gain_pwm_per_flow_s=100.)
    neutral = np.array([1500., 1500., 1500., 1500.])
    integral = {}
    drift = np.array([-.006, 0.])                      # card 26: ~0.024 m/s backwards
    out = [hold_stage(neutral, drift, h, trigger='near_obstacle', integral=integral)[0][2] for _ in range(250)]
    assert out[0] == pytest.approx(1500.+400*.006+100*.006*.02)
    assert out[-1] > out[0] and out[-1] <= 1510.        # grows over 5 s, never beyond the cap
    assert integral['pitch'] > 0
    big = {}
    for _ in range(1000):
        after, _ = hold_stage(neutral, np.array([-.05, 0.]), h, trigger='near_obstacle', integral=big)
    assert after[2] == 1510. and big.get('pitch', 0.) <= 10.  # anti-windup: integral stops at the cap
    hold_stage(neutral, drift, h, trigger=None, integral=integral)
    assert integral == {}                               # reset without a trigger
    frozen = {'pitch': 3.}
    hold_stage(np.array([1500., 1500., 1560., 1500.]), drift, h, trigger='near_obstacle', integral=frozen)
    assert frozen['pitch'] == 3.                        # network drives Pitch: frozen
    p_only, _ = hold_stage(neutral, drift, HoldConfig(enabled=True), trigger='near_obstacle', integral={})
    assert p_only[2] == pytest.approx(1500.+400*.006)   # ki = 0 keeps round 4/5 exactly


def test_round6_changes_only_the_hold_integral():
    r5 = PipelineConfig.from_json(ROOT/'configs/pipeline_v1_r5.json')
    r6 = PipelineConfig.from_json(ROOT/'configs/pipeline_v1_r6.json')
    assert r5.hold.integral_gain_pwm_per_flow_s == 0. and r6.hold.integral_gain_pwm_per_flow_s == 100.
    assert (r6.brake, r6.veto, r6.memory, r6.retreat) == (r5.brake, r5.veto, r5.memory, r5.retreat)
    from dataclasses import replace
    assert replace(r6.hold, integral_gain_pwm_per_flow_s=0.) == r5.hold


def test_round6b_changes_only_the_integral_gain():
    r6 = PipelineConfig.from_json(ROOT/'configs/pipeline_v1_r6.json')
    r6b = PipelineConfig.from_json(ROOT/'configs/pipeline_v1_r6b.json')
    assert r6b.hold.integral_gain_pwm_per_flow_s == 300.
    from dataclasses import replace
    assert replace(r6b, hold=replace(r6b.hold, integral_gain_pwm_per_flow_s=100.)) == r6
