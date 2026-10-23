import numpy as np
import pytest

from simulation.xy_coordinator import XYCoordinator


def command(pitch=1400., yaw=1500.):
    return np.array([1213., 1512., pitch, yaw])


def apply(c, flow=.0, distance=.6, **kwargs):
    return c.apply(command(**kwargs), flow_xy=np.array([flow, 0.]),
                   tof_m=np.full(64, distance), dt=.02)


def test_anti_rollback_preserves_braking_at_speed_and_unrelated_channels():
    c = XYCoordinator(anti_rollback=True)
    np.testing.assert_array_equal(apply(c, flow=.15), command())
    for _ in range(50):
        result = apply(c, flow=0.)
    assert result[2] == 1500.
    np.testing.assert_array_equal(result[[0, 1, 3]], command()[[0, 1, 3]])
    assert c.last_decision['reasons'] == ['anti_rollback']
    assert apply(c, flow=-.1, pitch=1560.)[2] == 1560.


def test_anti_rollback_is_smooth_between_stop_and_forward_flow():
    values = [apply(XYCoordinator(anti_rollback=True), flow=f)[2]
              for f in np.linspace(.02, .06, 41)]
    assert values[0] == 1500. and values[-1] == 1400.
    assert np.all(np.diff(values) <= 0)
    assert max(abs(np.diff(values))) < 4.
    c = XYCoordinator(anti_rollback=True)
    for flow in [.01, -.01] * 25:
        assert apply(c, flow=flow)[2] == 1500.


def test_turn_latch_preserves_entire_pulse_envelope_and_neutral():
    c = XYCoordinator(turn_latch=True)
    for yaw in [1350., 1650., 1520., 1501., 1500., 1500., 1490.]:
        result = apply(c, yaw=yaw)
        assert abs(result[3]-1500.) == abs(yaw-1500.)
        assert result[3] <= 1500.
        np.testing.assert_array_equal(result[:3], command(yaw=yaw)[:3])
    assert apply(c, yaw=1500.)[3] == 1500.


def test_latch_requires_threat_and_clear_dwell_before_choosing_new_sign():
    c = XYCoordinator(turn_latch=True)
    assert apply(c, distance=3., yaw=1350.)[3] == 1350.
    assert c.turn_sign == 0
    assert apply(c, yaw=1650.)[3] == 1650.
    for _ in range(39):
        assert apply(c, distance=1.1, yaw=1500.)[3] == 1500.
    assert c.turn_sign == 1
    apply(c, distance=.9, yaw=1500.)  # Interrupt the clear dwell.
    for _ in range(40):
        apply(c, distance=1.1, yaw=1500.)
    assert c.turn_sign == 0
    assert apply(c, yaw=1350.)[3] == 1350.
    c.reset()
    assert c.turn_sign == 0 and c.filtered_flow is None


def test_disabled_coordinator_is_identity():
    c = XYCoordinator()
    for flow, yaw in [(.2, 1350.), (-.2, 1650.), (0., 1500.)]:
        np.testing.assert_array_equal(apply(c, flow=flow, yaw=yaw), command(yaw=yaw))
        assert not c.last_decision['intervened']


def test_near_zero_window_keeps_forward_braking_and_ignores_small_yaw_for_acquisition():
    from simulation.xy_coordinator import XYConfig
    config = XYConfig(stopped_flow=-.025, forward_flow=.005, acquire_yaw_pwm=60.)
    for speed in [.006, .03, .06, .15]:
        c = XYCoordinator(anti_rollback=True, turn_latch=True, config=config)
        assert apply(c, flow=speed, yaw=1525.)[2] == 1400.
        assert c.turn_sign == 0
        apply(c, flow=speed, yaw=1440.)
        assert c.turn_sign == -1
        assert apply(c, flow=speed, yaw=1500.)[3] == 1500.
    c = XYCoordinator(anti_rollback=True, config=config)
    assert apply(c, flow=-.025)[2] == 1500.


def test_rollout_records_network_and_applied_commands_separately():
    from optimizer.rollout import simulate_policy_rollout
    from tests.test_evaluation_contract import ScriptedEnv

    class Policy:
        def step_np(self, *args, **kwargs):
            return command()

    class StillEnv(ScriptedEnv):
        def reset_obs(self):
            return np.r_[0., 0., np.full(64, .2)].astype(np.float32)

        def reset(self, seed=None):
            super().reset(seed)
            return self.reset_obs()

    _, metrics = simulate_policy_rollout(Policy(), env=StillEnv(), eval_steps=3,
        record_trace=True, coordinator=XYCoordinator(anti_rollback=True))
    assert metrics['coordinator']['intervention_steps'] == 3
    for row in metrics['trace']['control']:
        assert row['network_pwm'][2] == 1400.
        assert row['applied_pwm'][2] == 1500.
        assert row['rc_pwm'] == row['applied_pwm']
        assert row['coordinator']['reasons'] == ['anti_rollback']


def test_arena_mirror_reflects_geometry_and_preserves_sensor_frame():
    from evaluate.watch_mvp import make_environment
    a, b = make_environment('arena'), make_environment('arena_mirror')
    try:
        for x, y in zip(a.boxes, b.boxes):
            assert (x.x_min, x.x_max) == (y.x_min, y.x_max)
            assert (x.y_min, x.y_max) == (-y.y_max, -y.y_min)
        for x, y in zip(a.cylinders, b.cylinders):
            assert x.center_x == y.center_x and x.center_y == -y.center_y
        a.reset(seed=42); b.reset(seed=42)
        np.testing.assert_array_equal(a.physics.pos, b.physics.pos)
        np.testing.assert_array_equal(a.tof.body_ray_dirs, b.tof.body_ray_dirs)
    finally:
        a.close(); b.close()


def test_corner_pair_mirrors_pose_velocity_and_actual_tof_columns():
    from evaluate.watch_mvp import make_environment
    a, b = make_environment('corner_approach'), make_environment('corner_approach_mirror')
    try:
        a.reset(seed=42); b.reset(seed=42)
        np.testing.assert_allclose(a.physics.pos * [1, -1, 1], b.physics.pos)
        np.testing.assert_allclose(a.physics.vel * [1, -1, 1], b.physics.vel)
        assert a.physics.quaternion_to_euler(a.physics.quat)[2] == pytest.approx(
            -b.physics.quaternion_to_euler(b.physics.quat)[2])
        left, right = a._get_observation()[2:].reshape(8,8), b._get_observation()[2:].reshape(8,8)
        np.testing.assert_allclose(left[:, ::-1], right, atol=1e-6)
        assert not np.allclose(left, right)
    finally:
        a.close(); b.close()


def test_factorial_metrics_measure_reverse_travel_and_sustained_corner_exit():
    from evaluate.compare_xy_coordinator import summarize
    controls = [dict(network_pwm=command().tolist(), applied_pwm=command().tolist())]
    physics = [dict(time_s=t, position_m=pos, body_velocity_mps=vel) for t,pos,vel in [
        (0., [0,0,1], [0,0,0]), (.5, [3,3,1], [-1,0,0]),
        (1., [3,3,1], [-1,0,0]), (1.5, [2,2,1], [1,0,0]), (2., [2,2,1], [1,0,0])]]
    report = dict(scene='test', coordinator_mode='baseline', checkpoint_sha256='test',
        trace=dict(control=controls, physics=physics), metrics=dict(horizon_completed=True,
        feasible=True, survival_time_s=2., crashed=False, failure_reason=None,
        min_clearance_m=.3, yaw_bursts=dict(count=0)))
    result = summarize(report)
    assert result['corner_time_s'] == 1.
    assert result['corner_exits_1s'] == 1
    assert result['backward_distance_m'] == 1.
    assert result['max_continuous_backward_m'] == 1.


@pytest.mark.parametrize('dt', [0., -.02, float('nan')])
def test_invalid_dt_rejected(dt):
    with pytest.raises(ValueError):
        XYCoordinator().apply(command(), flow_xy=np.zeros(2), tof_m=np.ones(64), dt=dt)
