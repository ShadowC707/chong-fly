import math

import numpy as np
import pytest

from simulation.altitude_control import AltitudeHold, AltitudeSample, AltitudeTelemetryError


def sample(height=1., time=0., roll=0., pitch=0., valid=True):
    # Downward emitter sits 18 mm below the center, along body -Z.
    cosine = math.cos(roll)*math.cos(pitch)
    return AltitudeSample(height/cosine-.018, roll, pitch, time, valid)


def test_height_error_and_vertical_motion_have_opposing_feedback():
    low = AltitudeHold().update(sample(.8), now_s=0)
    high = AltitudeHold().update(sample(1.2), now_s=0)
    assert low.throttle_pwm > 1213 > high.throttle_pwm
    rising, falling = AltitudeHold(), AltitudeHold()
    rising.update(sample(.98), now_s=0)
    falling.update(sample(1.02), now_s=0)
    up = rising.update(sample(1., .02), now_s=.02)
    down = falling.update(sample(1., .02), now_s=.02)
    assert up.vertical_speed_mps > 0 > down.vertical_speed_mps
    assert up.throttle_pwm < down.throttle_pwm


def test_mount_and_tilt_compensation_do_not_invent_height_change():
    result = AltitudeHold().update(sample(1., roll=.3, pitch=.2), now_s=0)
    assert result.height_m == pytest.approx(1.)
    assert result.vertical_speed_mps == 0
    assert result.throttle_pwm > 1213  # replace vertical thrust lost to tilt


@pytest.mark.parametrize('fault', ['invalid', 'nan', 'stale', 'future', 'tilt', 'range'])
def test_bad_telemetry_revokes_autonomous_output_instead_of_guessing_throttle(fault):
    s = sample()
    if fault == 'invalid': s = AltitudeSample(s.range_m, 0, 0, 0, False)
    if fault == 'nan': s = AltitudeSample(float('nan'), 0, 0, 0, True)
    if fault == 'stale': s = sample(time=-.2)
    if fault == 'future': s = sample(time=.1)
    if fault == 'tilt': s = sample(roll=1.2)
    if fault == 'range': s = sample(height=10)
    with pytest.raises(AltitudeTelemetryError): AltitudeHold().update(s, now_s=0)


def test_duplicate_timestamp_and_abrupt_surface_jump_are_rejected():
    controller = AltitudeHold()
    controller.update(sample(), now_s=0)
    with pytest.raises(AltitudeTelemetryError): controller.update(sample(), now_s=.02)
    with pytest.raises(AltitudeTelemetryError): controller.update(sample(.3, .02), now_s=.02)


def test_composition_changes_only_throttle_and_reset_discards_old_history():
    controller = AltitudeHold()
    command = np.array([1900., 1550., 1600., 1350.])
    result = controller.apply(command, sample(.8), now_s=0)
    np.testing.assert_array_equal(result[1:], command[1:])
    assert command[0] == 1900 and result[0] != command[0]
    controller.reset()
    assert controller.update(sample(), now_s=0).vertical_speed_mps == 0


def test_controller_is_bounded_and_recovers_after_sustained_error():
    controller = AltitudeHold()
    values = [controller.update(sample(.4, i*.02), now_s=i*.02).throttle_pwm for i in range(200)]
    assert min(values) >= 1100 and max(values) <= 1400
    for i in range(200, 401):
        height = min(1.2, .4 + (i-200)*.01)
        result = controller.update(sample(height, i*.02), now_s=i*.02)
    assert result.throttle_pwm < 1213


def test_adapter_uses_only_sensor_fields():
    from types import SimpleNamespace
    from simulation.altitude_control import sample_from_observation
    obs = SimpleNamespace(laser_distance=.982, laser_valid=True,
                          attitude_euler=[0, 0, 0], sim_time=.02)
    result = sample_from_observation(obs)
    assert result == sample(time=.02)


def test_fault_remains_latched_until_explicit_reset():
    controller = AltitudeHold()
    with pytest.raises(AltitudeTelemetryError): controller.update(sample(valid=False), now_s=0)
    with pytest.raises(AltitudeTelemetryError, match='Latched'):
        controller.update(sample(time=.02), now_s=.02)
    controller.reset()
    assert controller.update(sample(time=.02), now_s=.02).height_m == pytest.approx(1.)


def test_rollout_reports_separate_control_contract_and_sensor_fault_stops_before_step():
    from types import SimpleNamespace
    from optimizer.rollout import simulate_policy_rollout, BENCHMARK_VERSION
    from tests.test_evaluation_contract import ScriptedEnv, RecordingPolicy
    env = ScriptedEnv()
    env.get_isaac_obs = lambda: SimpleNamespace(laser_distance=.982, laser_valid=False,
                                               attitude_euler=[0, 0, 0], sim_time=0.)
    _, metrics = simulate_policy_rollout(RecordingPolicy(), env=env, eval_steps=10,
                                         altitude_hold=AltitudeHold())
    assert metrics['fatal_failure'] and not metrics['feasible'] and env.steps == 0
    assert metrics['benchmark_version'] == BENCHMARK_VERSION
    assert metrics['altitude_control']['throttle_owner'] == 'altitude_controller'


def test_real_altitude_loop_holds_height_during_sustained_tilt_without_world_state_feedback():
    from simulation.drone_env import DroneSimulationEnv
    from optimizer.rollout import simulate_policy_rollout
    class TiltPolicy:
        def step_np(self, *args, **kwargs): return np.array([1800., 1500., 1570., 1500.])
    env = DroneSimulationEnv(engine='standalone', headless=True)
    env.boxes, env.cylinders = [], []
    env.room.x_min, env.room.x_max = -100, 100
    _, metrics = simulate_policy_rollout(TiltPolicy(), env=env, eval_steps=500,
                                         altitude_hold=AltitudeHold(), seed=42)
    assert metrics['feasible']
    assert abs(env.physics.pos[2]-1) < .08
    assert metrics['altitude_tracking']['max_abs_error_m'] < .12
