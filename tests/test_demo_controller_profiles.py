import numpy as np
import pytest
from demo.runner import DemoRunner
from demo.examples import recompute
from evaluate.phased_maneuver import install,SOURCE

def test_profile_matches_experiment_and_survives_reset():
    a=DemoRunner(controller_mode='phased_v2',scene='room',start=(2.8,0,1))
    b=DemoRunner(config=SOURCE/'runtime_config.json',scene='room',start=(2.8,0,1))
    try:
        b.system=install(b.system)
        for r in (a,b): r.resume();r.event(dict(type='mode',mode='AUTO'))
        saw_maneuver=False
        for _ in range(1800):
            a.advance();b.advance()
            np.testing.assert_array_equal(a.env.physics.pos,b.env.physics.pos)
            assert a.rows[-1]['plant_input_pwm']==b.rows[-1]['plant_input_pwm']
            saw_maneuver |= a.system.runtime.coordinator.last_decision.get('phased_phase')=='TURN'
        assert saw_maneuver
        a.event(dict(type='mode',mode='MANUAL'));a.advance(5)
        assert a.rows[-1]['plant_source']!='auto'
        a.reset()
        assert a.system.runtime.coordinator.phase=='CRUISE'
        assert a.manifest()['controller_mode']=='phased_v2'
    finally:a.env.close();b.env.close()

def test_profile_recording_recomputes_with_same_controller():
    r=DemoRunner(controller_mode='phased_v2',scene='room',max_duration_s=2,run_id='profile-test')
    try:
        r.resume();r.event(dict(type='mode',mode='AUTO'));r.advance(300)
        data=r.recording();again=recompute(data)
        assert data['frames']==again['frames']
        assert again['manifest']['controller_mode']=='phased_v2'
    finally:r.env.close()

def test_profiles_reject_ambiguous_selection():
    with pytest.raises(ValueError):DemoRunner(controller_mode='unknown')
    with pytest.raises(ValueError):DemoRunner(controller_mode='phased_v2',config='arbitrary.json')


def test_named_candidate_uses_canonical_bundle_and_keeps_historical_replay_identity():
    from contextlib import closing
    from demo.profiles import CANDIDATE_A_CONFIG, HISTORICAL_CANDIDATE_A_CONFIG
    from edge.runtime import sha256

    with closing(DemoRunner(controller_mode='candidate_a')) as live:
        assert live.config_path == CANDIDATE_A_CONFIG
        assert live.manifest()['controller_role'] == 'fixed-xy-neural-control'
        assert live.base_config.bundle_dir.endswith('k306_adaptive_tau_v1')

    if HISTORICAL_CANDIDATE_A_CONFIG.is_file():
        historical_sha = sha256(HISTORICAL_CANDIDATE_A_CONFIG)
        with closing(DemoRunner(controller_mode='candidate_a', recorded_config_sha256=historical_sha)) as old:
            assert old.manifest()['config_sha256'] == historical_sha


@pytest.mark.parametrize('profile', ['phased_recovery', 'phased_escape', 'phased_side_guard'])
def test_phase_profile_and_yaw_evaluator_use_the_same_coordinator(profile):
    from demo.yaw_experiment import install as install_yaw

    a = DemoRunner(controller_mode=profile)
    b = DemoRunner(controller_mode='candidate_a')
    try:
        b.system = install_yaw(b.system, profile=profile)
        assert a.system.manifest['resolved']['coordinator'] == b.system.manifest['resolved']['coordinator']
        assert a.system.manifest['executed_modules'] == b.system.manifest['executed_modules']
        for runner in (a, b):
            runner.resume()
            runner.event(dict(type='mode', mode='AUTO'))
            runner.advance(150)
        np.testing.assert_array_equal(a.env.physics.pos, b.env.physics.pos)
        assert a.rows[-1]['plant_input_pwm'] == b.rows[-1]['plant_input_pwm']
    finally:
        a.close()
        b.close()


def test_threat_gated_pitch_is_isolated_and_stop_has_priority():
    from simulation.threat_gated_xy import ThreatGatedXYCoordinator
    from simulation.xy_coordinator import XYCoordinator

    network = np.array([1213., 1520., 1470., 1480.])
    for distance, flow, stop, enabled in ((.84, 0., False, True), (.85, 0., False, False),
                                         (.84, -.005, False, False), (.84, 0., True, False)):
        native = XYCoordinator(anti_rollback=True, turn_latch=True, stop_hover=True)
        overlay = ThreatGatedXYCoordinator(anti_rollback=True, turn_latch=True, stop_hover=True)
        args = dict(flow_xy=[flow, 0.], tof_m=np.full(64, distance), dt=.02, stop_requested=stop)
        expected = native.apply(network, **args)
        if enabled:
            expected[2] = network[2]
        np.testing.assert_array_equal(overlay.apply(network, **args), expected)
