"""Step 7: the SITL loop can fly phased_side_guard; the default chain stays the historical one."""
from pathlib import Path

import pytest

from demo.profiles import CANDIDATE_A_CONFIG
from simulation.sitl_flight import SITL_PROFILES, build_controller, flight_steps

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'data/scenarios/reiteration_v1/sitl'


def test_default_chain_is_unchanged():
    from simulation.phased_yaw import SmoothPhasedCoordinator
    from demo.yaw_experiment import YawHandoverSystem
    system = build_controller(CANDIDATE_A_CONFIG, yaw_shaping=True, integration_profile='multi_rate')
    assert type(system) is YawHandoverSystem and type(system.runtime.coordinator) is SmoothPhasedCoordinator
    assert system.manifest['experiment']['profile'] == 'phased_smooth'


@pytest.mark.parametrize('profile, cls', [('phased_side_guard', 'simulation.side_clearance.SideClearanceCoordinator'),
                                          ('phased_failed_exit', 'simulation.failed_exit.FailedExitCoordinator')])
def test_sitl_builds_phased_profile_like_standalone(profile, cls):
    import importlib
    from demo.yaw_experiment import YawHandoverSystem
    from demo.runner import DemoRunner
    module, name = cls.rsplit('.', 1)
    coordinator = getattr(importlib.import_module(module), name)
    system = build_controller(CANDIDATE_A_CONFIG, yaw_shaping=True, integration_profile='multi_rate',
                              controller_profile=profile)
    assert type(system) is YawHandoverSystem and type(system.runtime.coordinator) is coordinator
    assert system.manifest['experiment']['profile'] == profile
    standalone = DemoRunner(controller_mode=profile, scene='room', seed=3, max_duration_s=1)
    try:
        ref = standalone.system
        assert type(ref) is type(system) and type(ref.runtime.coordinator) is type(system.runtime.coordinator)
        assert ref.runtime.coordinator.phased_config == system.runtime.coordinator.phased_config
        assert ref.manifest['experiment']['overlay'] == system.manifest['experiment']['overlay']
    finally:
        standalone.close()


@pytest.mark.parametrize('profile', ['pipeline_v1', 'pipeline_v1_r7', 'candidate_a', 'phased_swath'])
def test_profiles_without_a_sitl_path_are_refused(profile):
    assert profile not in SITL_PROFILES
    with pytest.raises(ValueError):
        build_controller(CANDIDATE_A_CONFIG, yaw_shaping=True, integration_profile='multi_rate',
                         controller_profile=profile)


def test_a_coordinator_profile_needs_the_yaw_handover_path():
    with pytest.raises(ValueError):
        build_controller(CANDIDATE_A_CONFIG, yaw_shaping=False, controller_profile='phased_side_guard')


@pytest.mark.skipif(not (SOURCE/'report.json').is_file(), reason='reiteration SITL source missing')
def test_flight_manifest_records_the_profile(tmp_path):
    import json
    steps = flight_steps(tmp_path/'flight', SOURCE, 'clear', 'legacy', integration_profile='multi_rate',
                         yaw_shaping=True, runtime_config=CANDIDATE_A_CONFIG, controller_profile='phased_side_guard',
                         horizon_s=.04, stop_at_s=None)
    view = next(steps)
    steps.close()
    manifest = json.loads((tmp_path/'flight/manifest.json').read_text())
    assert manifest['controller_profile'] == 'phased_side_guard'
    assert manifest['system']['experiment']['profile'] == 'phased_side_guard'
    assert type(view.system.runtime.coordinator).__name__ == 'SideClearanceCoordinator'
