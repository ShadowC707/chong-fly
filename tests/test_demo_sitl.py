from dataclasses import asdict
from types import SimpleNamespace

import pytest

from demo.scenes import make_environment
from demo.server import DemoService
from demo.sitl import SitlRunner, scene_for_run
from edge.session_receiver import SessionReceiver


def test_room_primitives_match_standalone_generator_for_same_seed_and_start():
    options = dict(obstacle_density_pct=18, layout_style='clustered')
    start = (.4, -.6, 1.)
    spec, details = scene_for_run('random_room', 20261022, start, -.7, options)
    env = make_environment('random_room', seed=20261022, start=start, options=options, collision_radius_m=.2)
    try:
        assert spec['room'] == asdict(env.room)
        assert spec['boxes'] == [asdict(b) for b in env.boxes]
        assert spec['cylinders'] == [asdict(c) for c in env.cylinders]
        assert details == env.demo_geometry_spec
        assert spec['start'] == list(start)
        assert spec['yaw'] == -.7
    finally:
        env.close()


def test_reference_room_requires_the_reference_pose():
    with pytest.raises(ValueError, match='0°'):
        scene_for_run('long_arena', 1, (0., 0., 1.), .5, {})


def test_unavailable_sitl_does_not_replace_or_start_standalone():
    service = DemoService()
    try:
        existing = service.new_run()
        with pytest.raises(ValueError, match='Docker Desktop'):
            service.new_run(backend='sitl')
        assert service.runner is existing and existing.paused
    finally:
        if service.runner: service.runner.close()


def test_user_pause_freezes_auto_but_heartbeat_loss_revokes_it():
    # The lifecycle does not require a Bullet or firmware process for this test.
    runner = object.__new__(SitlRunner)
    runner.view = SimpleNamespace(tick=2000, time_s=.5)
    runner.terminal, runner.paused, runner.revision = False, False, 1
    runner.request, runner.stop_requested = 'AUTO', True
    runner.manual = SimpleNamespace(clear=lambda: None)
    runner.wall_started, runner.wall_active = None, 0.
    runner.events, runner.input_schedule = [], []
    runner.receiver = SessionReceiver()
    runner.receiver.set_mode('AUTO', now_s=runner.time_s)
    runner.pause()
    assert runner.paused and runner.time_s == .5
    assert runner.stop_requested and not runner.receiver.gate.tripped

    service = DemoService(sitl_enabled=True)
    service.runner = runner
    service.heartbeat('owner', 0.)
    with pytest.raises(ValueError, match='Another browser'):
        service.heartbeat('observer', .5)
    service.poll(2.)
    assert runner.paused and runner.request == 'MANUAL'
    assert runner.receiver.reason == 'authority_revoked' and runner.receiver.gate.tripped
    service.heartbeat('observer', 2.)


def test_sitl_duration_aligns_to_control_boundaries():
    SitlRunner._validate_duration(2.04)
    with pytest.raises(ValueError, match='20 ms'):
        SitlRunner._validate_duration(2.03)
