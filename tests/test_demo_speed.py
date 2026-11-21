"""Live Standalone speed multiplier: only the Pacer's wall-time schedule changes."""
import hashlib
import json

import numpy as np
import pytest

from demo.recording import plain
from demo.runner import DemoRunner
from demo.server import DemoService, Pacer

SCENE = dict(controller_mode='phased_failed_exit', scene='random_room', seed=3934218795, start=(-.7, 0., 1.), yaw=0.,
             scene_options=dict(obstacle_density_pct=20, layout_style='clustered'), max_duration_s=180.)
STEPS = 3000  # 12 s of flight with AUTO from the first step


def paced_trajectory(speed):
    r = DemoRunner(run_id='speed', **SCENE)
    p = Pacer(r, speed)
    r.resume()
    r.event(dict(type='mode', mode='AUTO'))
    t, batches = 0., []
    p.poll(t)
    while r.step_index < STEPS:
        before = r.step_index
        t += .02
        p.poll(t)
        batches.append(r.step_index-before)
    rows = r.rows[:STEPS]
    digest = hashlib.sha256(json.dumps(plain(rows), sort_keys=True, allow_nan=False).encode()).hexdigest()
    r.close()
    return digest, rows, batches, r.step_dt


def test_trajectory_is_bitwise_identical_at_1x_and_4x():
    d1, rows1, b1, dt = paced_trajectory(1)
    d4, rows4, b4, _ = paced_trajectory(4)
    assert len(rows1) == len(rows4) == STEPS
    assert max(b1) == round(.02/dt) and max(b4) == 4*round(.02/dt)  # 4x really ran 4x the steps per poll
    for a, b in zip(rows1, rows4):
        assert a['time_s'] == b['time_s'] and a['position_m'] == b['position_m']
    assert d1 == d4


@pytest.mark.parametrize('speed', [1, 2, 4])
def test_per_poll_step_limit_scales_with_speed_and_physics_dt_is_fixed(speed):
    r = DemoRunner()
    r.resume()
    p = Pacer(r, speed)
    p.poll(0.)
    p.poll(.2)  # a long (not idle) gap: no catch-up beyond the scaled per-poll limit
    assert r.step_index == speed*round(.02/r.step_dt)
    assert r.step_dt == DemoRunner.step_dt
    n = r.step_index
    p.poll(.2+.02)
    assert r.step_index-n <= speed*round(.02/r.step_dt)
    p.poll(2.)  # idle gap still drops wall time
    assert r.step_index-n <= speed*round(.02/r.step_dt)
    r.close()


def test_invalid_speed_rejected():
    r = DemoRunner()
    for bad in (0, 3, 8, True, '2', None):
        with pytest.raises(ValueError):
            Pacer(r, bad)
    s = DemoService()
    with pytest.raises(ValueError):
        s.set_speed(3)
    r.close()


def test_service_speed_applies_to_standalone_and_persists_across_runs():
    s = DemoService()
    s.new_run()
    assert s.set_speed(4) == 4 and s.pacer.speed == 4
    s.new_run()
    assert s.pacer.speed == 4
    s.runner.close()


def test_speed_multiplier_unavailable_for_sitl(monkeypatch):
    import demo.sitl as sitl

    class FakeSitl(DemoRunner):
        def __init__(self, *, controller_profile=None, reference_profile=None, **kw):
            super().__init__(**kw)

    monkeypatch.setattr(sitl, 'SitlRunner', FakeSitl)
    s = DemoService(sitl_enabled=True)
    s.new_run()
    s.set_speed(4)
    s.new_run(backend='sitl', scene='random_room', seed=1)
    assert s.speed == 1 and s.pacer.speed == 1
    with pytest.raises(ValueError, match='Standalone'):
        s.set_speed(2)
    assert s.set_speed(1) == 1
    s.runner.close()
