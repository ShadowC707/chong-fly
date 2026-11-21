"""Step 8: live display summary (who controls, network share, memory points) and the SITL demo profile."""
from contextlib import closing

import numpy as np
import pytest

from demo.live_view import controller_of, running_share


def frame(network, coordinated, *, source='auto', stop=False, request='AUTO'):
    return dict(plant_source=source, stop_requested=stop, request=request,
                trace=dict(network_pwm=network, coordinated_pwm=coordinated, tick_time_s=0.))


N = [1400., 1500., 1550., 1480.]


def test_controller_attribution():
    assert controller_of(frame(N, [1400., 1505., 1545., 1480.]), {})['kind'] == 'network'      # within 10 PWM
    stage = dict(stages=[dict(stage='brake_no_rollback', after=N, reason='not_braking'),
                         dict(stage='side_veto', after=[1400., 1500., 1550., 1500.], reason='side_obstacle_yaw')])
    c = controller_of(frame(N, [1400., 1500., 1550., 1500.]), stage)
    assert c['kind'] == 'stage' and c['stage'] == 'side_veto' and c['label'] == 'Бічне вето'
    c = controller_of(frame(N, [1400., 1500., 1500., 1600.]), dict(phased_phase='TURN'))
    assert c['kind'] == 'phase' and 'поворот' in c['label']
    assert controller_of(frame(N, N, stop=True), {})['kind'] == 'stop'
    assert controller_of(frame(N, N, source='manual', request='MANUAL'), {})['kind'] == 'manual'
    assert controller_of(None, None)['kind'] == 'idle'


def test_running_share_is_incremental_and_counts_each_control_tick_once():
    class R:
        rows = []
    r = R()
    for i in range(10):  # 10 control ticks, each repeated over 5 physics rows; ticks 3 and 7 deviate
        coord = [1400., 1500., 1600., 1480.] if i in (3, 7) else N
        for _ in range(5):
            row = frame(N, coord)
            row['trace']['tick_time_s'] = i*.02
            r.rows.append(row)
        share, ticks = running_share(r)
    assert ticks == 10 and share == pytest.approx(.8)
    r.rows.append(dict(frame(N, N, stop=True)))  # STOP ticks are not analysed
    assert running_share(r) == (pytest.approx(.8), 10)


@pytest.mark.parametrize('profile, memory', [('phased_side_guard', False), ('pipeline_v1', True)])
def test_snapshot_carries_the_live_view(profile, memory):
    from demo.runner import DemoRunner
    with closing(DemoRunner(controller_mode=profile, scene='random_room', seed=20261021,
                            scene_options=dict(obstacle_density_pct=20, layout_style='uniform'),
                            max_duration_s=4)) as runner:
        runner.resume()
        runner.event(dict(type='mode', mode='AUTO'))
        while not runner.terminal:
            runner.advance(25)
        snapshot = runner.snapshot()
        live, frame = snapshot['live'], snapshot['frame']
    assert live['tof_fov_deg'] == 45. and live['share_tolerance_pwm'] == 10.
    assert live['network_share'] is not None and 0. <= live['network_share'] <= 1. and live['share_ticks'] > 100
    assert live['controller']['label']
    assert (live['memory_points'] is not None) == memory
    if memory:
        assert len(live['memory_points']) <= 400 and all(len(p) == 3 for p in live['memory_points'])
        assert live['memory_frame'] == 'controller_odometry' and len(live['estimated_position']) == 3
        # the controller's own position estimate stays close to the true pose over a short flight
        assert np.linalg.norm(np.subtract(live['estimated_position'], frame['position_m'])[:2]) < .1
    else:
        assert live['estimated_position'] is None and live['memory_frame'] is None


def test_sitl_demo_session_flies_phased_failed_exit(monkeypatch):
    import demo.sitl
    from demo.server import DemoService, SITL_DEMO_PROFILE
    captured = {}

    class Fake:
        terminal, paused, request = True, True, 'MANUAL'

        def __init__(self, **kwargs):
            captured.update(kwargs)

        def close(self):
            pass

    monkeypatch.setattr(demo.sitl, 'SitlRunner', Fake)
    service = DemoService(sitl_enabled=True)
    service.new_run(backend='sitl', scene='random_room', seed=1)
    assert SITL_DEMO_PROFILE == 'phased_failed_exit'
    assert captured['controller_profile'] == 'phased_failed_exit'
    assert captured['reference_profile'] == demo.sitl.REITERATION_PROFILE
    service.new_run(backend='sitl', scene='random_room', seed=1, controller_mode='phased_failed_exit')
    with pytest.raises(ValueError):
        service.new_run(backend='sitl', scene='random_room', seed=1, controller_mode='phased_side_guard')
    with pytest.raises(ValueError):
        service.new_run(backend='sitl', scene='random_room', seed=1, controller_mode='pipeline_v1')
