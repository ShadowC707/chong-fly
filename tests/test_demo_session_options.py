import pytest

from demo.runner import DemoRunner
from simulation.drone_env import BoxObstacle


def test_new_drone_radius_matches_visible_body_and_hits_before_old_envelope():
    r=DemoRunner(scene='room',start=(.7999,0,1))
    assert r.env.collision_radius_m==.2
    assert r.manifest()['geometry']['drone_visual']['collision_radius_m']==.2
    r.env.boxes=[BoxObstacle(1,1.2,-1,1,0,2.5)]
    r.env.physics.vel[:]=[.4,0,0]
    r.resume(); r.advance(5)
    assert r.terminal and r.pause_reason=='collision' and r.step_index==1


def test_duration_limit_can_be_extended_during_live_run_and_recomputes():
    from demo.examples import recompute
    r=DemoRunner(max_duration_s=1)
    r.resume(); r.advance(240)
    r.event({'type':'duration','seconds':2})
    r.advance(260)
    assert r.terminal and r.time_s==2
    assert r.pause_reason=='duration_limit'
    assert r.snapshot()['max_duration_s']==2
    data=r.recording()
    assert data['manifest']['max_duration_s']==1
    assert recompute(data)['frames']==data['frames']


def test_live_duration_rejects_already_elapsed_limit_without_changing_run():
    r=DemoRunner(max_duration_s=3); r.resume(); r.advance(300)
    with pytest.raises(ValueError,match='elapsed'):
        r.event({'type':'duration','seconds':1})
    assert not r.paused and r.snapshot()['max_duration_s']==3


def test_live_scene_runs_past_sixty_seconds_until_configured_limit():
    r=DemoRunner(max_duration_s=61)
    r.resume(); r.advance(15001)
    assert r.time_s>60 and not r.terminal
    r.advance(249)
    assert r.time_s==61 and r.terminal and r.pause_reason=='duration_limit'


@pytest.mark.parametrize('duration',[0,601,True,'60',float('nan')])
def test_invalid_duration_is_rejected(duration):
    with pytest.raises(ValueError,match='duration'):
        DemoRunner(max_duration_s=duration)


def test_legacy_recording_without_radius_or_duration_recomputes_old_envelope():
    from demo.examples import recompute
    r=DemoRunner(collision_radius_m=.1,max_duration_s=60)
    r.resume(); r.advance(10)
    data=r.recording()
    data['manifest'].pop('max_duration_s')
    rebuilt=recompute(data)
    assert rebuilt['manifest']['geometry']['collision_radius_m']==.1
    assert rebuilt['frames']==data['frames']


def test_larger_drone_density_passages_allow_diameter_with_margin():
    r=DemoRunner(scene='random_room',scene_options={'obstacle_density_pct':20})
    assert r.manifest()['geometry_spec']['minimum_random_surface_gap_m']==pytest.approx(.48)
    from demo.examples import recompute
    r.resume(); r.advance(15)
    assert recompute(r.recording())['manifest']['geometry']==r.manifest()['geometry']
