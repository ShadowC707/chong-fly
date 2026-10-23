import numpy as np
import pytest


def test_starts_are_fixed_valid_and_seed_changes_sensors_not_start_state():
    from evaluate.validate_xy_starts import STARTS, seed_probe
    from simulation.xy_scenes import DensityStartArena
    assert len(STARTS)==8
    for s in STARTS:
        env=DensityStartArena('arena_dense',start_pos=s['position_m'],start_yaw=s['yaw_rad'])
        try:
            a=env.reset(seed=41)
            np.testing.assert_allclose(env.physics.pos,s['position_m'])
            assert env.physics.quaternion_to_euler(env.physics.quat)[2]==pytest.approx(s['yaw_rad'],abs=1e-6)
            np.testing.assert_array_equal(env.physics.vel,[0,0,0])
            np.testing.assert_array_equal(a,env.reset(seed=41))
            assert env.specification()['initial_clearance_m']>=.2
        finally: env.close()
    p=seed_probe()
    assert p['same_seed_exact'] and p['different_seed_flow_differs']
    assert p['different_seed_tof_equal'] and p['same_physical_state']


def test_hazard_episodes_count_duration_and_longest_without_merging_gaps():
    from evaluate.validate_xy_starts import proximity_metrics
    rows=[dict(time_s=t,clearance_m=c) for t,c in [(0,None),(.004,.04),(.008,.04),(.012,.2),(.016,.08),(.020,.03)]]
    r=proximity_metrics(rows)
    assert r['below_005']['total_s']==pytest.approx(.012)
    assert r['below_005']['episodes']==2
    assert r['below_005']['longest_s']==pytest.approx(.008)
    assert r['below_010']['total_s']==pytest.approx(.016)
    assert r['below_010']['episodes']==2
    assert r['below_010']['longest_s']==pytest.approx(.008)


def test_admission_does_not_pass_collision_truncated_stop_or_long_close_approach():
    from evaluate.validate_xy_starts import admission
    row=dict(crashed=False,complete=True,min_clearance_m=.12,stop={'speed_target_passed':True},
        proximity={'below_010':{'longest_s':0.}})
    assert admission(row)['passed']
    for patch in [dict(crashed=True),dict(complete=False),dict(stop=None),
                  dict(min_clearance_m=.049),dict(proximity={'below_010':{'longest_s':.504}})]:
        assert not admission(dict(row,**patch))['passed']


def test_custom_start_rejected_for_non_density_scene():
    from evaluate.watch_mvp import make_environment
    with pytest.raises(ValueError,match='density'):
        make_environment('clear',arena_start={'start_pos':[0,0,1],'start_yaw':0.})


def test_original_density_start_remains_unchanged():
    from simulation.xy_scenes import DensityArena, DensityStartArena
    a,b=DensityArena('arena_dense'),DensityStartArena('arena_dense',start_pos=[0,0,1],start_yaw=0.)
    try:
        np.testing.assert_array_equal(a.reset(seed=7),b.reset(seed=7))
        for _ in range(3):
            u=[1213.,1500.,1527.,1500.]
            np.testing.assert_array_equal(a.step(u)[0],b.step(u)[0])
    finally: a.close(); b.close()
