import numpy as np
import pytest
from types import SimpleNamespace
from simulation.xy_coordinator import XYCoordinator
from evaluate.user_room_repair import braking_only_config, repeated_motion_windows, configure_view
from tests.test_edge_runtime import exported


def test_braking_revision_preserves_forward_brake_without_changing_latch():
    old=XYCoordinator(anti_rollback=True,turn_latch=True)
    cfg=braking_only_config(old.config)
    assert cfg.acquire_yaw_pwm==old.config.acquire_yaw_pwm
    assert cfg.threat_m==old.config.threat_m and cfg.clear_m==old.config.clear_m
    for flow in (.006,.019,.03,.06):
        c=XYCoordinator(anti_rollback=True,turn_latch=True,config=cfg)
        out=c.apply([1213,1502,1472,1489],flow_xy=[flow,0],tof_m=np.full(64,.7),dt=.02)
        np.testing.assert_array_equal(out,[1213,1502,1472,1489])
    c=XYCoordinator(anti_rollback=True,config=cfg)
    assert c.apply([1213,1500,1472,1500],flow_xy=[-.03,0],tof_m=np.ones(64),dt=.02)[2]==1500


def test_loop_metric_distinguishes_still_hover_and_back_and_forth_from_progress():
    t=np.arange(0,60.001,.02)
    p=np.column_stack([.2*np.sin(t),np.zeros_like(t)])
    assert repeated_motion_windows(t,p)
    assert not repeated_motion_windows(t,np.zeros_like(p))
    assert not repeated_motion_windows(t,np.column_stack([t*.2,np.zeros_like(t)]))
    assert not repeated_motion_windows(t[:100],p[:100])


def test_navigation_admission_rejects_silent_wait_and_explicit_block():
    from evaluate.user_room_repair import waiting_metrics
    def rows(phase,n):return [dict(decision=dict(phased_phase=phase)) for _ in range(n)]
    assert waiting_metrics(rows('WAIT_CLEAR',251))['unresolved_wait']
    assert not waiting_metrics(rows('WAIT_CLEAR',250))['unresolved_wait']
    assert not waiting_metrics(rows('STOP',1000))['unresolved_wait']
    assert waiting_metrics([dict(decision=dict(navigation_blocked=True))])['unresolved_wait']
    assert not waiting_metrics(rows('WAIT_CLEAR',200)+rows('CRUISE',1)+rows('WAIT_CLEAR',200))['unresolved_wait']


@pytest.mark.parametrize('variant',['xy_brake','smooth_retry'])
def test_experiment_rebuilds_manifest_and_receiver_system_before_first_tick(exported,tmp_path,variant):
    from edge.system import build_system
    from tests.test_system_assembly import config
    from demo.controllers import install
    from demo.yaw_experiment import install as smooth
    _,bundle=exported
    system=smooth(install(build_system(config(bundle,sensor_mode='multi_rate_v1',handover_mode='freeze_sync_v1',coordinator_mode='both',stop_hover=True))))
    before=system.manifest_sha256
    view=SimpleNamespace(tick=0,observed=None,system=system,integrated=SimpleNamespace(system=system),manifest={'system':system.manifest})
    configure_view(view,variant,tmp_path)
    assert view.system is view.integrated.system
    assert view.system.runtime is system.runtime
    if variant=='xy_brake':
        assert type(view.system.runtime.coordinator) is XYCoordinator
        assert view.manifest['system']['resolved']['coordinator']['config']['forward_flow']==.005
    else:
        from demo.yaw_experiment import YawHandoverSystem
        from simulation.phased_recovery import RecoveryPhasedCoordinator
        assert type(view.system.runtime.coordinator) is RecoveryPhasedCoordinator
        assert isinstance(view.system,YawHandoverSystem)
        assert view.manifest['system']['resolved']['coordinator']['config']['forward_flow']==.06
        assert view.manifest['system']['resolved']['coordinator']['max_retries']==3
    assert view.system.manifest_sha256!=before
    assert view.manifest['system']['resolved']['coordinator']['config']['acquire_yaw_pwm']==20.
    view.tick=80
    with pytest.raises(ValueError,match='frozen'):configure_view(view,'xy_control',tmp_path)
