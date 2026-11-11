import numpy as np
import pytest
from simulation.phased_coordinator import PhasedCoordinator
from simulation.xy_coordinator import XYCoordinator


def tick(c, flow=(0.,0.), distance=.7, stop=False, ranges=None):
    return c.apply([1213.,1500.,1530.,1500.],flow_xy=flow,
        tof_m=np.full(64,distance) if ranges is None else ranges,dt=.02,stop_requested=stop)


def test_disabled_exact_sequence():
    a=XYCoordinator(anti_rollback=True,turn_latch=True,stop_hover=True)
    b=PhasedCoordinator(enabled=False,anti_rollback=True,turn_latch=True,stop_hover=True)
    rng=np.random.default_rng(42)
    for i in range(100):
        args=dict(flow=rng.uniform(-.1,.2,2),distance=rng.uniform(.3,2),stop=i>80)
        np.testing.assert_array_equal(tick(a,**args),tick(b,**args))


def test_brakes_both_axes_even_when_antirollback_would_remove_brake():
    c=PhasedCoordinator(anti_rollback=True,turn_latch=True)
    out=tick(c,flow=(.015,.03))
    assert out[2]<1500 and out[1]>1500 and out[3]==1500
    assert out[0]==1213 and c.phase=='BRAKE'


def test_lateral_motion_prevents_turn():
    c=PhasedCoordinator()
    for _ in range(40): tick(c,flow=(0.,.06))
    assert c.phase=='BRAKE'


def test_direction_and_recovery_require_whole_grid():
    c=PhasedCoordinator()
    image=np.full((8,8),2.);image[:,:4]=.7
    for _ in range(10): out=tick(c,ranges=image)
    assert c.phase=='TURN' and out[3]==1640
    image[:]=2.;image[0,0]=.9
    for _ in range(20): tick(c,ranges=image)
    assert c.phase=='TURN'
    for _ in range(12): tick(c,distance=2.)
    assert c.phase=='RECOVER'
    for _ in range(30): out=tick(c,distance=2.)
    assert c.phase=='CRUISE' and out[2]==1530 and c.completed==1


def test_timeout_no_repeated_spin_until_clear():
    c=PhasedCoordinator()
    for _ in range(500): out=tick(c)
    assert c.phase=='FAILED' and c.failures==1 and out[3]==1500
    assert out[2]==1500
    c.reset()
    assert c.phase=='CRUISE' and c.failures==0


def test_stop_and_critical_delegate_to_original():
    a=XYCoordinator(stop_hover=True); b=PhasedCoordinator(stop_hover=True)
    for _ in range(30):
        np.testing.assert_array_equal(tick(a,stop=True,distance=1.2),tick(b,stop=True,distance=1.2))
    a.reset();b.reset()
    np.testing.assert_array_equal(tick(a,distance=.3),tick(b,distance=.3))
    assert b.phase=='CRITICAL'


def test_invalid_cancels_phase_and_raises():
    c=PhasedCoordinator();tick(c)
    with pytest.raises(ValueError): tick(c,flow=(np.nan,0))
    assert c.phase=='CRUISE'


def test_tie_alternates_and_reset_is_reproducible():
    c=PhasedCoordinator(); tick(c)
    assert c.direction==1
    for _ in range(100): tick(c,distance=2.)
    tick(c)
    assert c.direction==-1
    c.reset();tick(c)
    assert c.direction==1


def test_experiment_contract_overrides_priority_without_duplicate_keys():
    c=PhasedCoordinator(stop_hover=True)
    assert c.contract()['phased_enabled'] is True
    assert 'external authority' in c.contract()['priority']


def test_stop_cancels_active_turn_and_reset_clears_history():
    c=PhasedCoordinator(stop_hover=True)
    for _ in range(10): tick(c)
    assert c.phase=='TURN'
    assert tick(c,stop=True,distance=1.2)[3]==1500
    assert c.phase=='STOP' and c.last_decision['phased_event']=='cancelled'
    c.reset()
    assert c.phase=='CRUISE' and c.direction==0 and c.next_tie==1


def test_fast_flow_timeout_is_failure_not_turn():
    c=PhasedCoordinator()
    for _ in range(110): out=tick(c,flow=(.2,.1))
    assert c.phase=='FAILED' and c.failures==1
    assert out[3]==1500 and out[2]<1500 and out[1]>1500


def test_new_threat_interrupts_recovery_before_forward_cruise():
    c=PhasedCoordinator()
    for _ in range(10): tick(c)
    for _ in range(11): tick(c,distance=2.)
    assert c.phase=='RECOVER'
    out=tick(c,distance=.7)
    assert c.phase=='BRAKE' and out[2]==1500

def test_real_assembly_manual_revokes_and_reentry_resets_overlay():
    from edge.pinned_system import build_pinned_system
    from edge.system import RunConfig
    from evaluate.phased_maneuver import install, SOURCE
    from tests.test_flight_bridge import frame
    system=install(build_pinned_system(RunConfig.from_json(SOURCE/'runtime_config.json')))
    c=system.runtime.coordinator
    for _ in range(10): tick(c)
    assert c.phase=='TURN'
    manual=[1213.,1500.,1500.,1500.]
    result=system.tick(None,now_s=0.,request='MANUAL',manual_pwm=manual)
    assert result['lease'] is None and result['trace'].issued_pwm is None
    result=system.tick(frame(1),now_s=.02,request='SHADOW',manual_pwm=manual)
    assert c.phase=='CRUISE' and result['lease'] is None
    result=system.tick(None,now_s=.04,request='AUTO',manual_pwm=manual)
    assert result['lease'] is None and result['state']=='FAULT'


def test_critical_oscillations_preserve_episode_direction_and_budget():
    c=PhasedCoordinator()
    for _ in range(10): tick(c)
    sign=c.direction; elapsed=c.episode_elapsed
    for i in range(60):
        out=tick(c,distance=.46 if i%2 else .81)
        assert c.phase=='TURN' and c.direction==sign
        assert out[3]==1500+sign*140
        assert c.episode_elapsed>elapsed
        elapsed=c.episode_elapsed
    for i in range(500): tick(c,distance=.46 if i%2 else .81)
    assert c.phase=='FAILED' and c.failures==1 and not c.armed


def test_failed_episode_cannot_rearm_on_short_clear_flicker():
    c=PhasedCoordinator()
    for _ in range(250): tick(c)
    for _ in range(20): tick(c,distance=2.)
    tick(c,distance=.7)
    assert c.phase=='FAILED' and c.failures==1
    for _ in range(100): tick(c,distance=2.)
    tick(c,distance=.7)
    assert c.phase=='BRAKE' and c.episode_elapsed<.1


def test_stop_cannot_reset_budget_into_immediate_new_episode():
    c=PhasedCoordinator(stop_hover=True)
    for _ in range(10): tick(c)
    tick(c,stop=True,distance=.7)
    tick(c,distance=.7)
    assert c.phase=='WAIT_CLEAR' and not c.armed

def test_total_episode_budget_survives_recovery_interruptions():
    from simulation.phased_coordinator import PhasedConfig
    c=PhasedCoordinator(phased_config=PhasedConfig(episode_timeout_s=1.2))
    for i in range(100):
        tick(c,distance=2. if i%25>=10 else .7)
        if c.phase=='FAILED': break
    assert c.failures==1 and c.last_decision['phased_event']=='episode_timeout'
    assert c.episode_elapsed>=1.2-1e-9


def test_completed_maneuver_requires_fresh_clearance_to_rearm():
    c=PhasedCoordinator()
    for _ in range(10): tick(c)
    for _ in range(40): tick(c,distance=2.)
    assert c.completed==1 and not c.armed
    out=tick(c,distance=.7)
    assert c.phase=='WAIT_CLEAR' and out[3]==1500 and out[2]==1500
