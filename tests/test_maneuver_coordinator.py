import numpy as np
import pytest

from simulation.maneuver_coordinator import ManeuverConfig, ManeuverCoordinator
from simulation.xy_coordinator import XYCoordinator


def tick(c, *, yaw=1500., distance=.7, flow=(0., 0.), stop=False):
    return c.apply([1213.,1507.,1400.,yaw], flow_xy=flow,
                   tof_m=np.full(64,distance), dt=.02, stop_requested=stop)


def test_completed_pulse_releases_sign_in_neutral_distance_band():
    c=ManeuverCoordinator(turn_latch=True, gate_turn_by_flow=False)
    tick(c,yaw=1400.)
    for _ in range(9): tick(c,distance=.9)
    assert c.turn_sign==-1
    assert tick(c,distance=.9)[3]==1500.
    assert c.turn_sign==0
    assert c.last_decision['maneuver_release_event']
    assert tick(c,yaw=1700.)[3]==1700.


def test_short_gap_or_neutral_while_still_threatened_does_not_release():
    c=ManeuverCoordinator(turn_latch=True, gate_turn_by_flow=False)
    tick(c,yaw=1400.)
    for _ in range(9): tick(c,distance=.9)
    tick(c,distance=.79)
    for _ in range(9): tick(c,distance=.9)
    assert tick(c,yaw=1700.)[3]==1300.
    for _ in range(50): tick(c,distance=.7)
    assert c.turn_sign==-1


def test_non_neutral_recovery_does_not_count_as_completed_pulse():
    c=ManeuverCoordinator(turn_latch=True, gate_turn_by_flow=False)
    tick(c,yaw=1400.)
    for _ in range(30): tick(c,yaw=1460.,distance=.9)
    assert c.turn_sign==-1
    c.reset()
    assert c.turn_sign==0 and c.neutral_elapsed_s==0
    assert tick(c,yaw=1700.)[3]==1700.


def test_flow_gate_preserves_brake_but_delays_turn_while_fast():
    c=ManeuverCoordinator(gate_turn_by_flow=True)
    out=tick(c,yaw=1750.,flow=(.15,0.))
    np.testing.assert_array_equal(out[:3],[1213.,1507.,1400.])
    assert out[3]==1500.
    assert 'turn_flow_gate' in c.last_decision['reasons']
    for _ in range(50): out=tick(c,yaw=1750.)
    assert out[3]==1750.


def test_close_threat_restores_evasion_even_at_high_flow():
    c=ManeuverCoordinator()
    assert tick(c,yaw=1750.,distance=.3,flow=(.15,0.))[3]==1750.
    assert 1500 < tick(c,yaw=1750.,distance=.4,flow=(.15,0.))[3] < 1750


def test_neutral_remains_neutral_and_gate_never_reverses_direction():
    for yaw in (1200.,1500.,1800.):
        for distance in (.3,.4,.7,1.2):
            for flow in ((0.,0.),(.08,0.),(0.,.15),(-.15,0.)):
                out=tick(ManeuverCoordinator(),yaw=yaw,distance=distance,flow=flow)[3]-1500
                assert out*(yaw-1500)>=0
                assert abs(out)<=abs(yaw-1500)
                if yaw==1500.: assert out==0.


def test_stop_keeps_existing_priority_and_output():
    a=XYCoordinator(anti_rollback=True,turn_latch=True,stop_hover=True)
    b=ManeuverCoordinator(anti_rollback=True,turn_latch=True,stop_hover=True)
    for _ in range(40):
        np.testing.assert_array_equal(tick(a,yaw=1700.,distance=1.2,flow=(.1,.02),stop=True),
                                      tick(b,yaw=1700.,distance=1.2,flow=(.1,.02),stop=True))


def test_disabled_repairs_preserve_legacy_numbers_on_sequence():
    a=XYCoordinator(anti_rollback=True,turn_latch=True)
    b=ManeuverCoordinator(anti_rollback=True,turn_latch=True,
                          release_on_neutral=False,gate_turn_by_flow=False)
    rng=np.random.default_rng(42)
    for _ in range(150):
        args=dict(yaw=float(rng.uniform(1150,1850)),distance=float(rng.uniform(.3,1.3)),
                  flow=tuple(rng.uniform(-.1,.2,2)))
        np.testing.assert_array_equal(tick(a,**args),tick(b,**args))


@pytest.mark.parametrize('kwargs',[{'release_s':0},{'neutral_pwm':float('nan')},
                                  {'flow_full':.2},{'emergency_m':.6}])
def test_invalid_maneuver_contract_rejected(kwargs):
    with pytest.raises(ValueError): ManeuverConfig(**kwargs)
