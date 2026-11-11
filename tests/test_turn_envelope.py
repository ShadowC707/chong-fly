import math
import numpy as np
import pytest

from simulation.phased_coordinator import PhasedConfig, PhasedCoordinator
from simulation.turn_envelope import TurnEnvelope, resolve_turn_config


def tick(c, distance=.7, stop=False):
    return c.apply([1213.,1500.,1530.,1500.], flow_xy=[0.,0.],
                   tof_m=np.full(64,distance), dt=.02, stop_requested=stop)


def test_budget_derived_from_explicit_envelope_not_plant_state():
    base=PhasedConfig()
    resolved=resolve_turn_config(base,TurnEnvelope())
    assert resolved.turn_timeout_s==pytest.approx(224/(140/500*140))
    assert resolved.episode_timeout_s==8.
    assert resolved.yaw_pwm==140 and base.turn_timeout_s==4.
    assert resolve_turn_config(base,TurnEnvelope(min_full_scale_yaw_dps=200))==base


@pytest.mark.parametrize('kwargs',[
    {'min_full_scale_yaw_dps':0}, {'min_full_scale_yaw_dps':math.nan},
    {'min_full_scale_yaw_dps':True}, {'max_turn_s':math.inf},
    {'command_budget_deg':-1}, {'min_full_scale_yaw_dps':80},
    {'max_turn_s':9},
])
def test_invalid_or_unachievable_contract_is_rejected_not_silently_clamped(kwargs):
    with pytest.raises(ValueError):
        resolve_turn_config(PhasedConfig(),TurnEnvelope(**kwargs))


def test_does_not_shorten_existing_budget_or_accept_episode_without_recovery_reserve():
    with pytest.raises(ValueError):
        resolve_turn_config(PhasedConfig(),TurnEnvelope(min_full_scale_yaw_dps=300))
    with pytest.raises(ValueError):
        resolve_turn_config(PhasedConfig(episode_timeout_s=6.),TurnEnvelope())


def test_stationary_unresolved_threat_still_times_out_once_then_damps():
    c=PhasedCoordinator(phased_config=resolve_turn_config(PhasedConfig(),TurnEnvelope()))
    for _ in range(220): tick(c)
    assert c.phase=='TURN' and c.failures==0  # beyond former 4 s limit
    for _ in range(400): out=tick(c)
    assert c.phase=='FAILED' and c.failures==1
    assert c.turn_elapsed<=6. and c.episode_elapsed<=8.
    assert out[3]==1500 and not c.armed


def test_stop_cancels_extended_turn_and_cannot_reset_into_new_spin():
    c=PhasedCoordinator(stop_hover=True,
        phased_config=resolve_turn_config(PhasedConfig(),TurnEnvelope()))
    for _ in range(230): tick(c)
    assert c.phase=='TURN'
    assert tick(c,distance=1.2,stop=True)[3]==1500
    tick(c)
    assert c.phase=='WAIT_CLEAR' and not c.armed


def test_critical_oscillation_cannot_replenish_extended_budget():
    c=PhasedCoordinator(phased_config=resolve_turn_config(PhasedConfig(),TurnEnvelope()))
    tick(c)
    events=[]
    for i in range(500):
        tick(c,distance=.46 if i%2 else .81)
        if c.last_decision['phased_event']: events.append(c.last_decision['phased_event'])
    assert events.count('turn_timeout')==1 and c.phase=='FAILED'
    assert c.turn_elapsed<=6.


def test_resolved_budget_is_logged_and_default_assembly_stays_v2():
    from demo.runner import DemoRunner
    from demo.controllers import install
    r=DemoRunner(controller_mode='candidate_a')
    try:
        plain=install(r.system)
        assert plain.runtime.coordinator.phased_config.turn_timeout_s==4.
        configured=install(plain,turn_envelope=TurnEnvelope())
        contract=configured.manifest['experiment']['turn_envelope']
        assert contract['resolved_turn_timeout_s']==pytest.approx(40/7)
        assert contract['measured_progress'] is False
        assert configured.manifest['resolved']['coordinator']['phased_config']['turn_timeout_s']==pytest.approx(40/7)
    finally: r.env.close()
