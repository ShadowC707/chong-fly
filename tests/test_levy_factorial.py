"""Factor isolation, search gating, and physical-time experiment metrics."""
from dataclasses import replace

import numpy as np
import pytest

from simulation.xy_coordinator import XYCoordinator, BehaviorConfig


def step(c, distance=3.5, yaw=1500., tof=None):
    return c.apply([1213.,1500.,1527.,yaw],flow_xy=[0.,0.],
        tof_m=np.full(64,distance) if tof is None else tof,dt=.02)


def test_pause_preserves_wait_but_never_launches_in_grey_zone():
    c=XYCoordinator(levy_explore=True,behavior_config=BehaviorConfig(pause_search_wait=True))
    c.wait_target=.10
    for _ in range(4): step(c)
    elapsed=c.wait_elapsed
    for _ in range(100):
        assert step(c,distance=1.05)[3]==1500
        assert c.wait_elapsed==elapsed
        assert c.last_decision['search_event']=='wait_paused'
    step(c)
    assert c.last_decision['search_event']=='started'
    step(c,distance=.94)
    assert c.pulse_elapsed is None and c.wait_elapsed==0
    assert c.last_decision['search_event']=='preempted'


def test_directed_gate_checks_minimum_as_well_as_mean_without_consuming_randomness():
    c=XYCoordinator(levy_explore=True,behavior_config=BehaviorConfig(target_half_gate=True))
    c.wait_target=.02
    tof=np.full((8,8),2.); tof[0,0]=tof[0,7]=1.16
    for _ in range(20):
        assert step(c,tof=tof)[3]==1500
        assert c.last_decision['search_event']=='target_blocked'
    rng_state=c.rng.bit_generator.state
    for _ in range(20): step(c,tof=tof)
    assert c.rng.bit_generator.state==rng_state  # no random sign retry each tick
    step(c)
    assert c.last_decision['search_event']=='started'


def test_directed_gate_rejects_shallow_mean_and_chooses_open_left():
    c=XYCoordinator(levy_explore=True,behavior_config=BehaviorConfig(target_half_gate=True))
    c.wait_target=.02
    step(c,distance=1.5)
    assert c.last_decision['search_event']=='target_blocked'
    tof=np.full((8,8),1.3); tof[:,:4]=3.
    step(c,tof=tof)
    assert c.search_sign==-1
    assert c.last_decision['target_half_mean_m']==3.
    assert step(c,tof=tof)[3]<1500


@pytest.mark.parametrize('distance,yaw', [(.94,1500.),(3.5,1525.),(3.5,1475.)])
def test_large_pulse_retains_same_step_preemption(distance,yaw):
    c=XYCoordinator(levy_explore=True,behavior_config=BehaviorConfig(
        search_amplitude_pwm=240.,search_duration_s=.6,target_half_gate=True,pause_search_wait=True))
    c.wait_target=.02
    for _ in range(12): step(c)
    assert abs(c.last_decision['search_pwm'])>100
    assert step(c,distance=distance,yaw=yaw)[3]==yaw
    assert c.last_decision['search_cancelled_phase']=='pulse'


def test_factor_configs_only_change_intended_parameters():
    from evaluate.compare_levy_factorial import variant_configs
    configs=variant_configs()
    assert configs['levy_v1']==BehaviorConfig()
    assert configs['pause_only']==replace(configs['levy_v1'],pause_search_wait=True)
    assert configs['both']==replace(configs['directed_turn'],pause_search_wait=True)
    assert configs['directed_turn'].search_amplitude_pwm/500*200*configs['directed_turn'].search_duration_s/2==pytest.approx(28.8)
    assert all(c.search_preempt_m==.95 for c in configs.values())


def test_rollout_records_exact_earliest_minimum_at_physics_time():
    from optimizer.rollout import simulate_policy_rollout
    from tests.test_evaluation_contract import ScriptedEnv, RecordingPolicy
    class ClearanceEnv(ScriptedEnv):
        def step(self,action):
            obs,reward,done,info=super().step(action)
            info['clearance_m']=.1 if self.steps in (7,8) else .4
            return obs,reward,done,info
    _,r=simulate_policy_rollout(RecordingPolicy(),env=ClearanceEnv(),eval_steps=3,record_trace=True)
    event=r['min_clearance_event']
    assert event['time_s']==pytest.approx(.028)
    assert event['clearance_m']==.1
    np.testing.assert_allclose(event['position_m'],[.028,0,1])
    assert r['trace']['physics'][7]['clearance_m']==.1


def test_interior_time_is_time_weighted_and_excludes_stop():
    from evaluate.compare_levy_factorial import interior_metrics
    rows=[dict(time_s=t,position_m=[x,0,1]) for t,x in
          [(0,0),(1,0),(2,3),(3,3),(4,0),(5,0),(6,3),(7,3),(8,3)]]
    r=interior_metrics(rows,stop_at_s=6.)
    assert r['interior_time_before_stop_s']==3.
    assert r['interior_fraction_before_stop']==.5
    assert r['interior_time_after_first_wall_s']==2.
    assert r['returns_from_wall_1s']==1


@pytest.mark.parametrize('kw', [dict(pause_search_wait=1),dict(target_half_gate='yes'),
                                 dict(target_half_min_m=2.,target_half_mean_m=1.)])
def test_invalid_new_configuration_rejected(kw):
    with pytest.raises(ValueError): BehaviorConfig(**kw)
