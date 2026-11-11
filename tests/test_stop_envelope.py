import numpy as np
import pytest

from evaluate.measure_stop_envelope import stopping_metrics, CASES


def test_brief_zero_speed_is_not_stable_stop():
    time=np.arange(0.,4.01,.1)
    velocity=np.zeros((len(time),3));velocity[:11,0]=.2
    velocity[15:26,0]=.1
    result=stopping_metrics(time,np.zeros_like(velocity),velocity)
    assert result['first_below_s']==pytest.approx(1.1)
    assert result['settled_from_s']==pytest.approx(2.6)
    assert result['settle_observed_s']==pytest.approx(3.6)


def test_unfinished_settling_is_censored_not_reported_as_success():
    time=np.arange(0.,2.01,.1)
    v=np.zeros((len(time),3));v[:-5,0]=.2
    result=stopping_metrics(time,np.zeros_like(v),v)
    assert result['first_below_s'] is not None
    assert result['settled_from_s'] is None
    assert not result['settled']


def test_excursion_includes_sideways_motion_and_return_not_only_final_position():
    time=np.arange(4.)
    p=np.array([[0,0,1],[1,0,1],[1,1,1],[0,0,1]],float)
    result=stopping_metrics(time,p,np.zeros_like(p))
    assert result['max_excursion_m']==pytest.approx(np.sqrt(2))
    assert result['final_displacement_m']==0
    assert result['path_length_m']==pytest.approx(2+np.sqrt(2))
    assert result['swept_body_radius_m']==pytest.approx(.2+np.sqrt(2))


def test_grid_has_mirrored_lateral_states_and_no_tuned_coefficients():
    assert CASES['left_02']['velocity']==[0,.2,0]
    assert CASES['right_02']['velocity']==[0,-.2,0]
    assert CASES['pitch_forward_5']['pitch_deg']==5
    assert CASES['pitch_backward_5']['pitch_deg']==-5


@pytest.mark.parametrize('time,position,velocity',[
    ([0,0],np.zeros((2,3)),np.zeros((2,3))),
    ([0,1],np.zeros((2,2)),np.zeros((2,3))),
    ([0,1],np.zeros((2,3)),np.full((2,3),np.nan)),
])
def test_invalid_trace_rejected(time,position,velocity):
    with pytest.raises(ValueError):stopping_metrics(time,position,velocity)


def test_boundary_speed_is_not_stopped():
    time=np.arange(0,2.01,.1);v=np.zeros((len(time),3));v[:,0]=.03
    assert not stopping_metrics(time,np.zeros_like(v),v)['settled']


def test_real_stop_bench_preserves_pinned_chain_and_measures_full_horizon(tmp_path):
    import json
    from evaluate.measure_stop_envelope import run_trial
    report=run_trial(('forward_02',20261021,0.,str(tmp_path)))
    assert report['metrics']['horizon_s']==pytest.approx(8.)
    assert report['metrics']['settled'] and report['collision'] is None
    assert all(report['all_initial_sensors_fresh'].values())
    assert report['behavior_counts']['STOP_EVASION']==0
    assert .1<report['metrics']['max_excursion_m']<.3
    manifest=json.loads((tmp_path/report['id']/'manifest.json').read_text())
    assert manifest['system']['resolved']['coordinator']['config']['acquire_yaw_pwm']==20.
    assert manifest['system']['run_config']['coordinator_preset']=='original_v1'
    assert manifest['experiment']['injected_at_s']==1.
