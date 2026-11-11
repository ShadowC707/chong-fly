import json
from pathlib import Path
import numpy as np
import pytest
from simulation.sitl_twr import ThrustAdapter, thrust_to_weight


def parameters():
    return json.loads(Path('configs/ready_sim_reference_v1.json').read_text())['uncertainty_study']['nominal_dynamics']


def rows():
    return [dict(rc_pwm=p,mean_motor=(p-997)/1000) for p in (1070,1100,1150,1300,1500,1700,1900)]


@pytest.mark.parametrize('twr',[5.,10.])
def test_adapter_preserves_collective_thrust_without_changing_other_parameters(twr):
    ref=parameters();saved=ref.copy();a=ThrustAdapter(ref,twr,rows())
    assert ref==saved
    assert thrust_to_weight(a.dynamics)==pytest.approx(twr)
    assert {k:v for k,v in a.dynamics.items() if k!='max_rpm'}=={k:v for k,v in ref.items() if k!='max_rpm'}
    for pwm in (1100.,1213.,1300.,1400.):
        out=a.command([pwm,1510,1490,1640])
        u=(out[0]-997)/1000
        np.testing.assert_array_equal(out[1:],[1510,1490,1640])
        assert u*a.dynamics['max_rpm']==pytest.approx((pwm-1000)/1000*ref['max_rpm'])
    assert a.dynamics['max_rpm']<ref['max_rpm']


def test_nominal_profile_matches_original_adapter_exactly():
    from evaluate.sitl_neural import actuator_command
    p=parameters();a=ThrustAdapter(p,thrust_to_weight(p),rows())
    assert a.dynamics==p and a.scale==1.
    for pwm in np.linspace(1100,1400,50):
        x=[pwm,1520,1480,1640]
        np.testing.assert_array_equal(a.command(x),actuator_command(x,rows()))


def test_old_short_map_is_rejected_before_physics_not_extrapolated():
    old=[dict(rc_pwm=p,mean_motor=(p-997)/1000) for p in (1150,1200,1300)]
    with pytest.raises(ValueError,match='bracket'):ThrustAdapter(parameters(),5.,old)


@pytest.mark.parametrize('bad',[0,-1,float('nan'),float('inf')])
def test_invalid_twr_rejected(bad):
    with pytest.raises(ValueError):ThrustAdapter(parameters(),bad,rows())


def test_no_clipping_of_outside_reference_throttle_contract():
    a=ThrustAdapter(parameters(),5.,rows())
    for pwm in (1000.,1401.,float('nan')):
        with pytest.raises(ValueError):a.command([pwm,1500,1500,1500])


def test_twr_five_hover_and_upper_limit_are_not_confused():
    a=ThrustAdapter(parameters(),5.,rows())
    assert a.dynamics['max_rpm']==pytest.approx(13361.93955,abs=.001)
    assert a.contract()['physical_hover_motor']==pytest.approx(1/np.sqrt(5))
    assert a.contract()['requested_motor_bounds'][1]==pytest.approx(.8382019,abs=1e-6)


def test_hover_probe_covers_quantized_gap_without_parameter_search():
    from evaluate.sitl_twr_sensitivity import hover_probe_grid
    # Observed failure: inferred 1313.36, but 1314 -> .316 < 1/sqrt(10).
    grid=hover_probe_grid(1313.36)
    assert grid==[1311,1312,1313,1314,1315,1316]
    assert 1315 in grid
