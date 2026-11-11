import numpy as np
import pytest

from evaluate.sitl_neural import SceneSensors, actuator_command, scene_spec


def test_existing_sensor_adapter_preserves_body_flow_and_laser_offset():
    sensors=SceneSensors('clear')
    state=np.zeros(12); state[2]=1; state[3]=1
    flow,tof,alt=sensors.sample(state,0.)
    assert flow[0]>0 and flow[1]==0
    assert alt.range_m==pytest.approx(.982,abs=1e-6)
    assert tof.shape==(64,) and np.all((tof>=0)&(tof<=1))
    state[3:6]=[0,1,0];state[8]=np.pi/2
    rotated,_,_=sensors.sample(state,.02)
    np.testing.assert_allclose(rotated,flow,atol=1e-7)


def test_mirrored_geometry_produces_mirrored_tof_columns():
    state=np.zeros(12);state[2]=1;state[0]=.9
    _,left,_=SceneSensors('obstacle_left').sample(state,0.)
    _,right,_=SceneSensors('obstacle_right').sample(state,0.)
    np.testing.assert_allclose(left.reshape(8,8),right.reshape(8,8)[:,::-1],atol=1e-7)
    assert left.reshape(8,8)[:,:4].min()<left.reshape(8,8)[:,4:].min()


def test_actuator_adapter_changes_only_throttle_and_refuses_extrapolation():
    rows=[{'rc_pwm':1100,'mean_motor':.1},{'rc_pwm':1300,'mean_motor':.3}]
    command=np.array([1200.,1501.,1498.,1550.])
    result=actuator_command(command,rows)
    np.testing.assert_array_equal(result[1:],command[1:])
    assert result[0]==1200
    command[0]=1400
    with pytest.raises(ValueError): actuator_command(command,rows)


def test_scene_selection_fails_closed():
    with pytest.raises(ValueError):scene_spec('typo')
