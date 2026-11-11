import numpy as np
import pytest
from evaluate.maneuver_exit import ExitIntervention,post_switch_metrics


@pytest.mark.parametrize('channels', [(),(2,),(3,),(2,3)])
def test_channel_intervention_preserves_other_outputs_and_teacher_history(channels):
    class Network:
        default_dt=.02
        def reset_state(self):pass
        def step_np(self,*args,**kwargs):return np.array([1213.,1501.,1520.,1450.],np.float32)
    class Teacher:
        _escape_sign=None
        _cruise_recovery=1.
        def reset(self):self.n=0
        def step(self,x,dt):
            self.n+=1
            return np.array([1300.,1550.,1490.+self.n,1620.+self.n],np.float32)
    p=ExitIntervention(dict(candidate=Network()),Teacher(),channels)
    args=(np.zeros(2),np.ones(64))
    np.testing.assert_array_equal(p.step_np(*args),[1213,1501,1520,1450])
    p.intervene=True
    actual=p.step_np(*args);expected=np.array([1213,1501,1520,1450])
    expected[list(channels)]=np.array([1300,1550,1492,1622])[list(channels)]
    np.testing.assert_array_equal(actual,expected)
    np.testing.assert_array_equal(p.records[-1]['outputs']['candidate'],[1213,1501,1520,1450])


def test_post_switch_metrics_exclude_prefix_and_do_not_confuse_path_with_escape():
    time=np.array([0.,1.,2.,3.])
    pos=np.array([[0,0,1],[10,0,1],[11,0,1],[10,0,1]])
    result=post_switch_metrics(time,pos,np.array([.01,.2,.3,.4]),switch_s=1.)
    assert result['min_clearance_m']==.2
    assert result['path_m']==2.
    assert result['net_displacement_m']==0.
    with pytest.raises(ValueError):post_switch_metrics(time,pos,np.ones(4),switch_s=4.)
