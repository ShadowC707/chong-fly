import numpy as np
import pytest
import torch

from optimizer.probe_history_capacity import make_cases, endpoint_loss, memoryless_floor


@pytest.mark.parametrize('suffix', [5,12,25,50,150])
def test_contrast_cases_have_identical_suffix_and_distinct_teacher_context(suffix):
    data=make_cases((.08,.12),suffix)
    assert data['X'].shape==(12,50+suffix,74)
    for start in range(0,12,3):
        x,y=data['X'][start:start+3],data['Y'][start:start+3]
        assert torch.equal(x[0,50:],x[1,50:]) and torch.equal(x[0,50:],x[2,50:])
        assert y[0,-1,2]>1500 and y[1,-1,2]<1400
        torch.testing.assert_close(y[0,-1,2],y[2,-1,2],rtol=0,atol=0)


def test_terminal_objective_excludes_other_channels_and_earlier_frames():
    d=make_cases((.12,),5); p=d['Y'].clone()
    p[:,:-5]+=100; p[:,:,0]+=200; p[:,:,1]+=20;p[:,:,3]+=10
    assert endpoint_loss(p,d['Y']).item()==0
    p[:,-5:,2]+=50
    assert endpoint_loss(p,d['Y']).item()==pytest.approx(.01)


def test_memoryless_floor_equals_best_common_prediction_and_is_positive():
    d=make_cases((.08,.12),150)
    floor=memoryless_floor(d['Y'])
    p=d['Y'].clone()
    for start in range(0,len(p),3):
        p[start:start+3,-5:,2]=d['Y'][start:start+3,-5:,2].mean(dim=0)
    assert endpoint_loss(p,d['Y']).item()==pytest.approx(floor,rel=1e-6)
    assert floor>0


def test_invalid_sensor_schedule_is_rejected():
    for flows,suffix in [((.12,),0),((np.nan,),5),((1.2,),5),((.01,),5)]:
        with pytest.raises(ValueError):make_cases(flows,suffix)
