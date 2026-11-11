import torch
import pytest
from evaluate.diagnose_maneuver_gradients import alignment, descent_effect
from evaluate.diagnose_maneuver_gradients import capture_state, unchanged_state


def test_alignment_distinguishes_conflict_agreement_and_missing_signal():
    x=torch.tensor([1.,0.],dtype=torch.float64)
    assert alignment(x,-x)['cosine']==-1
    assert alignment(x,x)['cosine']==1
    assert alignment(x,torch.tensor([0.,1.],dtype=torch.float64))['cosine']==0
    assert alignment(x,torch.zeros_like(x))['cosine'] is None


def test_descent_derivative_sign_and_weighted_objective():
    target=torch.tensor([1.,2.],dtype=torch.float64)
    assert descent_effect(target,target)<0
    assert descent_effect(target,-target)>0
    assert descent_effect(target,torch.zeros_like(target))==0
    mixed=target+10*(-target)
    assert descent_effect(target,mixed)==pytest.approx(45.)


def test_readonly_audit_includes_tensor_parameters_and_extra_state():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.weight=torch.nn.Parameter(torch.ones(2));self.meta={'version':1}
        def get_extra_state(self):return self.meta
    p=Model();state=capture_state(p)
    assert unchanged_state(p,state)
    p.meta['version']=2
    assert not unchanged_state(p,state)
    p.meta['version']=1
    with torch.no_grad():p.weight[0]=2
    assert not unchanged_state(p,state)
