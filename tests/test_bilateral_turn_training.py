import pytest
import torch

from optimizer.bilateral_turn import bilateral_mask, magnitude_loss


def test_mask_requires_slow_bilateral_visible_threat_and_valid_strong_target():
    x=torch.ones(1,6,74);x[...,:2]=0
    x[...,2:66]=.5/3.5
    y=torch.full((1,6,4),1500.);y[...,3]=1620
    valid=torch.ones(1,6,dtype=torch.bool)
    x[0,1,2:66].reshape(8,8)[:,4:]=1
    x[0,2,0]=.06
    x[0,3,2:66]=.85/3.5
    valid[0,4]=False
    y[0,5,3]=1520
    assert bilateral_mask(x,y,valid).tolist()==[[True,False,False,False,False,False]]


def test_magnitude_target_does_not_force_teacher_sign_or_other_channels():
    target=torch.full((1,3,4),1500.);target[...,3]=1620
    pred=target.clone();pred[0,:,3]=torch.tensor([1470.,1530.,1500.]);pred.requires_grad_()
    mask=torch.ones(1,3,dtype=torch.bool)
    loss=magnitude_loss(pred,target,mask,normalizer=3.)
    loss.backward()
    assert torch.all(pred.grad[...,:3]==0)
    assert pred.grad[0,0,3]>0  # descent strengthens the existing negative turn
    assert pred.grad[0,1,3]<0  # descent strengthens the existing positive turn
    assert pred.grad[0,2,3]<0  # exactly zero uses teacher sign, not zero gradient
    assert pred.grad[0,0,3].item()==pytest.approx(-pred.grad[0,1,3].item())
    mirrored=3000-pred.detach();mirrored.requires_grad_()
    mirrored_loss=magnitude_loss(mirrored,3000-target,mask,normalizer=3.)
    mirrored_loss.backward()
    torch.testing.assert_close(mirrored_loss,loss)
    torch.testing.assert_close(mirrored.grad,-pred.grad)


def test_magnitude_mask_and_fixed_normalizer_preserve_partitioning():
    target=torch.full((2,2,4),1500.);target[...,3]=1620
    pred=torch.full((2,2,4),1510.,requires_grad=True)
    mask=torch.tensor([[True,False],[True,True]])
    full=magnitude_loss(pred,target,mask,normalizer=3.)
    pieces=sum(magnitude_loss(pred[i:i+1],target[i:i+1],mask[i:i+1],normalizer=3.) for i in range(2))
    torch.testing.assert_close(full,pieces)
    full.backward();assert torch.all(pred.grad[0,1]==0)
    empty=magnitude_loss(pred,target,torch.zeros_like(mask),normalizer=3.)
    assert empty.item()==0 and empty.requires_grad
    with pytest.raises(ValueError):magnitude_loss(pred,target,mask,normalizer=0)


def test_room_screen_does_not_accept_magnitude_at_the_cost_of_direction_or_motion():
    from optimizer.bilateral_turn import admission
    temporal=dict(direction=12,recovery=12,max_recovery_s=.4,narrow_amplitude=8,scenes=7,broad_joint=3)
    assert admission(dict(passed=True),temporal)['passed']
    for changes in (dict(direction=11),dict(recovery=11),dict(max_recovery_s=.82),
                    dict(narrow_amplitude=7),dict(scenes=6),dict(broad_joint=2)):
        assert not admission(dict(passed=True),dict(temporal,**changes))['passed']
    assert not admission(dict(passed=False),temporal)['passed']


def test_optional_objective_zero_preserves_existing_training_exactly():
    from optimizer.release_cruise import fit
    class TinyPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.command=torch.nn.Parameter(torch.tensor([1213.,1500.,1500.,1500.]))
        def forward(self,x,dt):return self.command.expand(*x.shape[:2],4),None
        def post_step(self):pass
    class Guard:
        steps=0
        def __call__(self,policy):self.steps+=1
    x=torch.ones(104,2,74);x[...,:2]=0
    y=torch.tensor([1213.,1500.,1530.,1620.]).expand(104,2,4).clone()
    data=dict(X=x,Y=y,valid=torch.ones(104,2,dtype=torch.bool),
        behavior=torch.zeros(104,2,dtype=torch.long),metadata=dict(behavior_names=['turn']))
    a,b,c=TinyPolicy(),TinyPolicy(),TinyPolicy()
    aux=dict(X=x[:1],Y=y[:1]);calls=[]
    def extra(pred,target,ids):
        calls.extend(ids.tolist())
        return pred.sum()*0
    left=fit(a,data,Guard(),aux,lambda _:None)
    right=fit(b,data,Guard(),aux,lambda _:None,extra_objective=extra)
    supplied=fit(c,data,Guard(),aux,lambda _:None,
        optimizer=torch.optim.Adam(c.parameters(),lr=.001))
    torch.testing.assert_close(a.command,b.command,atol=0,rtol=0)
    torch.testing.assert_close(a.command,c.command,atol=0,rtol=0)
    assert supplied == left
    with pytest.raises(ValueError,match='every trainable parameter'):
        fit(c,data,Guard(),aux,lambda _:None,optimizer=torch.optim.Adam(a.parameters(),lr=.001))
    assert torch.bincount(torch.tensor(calls)).tolist()==[10]*104
    for first,second in zip(left['progress'],right['progress']):
        assert second.pop('extra_loss')==0
        assert first==second
