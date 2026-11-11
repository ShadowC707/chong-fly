import torch
import pytest

from optimizer.probe_plastic_history import PlasticConstraints, anchor_inputs, protected_loss


def test_anchor_inputs_include_both_sides_clear_and_recovery():
    x=anchor_inputs()
    assert x.shape==(12,200,74)
    assert torch.all(x[:,75:,2:]==1)
    assert torch.all(x[:8,:,:2]==0)
    assert torch.all(x[:8,:25,2:]==1)
    assert torch.any(x[:4,25:75,2:34]<1) and torch.any(x[4:8,25:75,34:66]<1)


def test_protected_loss_ignores_throttle_and_penalizes_late_yaw():
    y=torch.full((12,200,4),1500.)
    p=y.clone();p[...,0]+=200
    assert protected_loss(p,y)==0
    p[:,125:,3]+=100
    assert protected_loss(p,y)>0


def test_plastic_constraints_keep_routes_amplitude_and_bounds(tiny_model_meta):
    from simulation.policy import ChongFlyMSPPolicy
    policy=ChongFlyMSPPolicy.from_meta(str(tiny_model_meta),preserve_signs=True,
        neutral_origin=True,sensor_encoding='proximity-v1',dt=.02)
    c=policy.cfc_network.cell; amplitude=c.A.detach().clone()
    guard=PlasticConstraints(policy)
    assert c.tau_raw.requires_grad and c._w_macro_param.requires_grad
    assert not c.A.requires_grad
    with torch.no_grad():
        c.tau_raw[0]=30.;c.tau_raw[1]=-30.
        c._w_macro_param.mul_(4.)
    policy.post_step();guard(policy)
    assert torch.equal(amplitude,c.A)
    assert float(c.tau.min())>=.005-1e-7
    assert float(c.tau.max())<=2.+1e-6
    assert float(c._effective_W().norm())<=guard.w_max_norm
    assert torch.all(c._effective_W()[~c._synapse_mask.bool()]==0)
    assert torch.all(c._effective_W()*c._synapse_sign>=0)
    with torch.no_grad():c.A[0]+=1
    with pytest.raises(ValueError,match='amplitude'):guard(policy)
