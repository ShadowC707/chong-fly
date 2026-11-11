from copy import deepcopy
import pytest
import torch
from optimizer.interface_attribution import hybrid_policy


def make_policy(meta):
    from simulation.policy import ChongFlyMSPPolicy
    return ChongFlyMSPPolicy.from_meta(str(meta),preserve_signs=True,
        neutral_origin=True,sensor_encoding='proximity-release-v1',dt=.02)


def test_hybrids_copy_only_selected_groups_and_full_pair_reproduces_new(tiny_model_meta):
    p=make_policy(tiny_model_meta);new=deepcopy(p)
    with torch.no_grad():
        new.sensor_layer.gain.mul_(1.3)
        new.cfc_network.cell.W_in.weight.mul_(.8)
        new.dn_head.proj.weight.mul_(1.2)
        new.dn_head.proj.bias[3]+=.1
    x=torch.rand(2,9,74)
    for inputs in (False,True):
        for head in (False,True):
            h=hybrid_policy(p,new,new_inputs=inputs,new_head=head)
            torch.testing.assert_close(h.sensor_layer.gain,(new if inputs else p).sensor_layer.gain,rtol=0,atol=0)
            torch.testing.assert_close(h.dn_head.proj.weight,(new if head else p).dn_head.proj.weight,rtol=0,atol=0)
            if inputs==head:
                torch.testing.assert_close(h(x,dt=.02)[0],(new if head else p)(x,dt=.02)[0],rtol=0,atol=0)
    before=p.sensor_layer.gain.detach().clone()
    with torch.no_grad():h.sensor_layer.gain.zero_()
    assert torch.equal(p.sensor_layer.gain,before)


def test_hybrid_rejects_uncontrolled_core_or_route_changes(tiny_model_meta):
    p=make_policy(tiny_model_meta);new=deepcopy(p)
    with torch.no_grad():new.cfc_network.cell.tau_raw.add_(.1)
    with pytest.raises(ValueError,match='Outside interface'):hybrid_policy(p,new,new_inputs=True,new_head=False)
    new=deepcopy(p)
    with torch.no_grad():new.dn_head.proj.route_mask.zero_()
    with pytest.raises(ValueError,match='Outside interface'):hybrid_policy(p,new,new_inputs=False,new_head=True)


def test_xy_head_update_preserves_inputs_core_throttle_roll_and_matches_forward(tiny_model_meta):
    from optimizer.interface_attribution import configure_xy_head,head_prediction
    from optimizer.maneuver_state import forward_hidden
    p=make_policy(tiny_model_meta)
    handles=configure_xy_head(p);before={n:v.detach().clone() for n,v in p.named_parameters()}
    x=torch.rand(2,5,74)
    with torch.no_grad():expected,h=forward_hidden(p,x)
    torch.testing.assert_close(head_prediction(p,h),expected,rtol=0,atol=0)
    optimizer=torch.optim.Adam([v for v in p.parameters() if v.requires_grad],lr=.01)
    head_prediction(p,h).sum().backward();optimizer.step();p.post_step()
    for n,v in p.named_parameters():
        if n.startswith('dn_head.'):
            torch.testing.assert_close(v[:2],before[n][:2],rtol=0,atol=0)
        else:torch.testing.assert_close(v,before[n],rtol=0,atol=0)
    assert not torch.equal(p.dn_head.proj.bias[2:],before['dn_head.proj.bias'][2:])
    for handle in handles:handle.remove()
