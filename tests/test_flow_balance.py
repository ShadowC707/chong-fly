import pytest
import torch

from optimizer.flow_balance import active_fast_weights
from optimizer.pretrain import pretrain_policy
from tests.test_pretrain_contract import TimedPolicy, dataset


def inputs():
    x=torch.zeros(2,5,74);x[...,2:]=1.
    x[0,1,0]=.2;x[0,1,2]=.1
    valid=torch.ones(2,5,dtype=torch.bool);valid[1,4]=False
    x[1,4,0]=.8;x[1,4,2]=.1  # Padding must not enter the rare group.
    w=valid[...,None].expand(-1,-1,3).float().clone()
    return x,valid,w


def test_mixture_preserves_channel_mass_and_roll_and_ignores_padding():
    x,valid,w=inputs()
    result,info=active_fast_weights(w,x,valid,mix=.25)
    assert info['frames']==1 and info['episodes']==1
    assert torch.equal(result[...,0],w[...,0])
    torch.testing.assert_close(result.sum((0,1)),w.sum((0,1)))
    assert result[0,1,1]==pytest.approx(.75+.25*9)
    assert result[0,1,2]==pytest.approx(.75+.25*9)
    assert torch.equal(result[~valid],torch.zeros_like(result[~valid]))
    assert torch.equal(w,valid[...,None].expand_as(w).float())


def test_zero_mix_is_bitwise_original():
    x,valid,w=inputs()
    assert torch.equal(active_fast_weights(w,x,valid,mix=0)[0],w)


def test_mask_is_based_on_inputs_and_covers_lateral_flow():
    x,valid,w=inputs();x[0,1,0]=0;x[0,1,1]=-.2
    assert active_fast_weights(w,x,valid,mix=.25)[1]['frames']==1
    x[0,1,2]=1
    with pytest.raises(ValueError,match='No active-fast'):active_fast_weights(w,x,valid,mix=.25)


@pytest.mark.parametrize('mix',[-.1,1.1,float('nan'),True])
def test_bad_mix_rejected(mix):
    x,valid,w=inputs()
    with pytest.raises(ValueError):active_fast_weights(w,x,valid,mix=mix)


def test_pretrain_computes_modifier_from_training_episodes_only():
    data=dataset();data['X'][...,:2]=0
    model=TimedPolicy()
    pretrain_policy(model,data,epochs=0,subset_ratio=1,seed=42,loss_mode='channel-v4')
    ids=model._pretrain_info['subset_indices'];held=model._pretrain_info['validation_indices']
    data['X'][held,:,0]=.2;data['X'][held,:,2]=.1
    with pytest.raises(ValueError,match='No active-fast'):
        pretrain_policy(TimedPolicy(),data,epochs=0,subset_ratio=1,seed=42,
                        loss_mode='channel-v4',active_fast_balance=.25)
    data['X'][ids[0],0,0]=.2;data['X'][ids[0],0,2]=.1
    trained=pretrain_policy(TimedPolicy(),data,epochs=1,subset_ratio=1,seed=42,
                           loss_mode='channel-v4',active_fast_balance=.25)
    info=trained._pretrain_info
    assert info['subset_indices']==ids and info['validation_indices']==held
    assert info['loss_contract']['active_fast_modifier']['frames']==1
    assert info['loss_contract']['active_fast_modifier']['channels']==['pitch','yaw']


def test_modifier_requires_explicit_channel_v4():
    with pytest.raises(ValueError,match='channel-v4'):
        pretrain_policy(TimedPolicy(),dataset(),active_fast_balance=.25)


def test_default_and_explicit_zero_have_identical_training():
    a,b=TimedPolicy(),TimedPolicy()
    for p,kwargs in ((a,{}),(b,dict(active_fast_balance=0.))):
        pretrain_policy(p,dataset(),epochs=2,subset_ratio=1,seed=42,loss_mode='channel-v4',**kwargs)
    assert torch.equal(a.bias,b.bias)
    assert a._pretrain_info==b._pretrain_info


def test_raw_scores_ignore_padding_and_throttle_and_report_episode_counts():
    from optimizer.continue_flow_reflex import score_raw
    x,valid,_=inputs()
    data=dict(X=x,Y=torch.full((2,5,4),1500.),valid=valid)
    prediction=data['Y'].clone()
    prediction[...,0]=1000.
    prediction[0,1,3]=1600.
    prediction[1,4,3]=2000.
    report=score_raw(prediction,data,[0,1])
    subset=report['neutral_yaw_active_fast']
    assert subset['frames']==1 and subset['episodes']==1
    assert subset['strong_yaw_frames']==1
    assert subset['mae_roll_pitch_yaw']==[0.,0.,100.]
    assert len(subset['per_episode'])==1
    assert report['all']['strong_yaw_frames']==1
