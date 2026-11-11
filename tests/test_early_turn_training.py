import numpy as np
import pytest
import torch
from optimizer.early_turn_training import correction_case, correction_loss


def test_correction_uses_full_reset_prefix_but_only_window_targets(tmp_path):
    time=np.array([0.,.02,5.98,6.,6.02,10.])
    segment=np.array([0,0,1,1,1,1])
    x=np.ones((6,74),np.float32); x[:,:2]=0
    y=np.full((6,4),1500,np.float32)
    path=tmp_path/'trace.npz'; np.savez(path,time=time,segment=segment,X=x,teacher=y)
    c=correction_case(path)
    assert c['time'].tolist()==[5.98,6.,6.02]
    assert c['mask'].tolist()==[[False,True,True]]
    assert c['X'].shape==(1,3,74)
    torch.save(c,tmp_path/'correction.pt')
    restored=torch.load(tmp_path/'correction.pt',weights_only=True)
    assert restored['time'].tolist()==c['time'].tolist()
    segment[4]=2; np.savez(path,time=time,segment=segment,X=x,teacher=y)
    with pytest.raises(ValueError,match='reset'): correction_case(path)


def test_correction_loss_is_joint_pitch_yaw_and_ignores_prefix():
    p=torch.full((1,3,4),1500.,requires_grad=True); y=p.detach().clone()
    y[0,0]=1000; y[0,1:,0:2]=1900; y[0,1:,2]=1400; y[0,1:,3]=1600
    mask=torch.tensor([[False,True,True]])
    loss=correction_loss(p,y,mask); assert loss.item()==pytest.approx(.04)
    loss.backward()
    assert torch.all(p.grad[:,:1]==0) and torch.all(p.grad[...,:2]==0)
    assert torch.all(p.grad[:,1:,2]>0) and torch.all(p.grad[:,1:,3]<0)
    with pytest.raises(ValueError): correction_loss(p,y,torch.zeros_like(mask))


def test_mirror_recomputes_teacher_and_keeps_window():
    from optimizer.early_turn_training import mirrored_case
    from generator.generate_reflex_dataset import make_teacher
    x=torch.ones(1,20,74);x[...,:2]=0
    x[:,5:15,2:66].reshape(1,10,8,8)[:,:,:,:4]=.5/3.5
    teacher=make_teacher('geometry-reflex-v5',noise_std_pwm=0,pwm_cruise_pitch=1530.)
    y=torch.from_numpy(np.stack([teacher.step(row,dt=.02) for row in x[0].numpy()])).unsqueeze(0)
    case=dict(X=x,Y=y,mask=torch.ones(1,20,dtype=torch.bool),time=torch.arange(20)*.02)
    pair=mirrored_case(case)
    assert pair['X'].shape==(2,20,74)
    torch.testing.assert_close(pair['Y'][0],y[0])
    torch.testing.assert_close(pair['Y'][1,:,3],3000-y[0,:,3])
    assert torch.equal(pair['mask'][0],pair['mask'][1])
