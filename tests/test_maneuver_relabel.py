import torch
from optimizer.probe_history_capacity import make_cases
from generator.generate_reflex_dataset import make_teacher
from optimizer.maneuver_relabel import replay_cruise_targets


def sample():
    d=make_cases((.12,),150)
    return dict(X=d['X'],Y=d['Y'],valid=torch.ones(d['X'].shape[:2],dtype=torch.bool),
        applied_actions=d['Y'].clone(),behavior=torch.zeros(d['X'].shape[:2],dtype=torch.long),
        metadata=dict(teacher_version='geometry-reflex-v5',dt=.02,
            teacher_parameters=make_teacher('geometry-reflex-v5',noise_std_pwm=0).parameters,
            behavior_names=['cruise','turn_right','turn_left','brake_left_drift','brake_right_drift','brake_forward']))


def test_relabel_preserves_observations_execution_and_nonpitch_targets():
    d=sample();old=d['Y'].clone()
    out=replay_cruise_targets(d,cruise_pwm=1530.)
    assert torch.equal(d['Y'],old)
    assert torch.equal(out['X'],d['X'])
    assert torch.equal(out['applied_actions'],d['applied_actions'])
    assert torch.equal(out['Y'][...,[0,1,3]],old[...,[0,1,3]])
    assert out['Y'][0,-1,2]==1482 # 1530 - 400*.12, not output clipping
    assert torch.equal(out['Y'][1,50:,2],old[1,50:,2]) # held brake unchanged
    assert out['metadata']['teacher_parameters']['pwm_cruise_pitch']==1530


def test_relabel_resets_each_episode_and_preserves_padding():
    d=sample();d['valid'][1,100:]=False;d['behavior'][1,100:]=-1
    d['Y'][1,100:]=1500
    out=replay_cruise_targets(d,cruise_pwm=1530.)
    assert torch.equal(out['Y'][1,100:],d['Y'][1,100:])
    assert (out['behavior'][1,100:]==-1).all()
    assert torch.equal(out['Y'][0],out['Y'][3]) # same clear history, fresh state


def test_relabel_rejects_incorrect_provenance():
    import pytest
    d=sample();d['Y'][0,0,3]+=30
    with pytest.raises(ValueError,match='replay'):
        replay_cruise_targets(d,cruise_pwm=1530.)
