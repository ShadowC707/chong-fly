import torch
from tests.test_pretrain_contract import dataset
from optimizer.continue_maneuver_data import combine


def source():
    d=dataset()
    d['metadata']['teacher_parameters']={}
    d['metadata']['episodes']=[dict(seed=i,scenario='test') for i in range(3)]
    return d


def test_combine_keeps_explicit_ids_and_padding_not_holdout():
    a,b=source(),source()
    for row in b['metadata']['episodes']: row['seed']+=10
    b['X']=b['X'][:,:3];b['Y']=b['Y'][:,:3]
    for k in ('applied_actions','valid','behavior'):b[k]=b[k][:,:3]
    joined=combine([(a,[0,2],'old'),(b,[1],'new')],steps=6)
    assert joined['metadata']['source_episodes']==[['old',0],['old',2],['new',1]]
    assert joined['valid'].sum(1).tolist()==[5,5,3]
    assert not joined['valid'][:,5:].any()
    assert (joined['behavior'][~joined['valid']]==-1).all()
    torch.testing.assert_close(joined['X'][2,:3],b['X'][1])


def test_combine_does_not_relabel_teacher_or_accept_duplicate_episode_seeds():
    import pytest
    a,b=source(),source()
    with pytest.raises(ValueError,match='seed'):
        combine([(a,[0],'old'),(b,[0],'new')],steps=6)
    b['metadata']['teacher_parameters']={'changed':True}
    with pytest.raises(ValueError,match='teacher_parameters'):
        combine([(a,[0],'old'),(b,[1],'new')],steps=6)
