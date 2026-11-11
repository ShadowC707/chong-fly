import torch
from optimizer.balanced_maneuvers import auxiliary_cases, behavioral_admission


def test_auxiliary_cases_use_same_teacher_and_identical_suffixes():
    history, cruise = auxiliary_cases(100)
    assert history['X'].shape == (24,150,74)
    for start in range(0,24,3):
        x,y=history['X'][start:start+3],history['Y'][start:start+3]
        assert torch.equal(x[0,50:],x[1,50:])
        assert torch.equal(x[1,50:],x[2,50:])
        assert y[1,-1,2] < y[2,-1,2]-100
        assert torch.equal(y[0,-1],y[2,-1])
    assert torch.all(cruise['Y'][0,:,2] == 1530)
    assert torch.all(cruise['Y'][1,:,2] == 1526)
    assert torch.all(cruise['Y'][2,:,2] == 1534)


def test_admission_does_not_confuse_stationary_survival_with_navigation():
    r=dict(temporal=dict(scenes=7,direction=12,recovery=12,narrow_amplitude=8,
                       max_recovery_s=.4),cruise=dict(final_pwm=[1213,1500,1500,1500]),
        handover=dict(passed=True))
    result=behavioral_admission(r, clear_speed=.016, history_error=18.)
    assert not result['passed']
    assert 'cruise_pitch' in result['reasons'] and 'clear_forward_speed' in result['reasons']
    r['cruise']['final_pwm'][2]=1528
    assert behavioral_admission(r,clear_speed=.2,history_error=18.)['passed']
    assert not behavioral_admission(r,clear_speed=.2,history_error=200.)['passed']
