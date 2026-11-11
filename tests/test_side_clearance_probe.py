import numpy as np
from evaluate.side_clearance_probe import pitch_probe


def test_probe_is_bounded_in_time_and_preserves_other_channels():
    action=np.array([1213.,1490.,1520.,1470.])
    for t in (5.99,6.,7.,8.):
        np.testing.assert_array_equal(pitch_probe(action,.04,t,'CRUISE',False)[0],action)
    changed,weight=pitch_probe(action,.04,6.5,'CRUISE',False)
    assert weight==1 and changed[2]==1484.
    np.testing.assert_array_equal(changed[[0,1,3]],action[[0,1,3]])
    assert 1484.<pitch_probe(action,.04,6.06,'CRUISE',False)[0][2]<1520.


def test_probe_never_weakens_existing_brake_or_steals_stop_and_phases():
    action=np.array([1213.,1490.,1400.,1470.])
    np.testing.assert_array_equal(pitch_probe(action,.04,6.5,'CRUISE',False)[0],action)
    action[2]=1520.
    for phase in ('BRAKE','TURN','RECOVER','FAILED','WAIT_CLEAR','CRITICAL','STOP'):
        np.testing.assert_array_equal(pitch_probe(action,.04,6.5,phase,False)[0],action)
    np.testing.assert_array_equal(pitch_probe(action,.04,6.5,'CRUISE',True)[0],action)
