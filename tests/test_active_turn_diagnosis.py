import numpy as np

from evaluate.diagnose_active_turn import mirror_observations, replay, threat_groups


def test_mirror_is_involution_and_preserves_longitudinal_channels():
    x=np.arange(2*74,dtype=np.float32).reshape(2,74)/200
    y=mirror_observations(x)
    np.testing.assert_array_equal(mirror_observations(y),x)
    np.testing.assert_array_equal(y[:,0],x[:,0])
    np.testing.assert_array_equal(y[:,1],-x[:,1])
    np.testing.assert_array_equal(y[:,2:66].reshape(-1,8,8),x[:,2:66].reshape(-1,8,8)[:,:,::-1])
    np.testing.assert_array_equal(y[:,66:],x[:,66:][:,[0,7,6,5,4,3,2,1]])


def test_replay_resets_on_each_recorded_segment_not_on_every_frame():
    class Accumulator:
        def reset_state(self): self.n=0
        def step_np(self,flow,tof,memory,dt):
            self.n+=1
            assert dt==.02
            return np.full(4,self.n,np.float32)
    y=replay(Accumulator(),np.ones((5,74),np.float32),np.array([2,2,3,3,3]))
    np.testing.assert_array_equal(y[:,0],[1,2,1,2,3])


def test_threat_groups_use_both_halves_flow_and_validity():
    x=np.ones((4,74),np.float32);x[:,:2]=0
    grid=x[:,2:66].reshape(4,8,8)
    grid[0,:,:4]=.5/3.5;grid[1]=.5/3.5;grid[2]=.5/3.5
    x[2,0]=.1
    groups=threat_groups(x,np.array([True,True,True,False]))
    np.testing.assert_array_equal(groups['slow_unilateral'],[True,False,False,False])
    np.testing.assert_array_equal(groups['slow_bilateral'],[False,True,False,False])
    np.testing.assert_array_equal(groups['active'],[True,True,True,False])


def test_intervention_changes_only_yaw_and_teacher_observes_before_switch():
    from evaluate.active_turn_counterfactual import YawIntervention
    class Network:
        default_dt=.02
        def reset_state(self):pass
        def step_np(self,*args,**kwargs):return np.array([1213.,1501.,1490.,1450.],np.float32)
    class Teacher:
        _escape_sign=None
        _cruise_recovery=1.
        def reset(self):self.n=0
        def step(self,x,dt):
            self.n+=1
            return np.array([1300.,1550.,1400.,1600.+self.n],np.float32)
    p=YawIntervention(dict(candidate=Network()),Teacher(),'candidate')
    args=(np.zeros(2),np.ones(64))
    np.testing.assert_array_equal(p.step_np(*args),[1213,1501,1490,1450])
    p.intervene=True
    np.testing.assert_array_equal(p.step_np(*args),[1213,1501,1490,1602])
    np.testing.assert_array_equal(p.records[-1]['outputs']['candidate'],[1213,1501,1490,1450])
