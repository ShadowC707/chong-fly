import numpy as np
from evaluate.practical_maneuver import ObservedPolicy, mask_episodes


class Policy:
    default_dt=.02
    def __init__(self,offset):self.offset=offset;self.h=offset;self.reset_state()
    def reset_state(self):self.n=0
    def step_np(self,flow,tof,memory,dt):
        self.n+=1
        return np.array([1213,1500,1526,1500+self.offset+self.n],np.float32)


class Teacher:
    def reset(self):self._escape_sign=None;self._cruise_recovery=1.;self.n=0
    def step(self,x,dt):self.n+=1;return np.array([1213,1500,1530,1500],np.float32)


def test_observer_returns_driver_unchanged_and_follows_resets():
    actual=Policy(10);reference=Policy(10)
    observed=ObservedPolicy({'before':Policy(0),'candidate':actual},Teacher(),'candidate')
    assert observed.h==10 # delegate diagnostics to the actual driver
    flow=np.zeros(2);tof=np.ones(64);memory=np.ones(8)
    for _ in range(3):
        np.testing.assert_array_equal(observed.step_np(flow,tof,memory),reference.step_np(flow,tof,memory,.02))
    observed.reset_state();reference.reset_state()
    np.testing.assert_array_equal(observed.step_np(flow,tof,memory),reference.step_np(flow,tof,memory,.02))
    assert observed.records[0]['segment']!=observed.records[-1]['segment']
    assert observed.records[-1]['phase']=='cruise'


def test_episodes_split_on_reset_and_gap_and_mark_censoring():
    rows=[dict(time_s=t,segment=s) for t,s in [(0,0),(.02,0),(.04,1),(.06,1),(.10,1)]]
    events=mask_episodes(rows,[True,True,True,False,True],end_s=.12)
    assert len(events)==3
    assert abs(events[0]['duration_s']-.04)<1e-9
    assert events[0]['right_censored'] # reset is not observed release
    assert not events[1]['right_censored']
    assert events[2]['right_censored']


def test_common_window_clips_samples_and_excludes_manual_motion():
    from evaluate.report_practical_maneuver import window_metrics
    data=dict(time=np.array([0., 1., 2., 3.]), auto=np.array([False, True, True, True]),
              velocity=np.array([[9., 0.], [-1., 0.], [2., 0.], [100., 0.]]),
              yaw=np.zeros(4), clearance=np.array([1., .05, .2, 0.]))
    metrics=window_metrics(data, 2.5)
    assert metrics['auto_s']==1.5
    assert metrics['distance_m']==2.
    assert metrics['reverse_s']==1.
    assert metrics['below_10cm_s']==1.
