import numpy as np
from evaluate.teacher_closed_loop import TeacherDriver


class Observer:
    default_dt = .02
    def reset_state(self): self.calls = 0
    def step_np(self, flow, tof, memory, dt):
        self.calls += 1
        return np.array([1213, 1500, 1526, 1500], np.float32)


class Teacher:
    def reset(self):
        self.calls=0; self._escape_sign=None; self._cruise_recovery=1.
    def step(self, x, dt):
        self.calls+=1
        return np.array([1213, 1500, 1530, 1510], np.float32)


def test_teacher_drives_first_call_without_baseline_takeover():
    p=TeacherDriver({'before': Observer()}, Teacher())
    np.testing.assert_array_equal(p.step_np(np.zeros(2),np.ones((8,8))), [1213,1500,1530,1510])
    assert p.records[0]['driver']=='teacher'
    assert p.records[0]['phase']=='cruise'
    assert p.records[0]['input'].shape==(74,)
    assert p.policies['before'].calls==1
    p.reset_state()
    p.step_np(np.zeros(2),np.ones(64))
    assert p.records[-1]['segment']==p.records[0]['segment']+1
    assert p.teacher.calls==1
