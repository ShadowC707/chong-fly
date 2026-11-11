import numpy as np

from evaluate.validate_maneuver_teacher import PolicyComparison


class Policy:
    default_dt = .02
    metadata = {}
    def __init__(self):
        self.calls = 0
    def reset_state(self):
        self.calls = 0
    def step_np(self, flow, tof, memory, dt):
        self.calls += 1
        return np.array([1500., 1500., 1500., 1500. + self.calls])


class Teacher:
    def reset(self):
        self.calls = 0
    def step(self, x, dt):
        self.calls += 1
        return np.array([1500., 1500., 1400., 1500. - self.calls])


def test_probe_compares_identical_inputs_without_changing_driver():
    p = PolicyComparison({'before': Policy(), 'control': Policy()}, Teacher(), 'before')
    flow = np.array([.1, -.05]); tof = np.ones(64)
    action = p.step_np(flow, tof)
    np.testing.assert_array_equal(action, [1500, 1500, 1500, 1501])
    np.testing.assert_array_equal(p.records[0]['input'][:2], flow.astype(np.float32))
    assert p.records[0]['outputs']['teacher'][2] == 1400
    flow[:] = 0
    assert p.records[0]['input'][0] != 0


def test_teacher_driver_and_reset_keep_separate_history_segments():
    p = PolicyComparison({'before': Policy()}, Teacher(), 'teacher')
    a = p.step_np(np.zeros(2), np.ones(64))
    assert a[3] == 1499
    p.reset_state()
    np.testing.assert_array_equal(p.step_np(np.zeros(2), np.ones(64)), a)
    assert len(p.records) == 2
    assert p.records[0]['segment'] != p.records[1]['segment']
