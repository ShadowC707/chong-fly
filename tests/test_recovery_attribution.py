import numpy as np

from evaluate.recovery_attribution import branch_step, replay_inputs


class Accumulator:
    def reset_state(self): self.h = np.zeros(1, np.float32)
    def step_np(self, flow, tof, memory, dt):
        self.h = self.h + flow[0]
        return np.array([0, 0, self.h[0], 0], np.float32)


def test_branches_do_not_mutate_live_state_or_input():
    p = Accumulator(); p.h = np.array([10.], np.float32)
    x = np.ones(74, np.float32); x[0] = -2
    assert branch_step(p, x)[2] == 8
    assert branch_step(p, x, reset=True)[2] == -2
    assert branch_step(p, x, flow_x=0)[2] == 10
    assert branch_step(p, x, steps=3)[2] == 4
    np.testing.assert_array_equal(p.h, [10])
    assert x[0] == -2


def test_replay_resets_at_recorded_segments():
    p = Accumulator()
    x = np.ones((3, 74), np.float32)
    predicted = replay_inputs(p, x, np.array([0, 0, 1]))
    np.testing.assert_array_equal(predicted[:, 2], [1, 2, 1])
