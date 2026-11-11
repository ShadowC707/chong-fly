import numpy as np

from evaluate.mvp_selection import mirror_scene, mirror_input, first_turn, PassivePair


def test_reflection_is_involution_and_changes_expected_side():
    scene = dict(start=[1., 2., 1.], yaw=.3, expected_sign=1,
                 room=dict(x_min=-4., x_max=4., y_min=-3., y_max=5., z_min=0., z_max=3.),
                 boxes=[dict(x_min=1., x_max=2., y_min=-2., y_max=-1., z_min=0., z_max=2.)],
                 cylinders=[dict(center_x=2., center_y=1., radius=.3, height=2.)])
    reflected = mirror_scene(scene)
    assert reflected['boxes'][0]['y_min'] == 1.
    assert reflected['expected_sign'] == -1
    assert mirror_scene(reflected) == scene
    x = np.arange(74, dtype=np.float32)
    np.testing.assert_array_equal(mirror_input(mirror_input(x)), x)
    assert mirror_input(x)[1] == -x[1]
    np.testing.assert_array_equal(mirror_input(x)[2:66].reshape(8,8), x[2:66].reshape(8,8)[:, ::-1])


def test_first_turn_requires_contiguous_same_sign_and_auto():
    t = np.arange(9)*.02
    pwm = np.full((9,4), 1500.)
    pwm[:,3] += [80,80,80,60,-60,0,-80,-90,-100]
    auto = np.array([False]*3+[True]*6)
    result = first_turn(t,pwm,auto,np.full(9,.9),-1)
    assert result['time_s'] == .12
    assert result['correct'] and result['sign'] == -1
    assert result['latency_s'] == .06
    assert first_turn(t,pwm,np.zeros(9,bool),np.full(9,.9),-1)['correct'] is False


class CounterPolicy:
    default_dt = .02
    def __init__(self): self.reset_state()
    def reset_state(self): self.n = 0
    def step_np(self, flow, tof, memory=None, dt=None):
        self.n += 1
        return np.full(4,1500.+self.n)


def test_observer_preserves_driver_and_resets_every_observer():
    observer = PassivePair(CounterPolicy(), CounterPolicy(), CounterPolicy())
    x = np.ones(74, np.float32)
    for expected in (1501,1502):
        np.testing.assert_array_equal(observer.step_np(x[:2],x[2:66],x[66:],dt=.02),np.full(4,expected))
    observer.reset_state()
    np.testing.assert_array_equal(observer.step_np(x[:2],x[2:66],x[66:],dt=.02),np.full(4,1501))
    assert observer.records[-1]['segment'] == 1
    assert len(observer.records) == 3
