import pytest
import torch

from optimizer.release_cruise import cruise_cases, cruise_loss, motion_admission


def test_cruise_cases_preserve_existing_teacher_contract():
    from optimizer.balanced_maneuvers import auxiliary_cases
    expected = auxiliary_cases(12)[1]
    actual = cruise_cases()
    for key in ('X', 'Y'):
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    assert actual['X'].shape == (9, 200, 74)


def test_cruise_loss_trains_navigation_without_training_throttle():
    target = torch.full((2, 3, 4), 1500.)
    prediction = target.clone()
    prediction[..., 0] = 1800
    prediction[..., 2] -= 30
    prediction.requires_grad_()
    loss = cruise_loss(prediction, target)
    assert loss.item() == pytest.approx((30 / 500) ** 2 / 3)
    loss.backward()
    assert torch.all(prediction.grad[..., 0] == 0)
    assert torch.all(prediction.grad[..., 2] < 0)


def test_motion_admission_rejects_stationary_survival_and_reverse_motion():
    good = dict(duration_s=10., auto_s=9., crashed=False, mean_forward_speed_mps=.2)
    assert motion_admission(1526., good)['passed']
    assert not motion_admission(1495., good)['passed']
    for speed in (.003, -.2, float('nan')):
        assert not motion_admission(1526., dict(good, mean_forward_speed_mps=speed))['passed']
    for change in (dict(crashed=True), dict(duration_s=2.), dict(auto_s=0.)):
        assert not motion_admission(1526., dict(good, **change))['passed']


def test_training_visits_each_episode_and_saves_safe_metadata(tmp_path):
    from optimizer.release_cruise import fit

    class TinyPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.command = torch.nn.Parameter(torch.tensor([1213.,1500.,1500.,1500.]))
            self.visits = []
            self.projected = False

        def forward(self, x, dt):
            if len(x) == 8:
                self.visits.extend(x[:,0,0].long().tolist())
            return self.command.expand(*x.shape[:2],4), None

        def post_step(self):
            self.projected = True

    class Guard:
        steps = 0

        def __call__(self, policy):
            assert policy.projected
            policy.projected = False
            self.steps += 1

    x = torch.ones(104,2,74)
    x[:,:,0] = torch.arange(104)[:,None]
    y = torch.tensor([1213.,1500.,1530.,1500.]).expand(104,2,4).clone()
    data = dict(X=x,Y=y,valid=torch.ones(104,2,dtype=torch.bool),
        behavior=torch.zeros(104,2,dtype=torch.long),metadata=dict(behavior_names=['cruise']))
    policy = TinyPolicy()
    result = fit(policy,data,Guard(),dict(X=x[:1],Y=y[:1]),lambda _:None)
    assert result['progress'][-1]['updates'] == 130
    assert torch.bincount(torch.tensor(policy.visits)).tolist() == [10]*104
    assert policy.command[0].item() == 1213.
    assert policy.command[2].item() > 1500.
    path = tmp_path/'metadata.pt'
    torch.save(result,path)
    assert torch.load(path,weights_only=True)['progress'] == result['progress']
