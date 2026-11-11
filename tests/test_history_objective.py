import pytest
import torch

from optimizer.history_objective import sequence_pitch_loss, sequence_metrics


def test_interior_error_is_trained_even_with_perfect_endpoint():
    target = torch.full((2, 200, 4), 1500.)
    prediction = target.clone()
    prediction[:, 80:120, 2] += 200
    prediction.requires_grad_()
    loss = sequence_pitch_loss(prediction, target)
    loss.backward()
    assert loss.item() == pytest.approx(0.032)
    assert torch.all(prediction.grad[:, 80:120, 2] > 0)
    assert torch.all(prediction.grad[:, -5:, :] == 0)


def test_exact_fast_maneuver_is_not_penalized_and_other_channels_are_separate():
    target = torch.full((1, 100, 4), 1500.)
    target[:, 25:50, 2] = 1200
    prediction = target.clone()
    prediction[..., [0, 1, 3]] += 100
    assert sequence_pitch_loss(prediction, target) == 0


def test_sequence_metrics_distinguish_endpoint_fit_from_sustained_fit():
    target = torch.full((3, 200, 4), 1500.)
    prediction = target.clone()
    prediction[:, 80:100, 2] = 1800
    result = sequence_metrics(prediction, target, suffix_start=50)
    assert result['terminal_max_error_pwm'] == 0
    assert result['suffix_max_error_pwm'] == 300
    assert result['suffix_mean_error_pwm'] == 40
    assert result['suffix_peak_to_peak_pwm'] == [300, 300, 300]
    assert not result['suffix_all_within_20_pwm']


def test_metrics_reject_invalid_or_empty_suffix():
    x = torch.zeros(1, 10, 4)
    with pytest.raises(ValueError):
        sequence_metrics(x, x, suffix_start=10)
    x[0, 0, 2] = float('nan')
    with pytest.raises(ValueError):
        sequence_pitch_loss(x, x)
