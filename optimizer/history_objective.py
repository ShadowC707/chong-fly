"""Full-history Pitch objective; no penalty on correctly labelled fast changes."""
import torch


def _validate(prediction, target):
    if (prediction.shape != target.shape or prediction.ndim != 3 or
            prediction.shape[-1] != 4 or min(prediction.shape[:2]) < 1):
        raise ValueError('Expected matching nonempty [batch, time, 4] commands')
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError('Nonfinite commands')


def sequence_pitch_loss(prediction, target):
    _validate(prediction, target)
    return ((prediction[..., 2] - target[..., 2]) / 500.).square().mean()


@torch.no_grad()
def sequence_metrics(prediction, target, *, suffix_start=50):
    _validate(prediction, target)
    if type(suffix_start) is not int or not 0 <= suffix_start < prediction.shape[1]:
        raise ValueError('Suffix must contain at least one frame')
    error = (prediction[..., 2] - target[..., 2]).abs()
    suffix = prediction[:, suffix_start:, 2]
    suffix_error = error[:, suffix_start:]
    return dict(full_mean_error_pwm=float(error.mean()), full_max_error_pwm=float(error.max()),
        terminal_max_error_pwm=float(error[:, -5:].max()),
        suffix_mean_error_pwm=float(suffix_error.mean()),
        suffix_max_error_pwm=float(suffix_error.max()),
        suffix_all_within_20_pwm=bool((suffix_error <= 20).all()),
        suffix_fraction_within_20_pwm=float((suffix_error <= 20).float().mean()),
        suffix_peak_to_peak_pwm=(suffix.amax(1) - suffix.amin(1)).cpu().tolist())
