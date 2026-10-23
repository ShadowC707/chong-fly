import numpy as np
import pytest


def test_padding_is_excluded_and_disconnected_frames_are_not_a_long_episode():
    from scripts.audit_training_coverage import summarize_mask
    mask = np.array([[1, 1, 0, 1, 0], [1, 0, 0, 0, 0]], dtype=bool)
    valid = np.array([[1, 1, 1, 1, 0], [0, 0, 0, 0, 0]], dtype=bool)
    y = np.full((2, 5, 4), 1500.)
    y[0, :, 3] = 1850
    result = summarize_mask(mask, valid, y, [0, 1], .02)
    assert result['frames'] == 3
    assert result['episodes'] == 1
    assert result['longest_run_s'] == pytest.approx(.04)
    assert result['episodes_with_0_5s_run'] == 0
    assert result['yaw_target_mean_pwm'] == 1850


def test_coarse_column_coverage_does_not_imply_full_height_probe_pattern():
    from scripts.audit_training_coverage import condition_masks
    x = np.zeros((1, 3, 74))
    x[..., 2:] = 1
    grid = x[..., 2:66].reshape(1, 3, 8, 8)
    grid[0, 0, 0, 0] = .1
    grid[0, 1, :, 0] = .1
    grid[0, 2, :, 0] = .1
    x[0, 2, 0] = .2
    masks = condition_masks(x, max_range=3.5, threat_m=.8)
    item = next(m for m in masks if m['name'] == 'left_d0.35_w1')
    assert item['coarse'].tolist() == [[True, True, True]]
    assert item['full_band'].tolist() == [[False, True, True]]
    assert item['full_band_low_flow'].tolist() == [[False, True, False]]


def test_no_matching_frames_produces_no_fake_zero_target():
    from scripts.audit_training_coverage import summarize_mask
    r = summarize_mask(np.zeros((1, 2), bool), np.ones((1, 2), bool),
                       np.zeros((1, 2, 4)), [0], .02)
    assert r['yaw_target_mean_pwm'] is None
    assert r['episodes'] == 0


def test_duplicate_or_overlapping_split_is_rejected():
    from scripts.audit_training_coverage import validate_split
    with pytest.raises(ValueError, match='split'):
        validate_split([0, 1], [1, 2], 3)
    with pytest.raises(ValueError, match='split'):
        validate_split([0, 0], [1, 2], 3)
    assert validate_split([0, 1], [2], 3) is None
