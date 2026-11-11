"""Measured heading and capture-time ToF alignment, independent of flight RC."""
from dataclasses import replace
import math

import numpy as np
import pytest

from edge.sensors import Measurement
from edge.survey_heading import HeadingConfig, HeadingSample, SurveyHeadingTracker


def heading(i, degrees, *, t=None, reference='scan-1', **kwargs):
    return HeadingSample(reference, i, i*.02 if t is None else t,
                         math.radians(degrees), **kwargs)


def tof(i, t, value=.5):
    return Measurement(i, t, np.full(64, value))


@pytest.mark.parametrize('angles,expected', [([179,-179,-177],183), ([-179,179,177],-183)])
def test_wrap_preserves_measured_turn_direction(angles, expected):
    tracker=SurveyHeadingTracker('scan-1')
    for i, angle in enumerate(angles):
        result=tracker.add_heading(heading(i,angle),now_s=i*.02)
    assert math.degrees(result.angle_rad)==pytest.approx(expected)


def test_held_tof_retains_capture_heading_and_identity_not_current_heading():
    tracker=SurveyHeadingTracker('scan-1')
    tracker.add_heading(heading(0,0),now_s=0)
    frame=tof(0,0)
    first=tracker.align_tof(frame,reference_id='scan-1',now_s=0)
    tracker.add_heading(heading(1,4),now_s=.02)
    held=tracker.align_tof(frame,reference_id='scan-1',now_s=.02)
    assert first.state=='new' and held.state=='held'
    assert replace(first,state='held')==held
    assert held.heading_rad==0 and held.heading_sample_id==0


def test_delayed_tof_uses_buffered_capture_heading():
    tracker=SurveyHeadingTracker('scan-1')
    for i in range(4):tracker.add_heading(heading(i,i*4),now_s=i*.02)
    aligned=tracker.align_tof(tof(0,.02),reference_id='scan-1',now_s=.06)
    assert math.degrees(aligned.heading_rad)==pytest.approx(4)
    assert aligned.heading_timestamp_s==.02


def test_capture_mismatch_is_bounded_not_silently_interpolated():
    tracker=SurveyHeadingTracker('scan-1')
    tracker.add_heading(heading(0,10,error_bound_rad=.01,timestamp_error_s=.001),now_s=0)
    aligned=tracker.align_tof(tof(0,.01),reference_id='scan-1',now_s=.02)
    assert math.degrees(aligned.heading_rad)==pytest.approx(10)
    assert aligned.heading_error_bound_rad==pytest.approx(.01+math.radians(240)*.011)


def test_only_future_heading_cannot_label_older_tof():
    tracker=SurveyHeadingTracker('scan-1')
    tracker.add_heading(heading(1,4),now_s=.02)
    with pytest.raises(ValueError,match='preceding'):tracker.align_tof(tof(0,0),reference_id='scan-1',now_s=.02)


def test_held_heading_does_not_extend_history_or_refresh_capture():
    tracker=SurveyHeadingTracker('scan-1');sample=heading(0,0)
    first=tracker.add_heading(sample,now_s=0)
    assert tracker.add_heading(sample,now_s=.02)==first
    with pytest.raises(ValueError,match='stale'):tracker.add_heading(sample,now_s=.05)
    assert tracker.faulted


@pytest.mark.parametrize('bad', [
    heading(0,1), heading(1,1,t=0), heading(1,1,valid=False),
    heading(1,30), heading(1,1,reference='other'), heading(1,1,t=.06),
    heading(1,1,error_bound_rad=math.nan), heading(1,1,timestamp_error_s=-1),
    heading(1,190), heading(1,1,error_bound_rad=.3),
])
def test_bad_heading_breaks_continuity_until_explicit_new_reference(bad):
    tracker=SurveyHeadingTracker('scan-1')
    tracker.add_heading(heading(0,0),now_s=0)
    with pytest.raises(ValueError):tracker.add_heading(bad,now_s=.06 if bad.timestamp_s==.06 else .02)
    assert tracker.faulted
    with pytest.raises(ValueError):tracker.add_heading(heading(2,2),now_s=.04)
    with pytest.raises(ValueError):tracker.align_tof(tof(1,.02),reference_id='scan-1',now_s=.04)
    with pytest.raises(ValueError):tracker.reset('scan-1')
    tracker.reset('scan-2')
    tracker.add_heading(heading(0,0,reference='scan-2'),now_s=.04)
    assert not tracker.faulted


@pytest.mark.parametrize('change', ['payload','timestamp','regressed','invalid','future','stale','reference'])
def test_tof_contract_rejects_bad_reuse(change):
    tracker=SurveyHeadingTracker('scan-1')
    tracker.add_heading(heading(0,0),now_s=0)
    frame=tof(2,0);tracker.align_tof(frame,reference_id='scan-1',now_s=0)
    now=.02
    bad={'payload':tof(2,0,.4),'timestamp':tof(2,.01),'regressed':tof(1,.01),
         'invalid':replace(frame,valid=False),'future':tof(3,.04),
         'stale':frame,'reference':frame}[change]
    if change=='stale':
        for i in range(1,7):tracker.add_heading(heading(i,0),now_s=i*.02)
        now=.12
    else:tracker.add_heading(heading(1,0),now_s=.02)
    with pytest.raises(ValueError):tracker.align_tof(bad,reference_id='other' if change=='reference' else 'scan-1',now_s=now)


def test_tof_payload_is_copied():
    tracker=SurveyHeadingTracker('scan-1');tracker.add_heading(heading(0,0),now_s=0)
    frame=tof(0,0);aligned=tracker.align_tof(frame,reference_id='scan-1',now_s=0)
    frame.value[:]=0
    assert aligned.normalized_ranges==(0.5,)*64
    with pytest.raises(ValueError):tracker.align_tof(frame,reference_id='scan-1',now_s=.01)


def test_capture_alignment_gap_and_clock_regression_rejected():
    tracker=SurveyHeadingTracker('scan-1');tracker.add_heading(heading(0,0),now_s=0)
    with pytest.raises(ValueError,match='alignment'):tracker.align_tof(tof(0,.03),reference_id='scan-1',now_s=.03)
    with pytest.raises(ValueError,match='clock'):tracker.add_heading(heading(1,1),now_s=.01)


def test_new_reference_discards_old_tof_and_heading_history():
    tracker=SurveyHeadingTracker('scan-1');tracker.add_heading(heading(0,0),now_s=0)
    tracker.align_tof(tof(0,0),reference_id='scan-1',now_s=0)
    tracker.reset('scan-2')
    tracker.add_heading(heading(0,50,t=.04,reference='scan-2'),now_s=.04)
    with pytest.raises(ValueError,match='preceding'):tracker.align_tof(tof(1,.02),reference_id='scan-2',now_s=.04)
    new=tracker.align_tof(tof(1,.04),reference_id='scan-2',now_s=.04)
    assert math.degrees(new.heading_rad)==pytest.approx(50)
    assert new.state=='new'


def test_uncertainty_over_budget_rejects_alignment_even_with_fresh_heading():
    tracker=SurveyHeadingTracker('scan-1')
    tracker.add_heading(heading(0,0,error_bound_rad=math.radians(8)),now_s=0.)
    with pytest.raises(ValueError,match='uncertainty'):
        tracker.align_tof(tof(0,.02),reference_id='scan-1',now_s=.02)


def test_tof_clock_uncertainty_contributes_to_declared_angle_bound():
    tracker=SurveyHeadingTracker('scan-1',HeadingConfig(tof_timestamp_error_s=.005))
    tracker.add_heading(heading(0,0),now_s=0.)
    result=tracker.align_tof(tof(0,0),reference_id='scan-1',now_s=0.)
    assert result.heading_error_bound_rad==pytest.approx(math.radians(240)*.005)


def test_fresh_tof_cannot_reuse_stale_heading_or_refresh_held_proof():
    tracker=SurveyHeadingTracker('scan-1');tracker.add_heading(heading(0,0),now_s=0.)
    frame=tof(0,0);tracker.align_tof(frame,reference_id='scan-1',now_s=0.)
    with pytest.raises(ValueError,match='fresh heading'):
        tracker.align_tof(frame,reference_id='scan-1',now_s=.05)


@pytest.mark.parametrize('kwargs', [{'max_age_s':0},{'history_s':.02},
    {'max_rate_rad_s':1000},{'max_error_rad':float('nan')},{'tof_timestamp_error_s':-1}])
def test_invalid_heading_budgets(kwargs):
    with pytest.raises(ValueError):HeadingConfig(**kwargs)
