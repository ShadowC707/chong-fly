from dataclasses import replace
import math

import numpy as np
import pytest

from edge.sensors import Measurement, SensorSnapshot, SensorBudgets, SensorTracker
from simulation.altitude_control import AltitudeHold, AltitudeSample, AltitudeTelemetryError


def snapshot(i, *, altitude=None, tof=None, attitude=None):
    t = i*.02
    return SensorSnapshot(i, t, Measurement(i,t,(0.,0.)),
        tof or Measurement(i,t,tuple(np.ones(64))),
        altitude or Measurement(i,t,(.982,0.,0.)),
        attitude or Measurement(i,t,(0.,0.)))


def test_channel_holds_keep_identity_and_capture_attitude():
    tracker = SensorTracker()
    first = snapshot(0)
    tracker.resolve(first, now_s=0.)
    second = snapshot(1, altitude=first.altitude, tof=first.tof,
                      attitude=Measurement(1,.02,(.1,.2)))
    resolved = tracker.resolve(second, now_s=.02)
    assert resolved['states'] == dict(flow='new', tof='held', altitude='held', attitude='new')
    assert resolved['altitude'].timestamp_s == 0.
    assert resolved['altitude'].roll_rad == 0.
    assert resolved['attitude'] == (.1,.2)


@pytest.mark.parametrize('kind', ['payload','timestamp','regressed_id','same_time_new_id','invalid','stale','future'])
def test_invalid_channel_does_not_commit_partial_snapshot(kind):
    tracker = SensorTracker()
    first = snapshot(1)
    tracker.resolve(first, now_s=.02)
    value = first.altitude
    if kind == 'payload': value=replace(value,value=(.98,.1,0.))
    if kind == 'timestamp': value=replace(value,timestamp_s=.03)
    if kind == 'regressed_id': value=replace(value,sample_id=0)
    if kind == 'same_time_new_id': value=replace(value,sample_id=2)
    if kind == 'invalid': value=replace(value,valid=False)
    if kind == 'future': value=replace(value,sample_id=2,timestamp_s=.05)
    i = 5 if kind == 'stale' else 2
    with pytest.raises(ValueError):
        tracker.resolve(snapshot(i,altitude=value),now_s=i*.02)
    # No fresh flow/tof state from the rejected transaction was committed.
    assert tracker.resolve(snapshot(2),now_s=.04)['states']['flow'] == 'new'


def test_held_altitude_does_not_create_derivative_and_uses_control_time():
    c = AltitudeHold()
    a = AltitudeSample(.962,0.,0.,0.,True)
    c.update(a,now_s=0.,allow_hold=True)
    for i in (1,2):
        result = c.update(a,now_s=i*.02,allow_hold=True)
        assert result.vertical_speed_mps == 0.
    assert c._integral == pytest.approx(.02*.04)
    b = AltitudeSample(.965,0.,0.,.06,True)
    result = c.update(b,now_s=.06,allow_hold=True)
    expected = -math.expm1(-.06/c.config.velocity_filter_tau_s)*(.003/.06)
    assert result.vertical_speed_mps == pytest.approx(expected)
    held = c.update(b,now_s=.08,allow_hold=True)
    assert held.vertical_speed_mps == result.vertical_speed_mps


@pytest.mark.parametrize('change', ['angles','stale','clock'])
def test_altitude_hold_rejects_invalid_reuse(change):
    c=AltitudeHold(); a=AltitudeSample(.982,0.,0.,0.,True)
    c.update(a,now_s=0.,allow_hold=True)
    b=replace(a,roll_rad=.1) if change=='angles' else a
    now=.08 if change=='stale' else (0. if change=='clock' else .02)
    with pytest.raises(AltitudeTelemetryError): c.update(b,now_s=now,allow_hold=True)


def test_new_sample_every_tick_preserves_synchronous_numbers():
    legacy, held = AltitudeHold(), AltitudeHold()
    for i in range(200):
        t=i*.02; a=AltitudeSample(.982+.005*math.sin(t),.01*math.sin(t),0.,t,True)
        assert legacy.update(a,now_s=t) == held.update(a,now_s=t,allow_hold=True)
        assert legacy._integral == held._integral


def test_shadow_freezes_integral_and_synchronizes_without_losing_estimator():
    c=AltitudeHold()
    for i in range(200):
        c.update(AltitudeSample(.902,0.,0.,i*.02,True),now_s=i*.02,integrate=False)
    assert c._integral == 0.
    before=(c._height,c._velocity,c._time)
    c.synchronize_output(1200.,now_s=3.98)
    assert (c._height,c._velocity,c._time) == before
    command=c.update(AltitudeSample(.902,0.,0.,4.,True),now_s=4.)
    assert abs(command.throttle_pwm-1200.) <= 3.+1e-10


def test_budgets_are_per_channel_and_explicit():
    tracker=SensorTracker(SensorBudgets(tof_max_age_s=.10,altitude_max_age_s=.06))
    first=snapshot(0); tracker.resolve(first,now_s=0.)
    assert tracker.resolve(snapshot(4,tof=first.tof),now_s=.08)['states']['tof']=='held'
    with pytest.raises(ValueError,match='altitude'):
        SensorTracker().resolve(snapshot(4,altitude=first.altitude),now_s=.08)


def test_hold_age_boundary_tolerates_only_float_roundoff():
    c=AltitudeHold(); a=AltitudeSample(.982,0.,0.,0.,True)
    c.update(a,now_s=0.,allow_hold=True)
    c.update(a,now_s=np.nextafter(.06,1.),allow_hold=True)
    with pytest.raises(AltitudeTelemetryError): c.update(a,now_s=.0601,allow_hold=True)
