import math

import pytest

from simulation.survey_heading_sensor import SimulatedHeadingSensor


def test_seeded_measurements_bound_error_and_reset_reference():
    a=SimulatedHeadingSensor('a',seed=7,initial_yaw_rad=3.)
    b=SimulatedHeadingSensor('a',seed=7,initial_yaw_rad=3.)
    for i in range(101):
        t=i*.02;truth=3.+t
        x=a.capture(truth,timestamp_s=t)
        assert x==b.capture(truth,timestamp_s=t)
        error=(x.angle_rad-t+math.pi)%(2*math.pi)-math.pi
        assert abs(error)<=x.error_bound_rad+1e-12
    a.reset('b',initial_yaw_rad=-1.,timestamp_s=3.)
    x=a.capture(-1.,timestamp_s=3.)
    assert x.sample_id==0 and x.reference_id=='b'
    assert abs(x.angle_rad)<=x.error_bound_rad


def test_repeated_capture_is_not_a_held_measurement_and_clock_cannot_regress():
    sensor=SimulatedHeadingSensor('a')
    sensor.capture(0.,timestamp_s=0.)
    with pytest.raises(ValueError):sensor.capture(0.,timestamp_s=0.)
    with pytest.raises(ValueError):sensor.capture(0.,timestamp_s=-.1)


def test_left_positive_measurement_is_opposite_to_right_positive_rc():
    sensor=SimulatedHeadingSensor('a',noise_bound_rad=0.,drift_rad_s=0.)
    sensor.capture(0.,timestamp_s=0.)
    assert sensor.capture(-.2,timestamp_s=.1).angle_rad==pytest.approx(-.2)


@pytest.mark.parametrize('kwargs', [{'noise_bound_rad':-1},{'drift_rad_s':float('nan')},
                                    {'timestamp_error_s':-1},{'seed':True}])
def test_invalid_sensor_model_is_rejected(kwargs):
    with pytest.raises(ValueError):SimulatedHeadingSensor('a',**kwargs)
