import numpy as np
from evaluate.probe_maneuver_history import histories
from generator.generate_reflex_dataset import make_teacher


def test_identical_suffix_retains_distinct_teacher_braking_context():
    cases=histories(); outputs={}
    for name,x in cases.items():
        np.testing.assert_array_equal(x[50:],cases['always_clear'][50:])
        teacher=make_teacher('geometry-reflex-v5',noise_std_pwm=0)
        for row in x: y=teacher.step(row,dt=.02)
        outputs[name]=y
    assert outputs['always_clear'][2]>1500
    assert outputs['threat_then_clear'][2]<1300
    np.testing.assert_array_equal(outputs['always_clear'],outputs['threat_then_settle'])
