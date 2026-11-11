import numpy as np
import torch
from evaluate.teacher_encoding_contract import histories,teacher_targets
from simulation.policy import SensorInputLayer


def test_current_encoding_hides_teacher_release_margin():
    x=histories()
    # A characterization of the current mismatch, not a desired future contract.
    assert not np.array_equal(x[0],x[1])
    encoder=SensorInputLayer(encoding='proximity-v1',learnable_scale=False)
    encoded=encoder(torch.from_numpy(x)).numpy()
    np.testing.assert_array_equal(encoded[0],encoded[1])
    y=teacher_targets(x)
    np.testing.assert_array_equal(y[0,:20],y[1,:20])
    assert y[0,-1,2]==1500 and y[1,-1,2]==1530
    assert y[0,-1,3]==1620 and y[1,-1,3]==1500
