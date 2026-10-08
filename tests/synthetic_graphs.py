"""Eight-node source→target examples, without FlyWire identities.

Only write to a caller-provided test directory. Separate flow and looming
relays feed four distinct motor outputs; one removable edge models a route gap.
"""
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from generator.graph_reducer import ReducedModel


def write_tiny_model(directory, *, looming_to_yaw=True, sparse=True):
    weights = np.zeros((8, 8), dtype=np.float32)
    weights[0, 2] = .4
    weights[1, 3] = .4
    weights[2, 4:8] = .2
    weights[2, 5] = -.2
    weights[3, 4:7] = .2
    if looming_to_yaw:
        weights[3, 7] = .2
    model = ReducedModel(
        W=csr_matrix(weights) if sparse else weights,
        k=8, reducer_name='test_example', cluster_map=np.arange(8, dtype=np.int32),
        sensor_index_map={'lptc_flow': [0], 'lc_looming': [1]},
        motor_index_map={'throttle': [4], 'roll': [5], 'pitch': [6], 'yaw': [7]},
        provenance={'source_kind': 'synthetic', 'purpose': 'test_fixture'},
    )
    return Path(model.save(str(directory))['meta'])
