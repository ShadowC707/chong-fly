import numpy as np
from optimizer.recovery_readout import fit_readout


def test_fit_uses_equal_group_weights_not_frame_counts():
    groups={'recover': (np.array([[1., 0.]]), np.array([2.])),
            'anchor': (np.array([[0., 1.]]), np.array([-1.]))}
    w=fit_readout(groups, np.zeros(2), regularization=.01)
    repeated={**groups, 'recover': (np.repeat(groups['recover'][0], 20, axis=0), np.full(20, 2.))}
    np.testing.assert_allclose(w, fit_readout(repeated, np.zeros(2), regularization=.01))
    np.testing.assert_allclose(w, [2/1.01, -1/1.01])


def test_zero_residual_keeps_reference_weights():
    h=np.array([[1., 2.], [-2., 1.]])
    w=np.array([.3, -.4])
    np.testing.assert_allclose(fit_readout({'anchor': (h, h@w)}, w, regularization=.1), w)
