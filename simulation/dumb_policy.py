"""Control experiment (step 8): phased_side_guard rules with the network replaced by a constant command.

The network still runs (same system, same timing); only its Roll/Pitch/Yaw are replaced, before the
coordinator, by a constant cruise command: Pitch 1505 (median network Pitch in phase CRUISE of the
phased_side_guard spread runs), Roll 1500, Yaw 1500. Throttle is left as given (the altitude regulator
owns it). Research-only; not a profile of the demo.
"""
import numpy as np

from simulation.side_clearance import SideClearanceCoordinator

CONSTANT_ROLL, CONSTANT_PITCH, CONSTANT_YAW = 1500., 1505., 1500.


class ConstantForwardSideClearance(SideClearanceCoordinator):
    def apply(self, network_pwm, **kwargs):
        command = np.asarray(network_pwm, dtype=float).copy()
        if command.shape == (4,):
            command[1:4] = (CONSTANT_ROLL, CONSTANT_PITCH, CONSTANT_YAW)
        return super().apply(command, **kwargs)


def make_dumb(runner):
    """Swap a phased_side_guard runner's coordinator to the constant-command variant in place."""
    coordinator = runner.system.runtime.coordinator
    if type(coordinator) is not SideClearanceCoordinator:
        raise ValueError('The dumb-policy control needs a phased_side_guard runner')
    coordinator.__class__ = ConstantForwardSideClearance
    return runner
