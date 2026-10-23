"""Shared ownership contract for navigation training and supervised evaluation.

The four-value RC interface is retained, but element zero from a navigation
policy is only a placeholder. Only AltitudeHold supplies the applied throttle.
Changing its calibration changes the experiment, not merely its metadata.
"""
from simulation.altitude_control import AltitudeHold

CONTROL_VERSION = 'navigation-altitude-v1'
NAVIGATION_INDICES = (1, 2, 3)
NAVIGATION_CHANNELS = ('roll', 'pitch', 'yaw')


def control_contract(altitude_hold=None):
    controller = AltitudeHold() if altitude_hold is None else altitude_hold
    return {
        'version': CONTROL_VERSION,
        'action_order': ['throttle', 'roll', 'pitch', 'yaw'],
        'learned_channels': list(NAVIGATION_CHANNELS),
        'learned_indices': list(NAVIGATION_INDICES),
        'altitude': controller.contract(),
    }
