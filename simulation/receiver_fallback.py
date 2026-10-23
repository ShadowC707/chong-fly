"""Local, sensor-only fallback for the simulated receiver, NOT FC firmware.

Keep the range estimator warm in AUTO, freeze its integral and track the actual
throttle. On loss of authority, resume local height hold and flow damping.
"""
import math
import numpy as np

from .altitude_control import AltitudeHold


class ReceiverFallback:
    def __init__(self):
        self.altitude = AltitudeHold()
        self.flow = None
        self.now = None

    def step(self, flow, sample, *, now_s, active, previous_pwm):
        flow = np.asarray(flow, dtype=float)
        if flow.shape != (2,) or not np.isfinite(flow).all():
            raise ValueError('Fallback needs valid local flow')
        dt = .004 if self.now is None else now_s-self.now
        self.flow = flow.copy() if self.flow is None else self.flow + (
            -math.expm1(-dt/.12))*(flow-self.flow)
        throttle = self.altitude.update(sample, now_s=now_s, allow_hold=True,
                                        integrate=active).throttle_pwm
        if not active:
            self.altitude.synchronize_output(float(previous_pwm[0]), now_s=now_s)
            throttle = float(previous_pwm[0])
        self.now = now_s
        return (throttle, float(1500+np.clip(400*self.flow[1], -100, 100)),
                float(1500-np.clip(400*self.flow[0], -100, 100)), 1500.)
