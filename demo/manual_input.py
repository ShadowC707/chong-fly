"""Physical keyboard codes to RC offsets, using simulation time only."""
from dataclasses import dataclass, asdict
import math
import numpy as np


@dataclass(frozen=True)
class InputConfig:
    amplitude_pwm: float = 100.
    slew_pwm_per_s: float = 300.

    def __post_init__(self):
        if not (math.isfinite(self.amplitude_pwm) and 0 < self.amplitude_pwm <= 500
                and math.isfinite(self.slew_pwm_per_s) and self.slew_pwm_per_s > 0):
            raise ValueError('Invalid demo input bounds')


class ManualInput:
    CODES = {'KeyA', 'KeyD', 'KeyW', 'KeyS', 'ArrowLeft', 'ArrowRight'}

    def __init__(self, config=None):
        self.config = config or InputConfig()
        self.keys = set()
        self.axes = np.zeros(3)

    def key(self, code, down):
        if code not in self.CODES or type(down) is not bool:
            raise ValueError('Invalid flight key event')
        if down: self.keys.add(code)
        else: self.keys.discard(code)

    def clear(self):
        self.keys.clear()

    def synchronize(self, pwm):
        # Follow the real actuator command while AUTO owns the plant. This is
        # a slew anchor, not a manual target or a hidden velocity brake.
        self.axes[:] = np.asarray(pwm, dtype=float)[1:] - 1500

    def advance(self, dt):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError('Input dt must be positive and finite')
        k = self.keys
        target = self.config.amplitude_pwm * np.array([
            int('KeyD' in k)-int('KeyA' in k),
            int('KeyW' in k)-int('KeyS' in k),
            int('ArrowRight' in k)-int('ArrowLeft' in k)])
        limit = self.config.slew_pwm_per_s * dt
        self.axes += np.clip(target-self.axes, -limit, limit)
        return self.axes + 1500

    def manifest(self):
        return dict(asdict(self.config), axes='roll,pitch,yaw',
                    release='slew to neutral; no velocity brake')
