"""Explicit, bounded conversion of a commanded yaw budget to wall-clock time.

This is an engineering assumption about an actuator envelope, not measured
angular progress. It reads no simulator state and does not identify hardware.
"""
from dataclasses import asdict, dataclass, replace
import math

from .phased_coordinator import PhasedConfig


@dataclass(frozen=True)
class TurnEnvelope:
    # Existing nominal budget: 140/500 * 200 deg/s * 4 s = 224 deg.
    command_budget_deg: float = 224.
    min_full_scale_yaw_dps: float = 140.
    max_turn_s: float = 6.

    def __post_init__(self):
        if any(type(v) not in (int,float) or not math.isfinite(v) or v<=0
               for v in asdict(self).values()):
            raise ValueError('Turn envelope requires finite positive real values')

    def contract(self, base):
        resolved=resolve_turn_config(base,self)
        return dict(**asdict(self),
            version='commanded-turn-envelope-v1', measured_progress=False,
            provenance='Explicit exploratory assumption; not measured G4 calibration',
            mapping='Linear RC yaw offset / 500 * full_scale_yaw_dps',
            resolved_turn_timeout_s=resolved.turn_timeout_s,
            episode_timeout_s=resolved.episode_timeout_s,
            limitation='Does not guarantee actual angle; stall, delay and saturation still possible')


def resolve_turn_config(base: PhasedConfig, envelope: TurnEnvelope) -> PhasedConfig:
    rate=base.yaw_pwm/500.*envelope.min_full_scale_yaw_dps
    seconds=envelope.command_budget_deg/rate
    if seconds < base.turn_timeout_s-1e-9:
        raise ValueError('Envelope must not silently shorten the established turn budget')
    if abs(seconds-base.turn_timeout_s)<=1e-9:
        seconds=base.turn_timeout_s
    if seconds > envelope.max_turn_s+1e-9:
        raise ValueError('Required turn time exceeds explicit wall-clock cap')
    if envelope.max_turn_s+base.recover_s > base.episode_timeout_s+1e-9:
        raise ValueError('Turn cap leaves no recovery reserve in episode budget')
    # Do not extend episode or braking budgets. A long braking phase can still
    # consume the shared episode budget before the full TURN allowance is used.
    return replace(base,turn_timeout_s=seconds)
