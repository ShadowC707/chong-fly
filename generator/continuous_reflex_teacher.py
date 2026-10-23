"""Research-only repair of support-count dilution in v4's spatial summary.

Only the force aggregation is continuous. The inherited turn latch, minimum
turn command, release hysteresis and flow gating are still a hybrid policy.
The original research identity is retained; v5 is its explicit collection version.
"""
import numpy as np

from generator.generate_reflex_dataset import ExpertReflexPolicy
from generator.reflex_contract import TEACHER_VERSION, TEACHER_AGGREGATION_VERSION


class ContinuousReflexTeacher(ExpertReflexPolicy):
    version = 'geometry-reflex-continuous-v1'
    training_ready = False
    aggregation_version = 'max-brake-force-weighted-direction-v1'

    def _summarize_repulsion(self, force, active):
        # A weaker additional ray must not reduce braking for an existing
        # closer obstacle. Max is bounded by the inherited distance clamp and
        # continuous, though not differentiable where the strongest ray swaps.
        peak = float(np.max(force))
        mass = float(np.sum(force))
        if mass == 0.:
            return 0., 0.
        # Use force, not a binary active count, as the directional weight.
        # |direction| <= 1; hence peak*direction -> 0 as all forces -> 0.
        # Uniform-depth obstacles retain the historical lateral summary.
        direction = float(np.dot(force, -self.nx))/mass
        return peak, peak*direction


class CorrectedExpertReflexPolicy(ContinuousReflexTeacher):
    """Versioned v5 collection teacher; behavior matches continuous-v1 exactly.

    This is still a research heuristic, not hardware flight admission.
    """
    version = TEACHER_VERSION
    aggregation_version = TEACHER_AGGREGATION_VERSION
