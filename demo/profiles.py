"""Repository configuration identities for Standalone profiles and recorded replay."""
from pathlib import Path

from edge.runtime import sha256

_CONFIGS = Path(__file__).resolve().parents[1]/'configs'
ORIGINAL_CONFIG = _CONFIGS/'mvp_runtime_v3.json'
CANDIDATE_A_CONFIG = _CONFIGS/'candidate_a_adaptive_tau_v1.json'
HISTORICAL_CANDIDATE_A_CONFIG = _CONFIGS/'demo_candidate_a.json'

OVERLAY_PROFILES = ('candidate_a_brake', 'phased_v2', 'phased_smooth', 'phased_recovery',
                    'phased_escape', 'phased_side_guard', 'phased_swath')
PHASED_PROFILES = tuple(p for p in OVERLAY_PROFILES if p.startswith('phased_'))
LIVE_PROFILE_LABELS = {
    'original': 'Попередній MVP',
    'candidate_a': 'Кандидат A · контроль',
    'phased_v2': 'Кандидат A · фазовий v2',
    'candidate_a_brake': 'Кандидат A · гальмування біля перешкоди',
    'phased_side_guard': 'Кандидат A · фазовий із бічним захистом',
    'phased_swath': 'Кандидат A · спостережена смуга · дослідний',
}

PROFILE_ROLES = {
    'original': 'historical-offline-control',
    'candidate_a': 'fixed-xy-neural-control',
    'phased_side_guard': 'behavioral-baseline',
    'candidate_a_brake': 'recorded-failure-control',
    'phased_swath': 'standalone-swath-experiment',
}


def profile_config(mode, recorded_config_sha256=None):
    if mode == 'original':
        return ORIGINAL_CONFIG
    if (recorded_config_sha256 is not None and HISTORICAL_CANDIDATE_A_CONFIG.is_file()
            and sha256(HISTORICAL_CANDIDATE_A_CONFIG) == recorded_config_sha256):
        return HISTORICAL_CANDIDATE_A_CONFIG
    return CANDIDATE_A_CONFIG
