"""Repository configuration identities for Standalone profiles and recorded replay."""
from pathlib import Path

from edge.runtime import sha256

_CONFIGS = Path(__file__).resolve().parents[1]/'configs'
ORIGINAL_CONFIG = _CONFIGS/'mvp_runtime_v3.json'
CANDIDATE_A_CONFIG = _CONFIGS/'candidate_a_adaptive_tau_v1.json'
HISTORICAL_CANDIDATE_A_CONFIG = _CONFIGS/'demo_candidate_a.json'
PIPELINE_V1_CONFIG = _CONFIGS/'pipeline_v1.json'
PIPELINE_CONFIGS = {'pipeline_v1': PIPELINE_V1_CONFIG,
                    'pipeline_v1_r1': _CONFIGS/'pipeline_v1_r1.json',
                    'pipeline_v1_r2': _CONFIGS/'pipeline_v1_r2.json',
                    'pipeline_v1_r3': _CONFIGS/'pipeline_v1_r3.json'}

OVERLAY_PROFILES = ('candidate_a_brake', 'phased_v2', 'phased_smooth', 'phased_recovery',
                    'phased_escape', 'phased_side_guard', 'phased_swath', 'pipeline_v1',
                    'pipeline_v1_r1', 'pipeline_v1_r2', 'pipeline_v1_r3')
PHASED_PROFILES = tuple(p for p in OVERLAY_PROFILES if p.startswith('phased_'))
LIVE_PROFILE_LABELS = {
    'original': 'Попередній MVP',
    'candidate_a': 'Кандидат A · контроль',
    'phased_v2': 'Кандидат A · фазовий v2',
    'candidate_a_brake': 'Кандидат A · гальмування біля перешкоди',
    'phased_side_guard': 'Кандидат A · фазовий із бічним захистом',
    'phased_swath': 'Кандидат A · спостережена смуга · дослідний',
    'pipeline_v1': 'Кандидат A · конвеєр v1 · дослідний',
    'pipeline_v1_r1': 'Кандидат A · конвеєр v1, коло 1 (пам\'ять) · дослідний',
    'pipeline_v1_r2': 'Кандидат A · конвеєр v1, коло 2 (гальмо) · дослідний',
    'pipeline_v1_r3': 'Кандидат A · конвеєр v1, коло 3 (відступ коридором) · дослідний',
}

PROFILE_ROLES = {
    'original': 'historical-offline-control',
    'candidate_a': 'fixed-xy-neural-control',
    'phased_side_guard': 'behavioral-baseline',
    'candidate_a_brake': 'recorded-failure-control',
    'phased_swath': 'standalone-swath-experiment',
    'pipeline_v1': 'standalone-pipeline-experiment',
    'pipeline_v1_r1': 'standalone-pipeline-experiment',
    'pipeline_v1_r2': 'standalone-pipeline-experiment',
    'pipeline_v1_r3': 'standalone-pipeline-experiment',
}


def profile_config(mode, recorded_config_sha256=None):
    if mode == 'original':
        return ORIGINAL_CONFIG
    if (recorded_config_sha256 is not None and HISTORICAL_CANDIDATE_A_CONFIG.is_file()
            and sha256(HISTORICAL_CANDIDATE_A_CONFIG) == recorded_config_sha256):
        return HISTORICAL_CANDIDATE_A_CONFIG
    return CANDIDATE_A_CONFIG
