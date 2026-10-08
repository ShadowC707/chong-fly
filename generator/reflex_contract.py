"""Versioned demonstration contract shared by collection and pretraining."""
DATASET_VERSION = 'reflex-v3'
LEGACY_TEACHER_VERSION = 'geometry-reflex-v2'
BRAKING_TEACHER_VERSION = 'geometry-reflex-v3'
TEACHER_VERSION = 'geometry-reflex-v4'
DEFAULT_DATASET_PATH = 'data/reflex_dataset_brake_first_v4.pt'
SCENARIOS = ('clear', 'obstacle_left', 'obstacle_right', 'obstacle_center',
             'drift_left', 'drift_right', 'drift_forward', 'drift_backward')
BEHAVIORS = ('cruise', 'turn_right', 'turn_left', 'brake_left_drift', 'brake_right_drift', 'brake_forward')
REQUIRED_PATHS = {'lptc_flow': ['roll', 'pitch'], 'lc_looming': ['pitch', 'yaw']}
