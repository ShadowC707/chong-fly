"""Versioned demonstration contract shared by collection and pretraining."""
DATASET_VERSION = 'reflex-v3'
TEACHER_VERSION = 'geometry-reflex-v2'
DEFAULT_DATASET_PATH = 'data/reflex_dataset_v3.pt'
SCENARIOS = ('clear', 'obstacle_left', 'obstacle_right', 'obstacle_center',
             'drift_left', 'drift_right', 'drift_forward', 'drift_backward')
BEHAVIORS = ('cruise', 'turn_right', 'turn_left', 'brake_left_drift', 'brake_right_drift', 'brake_forward')
REQUIRED_PATHS = {'lptc_flow': ['roll', 'pitch'], 'lc_looming': ['pitch', 'yaw']}
