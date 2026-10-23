"""Deterministic physical challenge scenes, separate from demonstration seeds."""
import numpy as np

from configs.flight_config import DEFAULT_DT
from simulation.drone_env import DroneSimulationEnv, RoomBoundaries, BoxObstacle

SCENE_VERSION = 'navigation-challenges-v1'
SCENES = ('clear', 'obstacle_left', 'obstacle_right', 'obstacle_center',
          'fast_center', 'drift_left', 'drift_right')


class NavigationScene(DroneSimulationEnv):
    """Reset keeps scenario geometry/velocity, including rollout's own reset.

    Large room avoids turning every long clear trajectory into a wall test.
    Every obstacle is a physical box used by both raycasting and collision.
    """
    def __init__(self, scenario):
        if scenario not in SCENES:
            raise ValueError('Unknown navigation challenge')
        self.scenario = scenario
        super().__init__(dt=DEFAULT_DT, engine='standalone', headless=True,
                         room=RoomBoundaries(-20, 20, -20, 20, 0, 4))
        self.cylinders, self.boxes = [], []
        if scenario.startswith('obstacle_') or scenario == 'fast_center':
            lo, hi = {'obstacle_left':(.02, .5), 'obstacle_right':(-.5, -.02)}.get(
                scenario, (-.35, .35))
            self.boxes = [BoxObstacle(1.1, 1.25, lo, hi, .2, 2.)]

    def reset(self, initial_pos=None, seed=None):
        super().reset(initial_pos=np.array([0., 0., 1.]) if initial_pos is None else initial_pos,
                      seed=seed)
        self.physics.quat[:] = [1., 0., 0., 0.]
        self.physics.vel[:] = [1.5 if self.scenario == 'fast_center' else .3,
                              .55 if self.scenario == 'drift_left' else
                              -.55 if self.scenario == 'drift_right' else 0., 0.]
        self.physics.omega[:] = 0
        return self._sample_observation(elapsed_dt=0.)
