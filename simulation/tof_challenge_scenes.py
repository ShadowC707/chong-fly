"""Small deterministic geometry challenges; development cases, not a holdout."""
from dataclasses import asdict
import math

from simulation.benchmark_scenes import NavigationScene
from simulation.drone_env import BoxObstacle, CylinderObstacle

SCENE_VERSION = 'tof-geometry-challenges-v1'
SCENES = ('thin_center_gap', 'thin_on_ray', 'thin_left', 'thin_right',
          'wide_center', 'near_left_far_right', 'near_right_far_left',
          'corner_left', 'corner_right', 'two_posts', 'post_cluster', 'low_cylinder')


class ToFChallengeScene(NavigationScene):
    def __init__(self, scenario):
        if scenario not in SCENES:
            raise ValueError('Unknown ToF challenge')
        super().__init__('clear')
        self.scenario = scenario
        if scenario.startswith('thin_'):
            y = {'thin_center_gap':0., 'thin_on_ray':.6*math.tan(math.radians(45/14)),
                 'thin_left':.12, 'thin_right':-.12}[scenario]
            self.boxes = [BoxObstacle(.6, .62, y-.005, y+.005, .2, 1.8)]
        elif scenario == 'wide_center':
            self.boxes = [BoxObstacle(.6, .65, -.15, .15, .2, 1.8)]
        elif scenario in ('near_left_far_right', 'near_right_far_left'):
            self.boxes = [BoxObstacle(.55, .65, .05, .3, .2, 1.8),
                          BoxObstacle(.9, 1.05, -.35, -.1, .2, 1.8)]
            if scenario == 'near_right_far_left':
                self.boxes = [BoxObstacle(b.x_min, b.x_max, -b.y_max, -b.y_min, b.z_min, b.z_max)
                              for b in self.boxes]
        elif scenario in ('corner_left', 'corner_right'):
            self.boxes = [BoxObstacle(.8, .9, -.1, 1.6, .1, 2.),
                          BoxObstacle(-.2, .9, .65, .75, .1, 2.)]
            if scenario == 'corner_right':
                self.boxes = [BoxObstacle(b.x_min, b.x_max, -b.y_max, -b.y_min, b.z_min, b.z_max)
                              for b in self.boxes]
        elif scenario == 'two_posts':
            self.cylinders = [CylinderObstacle(.7, y, .04, 1.8) for y in (-.2, .2)]
        elif scenario == 'post_cluster':
            self.cylinders = [CylinderObstacle(.6, .1, .03, 1.8),
                              CylinderObstacle(.85, -.15, .04, 1.8),
                              CylinderObstacle(1.1, .18, .05, 1.8)]
        elif scenario == 'low_cylinder':
            self.cylinders = [CylinderObstacle(.55, 0., .3, .8)]

    def specification(self):
        return dict(version=SCENE_VERSION, scenario=self.scenario, units='metres',
            frame='FLU: x forward, y left, z up', room=asdict(self.room),
            boxes=[asdict(b) for b in self.boxes], cylinders=[asdict(c) for c in self.cylinders],
            initial_position=[0., 0., 1.], initial_velocity=[.3, 0., 0.],
            initial_quaternion_wxyz=[1., 0., 0., 0.], collision_radius_m=self.collision_radius_m)
