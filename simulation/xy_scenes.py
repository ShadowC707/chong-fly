"""Paired local approaches to a room corner; experimental scenes, not holdout."""
from dataclasses import replace
import math

import numpy as np

from configs.flight_config import DEFAULT_DT
from simulation.drone_env import DroneSimulationEnv, RoomBoundaries, BoxObstacle, CylinderObstacle


DENSITY_SCENES = ('arena_sparse', 'arena_medium', 'arena_dense')


class DensityArena(DroneSimulationEnv):
    """Nested fixed layouts; conservative bounding-box gaps >= 1.3 m.

    Gap means free surface-to-surface width, before subtracting drone diameter.
    Every obstacle also has a >= 1.3 m passage to the room boundary.
    """
    def __init__(self, name):
        if name not in DENSITY_SCENES:
            raise ValueError('Unknown density arena')
        self.scene_name = name
        super().__init__(engine='standalone', headless=True, dt=DEFAULT_DT)
        self.room = RoomBoundaries(-4., 4., -4., 4., 0., 3.)
        self.boxes, self.cylinders = [], []
        locations = [(2.3,0.), (-2.3,2.3), (0.,-2.3), (2.3,2.3), (-2.3,0.),
                     (0.,2.3), (2.3,-2.3), (-2.3,-2.3)]
        count = dict(arena_sparse=2, arena_medium=5, arena_dense=8)[name]
        self.footprints = []
        for i,(x,y) in enumerate(locations[:count]):
            radius = (.25,.30,.35)[i%3]
            if i%3 == 1:
                self.boxes.append(BoxObstacle(x-radius,x+radius,y-radius,y+radius,0.,2.5))
            else:
                self.cylinders.append(CylinderObstacle(x,y,radius,2.5))
            self.footprints.append((x-radius,x+radius,y-radius,y+radius))
        if min(self._gaps()) < 1.3:
            raise ValueError('Arena does not meet the 1.3 m passage contract')

    def _gaps(self):
        boxes = self.footprints
        between = [math.hypot(max(0., a[0]-b[1], b[0]-a[1]),
                              max(0., a[2]-b[3], b[2]-a[3]))
                   for i,a in enumerate(boxes) for b in boxes[i+1:]]
        wall = [min(a[0]+4,4-a[1],a[2]+4,4-a[3]) for a in boxes]
        return min(between), min(wall)

    def specification(self):
        obstacle, wall = self._gaps()
        return dict(version='density-arena-v1', scene=self.scene_name,
            obstacle_count=len(self.footprints), minimum_obstacle_gap_m=obstacle,
            minimum_wall_gap_m=wall, gap_method='conservative XY bounding-box surface distances',
            minimum_required_gap_m=1.3, initial_position_m=[0.,0.,1.], initial_yaw_rad=0.,
            initial_velocity_world_mps=[0.,0.,0.], nested_layout=True)

    def reset(self, initial_pos=None, seed=None):
        super().reset(initial_pos=np.array([0.,0.,1.]) if initial_pos is None else initial_pos, seed=seed)
        self.physics.quat[:] = [1.,0.,0.,0.]
        self.physics.vel[:] = 0.
        self.physics.omega[:] = 0.
        return self._sample_observation(elapsed_dt=0.)


class DensityStartArena(DensityArena):
    """Same geometry/noise as the historical arena, with an explicit cold start."""
    def __init__(self, name, *, start_pos, start_yaw):
        position=np.asarray(start_pos,dtype=float)
        if position.shape!=(3,) or not np.isfinite(position).all() or not math.isfinite(start_yaw):
            raise ValueError('Invalid density start')
        self.start_pos=position.copy()
        self.start_yaw=float(start_yaw)
        super().__init__(name)
        from simulation.collisions import contact_and_clearance
        contact,self.initial_clearance=contact_and_clearance(position,position,self.room,
            self.boxes,self.cylinders,self.collision_radius_m)
        if contact is not None or self.initial_clearance<.2:
            self.close()
            raise ValueError('Density start needs >=0.2 m body clearance')

    def specification(self):
        spec=super().specification()
        spec.update(version='density-start-v1',initial_position_m=self.start_pos.tolist(),
            initial_yaw_rad=self.start_yaw,initial_clearance_m=self.initial_clearance,
            recurrent_initial_state='zero',initial_attitude='level',initial_angular_velocity_rads=[0.,0.,0.])
        return spec

    def reset(self, initial_pos=None, seed=None):
        if initial_pos is not None:
            raise ValueError('Explicit density start must not be overridden at reset')
        # Keep the same two sensor samples as DensityArena.reset; set final attitude
        # before the second sample, with no extra random draws.
        DroneSimulationEnv.reset(self,initial_pos=self.start_pos,seed=seed)
        self.physics.quat[:]=[math.cos(self.start_yaw/2),0.,0.,math.sin(self.start_yaw/2)]
        self.physics.vel[:]=0.
        self.physics.omega[:]=0.
        return self._sample_observation(elapsed_dt=0.)


def reflect_geometry_y(env):
    env.boxes = [replace(b, y_min=-b.y_max, y_max=-b.y_min) for b in env.boxes]
    env.cylinders = [replace(c, center_y=-c.center_y) for c in env.cylinders]
    env.room = replace(env.room, y_min=-env.room.y_max, y_max=-env.room.y_min)


class CornerApproachScene(DroneSimulationEnv):
    def __init__(self, *, mirrored=False):
        self.mirrored = mirrored
        super().__init__(engine='standalone', headless=True, dt=DEFAULT_DT)
        if mirrored:
            reflect_geometry_y(self)

    def specification(self):
        sign = -1 if self.mirrored else 1
        yaw = sign*math.pi/4
        return dict(version='local-corner-approach-v1', mirrored=self.mirrored,
            initial_position_m=[2.4, sign*2.7, 1.], initial_yaw_rad=yaw,
            initial_velocity_world_mps=[.3*math.cos(yaw), .3*math.sin(yaw), 0.],
            recurrent_initial_state='zero; fresh approach, not a restored baseline snapshot')

    def reset(self, initial_pos=None, seed=None):
        spec = self.specification()
        super().reset(initial_pos=np.array(spec['initial_position_m']) if initial_pos is None else initial_pos,
                      seed=seed)
        yaw = spec['initial_yaw_rad']
        self.physics.quat[:] = [math.cos(yaw/2), 0., 0., math.sin(yaw/2)]
        self.physics.vel[:] = spec['initial_velocity_world_mps']
        self.physics.omega[:] = 0.
        return self._sample_observation(elapsed_dt=0.)
