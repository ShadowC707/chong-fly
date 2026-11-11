"""Standalone-only synthetic heading and explicit initial-clearance check."""
import numpy as np
from simulation.survey_heading_sensor import SimulatedHeadingSensor


class SwathSimulationAdapter:
    def __init__(self, runner):
        self.system = runner.system
        nav = self.system.navigation
        if runner.step_index != 0 or runner.collision_radius_m != .2:
            raise ValueError('Swath requires a fresh session with a 20cm airframe')
        x, y = runner.start[:2]
        distances = [np.hypot(max(b.x_min-x, 0., x-b.x_max), max(b.y_min-y, 0., y-b.y_max))
                     for b in runner.env.boxes]
        distances.extend(np.hypot(c.center_x-x, c.center_y-y)-c.radius for c in runner.env.cylinders)
        room = runner.env.room
        distances.extend((x-room.x_min, room.x_max-x, y-room.y_min, room.y_max-y))
        if min(distances) < nav.observer.swath_config.bootstrap_radius_m:
            raise ValueError('Для цього профілю потрібна вільна початкова ділянка радіусом 1,35 м.')
        self.preflight = dict(required_radius_m=nav.observer.swath_config.bootstrap_radius_m,
            measured_clear_radius_m=float(min(distances)), source='simulation geometry, initial position only')
        nav.confirm_preflight()
        self.sensor = SimulatedHeadingSensor(nav.observer.reference_id, drift_rad_s=0., timestamp_error_s=0.)

    def capture(self, runner, now_s):
        yaw = runner.env.physics.quaternion_to_euler(runner.env.physics.quat)[2]
        return self.sensor.capture(yaw, timestamp_s=now_s)

    def contract(self):
        return dict(version='swath-synthetic-inputs-v1', preflight=self.preflight,
            heading_noise_deg=.25, heading_drift_deg_s=0., heading_timestamp_error_s=0.,
            heading_seed=self.sensor.seed, initial_heading_reference_rad=0.,
            geometry_input='preflight only; no runtime map or true-position input to navigation')
