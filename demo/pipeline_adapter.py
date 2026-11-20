"""Standalone-only synthetic heading for the pipeline memory, as in demo/swath_adapter.py.

Unlike the swath adapter there is no preflight clearance disk: the pipeline limits commands
and never needs free space to be proven, so any valid start is accepted. A memory fault
(heading or motion history interrupted) starts a new reference and a fresh heading sensor.
"""
from simulation.survey_heading_sensor import SimulatedHeadingSensor


class PipelineSimulationAdapter:
    def __init__(self, runner):
        self.system = runner.system
        self.navigation = self.system.navigation
        if runner.step_index != 0:
            raise ValueError('The pipeline memory requires a fresh session')
        self.base_reference = self.navigation.memory.reference_id
        self.sensor = SimulatedHeadingSensor(self.base_reference, drift_rad_s=0., timestamp_error_s=0.)

    def capture(self, runner, now_s):
        yaw = runner.env.physics.quaternion_to_euler(runner.env.physics.quat)[2]
        memory = self.navigation.memory
        if memory.faulted:
            reference = f'{self.base_reference}-r{memory.resets+1}'
            memory.reset(reference)
            self.sensor.reset(reference, initial_yaw_rad=yaw, timestamp_s=now_s)
        return self.sensor.capture(yaw, timestamp_s=now_s)

    def contract(self):
        return dict(version='pipeline-synthetic-inputs-v1', heading_noise_deg=.25, heading_drift_deg_s=0.,
            heading_timestamp_error_s=0., heading_seed=self.sensor.seed, initial_heading_reference_rad=0.,
            geometry_input='none; no map, truth position or preflight clearance reaches the controller',
            on_memory_fault='new reference, heading sensor restarted at the true yaw, memory empty')
