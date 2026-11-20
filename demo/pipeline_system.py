"""Sensor-only entry point of the pipeline_v1 profile: hazard memory is fed before every tick."""
from edge.system import FlightSystem
from .pipeline_adapter import PipelineSimulationAdapter


class PipelineSystem(FlightSystem):
    adapter_class = PipelineSimulationAdapter
    manifest_key = 'pipeline_inputs'

    def tick(self, frame, *, now_s, heading_sample=None, **kwargs):
        self.navigation.observe(frame, heading_sample, now_s=now_s)
        return super().tick(frame, now_s=now_s, **kwargs)
