"""Sensor-only entry point shared by live demo and evaluation installation."""
from .yaw_experiment import YawHandoverSystem


class SwathSystem(YawHandoverSystem):
    def tick(self, frame, *, now_s, heading_sample=None, **kwargs):
        self.navigation.observe(frame, heading_sample, now_s=now_s)
        return super().tick(frame, now_s=now_s, **kwargs)
