"""Nominal synchronous/multirate sensor acquisition schedule for simulation adapters."""
from dataclasses import replace
import numpy as np
from edge.flight_bridge import SensorFrame
from edge.sensors import Measurement, SensorSnapshot


class SampleSchedule:
    """Sample at the first 50-Hz tick after each nominal sensor deadline.

    Capture is at that tick (not backdated to the nominal deadline). This gives
    60/80-ms ToF and 20/40-ms range intervals for nominal 15/30 Hz.
    """
    def __init__(self, profile):
        self.profile=profile
        self.epochs={}
        self.cached={}

    def frame(self,i,flow,tof,altitude,*,invalid_altitude=False):
        t=i*.02
        if self.profile=='sync':
            return SensorFrame(i,t,flow,tof,replace(altitude,valid=not invalid_altitude))
        values=dict(flow=tuple(flow),tof=tuple(np.asarray(tof).ravel()),
                    altitude=(altitude.range_m,altitude.roll_rad,altitude.pitch_rad),
                    attitude=(altitude.roll_rad,altitude.pitch_rad))
        for name,hz in dict(flow=50,tof=15,altitude=30,attitude=50).items():
            epoch=int(np.floor(t*hz+1e-8))
            if self.epochs.get(name)!=epoch:
                self.epochs[name]=epoch
                self.cached[name]=Measurement(epoch,t,values[name])
        channels=self.cached.copy()
        if invalid_altitude: channels['altitude']=replace(channels['altitude'],valid=False)
        return SensorSnapshot(i,t,**channels)
