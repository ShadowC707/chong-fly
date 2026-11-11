"""Narrow Pi data-plane adapter. Bind only after a trusted local session reset.

No implicit rebinding on packet arrival, no remote reset or authority grant.
The existing offline RunConfig describes the core, not this outer I/O layer.
"""
from dataclasses import replace

from .hardware_wire import decode_sensor, encode_command, to_snapshot
from .session_receiver import CommandPacket
from .system import RunConfig, build_system


class SensorDecodeError(ValueError):
    """Invalid wire bytes, distinct from an expired clock mapping."""


class PiHardwareAdapter:
    def __init__(self,config,*,session_id,clock_mapping):
        if not isinstance(config,RunConfig) or config.sensor_mode!='multi_rate_v1':
            raise ValueError('Multi-rate RunConfig required')
        # build_system checks the pinned bundle and executing code, unchanged.
        self.system=build_system(replace(config,session_id=session_id))
        self.session_id=session_id
        self.clock_mapping=clock_mapping

    def handle_sensor(self,data,*,now_s):
        try:
            message=decode_sensor(data)
        except (ValueError,TypeError,OverflowError) as exc:
            raise SensorDecodeError(str(exc)) from exc
        if message.session_id!=self.session_id:
            return None  # old telemetry cannot reset hidden state or bind a session
        frame=to_snapshot(message,self.clock_mapping,now_s=now_s)
        request=message.local_mode
        if message.receiver_tripped and request=='AUTO': request='SHADOW'
        result=self.system.tick(frame,now_s=now_s,request=request,
            manual_pwm=message.manual_pwm,stop_requested=message.stop_requested)
        packet=CommandPacket(self.session_id,message.sequence,result['lease'])
        return encode_command(packet),result['trace']
