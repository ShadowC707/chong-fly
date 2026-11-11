"""Reference authority and multirate sensor integration shared by SITL runs."""
from edge.altitude_control import AltitudeHold, AltitudeSample
from edge.session_receiver import CommandPacket, SessionReceiver
from edge.sensors import SensorSnapshot
from edge.system import observe_receiver
from simulation.sensor_schedule import SampleSchedule


class IntegratedControl:
    def __init__(self, system, profile):
        if profile not in ('sync', 'multi_rate', 'tof_only', 'altitude_only'):
            raise ValueError('Unknown sensor profile')
        self.system = system
        self.schedule = SampleSchedule(profile)
        self.receiver = SessionReceiver()
        self.receiver.reset(session_id=system.config.session_id, now_s=0.)
        self.manual_altitude = AltitudeHold(system.runtime.altitude.config)
        self.manual_pwm = (self.manual_altitude.config.hover_pwm, 1500., 1500., 1500.)
        self.previous_pwm = self.manual_pwm
        self.rows = []

    def output(self, now_s):
        # Called on EVERY physical tick, even when no new packet is delivered.
        return self.receiver.output(now_s=now_s, fallback_pwm=self.manual_pwm)

    def tick(self, i, flow, tof, altitude, *, stop_requested=False, request='AUTO', manual_input=None):
        now = i*.02
        before = self.output(now)
        active = before['source'] == 'auto'
        frame = self.schedule.frame(i, flow, tof, altitude, invalid_altitude=not altitude.valid)
        if isinstance(frame, SensorSnapshot):
            m = frame.altitude
            held = AltitudeSample(*m.value, m.timestamp_s, m.valid)
        else:
            held = frame.altitude
        throttle = self.manual_altitude.update(held, now_s=now, allow_hold=True,
                                               integrate=not active).throttle_pwm
        if active:
            self.manual_altitude.synchronize_output(self.previous_pwm[0], now_s=now)
            throttle = self.previous_pwm[0]
        # Explicit neutral-axes simulation fallback; never reuse an old turn.
        axes = (manual_input.advance(.02) if manual_input is not None and request!='AUTO'
                else (1500., 1500., 1500.))
        self.manual_pwm = (throttle, *axes)
        manual = (throttle, *self.previous_pwm[1:]) if active else self.manual_pwm
        self.receiver.announce_sensor(i, now_s=now)
        if request=='AUTO' and self.receiver.gate.tripped: request = 'SHADOW'
        result = self.system.tick(frame, now_s=now, request=request,
                                  manual_pwm=manual, stop_requested=stop_requested)
        lease = result['lease']
        self.receiver.set_mode('AUTO' if lease is not None else
                               ('MANUAL' if request=='MANUAL' else 'SHADOW'), now_s=now)
        event = self.receiver.receive(CommandPacket(self.system.config.session_id, i, lease), now_s=now)
        trace = result['trace']
        if trace.issued_pwm is not None:
            trace = observe_receiver(trace, session_id=event.session_id, sequence=event.sequence,
                accepted_pwm=event.accepted_pwm, receiver_time_s=now, reason=event.reason)
        selected = self.output(now)
        self.previous_pwm = selected['applied_pwm']
        if selected['source'] == 'auto':
            self.manual_altitude.synchronize_output(self.previous_pwm[0], now_s=now)
            if manual_input is not None: manual_input.synchronize(self.previous_pwm)
        row = trace.to_dict()
        row.update(selected=selected, manual_pwm=list(manual), stop_requested=stop_requested)
        if 'yaw_handover' in result: row['yaw_handover'] = result['yaw_handover']
        self.rows.append(row)
        return dict(trace=trace, selected=selected, manual_pwm=manual, frame=frame, result=result)
