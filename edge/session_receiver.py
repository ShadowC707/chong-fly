"""Session-aware executable receiver specification; no radio or FC driver.

Local authority and the local sensor-send clock are trusted. Pi packet times
are never used to extend a lease. Session identifiers are not authentication.
"""
from dataclasses import dataclass
import math

if __package__:
    from .flight_bridge import CommandGate, CommandLease, pwm_array
else:
    from flight_bridge import CommandGate, CommandLease, pwm_array


@dataclass(frozen=True)
class CommandPacket:
    session_id: str
    sequence: int
    lease: CommandLease | None


@dataclass(frozen=True)
class ReceiverEvent:
    session_id: str
    sequence: int
    receiver_time_s: float
    accepted_pwm: tuple | None
    reason: str
    tripped: bool


class SessionReceiver:
    def __init__(self):
        self.gate = CommandGate()
        self.mode = 'MANUAL'
        self.session_id = None
        self._sessions = set()
        self._clock = None
        self.latest_sequence = -1
        self._received_sequence = -1
        self.sensor_sent_s = None
        self.reason = 'reset_required'

    def _time(self, now_s):
        if (not isinstance(now_s, (int, float)) or not math.isfinite(now_s)
                or now_s < 0 or (self._clock is not None and now_s < self._clock)):
            self._trip('invalid_receiver_clock')
            raise ValueError('Receiver clock must be finite and monotonic')
        self._clock = now_s

    def _trip(self, reason):
        self.gate.tripped = True
        self.gate.pwm = None
        self.reason = reason

    def reset(self, *, session_id, now_s):
        self._time(now_s)
        if self.mode != 'MANUAL':
            raise ValueError('Receiver reset requires MANUAL')
        if not isinstance(session_id, str) or not session_id.strip() or session_id in self._sessions:
            raise ValueError('Reset requires a new nonempty session identifier')
        self._sessions.add(session_id)
        self.session_id = session_id
        self.gate = CommandGate()
        self.latest_sequence = -1
        self._received_sequence = -1
        self.sensor_sent_s = None
        self.reason = 'awaiting_command'

    def set_mode(self, mode, *, now_s):
        self._time(now_s)
        if mode not in ('MANUAL', 'SHADOW', 'AUTO'):
            self._trip('invalid_mode')
            raise ValueError('Unknown receiver authority')
        if self.mode == 'AUTO' and mode != 'AUTO':
            self._trip('authority_revoked')
        self.mode = mode

    def announce_sensor(self, sequence, *, now_s):
        self._time(now_s)
        if (self.session_id is None or type(sequence) is not int
                or sequence <= self.latest_sequence):
            self._trip('invalid_sensor_sequence')
            raise ValueError('Sensor sequence must increase within a reset session')
        self.latest_sequence = sequence
        self.sensor_sent_s = now_s

    def receive(self, packet, *, now_s):
        self._time(now_s)
        if not isinstance(packet, CommandPacket):
            self._trip('malformed_packet')
            return ReceiverEvent('', -1, now_s, None, self.reason, True)
        if packet.session_id != self.session_id or self.session_id is None:
            return ReceiverEvent(packet.session_id, packet.sequence, now_s, None,
                                 'session_mismatch', self.gate.tripped)
        accepted = None
        if (type(packet.sequence) is not int or packet.sequence != self.latest_sequence
                or packet.sequence <= self._received_sequence):
            self._trip('sequence_rejected')
        elif packet.lease is None:
            if self.gate.pwm is not None:
                self._trip('sender_revoked')
            elif not self.gate.tripped:
                self.reason = 'awaiting_command'
        elif not isinstance(packet.lease, CommandLease) or packet.lease.sequence != packet.sequence:
            self._trip('malformed_lease')
        elif self.gate.accept(packet.lease, received_s=now_s,
                              auto_authorized=self.mode == 'AUTO',
                              sensor_sent_s=self.sensor_sent_s,
                              latest_sensor_sequence=self.latest_sequence):
            accepted = self.gate.pwm
            self.reason = 'accepted'
        else:
            self.reason = 'lease_rejected'
        if type(packet.sequence) is int:
            self._received_sequence = max(self._received_sequence, packet.sequence)
        return ReceiverEvent(packet.session_id, packet.sequence, now_s, accepted,
                             self.reason, self.gate.tripped)

    def output(self, *, now_s, fallback_pwm):
        self._time(now_s)
        fallback = tuple(float(x) for x in pwm_array(fallback_pwm))
        accepted = None
        # Before the first lease, SHADOW/admission is allowed to take time.
        # Once active, poll the deadline independently of packet arrivals.
        if self.gate.pwm is not None and not self.gate.tripped:
            accepted = self.gate.command(now_s=now_s, auto_authorized=self.mode == 'AUTO')
            if accepted is None:
                self.reason = 'watchdog_expired' if self.mode == 'AUTO' else 'authority_revoked'
        return dict(session_id=self.session_id, sequence=self.gate.sequence,
                    accepted_pwm=accepted, applied_pwm=fallback if accepted is None else accepted,
                    source='fallback' if accepted is None else 'auto',
                    tripped=self.gate.tripped, reason=self.reason)
