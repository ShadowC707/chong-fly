"""Bounded offline replay of raw capture v1. Never a live control source.

Recorded timestamps remain old notebook times; replay does not establish RF
freshness, calibration, device identity or a physically connected controller.
"""
import hashlib
import json
from pathlib import Path

from .gamepad_input import GamepadSample, GamepadState, _integer

MAX_BYTES = 8 * 1024 * 1024


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate capture JSON key')
        result[key] = value
    return result


def _constant(_):
    raise ValueError('Non-finite capture JSON number')


class ReplayGamepadSource:
    def __init__(self, report, *, sha256=None):
        if not isinstance(report, dict) or report.get('version') != 'gamepad-raw-capture-v1':
            raise ValueError('Expected raw capture v1')
        if (report.get('calibration_profile_applied') is not False
                or type(report.get('udp_sends')) is not int or report['udp_sends'] != 0
                or type(report.get('fc_attempts')) is not int or report['fc_attempts'] != 0):
            raise ValueError('Expected read-only raw capture')
        hz = report.get('requested_hz')
        _integer(hz, 1, 200)
        rows = report.get('samples')
        if not isinstance(rows, list) or not 1 <= len(rows) <= 12000:
            raise ValueError('Expected 1..12000 recorded samples')
        samples = []
        for row in rows:
            if not isinstance(row, dict) or set(row) != {
                    'sample_id', 'read_completed_us', 'source', 'api_status', 'state', 'connected'}:
                raise ValueError('Invalid raw sample fields')
            _integer(row['sample_id'], 0, 2**64 - 1)
            _integer(row['read_completed_us'], 0, 2**64 - 1)
            _integer(row['api_status'], 0, 2**32 - 1)
            if row['source'] not in ('mock', 'xinput', 'replay'):
                raise ValueError('Unknown recorded input source')
            if samples and (row['sample_id'] <= samples[-1].sample_id
                    or row['read_completed_us'] < samples[-1].read_completed_us):
                raise ValueError('Recorded sample identity/time regressed')
            raw_state = row['state']
            try:
                state = GamepadState(**raw_state) if isinstance(raw_state, dict) else None
            except TypeError as exc:
                raise ValueError('Invalid recorded state fields') from exc
            connected = row['api_status'] == 0
            if (type(row['connected']) is not bool or row['connected'] != connected
                    or connected != (state is not None)
                    or not connected and raw_state is not None):
                raise ValueError('Contradictory connected/state/status fields')
            samples.append(GamepadSample(row['sample_id'], row['read_completed_us'], 'replay', row['api_status'], state))
        self._samples = tuple(samples)
        self._index = 0
        self.sample_count = len(samples)
        self.requested_hz = hz
        self.sha256 = sha256

    def poll(self):
        if self._index < self.sample_count:
            result = self._samples[self._index]
            self._index += 1
            return result
        # The exhausted source has no new observation or time. Do not repeat
        # a held stick value, synthesize neutral, or re-date old input as fresh.
        last = self._samples[-1]
        return GamepadSample(last.sample_id, last.read_completed_us, 'replay', 1167, None)


def load_capture(path):
    with Path(path).open('rb') as stream:
        raw = stream.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError('Capture exceeds 8 MiB')
    try:
        report = json.loads(raw.decode('utf-8'), object_pairs_hook=_object, parse_constant=_constant)
    except (UnicodeError, RecursionError) as exc:
        raise ValueError('Invalid capture JSON encoding/nesting') from exc
    return ReplayGamepadSource(report, sha256=hashlib.sha256(raw).hexdigest())
