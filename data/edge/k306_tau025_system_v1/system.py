"""Explicit assembly and command evidence. NumPy + stdlib; no transport or arming.

Run as a module (repository) or system.py (flat bundle) to resolve a config and
write its manifest. Tick callers supply synchronous SensorFrames themselves.
"""
import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields, replace
import hashlib
import json
import math
from pathlib import Path
import re
import sys

if __package__:
    from .runtime import FlightRuntime, AltitudeHold, XYCoordinator, sha256
    from .flight_bridge import BridgeConfig, FlightBridge, pwm_array
else:
    from runtime import FlightRuntime, AltitudeHold, XYCoordinator, sha256
    from flight_bridge import BridgeConfig, FlightBridge, pwm_array


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


@dataclass(frozen=True)
class RunConfig:
    bundle_dir: str
    expected_metadata_sha256: str
    coordinator_mode: str = 'baseline'
    coordinator_preset: str = 'original_v1'
    levy_explore: bool = False
    stop_hover: bool = False
    seed: int = 42
    bridge: BridgeConfig = field(default_factory=BridgeConfig)
    session_id: str = 'offline'
    transport_mode: str = 'offline'

    def __post_init__(self):
        if not isinstance(self.bundle_dir, (str, Path)) or not str(self.bundle_dir).strip():
            raise ValueError('A bundle directory is required')
        object.__setattr__(self, 'bundle_dir', str(Path(self.bundle_dir).resolve()))
        if not isinstance(self.expected_metadata_sha256, str) or not re.fullmatch(
                '[0-9a-f]{64}', self.expected_metadata_sha256):
            raise ValueError('Expected SHA-256 must be explicit lowercase hex')
        if (self.coordinator_mode not in ('baseline', 'anti_rollback', 'turn_latch', 'both')
                or self.coordinator_preset not in ('original_v1', 'near_zero_v1')):
            raise ValueError('Unknown coordinator configuration')
        if any(type(v) is not bool for v in (self.levy_explore, self.stop_hover)):
            raise ValueError('Behavior flags must be boolean')
        if (self.levy_explore or self.stop_hover) and self.coordinator_mode != 'both':
            raise ValueError('Search/STOP requires both coordinator')
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError('Seed must be a nonnegative integer')
        if self.transport_mode != 'offline':
            raise ValueError('Only offline transport is implemented; no hardware fallback')
        if not isinstance(self.session_id, str) or not self.session_id.strip():
            raise ValueError('Nonempty trace session_id required')
        bridge = self.bridge
        if isinstance(bridge, dict):
            if set(bridge)-{f.name for f in fields(BridgeConfig)}:
                raise ValueError('Unknown bridge fields')
            bridge = BridgeConfig(**bridge)
        if not isinstance(bridge, BridgeConfig) or any(type(v) not in (int, float)
                for v in asdict(bridge).values()) or bridge.lease_s > .04:
            raise ValueError('Bridge bounds must be numeric; receiver lease limit is 0.04 s')
        object.__setattr__(self, 'bridge', bridge)

    @classmethod
    def from_json(cls, path):
        path = Path(path).resolve()
        data = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(data, dict) or set(data)-{f.name for f in fields(cls)}:
            raise ValueError('Unknown RunConfig fields')
        if isinstance(data.get('bundle_dir'), str):
            data['bundle_dir'] = str(path.parent/data['bundle_dir'])
        return cls(**data)

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class CommandTrace:
    session_id: str
    sequence: int | None
    tick_time_s: float | None
    sensor_time_s: float | None
    manifest_sha256: str
    authority: str
    reason: str
    fault: str | None
    fault_origin: str | None
    network_pwm: tuple | None
    coordinated_pwm: tuple | None
    supervised_pwm: tuple | None
    issued_pwm: tuple | None
    accepted_pwm: tuple | None = None
    receiver_status: str = 'unobserved'
    receiver_time_s: float | None = None
    receiver_reason: str | None = None
    coordinator_reasons: tuple = ()
    behavior_state: str | None = None

    @property
    def applied_pwm(self):
        return self.supervised_pwm

    def to_dict(self):
        result = asdict(self)
        result['applied_pwm'] = self.supervised_pwm
        stages = ('network_pwm', 'coordinated_pwm', 'supervised_pwm', 'issued_pwm', 'accepted_pwm')
        result['changed_channels'] = {}
        for previous, current in zip(stages, stages[1:]):
            a, b = getattr(self, previous), getattr(self, current)
            result['changed_channels'][current] = None if a is None or b is None else [
                name for i, name in enumerate(('throttle', 'roll', 'pitch', 'yaw')) if a[i] != b[i]]
        return result


def observe_receiver(trace, *, session_id, sequence, accepted_pwm, receiver_time_s, reason):
    """Attach an explicit receiver observation to THIS issue, without changing control.

    Receiver time is its own clock, not comparable with Pi tick time. None means
    a reported rejection; absence of a report remains 'unobserved'. Session is
    trace correlation only, not a session-aware wire protocol or authentication.
    """
    if (trace.issued_pwm is None or trace.receiver_status != 'unobserved'
            or session_id != trace.session_id or type(sequence) is not int or sequence != trace.sequence
            or type(receiver_time_s) not in (int, float) or not math.isfinite(receiver_time_s)
            or receiver_time_s < 0 or not isinstance(reason, str) or not reason.strip()):
        raise ValueError('Receiver observation does not match an unobserved issued command')
    accepted = None if accepted_pwm is None else tuple(float(x) for x in pwm_array(accepted_pwm))
    if accepted is not None and accepted != trace.issued_pwm:
        raise ValueError('Receiver reported PWM differs from issued command')
    return replace(trace, accepted_pwm=accepted,
                   receiver_status='rejected' if accepted is None else 'accepted',
                   receiver_time_s=float(receiver_time_s), receiver_reason=reason)


def _finite_time(value):
    # Diagnostics must not break MANUAL revocation or fault returns on bad inputs.
    try:
        return float(value) if math.isfinite(value) else None
    except (TypeError, ValueError, OverflowError):
        return None


class FlightSystem:
    def __init__(self, config, runtime, manifest):
        self.config = config
        self.runtime = runtime
        self.bridge = FlightBridge(runtime, config.bridge)
        self._manifest = deepcopy(manifest)
        self.manifest_sha256 = hashlib.sha256((_canonical(manifest)+'\n').encode('utf-8')).hexdigest()

    @property
    def manifest(self):
        return deepcopy(self._manifest)

    def write_manifest(self, path):
        # Fix LF bytes across Windows/Linux so trace identity equals file SHA-256.
        with Path(path).open('x', encoding='utf-8', newline='\n') as target:
            target.write(_canonical(self._manifest)+'\n')

    def tick(self, frame, *, now_s, request, manual_pwm, stop_requested=False):
        # Clear only diagnostics, including when Bridge skips runtime for MANUAL/fault.
        self.runtime.last_stages = {}
        result = self.bridge.tick(frame, now_s=now_s, request=request,
                                  manual_pwm=manual_pwm, stop_requested=stop_requested)
        stages = {k: tuple(float(x) for x in v) for k, v in self.runtime.last_stages.items()}
        lease = result['lease']
        xy = self.runtime.coordinator
        decision = xy.last_decision if xy is not None and 'coordinated_pwm' in stages else {}
        sequence = getattr(frame, 'sequence', None)
        trace = CommandTrace(
            session_id=self.config.session_id, sequence=sequence if type(sequence) is int else None,
            tick_time_s=_finite_time(now_s), sensor_time_s=_finite_time(getattr(frame, 'timestamp_s', None)),
            manifest_sha256=self.manifest_sha256, authority=result['state'], reason=result['reason'],
            fault=result['fault'], fault_origin=None if result['fault'] is None else (
                'runtime' if self.runtime.fault is not None else 'bridge'),
            network_pwm=stages.get('network_pwm'), coordinated_pwm=stages.get('coordinated_pwm'),
            supervised_pwm=stages.get('supervised_pwm'), issued_pwm=None if lease is None else lease.pwm,
            coordinator_reasons=tuple(decision.get('reasons', ())),
            behavior_state=decision.get('state'))
        return dict(result, trace=trace)


def build_system(run_config):
    """Assemble exactly the pinned implementation; no controller overrides or I/O."""
    if not isinstance(run_config, RunConfig):
        raise TypeError('build_system requires RunConfig')
    c = run_config
    bundle = Path(c.bundle_dir)
    if sha256(bundle/'metadata.json') != c.expected_metadata_sha256:
        raise ValueError('Pinned metadata checksum mismatch')
    runtime = FlightRuntime(bundle, coordinator_mode=c.coordinator_mode,
                            coordinator_preset=c.coordinator_preset, levy_explore=c.levy_explore,
                            stop_hover=c.stop_hover, seed=c.seed)
    executed = {}
    implementations = {'system.py': RunConfig, 'runtime.py': FlightRuntime,
                       'flight_bridge.py': FlightBridge, 'altitude_control.py': AltitudeHold,
                       'xy_coordinator.py': XYCoordinator}
    for name, cls in implementations.items():
        path = Path(sys.modules[cls.__module__].__file__).resolve()
        actual = sha256(path)
        if runtime.metadata['files'].get(name) != actual:
            raise ValueError('Executing code does not match pinned bundle: '+name)
        executed[name] = dict(path=str(path), sha256=actual)
    manifest = dict(version='flight-system-v1', run_config=c.to_dict(),
        bundle=dict(metadata_sha256=c.expected_metadata_sha256, files=runtime.metadata['files'],
                    source=runtime.metadata['source']), executed_modules=executed,
        resolved=dict(control_dt_s=runtime.policy.default_dt, action_order=runtime.metadata['action_order'],
            sensor_encoding=runtime.metadata['sensor_encoding'], tof_range_m=runtime.metadata['tof_range_m'],
            coordinator_mode=c.coordinator_mode, coordinator_preset=c.coordinator_preset,
            coordinator=None if runtime.coordinator is None else runtime.coordinator.contract(),
            altitude=runtime.altitude.contract(), bridge=asdict(c.bridge),
            bridge_admission_height_m=1., bridge_tof_range_m=3.5,
            bridge_slew_pwm_per_s=[150., 2000., 2000., 10000.],
            sensor_contract='strict_synchronous_v1', shadow_integrator='legacy_updates',
            clock_contract='neural dt 0.02; bridge slew uses arrival delta; sensor timestamps synchronous'),
        command_stages=['network_pwm', 'coordinated_pwm', 'supervised_pwm', 'issued_pwm', 'accepted_pwm'],
        limitations=['No scheduler, physical environment, transport or receiver execution.',
                     'Receiver acceptance requires an explicit observation; not inferred from issue.',
                     'Trace session_id is not a wire-protocol session or replay protection.'])
    return FlightSystem(c, runtime, manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--manifest-out', type=Path, required=True)
    args = parser.parse_args()
    system = build_system(RunConfig.from_json(args.config))
    # Exclusive creation prevents accidentally overwriting experiment evidence.
    system.write_manifest(args.manifest_out)
    print(json.dumps(dict(manifest=str(args.manifest_out), sha256=system.manifest_sha256,
                          status='assembled_offline'), indent=2))


if __name__ == '__main__':
    main()
