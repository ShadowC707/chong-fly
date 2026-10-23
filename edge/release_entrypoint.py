"""Portable offline integration check. No hardware connections or motor arming."""
import argparse
from dataclasses import replace
import hashlib
import importlib.abc
import json
from pathlib import Path
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root):
    path = root/'release.json'
    if digest(path) != (root/'release.sha256').read_text().strip():
        raise ValueError('Release metadata checksum mismatch')
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if manifest.get('version') != 'chong-offline-release-v1':
        raise ValueError('Unsupported release contract')
    for name, expected in manifest['files'].items():
        path = (root/name).resolve()
        if not path.is_relative_to(root) or digest(path) != expected:
            raise ValueError('Release file checksum mismatch: '+name)
    return manifest


FORBIDDEN = {'torch', 'pandas', 'scipy', 'simulation', 'optimizer', 'generator',
             'evaluate', 'edge', 'artifacts', 'core', 'configs'}


class RuntimeDependenciesOnly(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in FORBIDDEN:
            raise ImportError('Forbidden release dependency: '+fullname)


def self_check(root):
    manifest = verify(root)
    sys.meta_path.insert(0, RuntimeDependenciesOnly())
    sys.path[:0] = [str(root/'bundle'), str(root)]
    import numpy as np
    from flight_bridge import SensorFrame
    from runtime import AltitudeSample
    from sensors import Measurement, SensorSnapshot
    from system import RunConfig, build_system, observe_receiver
    from session_receiver import SessionReceiver, CommandPacket
    from transport import SimulatedTransport
    cfg = replace(RunConfig.from_json(root/'run_config.json'), session_id='self-check-0')
    system = build_system(cfg)
    receiver = SessionReceiver()
    receiver.reset(session_id=cfg.session_id, now_s=1000.)
    link = SimulatedTransport()
    accepted = 0
    accepted_sessions = set()
    tripped = old_rejected = False
    old_packet = None
    manual = (system.runtime.altitude.config.hover_pwm, 1500., 1500., 1500.)
    for i in range(120):
        t = i*.02
        if i == 80:
            receiver.set_mode('MANUAL', now_s=1000.+t)
            cfg = replace(cfg, session_id='self-check-1')
            receiver.reset(session_id=cfg.session_id, now_s=1000.+t)
            system = build_system(cfg)
        mode = 'SHADOW' if i < 30 or 80 <= i < 110 else 'AUTO'
        receiver.set_mode(mode, now_s=1000.+t)
        current = receiver.output(now_s=1000.+t, fallback_pwm=manual)
        tripped = tripped or current['tripped']
        sequence = i if i < 80 else i-80
        receiver.announce_sensor(sequence, now_s=1000.+t)
        stamp = 10.+t
        a = system.runtime.altitude.config
        sample = AltitudeSample(a.target_height_m-a.emitter_below_center_m, 0., 0., stamp, True)
        if cfg.sensor_mode == 'multi_rate_v1':
            frame = SensorSnapshot(sequence, stamp, Measurement(i, stamp, (0., 0.)),
                Measurement(i, stamp, (1.,)*64), Measurement(i, stamp, (sample.range_m, 0., 0.)),
                Measurement(i, stamp, (0., 0.)))
        else:
            frame = SensorFrame(sequence, stamp, (0., 0.), np.ones(64), sample)
        link.send('uplink', frame, now_s=t)
        delivered = link.drain(t+.004)[0]
        result = system.tick(delivered.payload, now_s=10.+t+.004,
            request='SHADOW' if receiver.gate.tripped else mode, manual_pwm=manual,
            stop_requested=cfg.stop_hover)
        packet = CommandPacket(cfg.session_id, sequence, result['lease'])
        if i == 59:
            old_packet = packet
        link.send('downlink', packet, now_s=t+.004, drop=i == 60)
        for delivered in link.drain(t+.008):
            receipt = receiver.receive(delivered.payload, now_s=1000.+t+.008)
            trace = result['trace']
            if trace.issued_pwm is not None:
                observe_receiver(trace, session_id=packet.session_id, sequence=sequence,
                    accepted_pwm=receipt.accepted_pwm, receiver_time_s=receipt.receiver_time_s,
                    reason=receipt.reason)
            if receipt.accepted_pwm is not None:
                accepted += 1
                accepted_sessions.add(packet.session_id)
        if i == 115:
            event = receiver.receive(old_packet, now_s=1000.+t+.009)
            old_rejected = event.reason == 'session_mismatch' and not receiver.gate.tripped
    forbidden = sorted(n for n in FORBIDDEN if n in sys.modules)
    passed = accepted > 0 and len(accepted_sessions) == 2 and tripped and old_rejected and not forbidden
    return dict(passed=passed, mode='offline_synthetic_telemetry_no_physics', accepted_commands=accepted,
        accepted_sessions=sorted(accepted_sessions), watchdog_tripped=tripped, old_session_rejected=old_rejected,
        forbidden_modules_loaded=forbidden, model_sha256=digest(root/'bundle/model.npz'),
        release_sha256=digest(root/'release.json'), numpy_version=np.__version__,
        bundle_metadata_sha256=manifest['bundle_metadata_sha256'],
        limitation='Synthetic clear telemetry; no hardware I/O, real-time scheduler or physical fallback validation.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-check', action='store_true', required=True)
    parser.parse_args()
    result = self_check(Path(__file__).resolve().parent)
    print(json.dumps(result, indent=2, allow_nan=False))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
