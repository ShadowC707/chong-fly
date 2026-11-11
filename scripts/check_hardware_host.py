"""One Windows host check: tests, native mock, raw-input replay, optional ESP build.

No serial ports, device discovery, firmware upload, FC output or remote peers.
Only the mock opens UDP, on loopback with ephemeral ports. Reports never qualify
physical timing, clock mapping or flight readiness.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
HARDWARE_TESTS = (
    'hardware_wire', 'esp32_host', 'hardware_mock', 'hardware_udp',
    'ground_peer_transport', 'esp32_foundation', 'foundation_loop',
    'gamepad_input', 'gamepad_replay', 'hardware_check',
    'usb_gamepad_probe', 'esp_flash_backup',
)


def protected_hashes():
    paths = [ROOT / 'configs/mvp_runtime_v3.json']
    paths += [ROOT / name for name in (
        'edge/system.py', 'edge/runtime.py', 'edge/flight_bridge.py',
        'edge/sensors.py', 'edge/session_receiver.py', 'edge/transport.py',
        'simulation/altitude_control.py', 'simulation/xy_coordinator.py')]
    paths += [p for p in (ROOT / 'data/edge/k306_tau025_integrated_v1').iterdir() if p.is_file()]
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def run_step(name, command, out, steps, *, execute=subprocess.run, env=None):
    print(f'{name}: running', flush=True)
    start = time.monotonic()
    try:
        result = execute([str(c) for c in command], cwd=ROOT, env=env, capture_output=True,
                         text=True, encoding='utf-8', errors='replace', timeout=1200)
        code, output = result.returncode, (result.stdout or '') + (result.stderr or '')
    except (subprocess.TimeoutExpired, OSError) as exc:
        partial = getattr(exc, 'output', None) or ''
        if isinstance(partial, bytes): partial = partial.decode('utf-8', errors='replace')
        code, output = -1, partial + '\n' + str(exc)
    log = out / f'{name}.log'
    log.write_text(output, encoding='utf-8')
    steps.append(dict(name=name, command=[str(c) for c in command], returncode=code,
                      elapsed_seconds=round(time.monotonic() - start, 3), log=log.name))
    if code != 0:
        raise RuntimeError(f'{name} failed ({code}); see {log}')
    print(f'{name}: passed', flush=True)


def finish_report(report, before, after, *, error):
    report.update(protected_before=before, protected_after=after,
                  protected_files_unchanged=before == after, error=error,
                  success=error is None and before == after and bool(report['steps'])
                          and all(s['returncode'] == 0 for s in report['steps']),
                  hardware_qualification=dict(esp_boot=False, pi_execution=False, wifi=False,
                      sensor_acquisition=False, clock_mapping=False, gamepad_rf_loss=False,
                      fc_control=False, flight=False),
                  timing_budget=dict(latest_frame_response_strictly_below_us=20000,
                      max_lease_from_esp_frame_us=40000, qualification='open',
                      reference='docs/PI_COMPUTE_AUDIT_UK.md'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, required=True, help='New directory; never overwritten')
    parser.add_argument('--full-suite', action='store_true', help='Run all repository tests instead of hardware tests')
    parser.add_argument('--build-firmware', action='store_true', help='Build foundation only; never upload')
    parser.add_argument('--build-python', type=Path, default=ROOT / '.venv-esp32-build/Scripts/python.exe')
    args = parser.parse_args()
    if os.name != 'nt': parser.error('This check currently requires Windows with MSVC x64 and Windows SDK')
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    temp = out / 'temp'
    temp.mkdir()
    env = {**os.environ, 'TMP': str(temp), 'TEMP': str(temp)}
    before = protected_hashes()
    report = dict(version='hardware-host-check-v1', recorded_utc=datetime.now(timezone.utc).isoformat(),
                  full_suite=args.full_suite, steps=[], firmware_build_requested=args.build_firmware,
                  limitation='Host execution and synthetic clocks/sensors only. No physical qualification.')
    error = None
    def step(name, command): run_step(name, command, out, report['steps'], env=env)
    try:
        tests = ['tests'] if args.full_suite else [f'tests/test_{name}.py' for name in HARDWARE_TESTS]
        step('pytest', [sys.executable, '-m', 'pytest', *tests, '-q', '--basetemp', out / 'pytest-temp',
                        '--junitxml', out / 'tests.xml'])
        xml = ET.parse(out / 'tests.xml').getroot()
        report['tests'] = {name: sum(int(s.get(name, 0)) for s in xml.iter('testsuite'))
                           for name in ('tests', 'errors', 'failures', 'skipped')}
        step('native_build', ['powershell', '-NoProfile', '-File', ROOT / 'firmware/esp32/build_host.ps1',
                              '-Foundation', '-OutDir', out / 'native'])
        for name, flags in [('normal', []), ('takeover', ['--takeover'])]:
            path = out / f'{name}.json'
            step(name, [sys.executable, '-m', 'edge.foundation_mock', '--foundation-exe',
                        out / 'native/foundation_host.exe', '--out', path, *flags])
            report[name] = json.loads(path.read_text(encoding='utf-8'))
        step('gamepad_mock', [sys.executable, '-m', 'edge.gamepad_probe', '--source', 'mock',
                             '--seconds', '.1', '--out', out / 'gamepad_mock.json'])
        step('gamepad_replay', [sys.executable, '-m', 'edge.gamepad_probe', '--source', 'replay',
                               '--replay', out / 'gamepad_mock.json', '--out', out / 'gamepad_replay.json'])
        if args.build_firmware:
            step('firmware_build', [args.build_python.resolve(), ROOT / 'firmware/esp32/pio_local.py',
                                   'run', '--project-dir', ROOT / 'firmware/esp32/app', '--jobs', '2'])
            build = ROOT / 'firmware/esp32/app/.pio/build/foundation'
            report['firmware_artifacts'] = {name: dict(size_bytes=(build / name).stat().st_size,
                sha256=hashlib.sha256((build / name).read_bytes()).hexdigest())
                for name in ('bootloader.bin', 'partitions.bin', 'firmware.bin')}
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    finish_report(report, before, protected_hashes(), error=error)
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(success=report['success'], error=error, report=str(out / 'report.json')), indent=2))
    return 0 if report['success'] else 1


if __name__ == '__main__': raise SystemExit(main())
