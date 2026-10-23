"""Host-side replay of trusted exported code, in an isolated Python subprocess.

This is import/dependency isolation, not a sandbox for untrusted Python. Hashes
establish integrity, not authenticity. Never edits the source bundle.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _verified_metadata(bundle):
    path = bundle/'metadata.json'
    if _digest(path) != (bundle/'metadata.sha256').read_text().strip():
        raise ValueError('Metadata checksum mismatch')
    meta = json.loads(path.read_text(encoding='utf-8'))
    required = {'model.npz', 'runtime.py', 'altitude_control.py', 'xy_coordinator.py',
                'benchmark.py', 'requirements.txt'}
    files = meta.get('files', {})
    if set(files) not in (required, required | {'flight_bridge.py'},
                          required | {'flight_bridge.py', 'system.py'},
                          required | {'flight_bridge.py', 'system.py', 'sensors.py'}):
        raise ValueError('Unsupported bundle file contract for isolated replay')
    for name, digest in files.items():
        if _digest(bundle/name) != digest:
            raise ValueError('Hash mismatch: '+name)
    return meta


def replay_bundle(bundle, inputs, cases, *, stop_requests=None, timeout_s=120, assembly=False):
    """Replay flow/tof/memory/altitude/times arrays through the bundled runtime.

    altitude is [range_m, roll_rad, pitch_rad]; every sample here is valid and
    synchronous. This numerical replay is not an asynchronous-sensor test.
    Each case starts with fresh state. supervised_pwm is runtime's legacy
    applied_pwm, before any FlightBridge or receiver processing.
    """
    bundle = Path(bundle).resolve()
    meta = _verified_metadata(bundle)  # verify BEFORE executing any package code
    if type(assembly) is not bool or (assembly and 'system.py' not in meta['files']):
        raise ValueError('Assembly replay requires a system-capable bundle')
    if not isinstance(cases, list) or not 1 <= len(cases) <= 8:
        raise ValueError('Replay requires one to eight explicit cases')
    allowed = {'coordinator_mode', 'coordinator_preset', 'levy_explore', 'stop_hover', 'seed'}
    if assembly: allowed |= {'sensor_mode','sensor_budgets','handover_mode'}
    if any(not isinstance(case, dict) or set(case)-allowed for case in cases):
        raise ValueError('Unsupported replay case fields')
    arrays = {name: np.asarray(value) for name, value in inputs.items()}
    if set(arrays) != {'flow', 'tof', 'memory', 'altitude', 'times'}:
        raise ValueError('Invalid replay input fields')
    if arrays['times'].ndim != 1:
        raise ValueError('Replay times must be a one-dimensional array')
    steps = len(arrays['times'])
    shapes = dict(flow=(steps, 2), tof=(steps, 64), memory=(steps, 8),
                  altitude=(steps, 3), times=(steps,))
    if not 1 <= steps <= 10000 or any(arrays[n].shape != shape
            or arrays[n].dtype.kind not in 'fi' or not np.isfinite(arrays[n]).all()
            for n, shape in shapes.items()):
        raise ValueError('Invalid finite replay array shapes/dtypes')
    stop = np.zeros((len(cases), steps), dtype=bool) if stop_requests is None else np.asarray(stop_requests)
    if stop.dtype != np.bool_ or stop.shape != (len(cases), steps):
        raise ValueError('STOP requests must be a boolean case-by-step array')
    worker = Path(__file__).with_name('_bundle_replay_worker.py').resolve()
    env = os.environ.copy()
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        env[name] = '1'
    with tempfile.TemporaryDirectory(prefix='chong-bundle-replay-') as temp:
        root = Path(temp)
        executed = root/'bundle'
        executed.mkdir()
        # Copy only verified contract files; exclude caches and unrelated scripts.
        for name in [*meta['files'], 'metadata.json', 'metadata.sha256']:
            shutil.copyfile(bundle/name, executed/name)
        copied_meta = _verified_metadata(executed)
        if copied_meta != meta:
            raise ValueError('Bundle changed while preparing replay')
        np.savez(root/'inputs.npz', **arrays, stop_requests=stop)
        (root/'cases.json').write_text(json.dumps(cases, allow_nan=False), encoding='utf-8')
        (root/'execution_options.json').write_text(json.dumps(dict(assembly=assembly)), encoding='utf-8')
        result = subprocess.run([sys.executable, '-I', str(worker), str(root)],
                                cwd=root, env=env, capture_output=True, text=True,
                                encoding='utf-8', errors='replace', timeout=timeout_s)
        if result.returncode:
            raise RuntimeError('Isolated bundle replay failed:\n'+result.stderr[-6000:])
        with np.load(root/'outputs.npz', allow_pickle=False) as saved:
            output = {name: saved[name].copy() for name in saved.files}
        expected_shapes = dict(network_pwm=(len(cases), steps, 4),
                               supervised_pwm=(len(cases), steps, 4),
                               hidden=(len(cases), steps, meta['hidden_size']))
        if set(output) != set(expected_shapes) or any(output[n].shape != shape
                or not np.isfinite(output[n]).all() for n, shape in expected_shapes.items()):
            raise ValueError('Invalid replay output arrays')
        report = json.loads((root/'execution.json').read_text(encoding='utf-8'))
        for name in ('runtime.py', 'altitude_control.py', 'xy_coordinator.py'):
            if report['executed_modules'][name]['sha256'] != meta['files'][name]:
                raise ValueError('Executed code differs from manifest: '+name)
        if report['assembly'] != assembly:
            raise ValueError('Assembly execution mode mismatch')
        if assembly and any(report['executed_modules'][name]['sha256'] != meta['files'][name]
                            for name in ('system.py', 'flight_bridge.py')):
            raise ValueError('Assembly code differs from manifest')
        if 'sensors.py' in meta['files'] and report['executed_modules']['sensors.py']['sha256'] != meta['files']['sensors.py']:
            raise ValueError('Sensor code differs from manifest')
        if not report['isolated_python'] or report['forbidden_modules_loaded']:
            raise ValueError('Replay isolation contract failed')
        if _verified_metadata(executed) != meta or _verified_metadata(bundle) != meta:
            raise ValueError('Bundle changed during replay')
        report.update(source_bundle=str(bundle), bundle_metadata_sha256=_digest(bundle/'metadata.json'),
                      cases=cases, steps=steps)
        output['report'] = report
        return output
