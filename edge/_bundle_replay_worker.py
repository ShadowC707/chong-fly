"""Private subprocess entry point. Imports runtime ONLY from the copied bundle."""
import hashlib
import importlib.abc
import json
from pathlib import Path
import sys


FORBIDDEN = {'torch', 'scipy', 'pandas', 'core', 'configs', 'artifacts', 'control',
             'simulation', 'optimizer', 'generator', 'evaluate', 'edge'}


class RejectRepositoryDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in FORBIDDEN:
            raise ImportError('Forbidden runtime dependency: '+fullname)


def main():
    root = Path(sys.argv[1]).resolve()
    bundle = root/'bundle'
    sys.meta_path.insert(0, RejectRepositoryDependencies())
    sys.path.insert(0, str(bundle))
    import numpy as np
    import runtime
    if (bundle/'flight_bridge.py').exists():
        import flight_bridge  # also check the deployment dependency boundary

    cases = json.loads((root/'cases.json').read_text(encoding='utf-8'))
    assembly = json.loads((root/'execution_options.json').read_text(encoding='utf-8'))['assembly']
    manifests = []
    if assembly:
        import system
    with np.load(root/'inputs.npz', allow_pickle=False) as saved:
        inputs = {name: saved[name].copy() for name in saved.files}
    steps = len(inputs['times'])
    outputs = {key: [] for key in ('network_pwm', 'supervised_pwm', 'hidden')}
    reasons = []
    for index, options in enumerate(cases):
        if assembly:
            assembled = system.build_system(system.RunConfig(bundle_dir=str(bundle),
                expected_metadata_sha256=hashlib.sha256((bundle/'metadata.json').read_bytes()).hexdigest(),
                **options))
            flight = assembled.runtime
            manifests.append(assembled.manifest)
        else:
            flight = runtime.FlightRuntime(bundle, **options)
        rows = {key: [] for key in outputs}
        case_reasons = []
        for i, t in enumerate(inputs['times']):
            sample = runtime.AltitudeSample(*inputs['altitude'][i], float(t), True)
            extra={}
            if assembly and getattr(assembled.config,'sensor_mode',None)=='multi_rate_v1':
                extra['allow_sensor_hold']=True
            result = flight.step(inputs['flow'][i], inputs['tof'][i], sample,
                                 now_s=float(t), memory_ring=inputs['memory'][i],
                                 stop_requested=bool(inputs['stop_requests'][index, i]),**extra)
            rows['network_pwm'].append(result['network_pwm'].copy())
            rows['supervised_pwm'].append(result['applied_pwm'].copy())
            rows['hidden'].append(flight.policy.h.copy())
            case_reasons.append(list(result['coordinator']['reasons']))
        for key in outputs:
            outputs[key].append(np.asarray(rows[key]))
        reasons.append(case_reasons)
    executed = {}
    for name in ('runtime', 'altitude_control', 'xy_coordinator', 'flight_bridge', 'system', 'sensors'):
        if name not in sys.modules:
            continue
        path = Path(sys.modules[name].__file__).resolve()
        if path.parent != bundle:
            raise RuntimeError('Module imported outside bundle: '+str(path))
        executed[path.name] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    forbidden = sorted(name for name in sys.modules if name.split('.')[0] in FORBIDDEN)
    if forbidden:
        raise RuntimeError('Forbidden modules loaded: '+repr(forbidden))
    np.savez(root/'outputs.npz', **{key: np.asarray(value) for key, value in outputs.items()})
    (root/'execution.json').write_text(json.dumps(dict(
        version='isolated-bundle-replay-v1', isolated_python=bool(sys.flags.isolated),
        execution_directory=str(bundle), executed_modules=executed,
        assembly=assembly, assembly_manifests=manifests,
        forbidden_modules_loaded=forbidden, reasons=reasons,
        numpy_version=np.__version__, python_version=sys.version,
        limitation='Import isolation for trusted code, not a security sandbox; no Bridge/Gate execution.'),
        allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
