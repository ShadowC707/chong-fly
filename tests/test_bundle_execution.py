import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tests.test_edge_runtime import exported


def sequence(steps=400):
    rng = np.random.default_rng(19)
    tof = np.ones((steps, 64), dtype=np.float32)
    tof[40:180, :24] = .15
    return dict(flow=rng.normal(0, .02, (steps, 2)).astype(np.float32),
                tof=tof, memory=np.ones((steps, 8), dtype=np.float32),
                altitude=np.tile([.982, 0., 0.], (steps, 1)),
                times=np.arange(steps)*.02)


def cases():
    return [dict(coordinator_mode='baseline'),
            dict(coordinator_mode='both', levy_explore=True, stop_hover=True, seed=19)]


def rehash(bundle, name):
    path = bundle/'metadata.json'
    meta = json.loads(path.read_text(encoding='utf-8'))
    meta['files'][name] = hashlib.sha256((bundle/name).read_bytes()).hexdigest()
    path.write_text(json.dumps(meta), encoding='utf-8')
    (bundle/'metadata.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest())


def test_isolated_bundle_replay_matches_reference_and_records_executed_code(exported):
    from edge.bundle_validation import replay_bundle
    from edge.runtime import FlightRuntime, AltitudeSample
    _, bundle = exported
    inputs = sequence()
    stop = np.zeros((2, 400), dtype=bool); stop[1, 300:] = True
    result = replay_bundle(bundle, inputs, cases(), stop_requests=stop)
    for c, options in enumerate(cases()):
        runtime = FlightRuntime(bundle, **options)
        for i, t in enumerate(inputs['times']):
            sample = AltitudeSample(*inputs['altitude'][i], t, True)
            want = runtime.step(inputs['flow'][i], inputs['tof'][i], sample,
                                now_s=t, memory_ring=inputs['memory'][i],
                                stop_requested=bool(stop[c, i]))
            np.testing.assert_allclose(result['network_pwm'][c, i], want['network_pwm'], atol=.02, rtol=0)
            np.testing.assert_allclose(result['supervised_pwm'][c, i], want['applied_pwm'], atol=.02, rtol=0)
            np.testing.assert_allclose(result['hidden'][c, i], runtime.policy.h, atol=2e-5, rtol=2e-4)
            assert result['report']['reasons'][c][i] == want['coordinator']['reasons']
    evidence = result['report']
    assert evidence['isolated_python'] and evidence['forbidden_modules_loaded'] == []
    assert Path(evidence['execution_directory']).resolve() != bundle.resolve()
    meta = json.loads((bundle/'metadata.json').read_text())
    for name, info in evidence['executed_modules'].items():
        assert Path(info['path']).parent == Path(evidence['execution_directory'])
        assert info['sha256'] == meta['files'][name]


def test_replay_executes_changed_bundle_implementation_not_repository(exported):
    from edge.bundle_validation import replay_bundle
    _, bundle = exported
    path = bundle/'runtime.py'
    # Legitimately re-manifest a test-only implementation with a recognizable output.
    path.write_text(path.read_text()+
                    '\n_original = FlightRuntime.step\n'
                    'def _marked(self, *args, **kwargs):\n'
                    '    result = _original(self, *args, **kwargs)\n'
                    '    result["applied_pwm"][3] = 1555.\n'
                    '    return result\n'
                    'FlightRuntime.step = _marked\n', encoding='utf-8')
    rehash(bundle, 'runtime.py')
    result = replay_bundle(bundle, sequence(3), [dict(coordinator_mode='baseline')])
    np.testing.assert_array_equal(result['supervised_pwm'][0, :, 3], [1555.]*3)


def test_replay_rejects_corrupt_bundle_before_execution(exported):
    from edge.bundle_validation import replay_bundle
    _, bundle = exported
    with (bundle/'runtime.py').open('a') as stream:
        stream.write('\nraise RuntimeError("should not execute")\n')
    with pytest.raises(ValueError, match='[Hh]ash'):
        replay_bundle(bundle, sequence(3), [dict(coordinator_mode='baseline')])


def test_replay_forbids_repository_and_training_imports(exported):
    from edge.bundle_validation import replay_bundle
    _, bundle = exported
    path = bundle/'runtime.py'
    path.write_text(path.read_text()+'\nimport pandas\n', encoding='utf-8')
    rehash(bundle, 'runtime.py')
    with pytest.raises(RuntimeError, match='Forbidden runtime dependency: pandas'):
        replay_bundle(bundle, sequence(3), [dict(coordinator_mode='baseline')])


@pytest.mark.parametrize('alter_bundle', [False, True])
def test_public_validator_defaults_to_bundled_code_and_detects_drift(exported, tmp_path, monkeypatch, alter_bundle):
    from edge import validate_numpy
    reference, bundle = exported
    run = tmp_path/'source'; run.mkdir()
    (run/'checkpoint.pt').write_bytes(b'synthetic provenance witness')
    path = bundle/'metadata.json'
    meta = json.loads(path.read_text())
    meta['source']['checkpoint_sha256'] = hashlib.sha256((run/'checkpoint.pt').read_bytes()).hexdigest()
    path.write_text(json.dumps(meta), encoding='utf-8')
    (bundle/'metadata.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest())
    monkeypatch.setattr(validate_numpy, 'load_research_run', lambda _: (reference, {'candidate': 'synthetic'}))
    if alter_bundle:
        source = bundle/'runtime.py'
        source.write_text(source.read_text()+
                          '\n_original = FlightRuntime.step\n'
                          'def _drift(self, *args, **kwargs):\n'
                          '    result = _original(self, *args, **kwargs)\n'
                          '    result["applied_pwm"][3] += 1.\n'
                          '    return result\n'
                          'FlightRuntime.step = _drift\n', encoding='utf-8')
        rehash(bundle, 'runtime.py')
        with pytest.raises(AssertionError, match='Bundled numerical parity failed'):
            validate_numpy.validate(run, bundle, steps=32)
    else:
        report = validate_numpy.validate(run, bundle, steps=32)
        assert report['passed'] and report['execution'] == 'bundled'
        assert report['execution_evidence']['isolated_python']
