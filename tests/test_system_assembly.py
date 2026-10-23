import hashlib
import json

import numpy as np
import pytest

from tests.test_edge_runtime import exported
from tests.test_flight_bridge import frame
from edge.runtime import FlightRuntime
from edge.flight_bridge import FlightBridge
from edge.system import RunConfig, build_system, observe_receiver


def config(bundle, **kw):
    return RunConfig(bundle_dir=str(bundle), expected_metadata_sha256=hashlib.sha256(
        (bundle/'metadata.json').read_bytes()).hexdigest(), **kw)


def test_runtime_exposes_stages_without_changing_legacy_output(exported):
    _, bundle = exported
    r = FlightRuntime(bundle, coordinator_mode='both')
    f = frame(0)
    result = r.step(f.flow_xy, f.tof_8x8, f.altitude, now_s=0.)
    assert result['applied_pwm'] is result['supervised_pwm']
    assert not np.shares_memory(result['coordinated_pwm'], result['supervised_pwm'])
    assert set(r.last_stages) == {'network_pwm', 'coordinated_pwm', 'supervised_pwm'}
    before = result['coordinated_pwm'].copy()
    f = frame(1)
    r.step(f.flow_xy, f.tof_8x8, f.altitude, now_s=.02)
    np.testing.assert_array_equal(result['coordinated_pwm'], before)


def test_config_file_resolves_paths_and_rejects_ambiguous_settings(exported, tmp_path):
    _, bundle = exported
    data = config(bundle).to_dict()
    data['bundle_dir'] = 'bundle'
    p = tmp_path/'run.json'
    p.write_text(json.dumps(data), encoding='utf-8')
    assert RunConfig.from_json(p).bundle_dir == str(bundle.resolve())
    for changes in ({'levy_explore': 'false'}, {'seed': True}, {'transport_mode': 'wifi'},
                    {'coordinator_mode': 'bogus'}, {'stop_hover': True},
                    {'bridge': {'lease_s': True}}, {'expected_metadata_sha256': 'x'}):
        with pytest.raises((ValueError, TypeError)):
            RunConfig(**(config(bundle).to_dict() | changes))
    data['dt'] = .04
    p.write_text(json.dumps(data), encoding='utf-8')
    with pytest.raises(ValueError, match='Unknown'):
        RunConfig.from_json(p)


def test_builder_manifest_pins_actual_code_and_is_not_mutable(exported):
    _, bundle = exported
    system = build_system(config(bundle, coordinator_mode='both', stop_hover=True))
    manifest = system.manifest
    assert manifest['bundle']['files']['model.npz']
    assert manifest['resolved']['control_dt_s'] == .02
    assert manifest['resolved']['coordinator_mode'] == 'both'
    for name, info in manifest['executed_modules'].items():
        assert info['sha256'] == manifest['bundle']['files'][name]
    manifest['resolved']['control_dt_s'] = 999
    assert system.manifest['resolved']['control_dt_s'] == .02
    with pytest.raises(ValueError, match='Pinned'):
        build_system(RunConfig(**(config(bundle).to_dict() | {'expected_metadata_sha256': '0'*64})))
    path = bundle/'xy_coordinator.py'
    path.write_text(path.read_text(encoding='utf-8')+'\n# different implementation\n', encoding='utf-8')
    meta = json.loads((bundle/'metadata.json').read_text(encoding='utf-8'))
    meta['files'][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (bundle/'metadata.json').write_text(json.dumps(meta), encoding='utf-8')
    (bundle/'metadata.sha256').write_text(hashlib.sha256((bundle/'metadata.json').read_bytes()).hexdigest())
    with pytest.raises(ValueError, match='Executing code'):
        build_system(config(bundle))


def test_system_preserves_bridge_and_records_real_stages(exported):
    _, bundle = exported
    system = build_system(config(bundle))
    direct = FlightBridge(FlightRuntime(bundle))
    # Derive manual snapshot from the reference rather than assume synthetic weights.
    probe = FlightRuntime(bundle)
    f = frame(0)
    manual = probe.step(f.flow_xy, f.tof_8x8, f.altitude, now_s=0.)['applied_pwm']
    for i in range(40):
        kwargs = dict(now_s=i*.02, request='AUTO' if i>=30 else 'SHADOW', manual_pwm=manual)
        want = direct.tick(frame(i), **kwargs)
        got = system.tick(frame(i), **kwargs)
        assert {k: got[k] for k in want} == want
        trace = got['trace']
        assert trace.supervised_pwm == got['preview_pwm']
        assert trace.issued_pwm == (None if got['lease'] is None else got['lease'].pwm)
        assert trace.accepted_pwm is None and trace.receiver_status == 'unobserved'
    assert got['lease'] is not None
    accepted = observe_receiver(trace, session_id=trace.session_id, sequence=trace.sequence,
                                accepted_pwm=trace.issued_pwm, receiver_time_s=100., reason='accepted')
    assert accepted.accepted_pwm == trace.issued_pwm
    assert trace.accepted_pwm is None
    rejected = observe_receiver(trace, session_id=trace.session_id, sequence=trace.sequence,
                                accepted_pwm=None, receiver_time_s=100., reason='expired')
    assert rejected.receiver_status == 'rejected'
    with pytest.raises(ValueError):
        observe_receiver(trace, session_id='wrong', sequence=trace.sequence,
                         accepted_pwm=trace.issued_pwm, receiver_time_s=100., reason='accepted')
    with pytest.raises(ValueError):
        observe_receiver(trace, session_id=trace.session_id, sequence=trace.sequence+1,
                         accepted_pwm=trace.issued_pwm, receiver_time_s=100., reason='accepted')
    manual_result = system.tick(None, now_s=.8, request='MANUAL', manual_pwm=None)
    assert manual_result['trace'].network_pwm is None
    bad = system.tick(frame(41, valid=False), now_s=41*.02, request='AUTO', manual_pwm=manual)
    assert bad['trace'].supervised_pwm is None and bad['trace'].issued_pwm is None
    assert bad['trace'].fault_origin == 'runtime'
    later = system.tick(frame(42), now_s=.84, request='AUTO', manual_pwm=manual)
    assert later['trace'].network_pwm is None
    json.dumps(later['trace'].to_dict(), allow_nan=False)


def test_assembly_is_executed_inside_isolated_bundle(exported):
    from edge.bundle_validation import replay_bundle
    from edge.validate_numpy import _replay_inputs
    _, bundle = exported
    inputs = _replay_inputs(100)
    cases = [dict(coordinator_mode='both', stop_hover=True, seed=19)]
    old = replay_bundle(bundle, inputs, cases)
    new = replay_bundle(bundle, inputs, cases, assembly=True)
    for name in ('network_pwm', 'supervised_pwm', 'hidden'):
        assert old[name].dtype == new[name].dtype
        assert old[name].tobytes() == new[name].tobytes()
    assert new['report']['assembly'] is True
    assert 'system.py' in new['report']['executed_modules']
    assert new['report']['assembly_manifests'][0]['resolved']['coordinator_mode'] == 'both'
    assert old['report']['reasons'] == new['report']['reasons']


def test_trace_keeps_stop_state_and_channel_changes(exported):
    _, bundle = exported
    system = build_system(config(bundle, coordinator_mode='both', stop_hover=True))
    result = system.tick(frame(0), now_s=0., request='SHADOW',
                         manual_pwm=[1213., 1500., 1500., 1500.], stop_requested=True)
    trace = result['trace']
    assert trace.behavior_state == 'STOP'
    assert 'stop_hover' in trace.coordinator_reasons
    assert trace.to_dict()['changed_channels']['issued_pwm'] is None
    # An invalid input time must still revoke authority and produce JSON-safe evidence.
    bad = system.tick(frame(1), now_s=float('nan'), request='AUTO', manual_pwm=None)
    assert bad['state'] == 'FAULT'
    assert bad['trace'].tick_time_s is None
    assert bad['trace'].network_pwm is None
    json.dumps(bad['trace'].to_dict(), allow_nan=False)


def test_structure_validator_detects_even_sub_tolerance_drift(exported, tmp_path):
    import shutil
    from edge.validate_structure import compare
    from tests.test_bundle_execution import rehash
    _, bundle = exported
    changed = tmp_path/'changed'
    shutil.copytree(bundle, changed)
    path = changed/'runtime.py'
    content = path.read_text(encoding='utf-8').replace(
        'self.h=h\n        return pwm', 'self.h=h\n        pwm[3] += np.float32(.001)\n        return pwm')
    path.write_text(content, encoding='utf-8')
    rehash(changed, 'runtime.py')
    report = compare(bundle, changed, steps=32)
    assert not report['passed']
    assert report['weights_unchanged']
    assert all(not r['stages']['network_pwm']['bitwise_equal'] for r in report['cases'])


def test_standalone_manifest_cli_matches_file_hash(exported, tmp_path):
    import subprocess
    import sys
    _, bundle = exported
    config_path = tmp_path/'config.json'
    config_path.write_text(json.dumps(config(bundle).to_dict()), encoding='utf-8')
    out = tmp_path/'manifest.json'
    # -I alone suppresses the script directory; explicitly add only this bundle.
    code = "import sys,runpy; sys.path.insert(0,sys.argv[1]); sys.argv=sys.argv[2:]; runpy.run_path(sys.argv[0],run_name='__main__')"
    result = subprocess.run([sys.executable, '-I', '-c', code, str(bundle), str(bundle/'system.py'),
                             '--config', str(config_path), '--manifest-out', str(out)],
                            cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['sha256'] == hashlib.sha256(out.read_bytes()).hexdigest()
    manifest = json.loads(out.read_text(encoding='utf-8'))
    assert all(str(bundle.resolve()) in info['path'] for info in manifest['executed_modules'].values())
