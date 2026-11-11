"""Recorded input stays offline evidence, including disconnects and old times."""
from copy import deepcopy
import json

import pytest

from edge.gamepad_input import GamepadState, MockGamepadSource, summarize
from edge.gamepad_probe import capture
from edge.gamepad_replay import ReplayGamepadSource, load_capture


def recording():
    state = GamepadState(42, 0x1000, 0, 255, -32768, 32767, -100, 100)
    source = MockGamepadSource([state, None, state], clock_us=iter([10, 20, 30]).__next__)
    return capture(source, count=3, hz=50, sleep=lambda _: None, clock=lambda: 0.)


def test_replay_preserves_values_disconnects_and_recorded_time_without_hardware(tmp_path):
    path = tmp_path / 'raw.json'
    report = recording()
    # Summary is derived again rather than trusted as evidence.
    report['summary'] = {'connected_samples': 999}
    path.write_text(json.dumps(report), encoding='utf-8')
    source = load_capture(path)
    samples = [source.poll() for _ in range(3)]
    assert [s.read_completed_us for s in samples] == [10, 20, 30]
    assert [s.connected for s in samples] == [True, False, True]
    assert {s.source for s in samples} == {'replay'}
    assert samples[0].state == samples[2].state
    assert samples[1].state is None
    assert summarize(samples)['connected_samples'] == 2
    assert source.sha256 and source.sample_count == 3 and source.requested_hz == 50
    end = source.poll()
    assert not end.connected and end.state is None and end.read_completed_us == 30


@pytest.mark.parametrize('mutate', [
    lambda r: r.update(version='future'),
    lambda r: r.update(samples=[]),
    lambda r: r.update(samples=r['samples'] * 4001),
    lambda r: r.update(calibration_profile_applied=True),
    lambda r: r.update(udp_sends=1),
    lambda r: r.update(fc_attempts=1),
    lambda r: r.update(requested_hz=True),
    lambda r: r['samples'][0].update(connected=False),
    lambda r: r['samples'][0].update(api_status=1167),
    lambda r: r['samples'][0].update(state=None),
    lambda r: r['samples'][0]['state'].update(left_x=32768),
    lambda r: r['samples'][0]['state'].update(buttons=True),
    lambda r: r['samples'][1].update(sample_id=0),
    lambda r: r['samples'][1].update(read_completed_us=9),
    lambda r: r['samples'][0].update(sample_id=-1),
    lambda r: r['samples'][0].update(source='made-up'),
])
def test_invalid_capture_never_becomes_plausible_input(mutate):
    report = deepcopy(recording())
    mutate(report)
    with pytest.raises(ValueError):
        ReplayGamepadSource(report)


@pytest.mark.parametrize('payload', [
    '{"version":"gamepad-raw-capture-v1","version":"future"}',
    '{"samples":NaN}',
    '[]',
    '{broken',
])
def test_loader_rejects_ambiguous_or_malformed_json(tmp_path, payload):
    path = tmp_path / 'bad.json'
    path.write_text(payload, encoding='utf-8')
    with pytest.raises(ValueError): load_capture(path)


def test_loader_bounds_file_read_before_json_parse(tmp_path):
    path = tmp_path / 'large.json'
    with path.open('wb') as stream:
        stream.truncate(8 * 1024 * 1024 + 1)
    with pytest.raises(ValueError, match='8 MiB'): load_capture(path)


def test_cli_replays_without_loading_xinput_and_keeps_output_exclusive(tmp_path, monkeypatch):
    from edge import gamepad_probe
    source = tmp_path / 'raw.json'
    output = tmp_path / 'replay.json'
    source.write_text(json.dumps(recording()), encoding='utf-8')
    monkeypatch.setattr('sys.argv', ['probe', '--source', 'replay', '--replay', str(source), '--out', str(output)])
    monkeypatch.setattr(gamepad_probe, 'WindowsXInputBackend', lambda: pytest.fail('Replay loaded a device API'))
    gamepad_probe.main()
    report = json.loads(output.read_text(encoding='utf-8'))
    assert report['source'] == 'replay' and report['replayed_input_sha256']
    assert [s['read_completed_us'] for s in report['samples']] == [10, 20, 30]
    assert report['udp_sends'] == report['fc_attempts'] == 0
    assert report['physical_connection_verified'] is False
    assert report['summary']['rf_freshness'] == 'unknown'
    with pytest.raises(FileExistsError): gamepad_probe.main()
