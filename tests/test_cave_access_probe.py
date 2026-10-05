import json
from types import SimpleNamespace
import pytest


def test_probe_lists_pinned_metadata_without_fetching_cells_or_saving_credentials():
    from generator.probe_cave import probe_access
    seen = []
    class Materialize:
        def get_versions(self): return [123, 122]
        def get_tables(self, *, version):
            seen.append(version)
            return ['annotations', 'synapses']
        def query_table(self, *args, **kwargs): pytest.fail('Discovery must not query cells')
    def factory(datastack, **kwargs):
        assert kwargs['write_server_cache'] is False and kwargs['max_retries'] == 0
        assert kwargs['auth_token_file'] == 'local-auth.json'
        return SimpleNamespace(materialize=Materialize())
    result = probe_access('test_stack', version=122, auth_token_file='local-auth.json', client_factory=factory)
    assert result['status'] == 'metadata_available'
    assert result['inspected_version'] == 122 and seen == [122]
    assert 'local-auth.json' not in json.dumps(result)


def test_auth_failure_reports_stage_without_leaking_exception_contents():
    from generator.probe_cave import probe_access
    class AuthException(Exception): pass
    def factory(*args, **kwargs): raise AuthException('token=DO_NOT_RECORD')
    result = probe_access('test_stack', client_factory=factory)
    assert result['status'] == 'auth_required'
    assert result['stage'] == 'client_initialization'
    assert 'DO_NOT_RECORD' not in json.dumps(result)


def test_requested_unavailable_version_does_not_silently_switch_to_latest():
    from generator.probe_cave import probe_access
    client = SimpleNamespace(materialize=SimpleNamespace(get_versions=lambda: [123]))
    result = probe_access('test_stack', version=122, client_factory=lambda *a, **k: client)
    assert result['status'] == 'version_unavailable'
    assert 'inspected_version' not in result


def test_discovery_does_not_require_access_to_production_or_fetch_cells():
    from generator.probe_cave import probe_access
    def factory(datastack, **kwargs):
        assert datastack is None
        return SimpleNamespace(info=SimpleNamespace(get_datastacks=lambda: ['flywire_fafb_public']))
    result = probe_access(list_datastacks=True, client_factory=factory)
    assert result['status'] == 'datastacks_available'
    assert result['available_datastacks'] == ['flywire_fafb_public']
    assert result['requested_datastack'] is None
    assert 'inspected_version' not in result


def test_forbidden_is_distinct_from_missing_auth_and_never_leaks_error():
    from generator.probe_cave import probe_access
    from requests.exceptions import HTTPError
    def factory(*a, **k):
        raise HTTPError('SECRET', response=SimpleNamespace(status_code=403))
    result = probe_access(client_factory=factory)
    assert result['status'] == 'access_denied'
    assert 'SECRET' not in json.dumps(result)
    assert result['next_action'] == 'check_datastack_permissions'


def test_discovery_rejects_snapshot_options():
    from generator.probe_cave import probe_access
    with pytest.raises(ValueError, match='list_datastacks'):
        probe_access(list_datastacks=True, version=123)
