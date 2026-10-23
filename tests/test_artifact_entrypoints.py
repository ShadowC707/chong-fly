"""Retired data must not trigger implicit regeneration or training."""
import importlib

import pytest


@pytest.mark.parametrize('name', ['generator.role_reducer', 'generator.audit_routes', 'bc_test'])
def test_manual_entrypoints_require_explicit_artifact_paths(name, monkeypatch):
    module = importlib.import_module(name)
    if hasattr(module, 'load_graph'):
        def unexpected_load(*args, **kwargs):
            raise AssertionError('Source must not be read before explicit CLI paths')
        monkeypatch.setattr(module, 'load_graph', unexpected_load)
    with pytest.raises(SystemExit) as exc:
        module.main([])
    assert exc.value.code == 2
