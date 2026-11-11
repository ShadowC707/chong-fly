"""An offline check must retain failure evidence without claiming qualification."""
from pathlib import Path
import subprocess

import pytest

from scripts.check_hardware_host import run_step, finish_report


def test_failed_child_stops_check_and_retains_its_output(tmp_path):
    steps = []
    with pytest.raises(RuntimeError, match='deliberate'):
        run_step('deliberate', ['fake'], tmp_path, steps,
                 execute=lambda *a, **kw: subprocess.CompletedProcess(a[0], 7, 'failure evidence'))
    assert steps[0]['returncode'] == 7
    assert (tmp_path / 'deliberate.log').read_text(encoding='utf-8') == 'failure evidence'


def test_successful_mock_never_qualifies_hardware_and_protected_change_fails():
    report = {'steps': [{'returncode': 0}]}
    finish_report(report, {'config': 'old'}, {'config': 'old'}, error=None)
    assert report['success'] is True
    assert all(value is False for value in report['hardware_qualification'].values())
    assert report['timing_budget']['qualification'] == 'open'
    finish_report(report, {'config': 'old'}, {'config': 'changed'}, error=None)
    assert report['success'] is False and report['protected_files_unchanged'] is False


def test_timeout_becomes_failure_evidence(tmp_path):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 1, output=b'partial output')
    steps = []
    with pytest.raises(RuntimeError, match='timeout'):
        run_step('timeout', ['fake'], tmp_path, steps, execute=timeout)
    assert steps[0]['returncode'] == -1
    assert 'partial output' in (tmp_path / 'timeout.log').read_text(encoding='utf-8')


def test_hardware_collection_does_not_import_training_dependencies():
    import sys
    from scripts.check_hardware_host import ROOT
    code = '''
import importlib.abc
import runpy
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
class BlockTraining(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'generator', 'torch', 'scipy', 'pandas', 'sklearn', 'numba'}:
            raise ImportError('Hardware test collection imported training: ' + fullname)
sys.meta_path.insert(0, BlockTraining())
runpy.run_path('tests/conftest.py')
'''
    result = subprocess.run([sys.executable, '-I', '-c', code], cwd=ROOT,
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
