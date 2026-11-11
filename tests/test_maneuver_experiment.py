import pytest

from demo.runner import DemoRunner
from edge.runtime import sha256
from evaluate.compare_maneuver_repair import install_experiment


def test_experiment_records_actual_overlay_and_command_identity():
    runner=DemoRunner()
    try:
        original=runner.system.manifest_sha256
        weights=runner.system.runtime.policy.arrays['W'].copy()
        runner.system=install_experiment(runner.system,'both')
        manifest=runner.system.manifest
        assert runner.system.manifest_sha256 != original
        assert manifest['experiment']['research_only']
        module=manifest['executed_modules']['maneuver_coordinator.py']
        assert sha256(module['path'])==module['sha256']
        assert manifest['resolved']['coordinator']['maneuver']['release_on_neutral']
        assert (runner.system.runtime.policy.arrays['W']==weights).all()
        runner.resume();runner.event(dict(type='mode',mode='AUTO'));runner.advance(150)
        trace=runner.rows[-1]['trace']
        assert trace['manifest_sha256']==runner.system.manifest_sha256
        assert trace['accepted_pwm'] is not None
        assert trace['accepted_pwm']==trace['issued_pwm']
        runner.event(dict(type='mode',mode='MANUAL'));runner.advance(5)
        assert runner.rows[-1]['plant_source']=='manual'
        assert runner.rows[-1]['trace']['issued_pwm'] is None
    finally:
        runner.env.close()


def test_baseline_identity_and_unknown_variant():
    runner=DemoRunner()
    try:
        assert install_experiment(runner.system,'baseline') is runner.system
        with pytest.raises(ValueError,match='Unknown'):
            install_experiment(runner.system,'invented')
    finally:
        runner.env.close()
