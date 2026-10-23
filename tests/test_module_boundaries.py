import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from tests.test_edge_runtime import exported
from tests.test_system_assembly import config


def test_nondefault_height_is_shared_by_controller_bridge_and_manifest(exported):
    from edge.system import build_system
    from edge.flight_bridge import SensorFrame
    from edge.runtime import AltitudeSample
    _, bundle = exported
    path = bundle/'metadata.json'
    meta = json.loads(path.read_text())
    meta['control_contract']['altitude']['config']['target_height_m'] = 1.2
    path.write_text(json.dumps(meta), encoding='utf-8')
    (bundle/'metadata.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest())
    system = build_system(config(bundle, coordinator_mode='both', stop_hover=True,
                                 handover_mode='freeze_sync_v1'))
    assert system.manifest['resolved']['bridge_admission_height_m'] == 1.2
    for i in range(40):
        t=i*.02
        frame=SensorFrame(i,t,(0.,0.),np.ones(64),AltitudeSample(1.182,0.,0.,t,True))
        result=system.tick(frame,now_s=t,request='AUTO',
                           manual_pwm=[1213.,1500.,1500.,1500.],stop_requested=True)
    assert result['lease'] is not None
    assert system.bridge.target_height_m == system.runtime.altitude.config.target_height_m


def test_bridge_cannot_issue_a_lease_the_receiver_will_always_reject():
    from edge.flight_bridge import BridgeConfig
    with pytest.raises(ValueError): BridgeConfig(lease_s=.05,max_age_s=.06)


def test_publication_is_stdlib_only_and_imports_no_generator(tmp_path):
    root=Path(__file__).resolve().parents[1]
    code=f'''import sys, importlib.abc
sys.path.insert(0, {str(root)!r})
class Deny(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch','numpy','scipy','pandas','generator','optimizer'):
            raise ImportError(fullname)
sys.meta_path.insert(0,Deny())
from artifacts.io import publication, write_json, file_sha256
from pathlib import Path
with publication(Path('out')) as stage: write_json(stage/'result.json',{{'ok':True}})
assert file_sha256('out/result.json')
'''
    result=subprocess.run([sys.executable,'-I','-c',code],cwd=tmp_path,capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_compatibility_exports_share_implementation():
    from artifacts.io import publication, write_json, file_sha256, publish_directory
    from artifacts.research import load_research_run
    from core.invariants import verify_recurrent_contract
    from simulation.sensor_schedule import SampleSchedule
    from generator import prepare_flywire_candidates as old
    from generator import source_contract
    from optimizer import benchmark_candidate, probe_yaw_recovery
    from evaluate import validate_sensor_handover
    assert old.publication is publication and old.write_json is write_json
    assert source_contract.file_sha256 is file_sha256
    assert source_contract.publish_directory is publish_directory
    assert benchmark_candidate.load_research_run is load_research_run
    assert probe_yaw_recovery.verify_recurrent_contract is verify_recurrent_contract
    assert validate_sensor_handover.SampleSchedule is SampleSchedule


def test_checkpoint_loader_no_longer_imports_experiment_entrypoints(tmp_path):
    root=Path(__file__).resolve().parents[1]
    code=f'''import sys
sys.path.insert(0,{str(root)!r})
import artifacts.research
import optimizer.mvp_constraints
assert not any(n in sys.modules for n in ('optimizer.benchmark_candidate',
    'optimizer.probe_yaw_recovery','generator.prepare_flywire_candidates',
    'optimizer.search_pipeline','optimizer.rollout'))
'''
    result=subprocess.run([sys.executable,'-I','-c',code],cwd=tmp_path,capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_import_smoke_entrypoint_does_not_execute_work_on_import():
    name = 'test_imports.py'
    import ast
    tree=ast.parse((Path(__file__).resolve().parents[1]/name).read_text(encoding='utf-8'))
    allowed=(ast.Import,ast.ImportFrom,ast.FunctionDef,ast.ClassDef,ast.Assign,ast.If)
    for node in tree.body:
        assert isinstance(node,allowed) or (isinstance(node,ast.Expr) and isinstance(node.value,ast.Constant)), name


def test_portable_release_runs_outside_repository_and_rejects_corruption(exported,tmp_path):
    from edge.release import build_release
    _, bundle=exported
    release=tmp_path/'portable'
    cfg=config(bundle,coordinator_mode='both',stop_hover=True,sensor_mode='multi_rate_v1',
               handover_mode='freeze_sync_v1')
    build_release(cfg,release)
    elsewhere=tmp_path/'elsewhere'; elsewhere.mkdir()
    command=[sys.executable,'-I',str(release/'run.py'),'--self-check']
    result=subprocess.run(command,cwd=elsewhere,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    report=json.loads(result.stdout)
    assert report['passed'] and report['accepted_commands']>0
    assert report['watchdog_tripped'] and report['old_session_rejected']
    assert report['forbidden_modules_loaded']==[]
    assert report['model_sha256']==hashlib.sha256((bundle/'model.npz').read_bytes()).hexdigest()
    with (release/'session_receiver.py').open('a') as f: f.write('\n# corrupted\n')
    result=subprocess.run(command,cwd=elsewhere,capture_output=True,text=True)
    assert result.returncode!=0 and 'checksum' in result.stderr.lower()
