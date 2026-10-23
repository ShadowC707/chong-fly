import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from edge.export_numpy import export_policy
from edge.runtime import NumpyPolicy, FlightRuntime
from simulation.altitude_control import AltitudeHold, AltitudeSample, AltitudeTelemetryError
from simulation.policy import ChongFlyMSPPolicy
from simulation.xy_coordinator import XYConfig, XYCoordinator
from tests.synthetic_graphs import write_tiny_model


@pytest.fixture
def exported(tmp_path):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        policy = ChongFlyMSPPolicy.from_meta(str(write_tiny_model(tmp_path/'graph')),
            mode='masked', preserve_signs=True, dt=.02, tau_init=.025,
            sensor_encoding='proximity-v1', neutral_origin=True)
    bundle = tmp_path/'bundle'
    export_policy(policy, bundle, source={'candidate': 'synthetic_test_only'})
    return policy, bundle


def test_multistep_policy_matches_hidden_state_and_pwm(exported):
    reference, bundle = exported
    runtime = NumpyPolicy(bundle)
    rng = np.random.default_rng(9)
    for i in range(400):
        flow = rng.uniform(-.3,.3,2).astype(np.float32)
        tof = rng.uniform(.04,1.,64).astype(np.float32) if i % 80 < 40 else np.ones(64,np.float32)
        memory = rng.uniform(0,1,8).astype(np.float32)
        want = reference.step_np(flow,tof,memory_ring=memory,dt=.02)
        got = runtime.step_np(flow,tof,memory_ring=memory,dt=.02)
        np.testing.assert_allclose(got,want,atol=.02,rtol=0)
        np.testing.assert_allclose(runtime.h,reference._hx.numpy()[0],atol=2e-5,rtol=2e-4)
    runtime.reset_state(); reference.reset_state()
    np.testing.assert_allclose(runtime.step_np(flow,tof),reference.step_np(flow,tof),atol=.02,rtol=0)


def test_pipeline_matches_coordinator_then_altitude_and_latches_fault(exported):
    reference,bundle=exported
    runtime=FlightRuntime(bundle,coordinator_mode='both')
    config=XYConfig(**runtime.metadata['xy_config'])
    xy=XYCoordinator(anti_rollback=True,turn_latch=True,config=config)
    altitude=AltitudeHold()
    for i in range(120):
        flow=np.array([.1*np.cos(i/20),.03],np.float32)
        tof=np.ones(64,np.float32); tof[10:30]=.15 if i<60 else 1.
        sample=AltitudeSample(.982,0.,0.,i*.02,True)
        raw=reference.step_np(flow,tof)
        desired=altitude.apply(xy.apply(raw,flow_xy=flow,tof_m=tof*3.5,dt=.02),sample,now_s=i*.02)
        actual=runtime.step(flow,tof,sample,now_s=i*.02)
        np.testing.assert_allclose(actual['network_pwm'],raw,atol=.02,rtol=0)
        np.testing.assert_allclose(actual['applied_pwm'],desired,atol=.02,rtol=0)
    with pytest.raises(AltitudeTelemetryError):
        runtime.step(flow,tof,AltitudeSample(.2,0.,0.,2.4,True),now_s=2.4)
    with pytest.raises(AltitudeTelemetryError):
        runtime.step(flow,tof,AltitudeSample(.982,0.,0.,2.42,True),now_s=2.42)
    runtime.reset()
    runtime.step(flow,tof,AltitudeSample(.982,0.,0.,0.,True),now_s=0.)


@pytest.mark.parametrize('corrupted', ['model.npz','metadata.json','altitude_control.py','flight_bridge.py'])
def test_bundle_detects_corruption(exported,corrupted):
    _,bundle=exported
    with (bundle/corrupted).open('ab') as stream:
        stream.write(b'corruption')
    with pytest.raises(ValueError,match='[Hh]ash|checksum'):
        NumpyPolicy(bundle)


def test_contract_rejects_unsupported_solver_even_with_updated_checksum(exported):
    _,bundle=exported
    path=bundle/'metadata.json'; metadata=json.loads(path.read_text())
    metadata['solver']='euler'; path.write_text(json.dumps(metadata))
    (bundle/'metadata.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(ValueError,match='contract'):
        NumpyPolicy(bundle)


def test_invalid_inputs_do_not_advance_neural_state(exported):
    _,bundle=exported; policy=NumpyPolicy(bundle)
    for tof,dt in [(np.full(64,np.nan),.02),(np.ones(64),.04),(np.full(64,2.),.02)]:
        with pytest.raises(ValueError):
            policy.step_np(np.zeros(2),tof,dt=dt)
        np.testing.assert_array_equal(policy.h,np.zeros_like(policy.h))


def test_export_refuses_unsupported_encoding_and_overwrite(exported,tmp_path):
    policy,bundle=exported
    with pytest.raises(FileExistsError):
        export_policy(policy,bundle,source={})
    policy.sensor_layer.encoding='distance-v1'
    with pytest.raises(ValueError,match='proximity'):
        export_policy(policy,tmp_path/'unsupported',source={})


def test_bundle_runs_outside_repository_without_torch_or_scipy(exported,tmp_path):
    _,bundle=exported
    code = '''
import sys, importlib.abc
class Reject(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','scipy','simulation','core','configs'}:
            raise AssertionError('Forbidden runtime dependency: '+fullname)
sys.meta_path.insert(0,Reject())
sys.path.insert(0,sys.argv[1])
from runtime import FlightRuntime, AltitudeSample
from flight_bridge import FlightBridge, CommandGate
import numpy as np
r=FlightRuntime(sys.argv[1],coordinator_mode='both',levy_explore=True,stop_hover=True)
result=r.step(np.zeros(2),np.ones(64),AltitudeSample(.982,0.,0.,0.,True),now_s=0.)
assert np.isfinite(result['applied_pwm']).all()
r.step(np.zeros(2),np.ones(64),AltitudeSample(.982,0.,0.,.02,True),now_s=.02,stop_requested=True)
assert 'torch' not in sys.modules and 'scipy' not in sys.modules
print('standalone-ok')
'''
    result=subprocess.run([sys.executable,'-I','-c',code,str(bundle)],cwd=tmp_path,
                          capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr
    assert 'standalone-ok' in result.stdout


@pytest.mark.parametrize('extra', [{}, {'pause_search_wait':True},
    {'target_half_gate':True,'search_amplitude_pwm':240.,'search_duration_s':.6},
    {'pause_search_wait':True,'target_half_gate':True,'search_amplitude_pwm':240.,'search_duration_s':.6}])
def test_exported_behavior_module_matches_source_through_search_preemption_stop(exported,extra):
    import importlib.util
    from simulation.xy_coordinator import BehaviorConfig
    _,bundle=exported
    # Load the copied standalone source, not the repository adapter.
    spec=importlib.util.spec_from_file_location('exported_xy_behavior_test',bundle/'xy_coordinator.py')
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    try:
        spec.loader.exec_module(module)
        a=XYCoordinator(anti_rollback=True,turn_latch=True,levy_explore=True,stop_hover=True,seed=19,
                        behavior_config=BehaviorConfig(**extra))
        b=module.XYCoordinator(anti_rollback=True,turn_latch=True,levy_explore=True,stop_hover=True,seed=19,
                               behavior_config=module.BehaviorConfig(**extra))
        states=set()
        for i in range(700):
            threat=250<=i<280
            raw=[1213.,1500.,1527.,1400. if threat else 1500.]
            kw=dict(flow_xy=[.02*np.cos(i/40), .01],tof_m=np.full(64,.6 if threat else 3.),
                    dt=.02,stop_requested=i>=500)
            np.testing.assert_array_equal(a.apply(raw,**kw),b.apply(raw,**kw))
            assert a.last_decision==b.last_decision
            states.add(a.last_decision['state'])
        assert {'SEARCH','STOP','CRUISE'} <= states
    finally:
        sys.modules.pop(spec.name,None)


def test_edge_runtime_stop_request_is_forwarded_and_disabled_request_rejected(exported):
    _,bundle=exported
    runtime=FlightRuntime(bundle,coordinator_mode='both',levy_explore=True,stop_hover=True)
    for i in range(40):
        result=runtime.step([.1,.1],np.ones(64),AltitudeSample(.982,0.,0.,i*.02,True),
                            now_s=i*.02,stop_requested=True)
    assert result['applied_pwm'][3]==1500
    assert result['applied_pwm'][1]>1500 and result['applied_pwm'][2]<1500
    assert result['coordinator']['state']=='STOP'
    disabled=FlightRuntime(bundle)
    with pytest.raises(ValueError,match='STOP'):
        disabled.step([0,0],np.ones(64),AltitudeSample(.982,0.,0.,0.,True),now_s=0.,stop_requested=True)
    np.testing.assert_array_equal(disabled.policy.h,np.zeros_like(disabled.policy.h))
