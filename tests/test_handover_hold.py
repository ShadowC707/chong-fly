from dataclasses import replace
import numpy as np
import pytest

from edge.flight_bridge import FlightBridge, SensorFrame
from edge.runtime import FlightRuntime
from edge.sensors import SensorBudgets
from edge.system import RunConfig, build_system
from tests.test_edge_runtime import exported
from tests.test_sensor_hold import snapshot
from tests.test_system_assembly import config


def test_multirate_bridge_holds_and_revokes_on_stale_channel(exported):
    _,bundle=exported
    bridge=FlightBridge(FlightRuntime(bundle),sensor_budgets=SensorBudgets(),handover_mode='freeze_sync_v1')
    first=snapshot(0)
    for i in range(5):
        s=snapshot(i,tof=first.tof)
        r=bridge.tick(s,now_s=i*.02,request='SHADOW',manual_pwm=[1213.,1500.,1500.,1500.])
        assert r['fault'] is None
    assert bridge.sensor_tracker.last_report['tof']['state']=='held'
    r=bridge.tick(snapshot(5,tof=first.tof),now_s=.10,request='SHADOW',manual_pwm=[1213.,1500.,1500.,1500.])
    assert r['fault'] is None
    r=bridge.tick(snapshot(6,tof=first.tof),now_s=.12,request='AUTO',manual_pwm=[1213.,1500.,1500.,1500.])
    assert r['state']=='FAULT' and r['lease'] is None
    assert 'tof' in r['fault']
    assert bridge.tick(None,now_s=.14,request='MANUAL',manual_pwm=None)['state']=='MANUAL'
    bridge.reset_fault()
    assert bridge.sensor_tracker.last_report=={}


def test_shadow_handover_tracking_and_auto_cancellation(exported):
    from copy import deepcopy
    _,bundle=exported
    runtime=FlightRuntime(bundle)
    bridge=FlightBridge(runtime,handover_mode='freeze_sync_v1')
    for i in range(101):
        t=i*.02
        s=snapshot(i)
        # Stable height below target makes the old integrator accumulate.
        from edge.runtime import AltitudeSample
        sample=AltitudeSample(.902,0.,0.,t,True)
        prediction=deepcopy(runtime.altitude).update(sample,now_s=t,integrate=False)
        manual=[prediction.throttle_pwm-15.,1500.,1500.,1500.]
        r=bridge.tick(SensorFrame(i,t,[0.,0.],np.ones(64),sample),now_s=t,
                      request='AUTO' if i==100 else 'SHADOW',manual_pwm=manual)
        assert runtime.altitude._integral==0.
    assert r['state']=='AUTO'
    assert r['lease'].pwm==tuple(manual)
    assert runtime.altitude._pwm==manual[0]
    history=runtime.altitude._time
    sample=replace(sample,timestamp_s=2.02)
    next_result=bridge.tick(SensorFrame(101,2.02,[0.,0.],np.ones(64),sample),now_s=2.02,
                            request='AUTO',manual_pwm=manual)
    assert abs(next_result['preview_pwm'][0]-manual[0]) <= 3.+1e-9
    assert runtime.altitude._time > history
    integral=runtime.altitude._integral
    sample=replace(sample,timestamp_s=2.04)
    r=bridge.tick(SensorFrame(102,2.04,[0.,0.],np.ones(64),sample),now_s=2.04,
                  request='SHADOW',manual_pwm=manual)
    assert r['lease'] is None and runtime.altitude._integral==integral


def test_system_manifest_and_trace_expose_new_contract(exported):
    _,bundle=exported
    system=build_system(config(bundle,sensor_mode='multi_rate_v1',handover_mode='freeze_sync_v1'))
    first=snapshot(0)
    system.tick(first,now_s=0.,request='SHADOW',manual_pwm=[1213.,1500.,1500.,1500.])
    result=system.tick(snapshot(1,tof=first.tof,altitude=first.altitude),now_s=.02,
                       request='SHADOW',manual_pwm=[1213.,1500.,1500.,1500.])
    assert result['trace'].to_dict()['sensor_channels']['altitude']['state']=='held'
    assert system.manifest['resolved']['sensor_contract']=='multi_rate_v1'
    assert system.manifest['resolved']['shadow_integrator']=='freeze_sync_v1'
    assert result['trace'].altitude_integral==0.
    with pytest.raises(ValueError,match='altitude'):
        build_system(config(bundle,sensor_mode='multi_rate_v1',sensor_budgets=SensorBudgets(altitude_max_age_s=.20)))


def test_held_downrange_cannot_hide_fresh_excessive_attitude(exported):
    from edge.sensors import Measurement
    _,bundle=exported
    b=FlightBridge(FlightRuntime(bundle),sensor_budgets=SensorBudgets(),handover_mode='freeze_sync_v1')
    first=snapshot(0)
    b.tick(first,now_s=0.,request='SHADOW',manual_pwm=[1213.,1500.,1500.,1500.])
    r=b.tick(snapshot(1,altitude=first.altitude,attitude=Measurement(1,.02,(.3,0.))),
             now_s=.02,request='AUTO',manual_pwm=[1213.,1500.,1500.,1500.])
    assert r['state']=='SHADOW' and not r['ready']
    assert b.runtime.altitude.last_command.height_m==1.


@pytest.mark.parametrize('height,delta',[(h,d) for h in (.92,1.,1.08) for d in (0.,-15.,15.)])
def test_exact_grid_freezes_and_limits_handover(exported,height,delta):
    from evaluate.validate_sensor_handover import telemetry_case
    _,bundle=exported
    case=telemetry_case(bundle,height,delta,'freeze_sync_v1','multi')
    assert case['passed']
    assert case['metrics']['sensor_hold_ticks']['altitude']>0
    assert case['metrics']['sensor_hold_ticks']['tof']>0


def test_new_sensor_path_preserves_isolated_synchronous_baseline(exported):
    from evaluate.validate_sensor_handover import verify_synchronous
    _,bundle=exported
    report=verify_synchronous(bundle,bundle,steps=100)
    assert report['passed']
    assert 'sensors.py' in report['after']['executed_modules']


def test_invalid_sample_at_engagement_never_advances_network_or_issues(exported):
    _,bundle=exported
    system=build_system(config(bundle,sensor_mode='multi_rate_v1',handover_mode='freeze_sync_v1'))
    for i in range(40):
        system.tick(snapshot(i),now_s=i*.02,request='SHADOW',manual_pwm=[1213.,1500.,1500.,1500.])
    previous=system.runtime.policy.h.copy()
    bad=snapshot(40)
    bad=replace(bad,tof=replace(bad.tof,valid=False))
    result=system.tick(bad,now_s=.8,request='AUTO',manual_pwm=[1213.,1500.,1500.,1500.])
    assert result['state']=='FAULT' and result['lease'] is None
    assert result['trace'].network_pwm is None
    np.testing.assert_array_equal(previous,system.runtime.policy.h)
    assert result['trace'].to_dict()['sensor_channels']['tof']['state']=='invalid'
    assert system.tick(snapshot(41),now_s=.82,request='AUTO',manual_pwm=None)['state']=='FAULT'
    system.tick(None,now_s=.84,request='MANUAL',manual_pwm=None)
    system.bridge.reset_fault()
    assert system.tick(snapshot(43),now_s=43*.02,request='SHADOW',
                       manual_pwm=[1213.,1500.,1500.,1500.])['fault'] is None


def test_new_modes_run_in_flat_bundle_without_repository_or_torch(exported,tmp_path):
    import subprocess,sys,json
    _,bundle=exported
    code='''
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from system import RunConfig,build_system
from sensors import Measurement,SensorSnapshot
b=Path(sys.argv[1])
s=build_system(RunConfig(bundle_dir=str(b),expected_metadata_sha256=hashlib.sha256((b/'metadata.json').read_bytes()).hexdigest(),coordinator_mode='both',stop_hover=True,sensor_mode='multi_rate_v1',handover_mode='freeze_sync_v1'))
for i in range(40):
 t=i*.02
 if i%4==0: tof=Measurement(i,t,(1.,)*64)
 if i%2==0: alt=Measurement(i,t,(.982,0.,0.))
 frame=SensorSnapshot(i,t,Measurement(i,t,(0.,0.)),tof,alt,Measurement(i,t,(0.,0.)))
 r=s.tick(frame,now_s=t,request='SHADOW' if i<30 else 'AUTO',manual_pwm=[1213.,1500.,1500.,1500.],stop_requested=True)
 assert r['fault'] is None
assert r['state']=='AUTO' and r['lease'] is not None
assert r['trace'].to_dict()['sensor_channels']['altitude']['state']=='held'
assert not any(n.split('.')[0] in ('torch','simulation','edge') for n in sys.modules)
print(json.dumps(dict(state=r['state'],accepted=r['trace'].accepted_pwm)))
'''
    result=subprocess.run([sys.executable,'-I','-c',code,str(bundle)],cwd=tmp_path,
                           text=True,capture_output=True,timeout=30)
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)==dict(state='AUTO',accepted=None)
