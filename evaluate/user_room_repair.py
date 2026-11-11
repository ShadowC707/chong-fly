"""Bounded user-room reproduction and braking-only ablation. No promotion."""
import argparse
from contextlib import closing
from dataclasses import replace
import json
from pathlib import Path
import shutil
import numpy as np

from evaluate.sitl_long_flight import digest, write, trace_metrics
from simulation.navigation_metrics import repeated_motion_windows, waiting_metrics


def braking_only_config(config):
    # Do NOT adopt near_zero_v1's separate change to latch acquisition.
    return replace(config,stopped_flow=-.025,forward_flow=.005)


def configure_view(view,variant,folder):
    """Only at the generator's pre-flight boundary, before FC starts/first tick."""
    from simulation.xy_coordinator import XYCoordinator
    from edge.system import FlightSystem
    if view.tick or view.observed is not None:raise ValueError('Configuration is frozen after first step')
    if variant not in ('smooth_control','xy_control','xy_brake','smooth_retry'):raise ValueError('Unknown variant')
    if variant=='smooth_control':return
    old=view.system.runtime.coordinator
    cls=XYCoordinator;system_cls=FlightSystem
    extra={}
    if variant=='smooth_retry':
        from simulation.phased_recovery import RecoveryPhasedCoordinator
        from demo.yaw_experiment import YawHandoverSystem
        cls=RecoveryPhasedCoordinator;system_cls=YawHandoverSystem
        extra['phased_config']=old.phased_config
    c=cls(anti_rollback=old.anti_rollback,turn_latch=old.turn_latch,config=old.config,
        stop_hover=old.stop_hover,behavior_config=old.behavior_config,seed=old.seed,**extra)
    if variant=='xy_brake':c.config=braking_only_config(c.config)
    runtime=view.system.runtime;runtime.coordinator=c
    manifest=view.system.manifest
    manifest['version']='user-room-repair-experiment-v1'
    manifest['resolved']['coordinator']=c.contract()
    manifest['experiment']=dict(variant=variant,braking_window_only=variant=='xy_brake',
        yaw_shaping=variant=='smooth_retry',phased=variant=='smooth_retry',neural_weights_changed=False,no_promotion=True)
    if variant=='smooth_retry':
        source=Path('simulation/phased_recovery.py')
        manifest['executed_modules'][source.name]=dict(path=str(source),sha256=digest(source))
    system=system_cls(view.system.config,runtime,manifest)
    view.system=system;view.integrated.system=system
    view.manifest['system']=system.manifest
    view.manifest['user_room_variant']=variant
    write(folder/'manifest.json',view.manifest)


def run(out,variant,geometry,reference=False):
    from demo.sitl import pinned_profile,SOURCE
    from evaluate.sitl_neural import flight_steps
    from evaluate.sitl_long_flight import long_scene
    from evaluate.sitl_lockstep import BIN
    rows,verified=pinned_profile()
    g=json.loads(geometry.read_text())
    spec=(long_scene() if reference else dict(room=g['room'],boxes=g['boxes'],cylinders=g['cylinders'],
        start=[-.7,0.,1.],yaw=0.,radius_m=g['collision_radius_m']))
    horizon=45. if reference else 180.;stop=40. if reference else None
    files=[p for d in ('simulation','demo','edge','evaluate') for p in Path(d).glob('*.py')]
    files += [geometry,Path('configs/demo_candidate_a.json'),Path('configs/ready_sim_reference_v1.json'),
        *Path('output/tau_continuation_v1/adaptive_tau/numpy').glob('*')]
    hashes={str(p):digest(p) for p in files if p.is_file()}
    out.mkdir(parents=True,exist_ok=False)
    write(out/'protocol.json',dict(variant=variant,scene=spec,horizon_s=horizon,stop_at_s=stop,
        controller='Frozen A; independent braking-window OR bounded WAIT_CLEAR retry treatment',
        sensors='SITL ideal 50/30/15Hz; NOT identical noisy standalone plant',target_twr=5.,
        loop_metric='30s windows every5s, path>1.5m AND both XY spans<=.8m; offline truth only',
        physical_criteria='No contact, full horizon, clearance>=.10m, height error<=.10m, continuous AUTO',
        navigation_criteria='No repeat-motion window, no continuous WAIT_CLEAR/FAILED >5s, no blocked latch',
        acquisition_yaw_pwm=20,braking_only_window=[-.025,.005] if variant=='xy_brake' else [.02,.06],
        no_training=True,no_search=True,no_promotion=True,firmware_sha256=digest(BIN),
        verified_profile=verified,input_sha256=hashes))
    for name in ('evaluate/user_room_repair.py','evaluate/sitl_neural.py','simulation/xy_coordinator.py',
                 'simulation/phased_coordinator.py','simulation/phased_yaw.py','simulation/phased_recovery.py'):
        dest=out/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(name,dest)
    folder=out/'flight'
    gen=flight_steps(folder,SOURCE,'user_room','sitl','reference_units',physics_dt_s=.00025,
        integration_profile='multi_rate',yaw_shaping=True,scene_override=spec,target_twr=5.,
        calibration_rows=rows,horizon_s=horizon,stop_at_s=stop)
    detailed=[];checkpoint=0.
    with closing(gen):
        view=next(gen);configure_view(view,variant,folder)
        while True:
            try:view=gen.send({})
            except StopIteration as end:
                result=end.value;break
            detailed.append(dict(time_s=view.time_s,decision=view.system.runtime.coordinator.last_decision.copy()))
            if view.time_s>=checkpoint:
                print(json.dumps(dict(variant=variant,time_s=view.time_s,position=view.state[:3].tolist(),
                    clearance=view.clearance,phase=view.decision.get('phase'))),flush=True);checkpoint+=30.
    with np.load(folder/'trace.npz') as a:
        result.update(trace_metrics(a['time_s'],a['state'],a['clearance'],a['acks']))
        # 50Hz view of position is sufficient for the coarse progress metric.
        ids=np.r_[np.arange(0,len(a['time_s'])-1,80),len(a['time_s'])-1]
        result['repeat_motion_windows']=repeated_motion_windows(a['time_s'][ids],a['state'][ids,:2])
    result['variant']=variant;result['stop_requested']=stop is not None
    result.update(waiting_metrics(detailed))
    if stop is None:
        result['stop_passed']=None;result['stop_last_second_max_speed_mps']=None
    result['sources_unchanged']=all(digest(k)==v for k,v in hashes.items())
    result['practical_passed']=bool(result['completed'] and not result['error'] and result['min_clearance_m']>=.1
        and result['max_height_error_m']<=.1 and result['authority_passed'] and not result['repeat_motion_windows']
        and not result['unresolved_wait'])
    write(out/'report.json',result);write(out/'coordinator_details.json',detailed)
    print(json.dumps(result,indent=2),flush=True)
    if result['error'] or not result['clock_passed'] or not result['sources_unchanged']:raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--variant',choices=['smooth_control','xy_control','xy_brake','smooth_retry'],required=True)
    p.add_argument('--geometry',type=Path,default=Path('output/user_room_audit_v1/geometry.json'))
    p.add_argument('--reference',action='store_true')
    a=p.parse_args();run(a.out,a.variant,a.geometry,a.reference)
