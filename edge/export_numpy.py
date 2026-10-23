"""Host-side exact export. PyTorch is required here, never in the edge bundle."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from artifacts.research import load_research_run
from simulation.control_contract import control_contract
from simulation.xy_coordinator import XYConfig, BehaviorConfig

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export_policy(policy,output,*,source):
    output=Path(output)
    if output.exists():
        raise FileExistsError('Export requires a new output directory')
    c=policy.cfc_network.cell
    if (policy.sensor_layer.encoding!='proximity-v1' or c.connectivity!='structured'
            or c.solver_type!='exponential_euler' or policy.sensor_dim not in (66,74)
            or policy.default_dt!=.02 or c.mode not in ('fixed','masked')):
        raise ValueError('Export requires structured proximity-v1 / exponential_euler at dt=0.02')
    if c._architecture_contract.get('sensor_encoding_parameters') != {'threshold_m':.8,'max_range_m':3.5}:
        raise ValueError('Unsupported proximity-v1 encoding parameters')
    if any(p.dtype != torch.float32 for p in policy.parameters()):
        raise ValueError('Export supports float32 parameters only; no silent precision conversion')
    def array(t):
        a=t.detach().cpu().numpy().astype(np.float32,copy=True)
        if not np.isfinite(a).all():
            raise ValueError('Nonfinite export tensor')
        return a
    W=c._effective_W().detach()
    if c.mode=='masked':
        W=W*c._synapse_mask
    head=policy.dn_head.proj
    arrays=dict(sensor_gain=array(policy.sensor_layer.gain),sensor_bias=array(policy.sensor_layer.bias),
        W=array(W),W_in=array(c.W_in.weight*c.W_in.route_mask),
        input_bias=array(c.W_in.bias*c.W_in.bias_mask),b=array(c.b),A=array(c.A),tau=array(c.tau),
        dn_weight=array(head.weight*head.route_mask),dn_bias=array(head.bias*head.bias_mask))
    output.mkdir(parents=True)
    np.savez_compressed(output/'model.npz',**arrays)
    for name,path in dict(runtime=ROOT/'edge/runtime.py',system=ROOT/'edge/system.py',sensors=ROOT/'edge/sensors.py',flight_bridge=ROOT/'edge/flight_bridge.py',benchmark=ROOT/'edge/benchmark.py',
        altitude_control=ROOT/'simulation/altitude_control.py',xy_coordinator=ROOT/'simulation/xy_coordinator.py').items():
        shutil.copyfile(path,output/(name+'.py'))
    (output/'requirements.txt').write_text('numpy>=1.26,<3\n',encoding='utf-8')
    original=asdict(XYConfig())
    meta=dict(format='chong-fly-numpy-v1',solver='exponential_euler',sensor_encoding='proximity-v1',
        matrix_orientation='source_target_h_at_W',dtype='float32',dt=.02,
        hidden_size=c.hidden_size,sensor_dim=policy.sensor_dim,
        tof_range_m=3.5,proximity_threshold_m=.8,
        action_order=['throttle','roll','pitch','yaw'],pwm=dict(min=1000.,mid=1500.,half=500.,max=2000.),
        source=source,neural_contract=c.get_extra_state(),control_contract=control_contract(),
        xy_config=original,xy_presets=dict(original_v1=original,near_zero_v1=asdict(
            XYConfig(stopped_flow=-.025,forward_flow=.005,acquire_yaw_pwm=60.))),
        default_coordinator_mode='baseline',parameter_bytes=sum(a.nbytes for a in arrays.values()),
        optional_behaviors=dict(version='xy-coordinator-v2', default_enabled=False,
                                config=asdict(BehaviorConfig())),
        export_torch_version=str(torch.__version__),export_numpy_version=np.__version__,
        limitations=['Research checkpoint; export does not grant flight admission.',
            'Checksum detects changed files, not authenticity against an attacker rewriting the whole bundle.',
            'Neural throttle is a placeholder; use FlightRuntime with valid altitude telemetry.',
            'No serial/Bluetooth, motor arming, scheduler, or hardware authority arbiter included.'])
    meta['files']={p.name:digest(p) for p in sorted(output.iterdir()) if p.is_file()}
    (output/'metadata.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    (output/'metadata.sha256').write_text(digest(output/'metadata.json')+'\n',encoding='ascii')
    return meta


def export_run(run,output):
    run=Path(run)
    policy,checkpoint=load_research_run(run)
    report=json.loads((run/'report.json').read_text(encoding='utf-8'))
    return export_policy(policy,output,source=dict(candidate=checkpoint['candidate'],
        checkpoint_sha256=digest(run/'checkpoint.pt'),report_sha256=digest(run/'report.json'),
        graph_meta_sha256=checkpoint['meta_sha256'],
        continuation=checkpoint.get('continuation'),training_constraints=checkpoint.get('training_constraints'),
        mvp_admission=report.get('mvp_admission'),training_ready=checkpoint['training_ready']))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=ROOT/'data/selection/mvp_frozen_core_v1/trial_00003/after')
    parser.add_argument('--out-dir',type=Path,required=True)
    args=parser.parse_args(); torch.set_num_threads(1)
    m=export_run(args.run_dir,args.out_dir)
    print(json.dumps(dict(output=str(args.out_dir.resolve()),candidate=m['source']['candidate'],
                         parameter_bytes=m['parameter_bytes'],model_file_bytes=(args.out_dir/'model.npz').stat().st_size)))


if __name__=='__main__':
    main()
