"""Host-only multi-step PyTorch/NumPy validation of an exported checkpoint."""
import argparse
import json
import os
from pathlib import Path

for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'

import numpy as np
import torch

from edge.runtime import FlightRuntime, sha256
from artifacts.research import load_research_run
from simulation.altitude_control import AltitudeHold, AltitudeSample
from simulation.xy_coordinator import XYConfig, XYCoordinator


def _validate_repository(run,bundle,steps=2000):
    reference,checkpoint=load_research_run(run)
    cases=[('baseline','original_v1',False),('both','original_v1',False),
           ('both','near_zero_v1',False),('both','original_v1',True)]
    runtimes=[FlightRuntime(bundle,coordinator_mode=mode,coordinator_preset=preset,
        levy_explore=behavior,stop_hover=behavior,seed=20261021) for mode,preset,behavior in cases]
    meta=runtimes[0].metadata
    if meta['source']['checkpoint_sha256']!=sha256(Path(run)/'checkpoint.pt'):
        raise ValueError('Validation checkpoint differs from export source')
    controllers=[AltitudeHold() for _ in cases]
    coordinators=[None if mode=='baseline' else XYCoordinator(anti_rollback=True,turn_latch=True,
        config=XYConfig(**meta['xy_presets'][preset]),levy_explore=behavior,stop_hover=behavior,
        seed=20261021) for mode,preset,behavior in cases]
    frames=_replay_inputs(steps)
    raw_error=hidden_error=0.
    applied_errors=np.zeros(len(cases)); decision_mismatches=np.zeros(len(cases),dtype=int)
    for i in range(steps):
        flow=frames['flow'][i]
        tof=frames['tof'][i].reshape(8,8)
        memory=frames['memory'][i]
        sample=AltitudeSample(*frames['altitude'][i],frames['times'][i],True)
        expected=reference.step_np(flow,tof,memory_ring=memory,dt=.02)
        for j,(runtime,altitude,xy) in enumerate(zip(runtimes,controllers,coordinators)):
            stop=bool(cases[j][2] and i>=steps-250)
            actual=runtime.step(flow,tof,sample,now_s=i*.02,memory_ring=memory,stop_requested=stop)
            coordinated=expected.copy() if xy is None else xy.apply(expected,flow_xy=flow,tof_m=tof*3.5,
                                                                   dt=.02,stop_requested=stop)
            applied=altitude.apply(coordinated,sample,now_s=i*.02)
            want_reasons=[] if xy is None else xy.last_decision['reasons']
            decision_mismatches[j]+=actual['coordinator']['reasons']!=want_reasons
            raw_error=max(raw_error,float(abs(actual['network_pwm']-expected).max()))
            hidden_error=max(hidden_error,float(abs(runtime.policy.h-reference._hx.numpy()[0]).max()))
            applied_errors[j]=max(applied_errors[j],float(abs(actual['applied_pwm']-applied).max()))
            if (not np.allclose(actual['network_pwm'],expected,atol=.02,rtol=0)
                    or not np.allclose(runtime.policy.h,reference._hx.numpy()[0],atol=2e-5,rtol=2e-4)
                    or not np.allclose(actual['applied_pwm'],applied,atol=.02,rtol=0)):
                raise AssertionError(f'Numerical parity failed at step {i}, mode {cases[j]}')
    if decision_mismatches.any():
        raise AssertionError('Coordinator decisions differ')
    return dict(version='numpy-parity-v1',execution='repository',passed=True,candidate=checkpoint['candidate'],steps=steps,
        simulated_time_s=steps*.02,source_checkpoint_sha256=sha256(Path(run)/'checkpoint.pt'),
        bundle_metadata_sha256=sha256(Path(bundle)/'metadata.json'),
        max_network_pwm_error=raw_error,max_hidden_state_error=hidden_error,
        pipelines=[dict(mode=mode,preset=preset,levy_stop=behavior,max_applied_pwm_error=float(applied_errors[j]),
            intervention_reason_mismatches=int(decision_mismatches[j])) for j,(mode,preset,behavior) in enumerate(cases)],
        tolerance=dict(pwm_atol=.02,pwm_rtol=0,hidden_atol=2e-5,hidden_rtol=2e-4),
        conditions='Synthetic temporal sequence: clear, near/far narrow/broad left/right threats, signed flow, zero-drive recovery, varying memory, valid altitude samples.',
        limitation='Numerical replay on identical inputs, not a new closed-loop flight evaluation.')


def _replay_inputs(steps):
    rng = np.random.default_rng(20261021)
    frames = dict(flow=[], tof=[], memory=[], altitude=[], times=[])
    for i in range(steps):
        phase = i % 400
        flow = (rng.normal(0, .01, 2)+[.15*np.sin(i/45), .1*np.cos(i/37)]).astype(np.float32)
        tof = np.ones((8, 8), np.float32)
        if 60 <= phase < 220:
            width = (1, 2, 4)[(i//400) % 3]
            cols = slice(0, width) if (i//200) % 2 else slice(8-width, 8)
            tof[:, cols] = np.float32((.4 if phase < 140 else .7)/3.5)
        if phase >= 300:
            flow[:] = 0.
        frames['flow'].append(flow)
        frames['tof'].append(tof.ravel())
        frames['memory'].append(rng.uniform(0, 1, 8).astype(np.float32))
        frames['altitude'].append([.982+.003*np.sin(i/50), 0., 0.])
        frames['times'].append(i*.02)
    return {name: np.asarray(values) for name, values in frames.items()}


def validate(run, bundle, steps=2000, *, execution='bundled'):
    """Default validates the package's code, not a host implementation using its weights."""
    if type(steps) is not int or not 1 <= steps <= 10000:
        raise ValueError('steps must be an integer in 1..10000')
    if execution == 'repository':
        return _validate_repository(run, bundle, steps)
    if execution != 'bundled':
        raise ValueError('execution must be bundled or repository')
    from edge.bundle_validation import replay_bundle, _verified_metadata
    from simulation.altitude_control import AltitudeConfig
    from simulation.xy_coordinator import BehaviorConfig

    reference, checkpoint = load_research_run(run)
    meta = _verified_metadata(Path(bundle))
    if meta['source']['checkpoint_sha256'] != sha256(Path(run)/'checkpoint.pt'):
        raise ValueError('Validation checkpoint differs from export source')
    cases = [dict(coordinator_mode=mode, coordinator_preset=preset,
                  levy_explore=behavior, stop_hover=behavior, seed=20261021)
             for mode, preset, behavior in [('baseline', 'original_v1', False),
                 ('both', 'original_v1', False), ('both', 'near_zero_v1', False),
                 ('both', 'original_v1', True)]]
    frames = _replay_inputs(steps)
    stop = np.zeros((len(cases), steps), dtype=bool)
    stop[-1, max(0, steps-250):] = True
    actual = replay_bundle(bundle, frames, cases, stop_requests=stop)
    controllers = [AltitudeHold(AltitudeConfig(**meta['control_contract']['altitude']['config'])) for _ in cases]
    coordinators = [None if c['coordinator_mode'] == 'baseline' else XYCoordinator(
        anti_rollback=True, turn_latch=True, config=XYConfig(**meta['xy_presets'][c['coordinator_preset']]),
        levy_explore=c['levy_explore'], stop_hover=c['stop_hover'], seed=c['seed'],
        behavior_config=BehaviorConfig(**meta['optional_behaviors']['config'])) for c in cases]
    raw_error = hidden_error = 0.
    applied_errors = np.zeros(len(cases))
    mismatches = np.zeros(len(cases), dtype=int)
    for i, t in enumerate(frames['times']):
        flow, tof, memory = frames['flow'][i], frames['tof'][i], frames['memory'][i]
        expected = reference.step_np(flow, tof, memory_ring=memory, dt=.02)
        sample = AltitudeSample(*frames['altitude'][i], t, True)
        for j, (altitude, xy) in enumerate(zip(controllers, coordinators)):
            coordinated = expected.copy() if xy is None else xy.apply(expected, flow_xy=flow,
                tof_m=tof*3.5, dt=.02, stop_requested=bool(stop[j, i]))
            supervised = altitude.apply(coordinated, sample, now_s=t)
            want_reasons = [] if xy is None else xy.last_decision['reasons']
            mismatches[j] += actual['report']['reasons'][j][i] != want_reasons
            raw_error = max(raw_error, float(abs(actual['network_pwm'][j, i]-expected).max()))
            hidden_error = max(hidden_error, float(abs(actual['hidden'][j, i]-reference._hx.numpy()[0]).max()))
            applied_errors[j] = max(applied_errors[j], float(abs(actual['supervised_pwm'][j, i]-supervised).max()))
            if (not np.allclose(actual['network_pwm'][j, i], expected, atol=.02, rtol=0)
                    or not np.allclose(actual['hidden'][j, i], reference._hx.numpy()[0], atol=2e-5, rtol=2e-4)
                    or not np.allclose(actual['supervised_pwm'][j, i], supervised, atol=.02, rtol=0)):
                raise AssertionError(f'Bundled numerical parity failed at step {i}, case {j}')
    if mismatches.any():
        raise AssertionError('Bundled coordinator decisions differ')
    evidence = {key: value for key, value in actual['report'].items() if key != 'reasons'}
    return dict(version='numpy-parity-v2', execution='bundled', passed=True,
        candidate=checkpoint['candidate'], steps=steps, simulated_time_s=steps*.02,
        source_checkpoint_sha256=sha256(Path(run)/'checkpoint.pt'),
        bundle_metadata_sha256=sha256(Path(bundle)/'metadata.json'),
        max_network_pwm_error=raw_error, max_hidden_state_error=hidden_error,
        pipelines=[dict(mode=c['coordinator_mode'], preset=c['coordinator_preset'],
            levy_stop=c['stop_hover'], max_applied_pwm_error=float(applied_errors[j]),
            intervention_reason_mismatches=int(mismatches[j])) for j, c in enumerate(cases)],
        execution_evidence=evidence,
        tolerance=dict(pwm_atol=.02, pwm_rtol=0, hidden_atol=2e-5, hidden_rtol=2e-4),
        conditions='Same deterministic temporal stimuli as repository parity; each case starts with fresh state.',
        limitation='Synchronous numerical replay; no Bridge/Gate, transport, sensor hold, or flight admission.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=Path('data/selection/mvp_frozen_core_v1/trial_00003/after'))
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--execution',choices=('bundled','repository'),default='bundled')
    args=parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Use a new validation report filename')
    torch.set_num_threads(1)
    report=validate(args.run_dir,args.bundle,execution=args.execution)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
