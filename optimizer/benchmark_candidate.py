"""Evaluate an immutable research checkpoint in named physical challenge scenes."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from configs.flight_config import CONTROL_DT
from generator.prepare_flywire_candidates import publication, write_json
from generator.source_contract import file_sha256
from optimizer.rollout import simulate_policy_rollout
from simulation.altitude_control import AltitudeHold
from simulation.control_contract import control_contract
from simulation.benchmark_scenes import NavigationScene, SCENES, SCENE_VERSION
from simulation.policy import ChongFlyMSPPolicy
from simulation.policy_diagnostics import temporal_response_probe


class ToFAblation(torch.nn.Module):
    """Replace only ToF by clear readings; keep real flow and memory unchanged."""
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def reset_state(self):
        self.policy.reset_state()

    def step_np(self, flow_xy, tof_8x8, memory_ring=None, dt=None):
        return self.policy.step_np(flow_xy, np.ones_like(tof_8x8),
                                   memory_ring=memory_ring, dt=dt)

    def forward(self, obs, **kwargs):
        ablated = obs.clone()
        ablated[..., 2:66] = 1
        return self.policy(ablated, **kwargs)


def load_research_run(run):
    run = Path(run)
    report = json.loads((run/'report.json').read_text(encoding='utf-8'))
    if file_sha256(run/'checkpoint.pt') != report['checkpoint_sha256']:
        raise ValueError('Checkpoint hash differs from its recorded diagnostic')
    checkpoint = torch.load(run/'checkpoint.pt', map_location='cpu', weights_only=True)
    if (checkpoint.get('format_version') != 'candidate-research-checkpoint-v1'
        or checkpoint.get('training_ready') is not False or checkpoint.get('dt') != CONTROL_DT):
        raise ValueError('Expected a current research checkpoint at control dt')
    if checkpoint.get('control_contract') != control_contract():
        raise ValueError('Checkpoint control contract differs from the current controller')
    metas = list((run/'graph').glob('meta_*.json'))
    if len(metas) != 1 or file_sha256(metas[0]) != checkpoint['meta_sha256']:
        raise ValueError('Graph metadata hash differs from checkpoint')
    metadata = json.loads(metas[0].read_text(encoding='utf-8'))
    if 'graph_file_sha256' in checkpoint:
        for filename, digest in checkpoint['graph_file_sha256'].items():
            path = (run/'graph'/filename).resolve()
            if not path.is_relative_to((run/'graph').resolve()) or file_sha256(path) != digest:
                raise ValueError('Graph file hash differs from checkpoint')
    for key in ('w_file', 'cmap_file'):
        if not (metas[0].parent/metadata[key]).resolve().is_relative_to((run/'graph').resolve()):
            raise ValueError('Graph reference leaves the checkpoint bundle')
    policy = ChongFlyMSPPolicy.from_meta(str(metas[0]), dt=checkpoint['dt'], mode=checkpoint['mode'],
        preserve_signs=checkpoint['preserve_signs'],
        tau_init=checkpoint.get('tau_init', .1),
        sensor_encoding=checkpoint.get('sensor_encoding', 'distance-v1'),
        neutral_origin=checkpoint.get('neutral_origin', False))
    policy.load_state_dict(checkpoint['state_dict'], strict=True)
    policy.eval()
    return policy, checkpoint


def benchmark_run(run, output, *, steps=1000, seeds=(20261007,), ablate_tof=False):
    if type(steps) is not int or not 1 <= steps <= 2000:
        raise ValueError('Use 1..2000 control steps per scene')
    if not seeds or len(seeds) > 3 or len(set(seeds)) != len(seeds) or any(type(s) is not int for s in seeds):
        raise ValueError('Use one to three distinct integer scene seeds')
    policy, checkpoint = load_research_run(run)
    if type(ablate_tof) is not bool:
        raise ValueError('ablate_tof must be an explicit boolean')
    if ablate_tof:
        policy = ToFAblation(policy)
    rows = []
    with publication(Path(output)) as stage:
        for scenario in SCENES:
            for seed in seeds:
                env = NavigationScene(scenario)
                try:
                    _, metrics = simulate_policy_rollout(policy, env=env, eval_steps=steps,
                        seed=seed, dt=CONTROL_DT, altitude_hold=AltitudeHold())
                    rows.append({'scenario':scenario, 'seed':seed, **metrics})
                    print(f'{scenario} seed={seed}: crashed={metrics["crashed"]}, '
                          f'survived={metrics["survival_time_s"]:.3f}s', flush=True)
                finally:
                    env.close()
        root = Path(__file__).resolve().parents[1]
        report = {'format_version':SCENE_VERSION, 'research_only':True, 'training_ready':False,
                  'checkpoint_sha256':file_sha256(Path(run)/'checkpoint.pt'),
                  'candidate':checkpoint['candidate'], 'control_dt':CONTROL_DT, 'steps':steps,
                  'ablate_tof':ablate_tof,
                  'seeds':list(seeds), 'scenes':rows, 'temporal_response':temporal_response_probe(policy),
                  'code_sha256':{name:file_sha256(root/name) for name in
                    ['optimizer/benchmark_candidate.py', 'optimizer/rollout.py',
                     'simulation/benchmark_scenes.py', 'simulation/drone_env.py',
                     'simulation/policy.py', 'core/models.py', 'core/routing.py',
                     'simulation/altitude_control.py', 'simulation/memory.py',
                     'simulation/metrics.py', 'simulation/avionics_filter.py',
                     'simulation/pmw3901_emulator.py', 'simulation/policy_diagnostics.py',
                     'configs/flight_config.py']},
                  'limitations':['Provisional standalone physics, not validated against assembled hardware.',
                                 'Completion in a few fixed scenes is not a flight admission.']}
        write_json(stage/'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--seeds', type=int, nargs='+', default=[20261007])
    parser.add_argument('--ablate-tof', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    benchmark_run(args.run, args.out_dir, steps=args.steps, seeds=tuple(args.seeds), ablate_tof=args.ablate_tof)


if __name__ == '__main__':
    main()
