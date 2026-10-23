"""Load an exact research artifact without importing benchmark/probe entrypoints."""
import json
from pathlib import Path
import torch

from artifacts.io import file_sha256
from configs.flight_config import CONTROL_DT
from simulation.control_contract import control_contract
from simulation.policy import ChongFlyMSPPolicy


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
    if 'training_constraints' in checkpoint:
        from optimizer.mvp_constraints import RecurrentConstraints
        # requires_grad is not in state_dict. Restore freeze flags and verify
        # the saved bound without altering inference weights on load.
        RecurrentConstraints(policy, metadata=checkpoint['training_constraints'])
    policy.eval()
    return policy, checkpoint
