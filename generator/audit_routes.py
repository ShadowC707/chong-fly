"""Compare directed sensor→motor reachability before and after graph reduction.

Read-only graph audit: it never adds edges or changes input/output mappings.
Paths establish a necessary condition for learning, not biological validity,
observability, controllability, or preservation of recurrent dynamics.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from core.routing import motor_mask, sensor_motor_paths
from generator.graph_reducer import ReducedModel, load_graph
from generator.reflex_contract import REQUIRED_PATHS
from generator.role_reducer import RolePreservingReducer


def audit_routes(source, models):
    sensors, motors = source._build_index_maps(np.arange(source.N))
    adjacency = sp.coo_matrix((source.signed_weights, (source.pre_idx, source.post_idx)),
                              shape=(source.N, source.N)).tocsr()
    adjacency.sum_duplicates()
    adjacency.eliminate_zeros()
    raw_paths = sensor_motor_paths(adjacency, sensors, motor_mask(source.N, motors))
    successors = {}
    for group, indices in sensors.items():
        destinations = np.unique(adjacency[indices].indices)
        successors[group] = sorted(set(source.nodes_df.iloc[destinations].cell_type.tolist()))
    report = {
        'audit_version': 'sensor-routes-v1',
        'source_sha256': source.source_sha256,
        'source_acquisition': source.source_provenance,
        'source_nodes': source.N,
        'source_nonzero_edges': adjacency.nnz,
        'source_sensor_motor_paths': raw_paths,
        'source_direct_sensor_successor_types': successors,
        'models': [],
    }
    for model in models:
        if model.provenance.get('source_sha256') != source.source_sha256:
            raise ValueError('Reduced model source fingerprint differs from the audited source')
        paths = sensor_motor_paths(model.W, model.sensor_index_map, motor_mask(model.k, model.motor_index_map))
        comparison = {}
        for group, channels in REQUIRED_PATHS.items():
            for channel in channels:
                before = raw_paths[group]['minimum_hops'][channel] is not None
                after = paths[group]['minimum_hops'][channel] is not None
                comparison[f'{group}->{channel}'] = {
                    (True, True): 'present_in_both',
                    (True, False): 'lost_after_reduction',
                    (False, True): 'introduced_after_reduction',
                    (False, False): 'missing_in_source_or_mapping',
                }[before, after]
        report['models'].append({
            'k': model.k, 'reducer': model.reducer_name,
            'declared_source_kind': model.provenance.get('source_kind', 'unverified'),
            'sensor_motor_paths': paths, 'required_routes': comparison,
        })
    return report


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=root/'data/raw_connectome')
    parser.add_argument('--config', type=Path, default=root/'configs/cell_mapping.json')
    parser.add_argument('--polarity-map', type=Path, help='The same explicit polarity map used for reduction')
    parser.add_argument('--allow-legacy-source', action='store_true', help='Explicitly audit unverified historical raw data')
    parser.add_argument('--models', type=Path, nargs='+', default=[
        root/f'data/reduced_models/meta_role_degree_k{k}.json' for k in (128, 256)])
    parser.add_argument('--output', type=Path, default=root/'data/sensor_route_audit_v1.json')
    args = parser.parse_args(argv)
    polarity = json.loads(args.polarity_map.read_text(encoding='utf-8')) if args.polarity_map else None
    nodes, edges, pre, post, weights = load_graph(str(args.data_dir), allow_legacy=args.allow_legacy_source,
                                                polarity_map=polarity)
    source = RolePreservingReducer(nodes, edges, json.loads(args.config.read_text(encoding='utf-8')),
                                   pre, post, weights)
    models = [ReducedModel.load(str(path), allow_legacy=False) for path in args.models]
    report = audit_routes(source, models)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    for row in report['models']:
        print(f"k={row['k']}: {row['required_routes']}")


if __name__ == '__main__':
    main()
