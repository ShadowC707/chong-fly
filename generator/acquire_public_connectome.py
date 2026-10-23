"""Acquire a pinned FlyWire 783 subgraph using published labels and CAVE synapses.

The interface scope is a diagnostic induced subgraph, not a complete sensory
pathway. No aliases, intermediate neurons or neurotransmitter signs are guessed.
"""
import argparse
import json
from pathlib import Path

from generator.circuit_extractor import load_cell_mapping, build_target_classes, build_exclusion_sets
from generator.published_annotations import select_annotations, annotation_source
from generator.cave_source import fetch_cave_connectivity
from generator.source_contract import write_bundle, manifest_sha256


def prepare_selection(annotations, config, scope, *, allow_namespace_union=False):
    if scope == 'interface':
        groups = list(config['sensor_groups'].values()) + list(config['motor_groups'].values())
        targets = [name for group in groups if isinstance(group, dict) for name in group.get('cell_types', [])]
    elif scope == 'configured':
        targets = build_target_classes(config, config.get('include_central_complex', True))
    else:
        raise ValueError('Choose interface or configured scope explicitly')
    excluded_classes, excluded_types = build_exclusion_sets(config)
    nodes, selection = select_annotations(annotations, version=783, target_classes=targets,
        excluded_classes=excluded_classes, excluded_types=excluded_types,
        allow_namespace_union=allow_namespace_union)
    selection['scope'] = scope
    selection['cell_mapping_sha256'] = manifest_sha256(config)
    selection['interface_groups'] = {
        f'{kind}.{name}': {
            'nodes':int(nodes.cell_type.isin(group['cell_types']).sum()),
            'sides':nodes.loc[nodes.cell_type.isin(group['cell_types']), 'side'].value_counts().to_dict()}
        for kind in ('sensor_groups','motor_groups') for name, group in config[kind].items()
        if isinstance(group, dict) and 'cell_types' in group}
    return nodes, selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', type=Path, default=Path('data/annotations_flywire783_v2_1_0/neurons.tsv'))
    parser.add_argument('--config')
    parser.add_argument('--scope', required=True, choices=['interface','configured'])
    parser.add_argument('--out-dir', required=True, type=Path)
    parser.add_argument('--report', type=Path, default=Path('data/flywire_public_selection_report.json'))
    parser.add_argument('--max-rows', type=int, default=50000)
    parser.add_argument('--selection-only', action='store_true', help='Audit annotations without requesting connectivity')
    parser.add_argument('--allow-namespace-union', action='store_true',
                        help='Explicit diagnostic union of colliding names; not a biological mapping approval')
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError('Refusing to overwrite an existing raw graph')
    if args.max_rows <= 0:
        parser.error('--max-rows must be positive')
    cfg = load_cell_mapping(args.config)
    nodes, selection = prepare_selection(args.annotations, cfg, args.scope,
                                         allow_namespace_union=args.allow_namespace_union)
    report = {'datastack':'flywire_fafb_public', 'materialization_version':783,
              'annotation_source':annotation_source(), 'selection':selection}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)
    if args.selection_only:
        return 0
    from caveclient import CAVEclient
    client = CAVEclient('flywire_fafb_public', version=783, max_retries=0, write_server_cache=False)
    nodes, edges, source = fetch_cave_connectivity(client, nodes,
        datastack='flywire_fafb_public', version=783, synapse_table='synapses_nt_v1',
        annotation_source=annotation_source(), selection=selection, max_rows=args.max_rows)
    source['cell_mapping_sha256'] = manifest_sha256(cfg)
    source['synapse_quality_filter'] = 'none; all returned rows between selected roots'
    summary = {'dataset':'cave', 'num_nodes':len(nodes), 'num_edges':len(edges),
               'total_synapses':int(edges.weight.sum()), 'purpose':'diagnostic induced subgraph',
               'selection':selection, 'motor_mapping_biologically_verified':False}
    write_bundle(nodes, edges, args.out_dir, source, summary)
    print(json.dumps({'output':str(args.out_dir), **summary}, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
