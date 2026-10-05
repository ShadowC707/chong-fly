"""Diagnostic directed projection from pinned, explicitly selected FlyWire roots.

This report is not a trainable graph or an RC motor mapping. Zero connections
are valid evidence; incomplete queries fail instead of publishing a report.
"""
import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import io
import json
from pathlib import Path
from unittest.mock import patch

from generator.cave_source import _fetch_complete
from generator.published_annotations import annotation_source, select_annotations
from generator.source_contract import canonical_ids, manifest_sha256


def select_pathway(annotations, registry, pathway):
    if (not isinstance(registry, dict)
            or registry.get('format_version') != 'flight-candidate-registry-v1'
            or type(registry.get('materialization_version')) is not int
            or registry['materialization_version'] != 783
            or registry.get('annotation_sha256') != annotation_source()['sha256']):
        raise ValueError('Registry must pin the supported annotation release and snapshot')
    pathways, candidates = registry.get('pathways'), registry.get('candidates')
    if not isinstance(pathways, dict) or not isinstance(candidates, dict):
        raise ValueError('Registry requires pathways and candidates')
    spec = pathways.get(pathway)
    if not isinstance(spec, dict):
        raise ValueError('Unknown candidate pathway')
    for direction in ('pre','post'):
        names = spec.get(direction)
        if (not isinstance(names, list) or not names
                or any(not isinstance(name, str) or not name for name in names)
                or len(set(names)) != len(names)):
            raise ValueError('Pathway endpoints must be nonempty unique candidate lists')
    targets = sorted(set(spec['pre'] + spec['post']))
    selectors = {}
    for name in targets:
        entry = candidates.get(name)
        if (not isinstance(entry, dict) or entry.get('annotation_status') != 'resolved'
                or not isinstance(entry.get('selector'), dict)):
            raise ValueError(f'Unresolved annotation identity: {name}')
        selectors[name] = entry['selector']
    nodes, selection = select_annotations(annotations, version=783, target_classes=targets,
                                         selectors=selectors)
    selection['strategy'] = 'directed_candidate_projection'
    selection['registry_sha256'] = manifest_sha256(registry)
    selection['pathway'] = pathway
    return nodes, selection, spec


def audit_candidate_pathway(client, annotations, registry, pathway, *, max_rows=50000):
    if type(max_rows) is not int or max_rows <= 0:
        raise ValueError('max_rows must be a positive integer')
    nodes, selection, spec = select_pathway(annotations, registry, pathway)
    pre = nodes[nodes.cell_type.isin(spec['pre'])]
    post = nodes[nodes.cell_type.isin(spec['post'])]
    filters = {'pre_pt_root_id':[int(root) for root in pre.root_id],
               'post_pt_root_id':[int(root) for root in post.root_id]}
    synapses = _fetch_complete(client.materialize, 'synapses_nt_v1', 783,
                               filters, 'pre_pt_root_id', max_rows)
    source = {'source_kind':'cave', 'datastack':'flywire_fafb_public',
        'materialization_version':783, 'synapse_table':'synapses_nt_v1',
        'annotation_table':None, 'annotation_source':{
            **annotation_source(), 'type_matching':'explicit_per_target_selectors'},
        'completeness':'query-count-checked', 'selection':selection,
        'query_filters':{column:[str(root) for root in roots] for column, roots in filters.items()},
        'query_scope':'directed pre-to-post projection only; no intermediate-neuron search',
        'synapse_quality_filter':'none; all returned rows matching the root filters',
        'synapse_rows':len(synapses)}
    edges = []
    if not synapses.empty:
        if synapses.id.duplicated().any():
            raise ValueError('Duplicate synapse IDs across query partitions')
        for key in ('pre_pt_root_id','post_pt_root_id'):
            synapses[key] = canonical_ids(synapses[key], key)
        edges = (synapses.groupby(['pre_pt_root_id','post_pt_root_id']).size()
                 .reset_index(name='weight')
                 .rename(columns={'pre_pt_root_id':'pre_id','post_pt_root_id':'post_id'})
                 .to_dict('records'))
    source['synapse_id_set_sha256'] = manifest_sha256(
        sorted(str(value) for value in synapses.id) if not synapses.empty else [])
    return {'format_version':'candidate-projection-audit-v2',
        'checked_at_utc':datetime.now(timezone.utc).isoformat(), 'source':source,
        'nodes':nodes.to_dict('records'), 'edges':edges,
        'selected_pre_count':len(pre), 'selected_post_count':len(post),
        'edge_pairs':len(edges), 'total_synapses':len(synapses),
        **projection_summary(nodes, edges, post.root_id.tolist()),
        'laterality_semantics':'annotation side; not visual receptive field or drone turn direction',
        'rc_mapping_validated':False, 'synaptic_signs_assigned':False,
        'functional_identity_evidence':{name:registry['candidates'][name].get('evidence', {})
                                        for name in selection['target_classes']}}


def projection_summary(nodes, edges, post_roots):
    """Summarize validated projection edges without treating same-root rows as inter-neuron evidence."""
    sides = dict(zip(nodes.root_id, nodes.side))
    inter = [edge for edge in edges if edge['pre_id'] != edge['post_id']]
    same = [edge for edge in edges if edge['pre_id'] == edge['post_id']]
    thresholds = {}
    for threshold in (1,5,10,20):
        kept = [edge for edge in inter if edge['weight'] >= threshold]
        ipsi = sum(edge['weight'] for edge in kept if sides[edge['pre_id']] == sides[edge['post_id']]
                   and sides[edge['pre_id']] in {'left','right'})
        contra = sum(edge['weight'] for edge in kept
                     if {sides[edge['pre_id']],sides[edge['post_id']]} == {'left','right'})
        total = sum(edge['weight'] for edge in kept)
        thresholds[str(threshold)] = {
            'edge_pairs':len(kept), 'total_synapses':total,
            'connected_pre_roots':sorted({edge['pre_id'] for edge in kept}),
            'connected_post_roots':sorted({edge['post_id'] for edge in kept}),
            'ipsilateral_synapses':ipsi, 'contralateral_synapses':contra,
            'center_involving_synapses':total-ipsi-contra,
            'per_post_root':{root:{'side':sides[root],
                'synapses':sum(edge['weight'] for edge in kept if edge['post_id'] == root),
                'connected_pre_count':sum(edge['post_id'] == root for edge in kept)}
                for root in post_roots}}
    return {'thresholds':thresholds,
        'same_root_connections':{'edge_pairs':len(same),'synapses':sum(edge['weight'] for edge in same)},
        'inter_neuron_connections':{'edge_pairs':len(inter),'synapses':sum(edge['weight'] for edge in inter)},
        'same_root_policy':'preserved in raw edges; excluded from inter-neuron thresholds; authenticity not inferred',
        'threshold_scope':'inter_neuron_edges_only',
        'threshold_semantics':'minimum unsigned synapse count per directed neuron pair, inclusive',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry', type=Path, default=Path('configs/flight_candidates_783.json'))
    parser.add_argument('--annotations', type=Path, default=Path('data/annotations_flywire783_v2_1_0/neurons.tsv'))
    parser.add_argument('--pathway', default='lplc2_dnp06')
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--max-rows', type=int, default=50000)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Refusing to overwrite an existing audit')
    if args.max_rows <= 0:
        parser.error('--max-rows must be positive')
    registry = json.loads(args.registry.read_text(encoding='utf-8'))
    # Validate local identity before any credential loading or network access.
    select_pathway(args.annotations, registry, args.pathway)
    import requests
    original_request = requests.sessions.Session.request
    def bounded_request(session, method, url, **kwargs):
        kwargs['timeout'] = (8, 60)
        return original_request(session, method, url, **kwargs)
    try:
        # Serial CLI only. Native auth stays inside CAVEclient; error text and
        # library output are suppressed because they may contain request details.
        with patch.object(requests.sessions.Session, 'request', bounded_request), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            from caveclient import CAVEclient
            client = CAVEclient('flywire_fafb_public', version=783, max_retries=0,
                                write_server_cache=False)
            report = audit_candidate_pathway(client, args.annotations, registry,
                                             args.pathway, max_rows=args.max_rows)
    except Exception as exc:
        print(json.dumps({'status':'failed', 'error_type':type(exc).__name__,
                          'report_written':False}))
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({'status':'complete', 'out':str(args.out), 'edge_pairs':report['edge_pairs'],
                      'total_synapses':report['total_synapses']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
