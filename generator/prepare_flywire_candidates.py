"""Acquire official filtered FlyWire 783 data, then prepare a small audited grid.

python -m generator.prepare_flywire_candidates acquire --out-dir <new-source>
python -m generator.prepare_flywire_candidates build --source <source> --out-dir <new-grid>
All publication is atomic and refuses existing output directories.
"""
import argparse
import contextlib
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import scipy.sparse as sp

from generator.filtered_candidates import (SYNAPSE_VIEW, CONNECTION_VIEW, fetch_view, clean_synapses,
    SIGN_ASSUMPTIONS, assign_polarities, select_bridges, compare_reachability)
from generator.published_annotations import select_annotations, annotation_source, ANNOTATION_SHA256
from generator.source_contract import file_sha256, manifest_sha256, write_bundle, read_bundle, publish_directory

DEFAULT_CONFIG = Path('configs/filtered_candidates_783_v1.json')
ANNOTATIONS = Path('data/annotations_flywire783_v2_1_0/neurons.tsv')


def validate_config(config):
    if (config.get('format_version') != 'filtered-flight-candidates-v1'
        or config.get('datastack') != 'flywire_fafb_public'
        or config.get('materialization_version') != 783
        or config.get('annotation_sha256') != ANNOTATION_SHA256
        or config.get('synapse_view') != SYNAPSE_VIEW
        or config.get('discovery_view') != CONNECTION_VIEW
        or config.get('polarity_policy') != SIGN_ASSUMPTIONS):
        raise ValueError('Unsupported source/polarity contract')
    for field in ('pair_min','bridge_budget'):
        if type(config.get(field)) is not int or config[field] < 1:
            raise ValueError('Positive integer selection budgets required')
    if not 0 <= config.get('bridge_min_prediction_confidence',-1) <= 1:
        raise ValueError('Invalid bridge NT confidence')
    sensor = config.get('sensor_contract',{})
    expected = {'dimension':74,'flow':[0,2],'tof':[2,66],'memory':[66,74],
                'memory_mapping':'disconnected','light_sensors':'not_integrated'}
    if any(sensor.get(k) != v for k,v in expected.items()):
        raise ValueError('Unsupported sensor contract')
    if config.get('families') != ['saccade_core_identity','interface_identity','bridge_identity','bridge_role_minimum']:
        raise ValueError('Unsupported candidate families')


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


@contextlib.contextmanager
def publication(output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Refusing to replace existing source/candidates')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.prepare-', dir=output.parent) as tmp:
        stage = Path(tmp)/'result'; stage.mkdir()
        yield stage
        publish_directory(stage, output)


def select_seeds(config, annotations):
    registry = json.loads(Path(config['identity_registry']).read_text(encoding='utf-8'))
    if registry['annotation_sha256'] != ANNOTATION_SHA256 or config['materialization_version'] != 783:
        raise ValueError('Identity source version mismatch')
    selectors = {name:registry['candidates'][name]['selector'] for name in config['seed_candidates']}
    selectors.update(config['flow_selectors'])
    nodes, selection = select_annotations(annotations, version=783, target_classes=list(selectors), selectors=selectors)
    selection['identity_registry_sha256'] = manifest_sha256(registry)
    return nodes, selection


def acquire(config_path, annotations, output, max_rows=50000):
    config = json.loads(Path(config_path).read_text(encoding='utf-8'))
    validate_config(config)
    if Path(output).exists():
        raise FileExistsError('Source output exists')
    seeds, selection = select_seeds(config, annotations)
    all_nodes = pd.read_csv(annotations, sep='\t', dtype=str, keep_default_na=False)
    flow = list(config['flow_selectors'])
    sensory = seeds.loc[seeds.cell_type.isin(flow+['LPLC2']), 'root_id'].tolist()
    motor = seeds.loc[seeds.cell_type.isin(['DNp06','DNp15','DNae014','DNb01']), 'root_id'].tolist()
    import requests
    original_request = requests.sessions.Session.request
    def bounded(session, method, url, **kwargs):
        kwargs.setdefault('timeout', (10,90))
        return original_request(session,method,url,**kwargs)
    requests.sessions.Session.request = bounded
    try:
        from caveclient import CAVEclient
        client = CAVEclient('flywire_fafb_public', version=783, max_retries=0, write_server_cache=False)
        views = client.materialize.get_views(version=783)
        if SYNAPSE_VIEW not in views or CONNECTION_VIEW not in views:
            raise ValueError('Required official filtered views unavailable at 783')
        metadata = {name:client.materialize.get_view_metadata(name, materialization_version=783)
                    for name in (SYNAPSE_VIEW,CONNECTION_VIEW)}
        outgoing, log_out = fetch_view(client.materialize, CONNECTION_VIEW,
                                      {'pre_pt_root_id':sensory}, max_rows=max_rows)
        incoming, log_in = fetch_view(client.materialize, CONNECTION_VIEW,
                                     {'post_pt_root_id':motor}, max_rows=max_rows)
        # Eligibility is explicit model scope, not evidence that other neurons are irrelevant.
        nt = all_nodes.known_nt.where(all_nodes.known_nt != '', all_nodes.top_nt)
        eligible_mask = (nt.isin(config['polarity_policy']) & all_nodes.side.isin(['left','right','center'])
                         & (pd.to_numeric(all_nodes.top_nt_conf) >= config['bridge_min_prediction_confidence']))
        eligible = set(all_nodes.loc[eligible_mask,'root_id'])
        bridges = select_bridges(outgoing,incoming,eligible,set(seeds.root_id),
                                 pair_min=config['pair_min'], budget=config['bridge_budget'])
        bridge_nodes = all_nodes[all_nodes.root_id.isin([r['root_id'] for r in bridges])].copy()
        bridge_nodes['source_cell_type'] = bridge_nodes.cell_type
        bridge_nodes['source_hemibrain_type'] = bridge_nodes.hemibrain_type
        # Never coalesce namespaces or silently label unnamed neurons as one type.
        bridge_nodes['cell_type'] = ['cell_type:'+r.cell_type if r.cell_type else
                                    'hemibrain_type:'+r.hemibrain_type if r.hemibrain_type else
                                    'root_id:'+r.root_id for r in bridge_nodes.itertuples()]
        nodes = pd.concat([seeds,bridge_nodes], ignore_index=True).fillna('').sort_values('root_id').reset_index(drop=True)
        roots = nodes.root_id.tolist()
        synapses, log_syn = fetch_view(client.materialize,SYNAPSE_VIEW,
            {'pre_pt_root_id':roots,'post_pt_root_id':roots}, max_rows=max_rows)
        clean, edges, quality = clean_synapses(synapses,pair_min=config['pair_min'])
        if edges.empty:
            raise ValueError('Empty cleaned graph')
        with publication(output) as stage:
            raw = stage/'raw'; raw.mkdir()
            outgoing.to_parquet(raw/'discovery_outgoing.parquet',index=False)
            incoming.to_parquet(raw/'discovery_incoming.parquet',index=False)
            synapses.to_parquet(raw/'official_filtered_synapses.parquet',index=False)
            clean.to_parquet(stage/'clean_synapses.parquet',index=False)
            nodes.to_csv(stage/'nodes.csv',index=False)
            edges.to_parquet(stage/'edges.parquet',index=False)
            write_json(stage/'config.json',config)
            write_json(stage/'selection.json',selection)
            write_json(stage/'bridges.json',bridges)
            write_json(stage/'view_metadata.json',metadata)
            write_json(stage/'identity_registry.json',json.loads(Path(config['identity_registry']).read_text(encoding='utf-8')))
            write_json(stage/'query_log.json',log_out+log_in+log_syn)
            manifest = {'format_version':'filtered783-source-v1','status':'complete',
                'created_at_utc':datetime.now(timezone.utc).isoformat(),
                'datastack':'flywire_fafb_public','materialization_version':783,
                'annotation_source':annotation_source(),'config_sha256':manifest_sha256(config),
                'seed_root_ids':seeds.root_id.tolist(), 'bridge_root_ids':[r['root_id'] for r in bridges],
                'quality':quality,'query_completeness':'count-checked disjoint partitions; no offset pagination',
                'scope':'induced seed+bounded two-hop bridges; not a complete brain circuit',
                'server_deduplication':'official valid_synapses_nt_v2 via valid_synapses_nt_np_v6; no local geometric recreation',
                'same_root_policy':'retained in raw response, excluded from clean graph; not asserted to be real autapses',
                'files':{str(p.relative_to(stage)).replace('\\','/'):file_sha256(p) for p in stage.rglob('*') if p.is_file()}}
            write_json(stage/'manifest.json',manifest)
    finally:
        requests.sessions.Session.request = original_request
    return {'source':str(output),'nodes':len(nodes),'edges':len(edges),'bridges':len(bridges),'quality':quality}


def read_source(source):
    source = Path(source)
    manifest = json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('format_version') != 'filtered783-source-v1' or manifest.get('status') != 'complete':
        raise ValueError('Incomplete or incompatible filtered source')
    required = {'raw/discovery_outgoing.parquet','raw/discovery_incoming.parquet',
                'raw/official_filtered_synapses.parquet','clean_synapses.parquet','nodes.csv','edges.parquet',
                'config.json','selection.json','bridges.json','view_metadata.json','query_log.json'}
    if (not required <= set(manifest.get('files',{})) or manifest.get('materialization_version') != 783
        or manifest.get('datastack') != 'flywire_fafb_public'
        or manifest.get('annotation_source',{}).get('sha256') != ANNOTATION_SHA256):
        raise ValueError('Missing source evidence')
    for name,digest in manifest['files'].items():
        path = (source/name).resolve()
        if not path.is_relative_to(source.resolve()) or file_sha256(path) != digest:
            raise ValueError('Filtered source SHA256 mismatch')
    config = json.loads((source/'config.json').read_text(encoding='utf-8'))
    validate_config(config)
    if manifest_sha256(config) != manifest['config_sha256']:
        raise ValueError('Source config fingerprint mismatch')
    nodes = pd.read_csv(source/'nodes.csv',dtype=str,keep_default_na=False)
    edges = pd.read_parquet(source/'edges.parquet')
    return nodes,edges,manifest,config


def smoke(meta_path, *, control=False):
    import torch
    from generator.graph_reducer import ReducedModel
    model = ReducedModel.load(str(meta_path),allow_legacy=False)
    torch.manual_seed(42)
    if control:
        h = np.linspace(0,1,model.k)
        for _ in range(20): h = np.tanh(h @ model.W)
        return {'kind':'anatomical recurrent matrix only', 'steps':20, 'finite':bool(np.isfinite(h).all())}
    from simulation.policy import ChongFlyMSPPolicy
    from simulation.control_contract import control_contract
    from configs.flight_config import CONTROL_DT
    policy = ChongFlyMSPPolicy.from_meta(str(meta_path), connectivity='structured', dt=CONTROL_DT)
    policy.eval()
    observations = torch.zeros(1,74); observations[:,2:] = 1
    h = None
    with torch.no_grad():
        for _ in range(20):
            action,h = policy(observations,h)
            if not torch.isfinite(action).all() or not torch.isfinite(h).all():
                raise ValueError('Nonfinite policy smoke')
    # A smoke test is not a sign-of-response or behavioral test.
    return {'kind':'structured 74-input policy', 'steps':20,'finite':True, 'dt':CONTROL_DT,
            'dtype':'float32',
            'output_shape':list(action.shape),'control_contract':control_contract(),
            'behavior_validated':False}


def build(source, output):
    from generator.role_reducer import RolePreservingReducer
    from generator.audit_routes import audit_routes
    from generator.graph_reducer import ReducedModel
    from generator.validate_flywire_candidates import quality_audit
    nodes, edges, manifest, config = read_source(source)
    quality = quality_audit(source)
    root_sets = {'saccade_core':set(nodes.loc[nodes.cell_type.isin(['VES041','DNae014','DNb01']),'root_id']),
                 'interface':set(manifest['seed_root_ids']), 'bridge':set(nodes.root_id)}
    registry = {'format_version':'filtered783-candidate-registry-v1',
                'source_manifest_sha256':file_sha256(Path(source)/'manifest.json'),
                'config_sha256':manifest['config_sha256'],'sensor_contract':config['sensor_contract'],
                'candidates':[], 'training_ready':False,
                'source_quality_review_required':quality['quality_review_required']}
    with publication(output) as stage:
        write_json(stage/'source_quality_audit.json',quality)
        for family,roots in root_sets.items():
            selected = nodes[nodes.root_id.isin(roots)].reset_index(drop=True)
            selected_edges = edges[edges.pre_id.isin(roots) & edges.post_id.isin(roots)].copy()
            bundle = stage/'sources'/family
            acquisition = {'source_kind':'cave','datastack':'flywire_fafb_public','materialization_version':783,
                'synapse_table':SYNAPSE_VIEW,'annotation_table':None,'annotation_source':annotation_source(),
                'completeness':'query-count-checked','parent_manifest_sha256':registry['source_manifest_sha256'],
                'selection':{'family':family,'root_ids':selected.root_id.tolist()},
                'quality':manifest['quality'],'pair_min':config['pair_min']}
            write_bundle(selected,selected_edges,bundle,acquisition,{'purpose':'candidate anatomy; not training approval'})
            selected,selected_edges,_ = read_bundle(bundle)
            # read_bundle defaults CSV null handling; retain explicit NT fields from source.
            selected = selected.fillna('')
            selected['idx'] = np.arange(len(selected))
            index = dict(zip(selected.root_id,selected.idx))
            pre = selected_edges.pre_id.map(index).to_numpy(dtype=np.int32)
            post = selected_edges.post_id.map(index).to_numpy(dtype=np.int32)
            signs, polarity = assign_polarities(selected)
            selected.attrs['polarity_contract'] = {'version':'nt-assumptions-v1','biologically_verified':False,'per_root':polarity}
            weights = selected_edges.weight.to_numpy(dtype=float)*signs[pre]
            reducer = RolePreservingReducer(selected,selected_edges,config['mapping'],pre,post,weights)
            ks = sorted({reducer.N, reducer.minimum_k}) if family == 'bridge' else [reducer.N]
            full = sp.coo_matrix((weights,(pre,post)),shape=(reducer.N,reducer.N)).tocsr()
            for k in ks:
                name = f'{family}_role_k{k}'
                model = reducer(k)
                model.provenance.update(sensor_contract=config['sensor_contract'], decoder_status=config['decoder_status'],
                    training_ready=False, preparation_version='filtered-flight-candidates-v1')
                control = family == 'saccade_core'
                paths = model.save(str(stage/name))
                loaded = ReducedModel.load(paths['meta'],allow_legacy=False)
                if not np.array_equal(loaded.cluster_map,model.cluster_map) or (loaded.W != model.W).nnz:
                    raise ValueError('Candidate round-trip mismatch')
                sources = sorted(set(i for values in reducer._sensor_node_sets.values() for i in values))
                targets = sorted(set(i for values in reducer._motor_node_sets.values() for i in values))
                pair_audit = compare_reachability(full,loaded.W,loaded.cluster_map,sources,targets)
                route_audit = None if control else audit_routes(reducer,[loaded])
                reasons = ['anatomical_control_without_sensor_contract'] if control else []
                if not control:
                    reasons.extend(key+':'+value for key,value in route_audit['models'][0]['required_routes'].items()
                                   if value != 'present_in_both')
                if not pair_audit['passed']: reasons.append('individual_interface_reachability_changed')
                checks = smoke(paths['meta'],control=control)
                row = {'candidate':name,'family':family,'k':k,'source_nodes':reducer.N,'minimum_k':reducer.minimum_k,
                    'edges':loaded.W.nnz,'source_edges':len(selected_edges),
                    'meta_file':str(Path(paths['meta']).relative_to(stage)).replace('\\','/'),
                    'source_bundle':f'sources/{family}', 'route_audit':route_audit,'interface_pair_audit':pair_audit,
                    'smoke':checks,'structural_admission':'rejected' if reasons else 'passed',
                    'rejection_reasons':reasons,'training_ready':False,
                    'source_quality_review_required':quality['quality_review_required'],
                    'open_scientific_gates':['sensor encoding and RC decoder validation','NT/receptor sign assumptions',
                                             'short pretrain and held-out navigation validation'] +
                                             (['review close detections retained by official filter'] if quality['quality_review_required'] else []),
                    'low_confidence_prediction_roots':[r['root_id'] for r in polarity
                                                      if r['basis']=='EM_prediction_top_nt' and r['prediction_confidence']<0.5]}
                write_json(stage/name/'audit.json',row)
                registry['candidates'].append(row)
        registry['files'] = {str(p.relative_to(stage)).replace('\\','/'):file_sha256(p)
                             for p in stage.rglob('*') if p.is_file()}
        write_json(stage/'registry.json',registry)
    return {'output':str(output),'candidates':[{k:r[k] for k in ('candidate','k','edges','minimum_k','structural_admission','rejection_reasons')}
                                             for r in registry['candidates']]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=['acquire','build'])
    parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    parser.add_argument('--annotations',type=Path,default=ANNOTATIONS)
    parser.add_argument('--source',type=Path)
    parser.add_argument('--out-dir',type=Path,required=True)
    parser.add_argument('--max-rows',type=int,default=50000)
    args = parser.parse_args()
    try:
        if args.operation == 'acquire':
            # CAVE may emit response bodies on error. Never echo external exception text.
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                result = acquire(args.config,args.annotations,args.out_dir,args.max_rows)
        else:
            if args.source is None: parser.error('build requires --source')
            result = build(args.source,args.out_dir)
    except Exception as exc:
        print(json.dumps({'status':'failed','stage':args.operation,'error_type':type(exc).__name__,
                          'ready_output_published':False}))
        return 2
    print(json.dumps(result,indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
