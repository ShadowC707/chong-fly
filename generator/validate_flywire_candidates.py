"""Offline source/candidate verification and functional smoke; never grants training approval."""
import argparse
from collections import deque
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import cKDTree

from generator.filtered_candidates import assign_polarities, clean_synapses, compare_reachability
from generator.prepare_flywire_candidates import read_source, write_json, smoke
from generator.source_contract import file_sha256, read_bundle


def verify_files(directory, files):
    directory = Path(directory).resolve()
    if not files:
        raise ValueError('Missing artifact SHA256 inventory')
    for name,digest in files.items():
        path = (directory/name).resolve()
        if not path.is_relative_to(directory) or file_sha256(path) != digest:
            raise ValueError('Artifact SHA256 mismatch')


def quality_audit(source):
    source = Path(source)
    nodes, edges, manifest, config = read_source(source)
    syn = pd.read_parquet(source/'raw/official_filtered_synapses.parquet')
    clean, recreated, audit = clean_synapses(syn,pair_min=config['pair_min'])
    pd.testing.assert_frame_equal(recreated,edges)
    pd.testing.assert_frame_equal(clean,pd.read_parquet(source/'clean_synapses.parquet'))
    if audit != manifest['quality']:
        raise ValueError('Source cleaning audit mismatch')
    counts = syn.groupby(['pre_pt_root_id','post_pt_root_id']).size().to_dict()
    roots = set(nodes.root_id)
    report = {'source_manifest_sha256':file_sha256(source/'manifest.json'),
              'cleft_min':float(syn.cleft_score.min()),'query_rows':len(syn),
              'discovery_crosscheck':{},'close_detection_pairs':[],
              'policy':'trust official validity membership; flag close detections, do not silently re-deduplicate'}
    for name in ('outgoing','incoming'):
        frame = pd.read_parquet(source/f'raw/discovery_{name}.parquet')
        frame = frame[frame.pre_pt_root_id.isin(roots) & frame.post_pt_root_id.isin(roots)]
        if any(r.n_syn != counts.get((r.pre_pt_root_id,r.post_pt_root_id),0) for r in frame.itertuples()):
            raise ValueError('Connection view and individual synapse counts differ')
        report['discovery_crosscheck'][name] = {'pairs':len(frame),'mismatches':0}
    for (pre,post), group in syn.groupby(['pre_pt_root_id','post_pt_root_id']):
        if len(group)<2: continue
        rows = group.reset_index(drop=True)
        points = np.stack(group.pre_pt_position)
        for i,j in sorted(cKDTree(points).query_pairs(100)):
            report['close_detection_pairs'].append({'pre':pre,'post':post,
                'distance_nm':float(np.linalg.norm(points[i]-points[j])),
                'synapse_ids':[str(rows.iloc[i].id),str(rows.iloc[j].id)],
                'pre_supervoxels':[str(rows.iloc[i].pre_pt_supervoxel_id),str(rows.iloc[j].pre_pt_supervoxel_id)],
                'post_supervoxels':[str(rows.iloc[i].post_pt_supervoxel_id),str(rows.iloc[j].post_pt_supervoxel_id)]})
    report['quality_review_required'] = bool(report['close_detection_pairs'])
    report['uncertainty'] = ('Server validity membership is verified; precise deduplication grouping/tie-breaking and '
                             'FAFB14 to FAFB14.1 coordinate effects are not independently established. '
                             'Nearby rows are candidates for review, not proven duplicate detections.')
    return report


def make_reducer(nodes,edges,config):
    from generator.role_reducer import RolePreservingReducer
    nodes = nodes.fillna('').copy()
    nodes['idx'] = np.arange(len(nodes))
    index = dict(zip(nodes.root_id,nodes.idx))
    pre = edges.pre_id.map(index).to_numpy(dtype=np.int32)
    post = edges.post_id.map(index).to_numpy(dtype=np.int32)
    signs,records = assign_polarities(nodes)
    nodes.attrs['polarity_contract'] = {'version':'nt-assumptions-v1','biologically_verified':False,'per_root':records}
    weights = edges.weight.to_numpy(dtype=float)*signs[pre]
    return RolePreservingReducer(nodes,edges,config['mapping'],pre,post,weights)


def directed_witness(reducer, source_types, target_types):
    nodes = reducer.nodes_df
    targets = set(nodes.loc[nodes.cell_type.isin(target_types),'idx'])
    parents = {int(i):None for i in nodes.loc[nodes.cell_type.isin(source_types),'idx']}
    queue = deque(parents)
    adjacency = sp.coo_matrix((np.ones(len(reducer.pre_idx)),(reducer.pre_idx,reducer.post_idx)),
                              shape=(reducer.N,reducer.N)).tocsr()
    while queue:
        node = queue.popleft()
        if node in targets:
            path=[]
            while node is not None:
                path.append({'root_id':reducer.root_ids[node],'cell_type':nodes.iloc[node].cell_type})
                node=parents[node]
            return path[::-1]
        for nxt in adjacency.indices[adjacency.indptr[node]:adjacency.indptr[node+1]]:
            if int(nxt) not in parents:
                parents[int(nxt)]=node; queue.append(int(nxt))
    return None


def check_structural_admission(row, reducer, model):
    """Recompute per-neuron reachability and admission, including rejected controls."""
    from generator.audit_routes import audit_routes
    if row.get('family') not in {'saccade_core', 'interface', 'bridge'}:
        raise ValueError('Unknown candidate structure family')
    expected = {'k':model.k, 'source_nodes':reducer.N, 'minimum_k':reducer.minimum_k,
                'edges':model.W.nnz, 'source_edges':len(reducer.signed_weights)}
    if any(row.get(key) != value for key,value in expected.items()):
        raise ValueError('Candidate structure differs from registry')
    full = sp.coo_matrix((reducer.signed_weights, (reducer.pre_idx, reducer.post_idx)),
                         shape=(reducer.N, reducer.N)).tocsr()
    sources = sorted({i for values in reducer._sensor_node_sets.values() for i in values})
    targets = sorted({i for values in reducer._motor_node_sets.values() for i in values})
    pairs = compare_reachability(full, model.W, model.cluster_map, sources, targets)
    control = row['family'] == 'saccade_core'
    routes = None if control else audit_routes(reducer, [model])
    reasons = ['anatomical_control_without_sensor_contract'] if control else [
        key+':'+value for key,value in routes['models'][0]['required_routes'].items()
        if value != 'present_in_both']
    if not pairs['passed']:
        reasons.append('individual_interface_reachability_changed')
    status = 'rejected' if reasons else 'passed'
    if (row.get('interface_pair_audit') != pairs or row.get('route_audit') != routes
        or row.get('rejection_reasons') != reasons or row.get('structural_admission') != status):
        raise ValueError('Candidate admission/audit differs from recomputed structure')
    return {'structural_admission':status, 'rejection_reasons':reasons,
            'interface_pair_audit':pairs, 'route_audit':routes}


def functional_smoke(meta_path):
    import torch
    from simulation.policy import ChongFlyMSPPolicy
    torch.manual_seed(42)
    from configs.flight_config import CONTROL_DT
    policy = ChongFlyMSPPolicy.from_meta(str(meta_path),connectivity='structured', dt=CONTROL_DT).double().eval()
    base = torch.zeros(1,20,74,dtype=torch.float64); base[:,:,2:] = 0.5
    base.requires_grad_()
    output,_ = policy(base)
    gradients={}
    for channel,index in [('roll',1),('pitch',2),('yaw',3)]:
        grad = torch.autograd.grad(output[0,-1,index],base,retain_graph=True)[0]
        if not torch.isfinite(grad).all(): raise ValueError('Nonfinite functional smoke gradient')
        gradients[channel] = {'flow':float(grad[:,:,:2].abs().max()),
                              'tof':float(grad[:,:,2:66].abs().max()),
                              'memory':float(grad[:,:,66:].abs().max())}
    if any(row['memory'] != 0 for row in gradients.values()):
        raise ValueError('Memory channels unexpectedly connected')
    from simulation.policy_diagnostics import response_probe
    native_response = response_probe(policy.float())
    return {'seed':42,'steps':20,'dt':CONTROL_DT,'dtype':'float64 diagnostic','max_absolute_input_gradient':gradients,
            'float32_response':native_response, 'finite':True,'behavior_validated':False}


def validate(source, candidates):
    from generator.graph_reducer import ReducedModel
    from generator.audit_routes import audit_routes
    source, candidates = Path(source),Path(candidates)
    nodes,edges,manifest,config=read_source(source)
    registry=json.loads((candidates/'registry.json').read_text(encoding='utf-8'))
    if (registry.get('format_version') != 'filtered783-candidate-registry-v1'
        or registry['source_manifest_sha256'] != file_sha256(source/'manifest.json')
        or registry['config_sha256'] != manifest['config_sha256']):
        raise ValueError('Candidate source/config provenance mismatch')
    verify_files(candidates,registry['files'])
    quality=quality_audit(source)
    report={'format_version':'filtered783-validation-v2','quality':quality,'candidates':[],
            'registry_sha256':file_sha256(candidates/'registry.json'),'training_ready':False,
            'versions':{name:importlib.metadata.version(name) for name in ['numpy','pandas','scipy','torch','caveclient','pyarrow']}}
    for row in registry['candidates']:
        selected,se,_=read_bundle(candidates/row['source_bundle'])
        expected=nodes[nodes.root_id.isin(selected.root_id)].reset_index(drop=True)
        # Match the exact root identities, roles and transmitter records against the parent.
        for col in ['root_id','cell_type','side','top_nt','known_nt']:
            if selected[col].fillna('').astype(str).tolist()!=expected[col].fillna('').astype(str).tolist():
                raise ValueError('Candidate node annotations differ from parent source')
        if not np.allclose(selected.top_nt_conf.astype(float),expected.top_nt_conf.astype(float),rtol=0,atol=1e-15):
            raise ValueError('Candidate NT confidence differs from parent source')
        ee=edges[edges.pre_id.isin(selected.root_id)&edges.post_id.isin(selected.root_id)].reset_index(drop=True)
        pd.testing.assert_frame_equal(se,ee)
        reducer=make_reducer(selected,se,config)
        meta_path=candidates/row['meta_file']
        loaded=ReducedModel.load(str(meta_path),allow_legacy=False)
        rebuilt=reducer(row['k'])
        if ((loaded.W!=rebuilt.W).nnz or not np.array_equal(loaded.cluster_map,rebuilt.cluster_map)
            or loaded.sensor_index_map!=rebuilt.sensor_index_map or loaded.motor_index_map!=rebuilt.motor_index_map
            or loaded.provenance['source_sha256']!=rebuilt.provenance['source_sha256']
            or loaded.provenance['source_root_ids']!=rebuilt.provenance['source_root_ids']
            or loaded.provenance['polarity_contract']!=rebuilt.provenance['polarity_contract']):
            raise ValueError('Candidate is not reproducible from source, signs and mapping')
        admission = check_structural_admission(row, reducer, loaded)
        control=row['family']=='saccade_core'
        checks=smoke(meta_path,control=control)
        functional=None if control else functional_smoke(meta_path)
        # Worst-case removal of both members of every flagged close pair, research only.
        syn=pd.read_parquet(source/'clean_synapses.parquet')
        suspect={sid for pair in quality['close_detection_pairs'] for sid in pair['synapse_ids']}
        syn=syn[~syn.id.isin(suspect)]
        _,sensitivity_edges,_=clean_synapses(syn,pair_min=config['pair_min'])
        sensitivity_edges=sensitivity_edges[sensitivity_edges.pre_id.isin(selected.root_id)&
                                             sensitivity_edges.post_id.isin(selected.root_id)]
        sensitivity=make_reducer(selected,sensitivity_edges,config)
        sensitivity_routes=None if control else audit_routes(sensitivity,[sensitivity(sensitivity.N)])
        path=directed_witness(reducer,['LPLC2'],['DNae014','DNb01']) if not control else None
        thresholds={}
        raw=pd.read_parquet(source/'clean_synapses.parquet')
        for threshold in [1,5,10,20]:
            _,te,_=clean_synapses(raw,pair_min=threshold)
            te=te[te.pre_id.isin(selected.root_id)&te.post_id.isin(selected.root_id)]
            if te.empty or control: continue
            tr=make_reducer(selected,te,config)
            thresholds[str(threshold)]=audit_routes(tr,[tr(tr.N)])['models'][0]['required_routes']
        report['candidates'].append({'candidate':row['candidate'],'reproducible':True,'smoke':checks,
            'functional_smoke':functional,'tof_yaw_witness':path,'threshold_sensitivity':thresholds,
            'remove_all_close_rows_sensitivity':None if control else sensitivity_routes['models'][0]['required_routes'],
            'source_quality_review_required':quality['quality_review_required'],
            **admission, 'training_ready':False})
    root = Path(__file__).resolve().parents[1]
    dependencies = ['generator/validate_flywire_candidates.py', 'generator/prepare_flywire_candidates.py',
                    'generator/filtered_candidates.py', 'generator/role_reducer.py',
                    'generator/graph_reducer.py', 'generator/source_contract.py', 'generator/audit_routes.py',
                    'core/models.py', 'core/routing.py', 'core/contracts.py', 'simulation/policy.py',
                    'simulation/control_contract.py', 'simulation/policy_diagnostics.py', 'configs/flight_config.py']
    report['code_sha256']={name:file_sha256(root/name) for name in dependencies}
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--candidates',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    if args.report.exists(): raise FileExistsError('Use a new validation report path')
    report=validate(args.source,args.candidates)
    write_json(args.report,report)
    print(json.dumps({'report':str(args.report),'verified_candidates':len(report['candidates']),
                      'close_detection_pairs':len(report['quality']['close_detection_pairs']),
                      'training_ready':False},indent=2))


if __name__=='__main__': main()
