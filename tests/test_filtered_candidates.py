"""Tiny synthetic fixtures for the measured-data preparation pipeline."""
import numpy as np
import pandas as pd
import pytest

from generator.filtered_candidates import fetch_view, clean_synapses, assign_polarities, select_bridges, compare_reachability
from generator.prepare_flywire_candidates import publication, read_source, validate_config
from generator.validate_flywire_candidates import verify_files


class View:
    def __init__(self, frame, fault=None):
        self.frame, self.fault, self.calls = frame, fault, []

    def query_view(self, name, **kw):
        self.calls.append(kw)
        frame = self.frame
        for key, values in kw['filter_in_dict'].items():
            frame = frame[frame[key].isin(values)]
        if kw.get('get_counts'):
            return pd.DataFrame({'count': [len(frame)]})
        return frame.iloc[:-1] if self.fault == 'truncate' else frame.copy()


def synapses():
    return pd.DataFrame({'id': [1, 2, 3], 'pre_pt_root_id': [11, 11, 12],
                         'post_pt_root_id': [12, 11, 13], 'cleft_score': [51., 60., 80.]})


def test_view_count_split_is_pinned_and_exact():
    api = View(synapses())
    result, log = fetch_view(api, 'valid_synapses_nt_np_v6', {'pre_pt_root_id': [11, 12]}, max_rows=2)
    assert len(result) == 3 and len(log) == 3
    assert all(c['materialization_version'] == 783 for c in api.calls)
    assert set(result.pre_pt_root_id) == {'11', '12'}


def test_truncation_duplicate_ids_and_float_roots_fail():
    with pytest.raises(ValueError, match='count'):
        fetch_view(View(synapses(), 'truncate'), 'valid_synapses_nt_np_v6', {'pre_pt_root_id': [11, 12]})
    for column in ['id', 'pre_pt_root_id']:
        frame = synapses()
        if column == 'id':
            frame.loc[1, column] = 1
        else:
            frame[column] = frame[column].astype(float)
        with pytest.raises(ValueError):
            fetch_view(View(frame), 'valid_synapses_nt_np_v6', {'pre_pt_root_id': [11, 12]})


def test_clean_is_separate_and_threshold_after_aggregation():
    frame = synapses()
    before = frame.copy(deep=True)
    clean, edges, audit = clean_synapses(frame, pair_min=1)
    pd.testing.assert_frame_equal(frame, before)
    assert len(clean) == 2 and audit['same_root_rows'] == 1
    assert edges.weight.sum() == 2
    assert clean_synapses(frame, pair_min=2)[1].empty


@pytest.mark.parametrize('score', [50, float('nan'), float('inf')])
def test_official_filtered_view_contract_is_checked(score):
    frame = synapses()
    frame.loc[0, 'cleft_score'] = score
    with pytest.raises(ValueError, match='cleft'):
        clean_synapses(frame, pair_min=1)


def test_nt_signs_are_explicit_assumptions_unknowns_fail():
    nodes = pd.DataFrame({'root_id': ['11','12','13'], 'top_nt': ['acetylcholine','gaba','glutamate'],
                          'top_nt_conf': ['0.9','0.8','0.7'], 'known_nt': ['','','']})
    signs, records = assign_polarities(nodes)
    assert signs.tolist() == [1, -1, -1]
    assert all(not r['synaptic_sign_verified'] for r in records)
    nodes.loc[2, 'top_nt'] = 'unknown'
    with pytest.raises(ValueError, match='polarity'):
        assign_polarities(nodes)


def test_bridges_require_both_directed_legs_and_deterministic_budget():
    outgoing = pd.DataFrame({'pre_pt_root_id': ['11','11','11'], 'post_pt_root_id': ['21','22','23'], 'n_syn': [8,10,20]})
    incoming = pd.DataFrame({'pre_pt_root_id': ['21','22','99'], 'post_pt_root_id': ['31']*3, 'n_syn': [9,10,50]})
    result = select_bridges(outgoing, incoming, {'21','22','23','99'}, {'11','31'}, pair_min=5, budget=1)
    assert [r['root_id'] for r in result] == ['22']


def test_reduction_cannot_invent_route_by_merging_disconnected_intermediates():
    # A->x and y->B: merging x/y invents a path A->B.
    full = np.zeros((4,4)); full[0,1] = full[2,3] = 1
    reduced = np.zeros((3,3)); reduced[0,1] = reduced[1,2] = 1
    report = compare_reachability(full, reduced, np.array([0,1,1,2]), [0], [3])
    assert report['introduced_pairs'] == [[0,3]]
    assert not report['passed']


def test_failed_publication_does_not_leave_ready_directory(tmp_path):
    output = tmp_path/'result'
    with pytest.raises(RuntimeError):
        with publication(output) as stage:
            (stage/'partial').write_text('partial')
            raise RuntimeError('simulated parquet/network failure')
    assert not output.exists()
    with publication(output) as stage:
        (stage/'complete').write_text('complete')
    with pytest.raises(FileExistsError):
        with publication(output): pass
    assert (output/'complete').read_text() == 'complete'


@pytest.mark.parametrize('field,value', [('materialization_version',784),('pair_min',0),('bridge_budget',-1),
                                        ('synapse_view','synapses_nt_v1'),('polarity_policy',{'unknown':1}),
                                        ('sensor_contract',{'dimension':82})])
def test_config_cannot_mislabel_ignored_quality_or_sensor_settings(field,value):
    import json
    from pathlib import Path
    config = json.loads(Path('configs/filtered_candidates_783_v1.json').read_text())
    config[field] = value
    with pytest.raises(ValueError): validate_config(config)


def test_filtered_source_loader_rejects_missing_evidence_even_if_complete(tmp_path):
    import json
    (tmp_path/'manifest.json').write_text(json.dumps({'format_version':'filtered783-source-v1',
        'status':'complete','files':{}}))
    with pytest.raises(ValueError): read_source(tmp_path)


def test_candidate_hash_check_fails_before_loading_modified_weights(tmp_path):
    from generator.source_contract import file_sha256
    artifact = tmp_path/'w.npz'; artifact.write_bytes(b'original')
    files = {'w.npz':file_sha256(artifact)}
    verify_files(tmp_path,files)
    artifact.write_bytes(b'changed')
    with pytest.raises(ValueError,match='SHA256'): verify_files(tmp_path,files)
    with pytest.raises(ValueError): verify_files(tmp_path,{})


def test_required_route_loss_is_reported():
    full=np.array([[0,1,0],[0,0,1],[0,0,0]])
    reduced=full.copy(); reduced[1,2]=0
    audit=compare_reachability(full,reduced,np.arange(3),[0],[2])
    assert audit['lost_pairs']==[[0,2]] and not audit['passed']


def test_unique_rows_from_same_pair_remain_distinct_contacts():
    frame=synapses(); frame.loc[1,'post_pt_root_id']=12
    clean,edges,_=clean_synapses(frame,pair_min=2)
    assert len(clean)==3 and edges.weight.tolist()==[2]


def test_single_root_over_budget_cannot_be_published_as_complete():
    with pytest.raises(ValueError,match='budget'):
        fetch_view(View(synapses()),'valid_synapses_nt_np_v6',{'pre_pt_root_id':[11]},max_rows=1)


def test_count_schema_cannot_be_a_random_id():
    class Broken(View):
        def query_view(self,name,**kwargs):
            if kwargs.get('get_counts'): return pd.DataFrame({'id':[3]})
            return super().query_view(name,**kwargs)
    with pytest.raises(ValueError,match='count'):
        fetch_view(Broken(synapses()),'valid_synapses_nt_np_v6',{'pre_pt_root_id':[11,12]})


def test_empty_view_has_explicit_zero_count():
    result,log=fetch_view(View(synapses()),'valid_synapses_nt_np_v6',{'pre_pt_root_id':[99]})
    assert result.empty and log[0]['returned_rows']==log[0]['expected_rows']==0


def test_bridge_unknown_or_unannotated_neuron_cannot_enter_by_ranking():
    outgoing=pd.DataFrame({'pre_pt_root_id':['11','11'],'post_pt_root_id':['21','22'],'n_syn':[100,10]})
    incoming=pd.DataFrame({'pre_pt_root_id':['21','22'],'post_pt_root_id':['31','31'],'n_syn':[100,10]})
    assert select_bridges(outgoing,incoming,{'22'},{'11','31'},pair_min=5,budget=2)[0]['root_id']=='22'


def test_build_real_pipeline_with_tiny_explicitly_mocked_acquisition(tmp_path,monkeypatch):
    import json
    from pathlib import Path
    import generator.prepare_flywire_candidates as prep
    # These are test identities, NOT measured FlyWire data. Only read_source is mocked.
    labels=['HSN','LPLC2','DNp06','DNp15','DNae014','DNb01','VES041']
    nodes=pd.DataFrame({'root_id':[str(i+11) for i in range(7)],'cell_type':labels,
        'side':['left']*7,'top_nt':['acetylcholine']*7,'top_nt_conf':['0.8']*7,'known_nt':['']*7})
    edges=pd.DataFrame({'pre_id':['11','12','12','17'],'post_id':['14','13','15','16'],'weight':[5]*4})
    cfg=json.loads(Path('configs/filtered_candidates_783_v1.json').read_text())
    manifest={'seed_root_ids':nodes.root_id.tolist(),'config_sha256':'mock_test_only','quality':{}}
    source=tmp_path/'source'; source.mkdir(); (source/'manifest.json').write_text('{}')
    monkeypatch.setattr(prep,'read_source',lambda _: (nodes,edges,manifest,cfg))
    from generator import validate_flywire_candidates as validator
    monkeypatch.setattr(validator,'quality_audit',lambda _: {'quality_review_required':False,'close_detection_pairs':[]})
    output=tmp_path/'models'
    with pytest.warns(UserWarning,match='Motor mapping'):
        result=prep.build(source,output)
    assert len(result['candidates'])==3  # no redundant min-k duplicate when minimum == N
    registry=json.loads((output/'registry.json').read_text())
    assert not registry['training_ready']
    assert all(row['smoke']['finite'] for row in registry['candidates'])
    verify_files(output,registry['files'])


@pytest.mark.parametrize('winerror', [5, 32, 33])
def test_publication_retries_transient_windows_lock(tmp_path, monkeypatch, winerror):
    import generator.source_contract as contract
    original = contract.os.rename
    calls = []
    def rename(source, destination):
        calls.append(1)
        if len(calls) == 1:
            error = PermissionError('simulated Windows file lock')
            error.winerror = winerror
            raise error
        return original(source, destination)
    monkeypatch.setattr(contract.os, 'rename', rename)
    monkeypatch.setattr(contract.time, 'sleep', lambda _: None)
    output = tmp_path / 'published'
    with publication(output) as stage:
        (stage / 'data').write_text('complete')
    assert (output / 'data').read_text() == 'complete'
    assert len(calls) == 2


def test_publication_permanent_lock_is_bounded_and_leaves_no_ready_output(tmp_path, monkeypatch):
    import generator.source_contract as contract
    calls = []
    def rename(*args):
        calls.append(1)
        error = PermissionError('persistent lock')
        error.winerror = 5
        raise error
    monkeypatch.setattr(contract.os, 'rename', rename)
    monkeypatch.setattr(contract.time, 'sleep', lambda _: None)
    output = tmp_path / 'published'
    with pytest.raises(PermissionError):
        with publication(output) as stage:
            (stage / 'data').write_text('complete')
    assert len(calls) == 4
    assert not output.exists()


@pytest.mark.parametrize('field', ['structural_admission', 'interface_pair_audit', 'rejection_reasons', 'edges'])
def test_validator_recomputes_admission_instead_of_trusting_registry(field):
    from generator.validate_flywire_candidates import check_structural_admission
    from generator.role_reducer import RolePreservingReducer
    from generator.audit_routes import audit_routes
    from tests.test_role_reduction import graph
    reducer = RolePreservingReducer(**graph())
    model = reducer(reducer.N)
    sources = sorted({i for values in reducer._sensor_node_sets.values() for i in values})
    targets = sorted({i for values in reducer._motor_node_sets.values() for i in values})
    audit = compare_reachability(model.W, model.W, model.cluster_map, sources, targets)
    row = {'family':'bridge', 'k':model.k, 'source_nodes':reducer.N,
           'minimum_k':reducer.minimum_k, 'edges':model.W.nnz,
           'source_edges':len(reducer.signed_weights),
           'route_audit':audit_routes(reducer, [model]), 'interface_pair_audit':audit,
           'structural_admission':'passed', 'rejection_reasons':[]}
    check_structural_admission(row, reducer, model)
    row[field] = {'structural_admission':'rejected', 'interface_pair_audit':{},
                  'rejection_reasons':['fabricated'], 'edges':0}[field]
    with pytest.raises(ValueError, match='admission|audit|structure'):
        check_structural_admission(row, reducer, model)
