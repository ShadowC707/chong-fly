from copy import deepcopy
from types import SimpleNamespace

import pandas as pd
import pytest


def candidate_fixture(tmp_path, monkeypatch):
    from generator import published_annotations as annotations
    from generator.source_contract import file_sha256
    path = tmp_path/'annotations.tsv'
    pd.DataFrame({'root_id':['101','102','103','104'], 'cell_type':['']*4,
        'hemibrain_type':['LPLC2','LPLC2','DNp06','DNp06'],
        'side':['left','right','left','right'],
        'super_class':['visual_projection']*2+['descending']*2}).to_csv(path, sep='\t', index=False)
    monkeypatch.setattr(annotations, 'ANNOTATION_SHA256', file_sha256(path))
    registry = {'format_version':'flight-candidate-registry-v1',
        'materialization_version':783, 'annotation_sha256':file_sha256(path),
        'candidates':{name:{'annotation_status':'resolved',
            'selector':{'namespace':'hemibrain_type', 'value':name}} for name in ['LPLC2','DNp06']},
        'pathways':{'loom':{'pre':['LPLC2'],'post':['DNp06']}}}
    class Materialize:
        def __init__(self):
            # Reverse and unselected edges must not enter the directed projection.
            self.frame = pd.DataFrame({'id':range(10),
                'pre_pt_root_id':[101,101,102,102,102,102,102,102,103,999],
                'post_pt_root_id':[103,103,103,104,104,104,104,104,101,104]})
            self.calls = []

        def query_table(self, table, **kwargs):
            self.calls.append(kwargs)
            assert table == 'synapses_nt_v1'
            assert kwargs['materialization_version'] == 783
            frame = self.frame
            for column, allowed in kwargs['filter_in_dict'].items():
                frame = frame[frame[column].isin(allowed)]
            if kwargs.get('get_counts'):
                assert kwargs['limit'] == 1
                return pd.DataFrame({'count':[len(frame)]})
            return frame.copy()
    return path, registry, SimpleNamespace(materialize=Materialize())


def test_projection_is_directed_count_checked_and_thresholds_apply_to_pairs(tmp_path, monkeypatch):
    from generator.audit_candidate_pathway import audit_candidate_pathway
    path, registry, client = candidate_fixture(tmp_path, monkeypatch)
    original = deepcopy(registry)
    report = audit_candidate_pathway(client, path, registry, 'loom', max_rows=6)
    assert registry == original
    assert report['total_synapses'] == 8
    assert report['edge_pairs'] == 3
    assert report['thresholds']['1']['ipsilateral_synapses'] == 7
    assert report['thresholds']['1']['contralateral_synapses'] == 1
    assert report['thresholds']['5']['total_synapses'] == 5
    assert report['thresholds']['5']['connected_post_roots'] == ['104']
    assert report['thresholds']['10']['connected_post_roots'] == []
    assert report['thresholds']['1']['per_post_root']['103']['synapses'] == 3
    assert report['rc_mapping_validated'] is False
    assert report['source']['query_filters'] == {'pre_pt_root_id':['101','102'],
                                                'post_pt_root_id':['103','104']}
    assert all(call['limit'] <= 7 for call in client.materialize.calls)


def test_zero_connections_is_a_valid_negative_result_with_all_targets_reported(tmp_path, monkeypatch):
    from generator.audit_candidate_pathway import audit_candidate_pathway
    path, registry, client = candidate_fixture(tmp_path, monkeypatch)
    client.materialize.frame = client.materialize.frame.iloc[:0]
    report = audit_candidate_pathway(client, path, registry, 'loom')
    assert report['edges'] == [] and report['total_synapses'] == 0
    assert report['thresholds']['1']['per_post_root']['103']['synapses'] == 0
    assert len(client.materialize.calls) == 1


def test_same_root_rows_do_not_inflate_inter_neuron_connectivity(tmp_path, monkeypatch):
    from generator.audit_candidate_pathway import audit_candidate_pathway
    path, registry, client = candidate_fixture(tmp_path, monkeypatch)
    registry['pathways']['loom'] = {'pre':['LPLC2','DNp06'],'post':['LPLC2','DNp06']}
    client.materialize.frame = pd.concat([client.materialize.frame,
        pd.DataFrame({'id':range(20,70),'pre_pt_root_id':[101]*50,'post_pt_root_id':[101]*50})],ignore_index=True)
    report = audit_candidate_pathway(client,path,registry,'loom')
    assert report['total_synapses'] == 59
    assert report['same_root_connections'] == {'edge_pairs':1,'synapses':50}
    assert report['inter_neuron_connections'] == {'edge_pairs':4,'synapses':9}
    assert report['thresholds']['1']['total_synapses'] == 9
    assert report['thresholds']['20']['total_synapses'] == 0
    assert report['threshold_scope'] == 'inter_neuron_edges_only'


@pytest.mark.parametrize('fault', ['ambiguous','hash','version','missing_candidate','empty_pre',
                                 'duplicate_pre','missing_pathway','bad_budget','bool_budget'])
def test_invalid_registry_or_query_cannot_start_network_access(tmp_path, monkeypatch, fault):
    from generator.audit_candidate_pathway import audit_candidate_pathway
    path, registry, client = candidate_fixture(tmp_path, monkeypatch)
    budget = 50000
    if fault == 'ambiguous': registry['candidates']['DNp06']['annotation_status'] = 'ambiguous'
    elif fault == 'hash': registry['annotation_sha256'] = '0'*64
    elif fault == 'version': registry['materialization_version'] = 630
    elif fault == 'missing_candidate': del registry['candidates']['DNp06']
    elif fault == 'empty_pre': registry['pathways']['loom']['pre'] = []
    elif fault == 'duplicate_pre': registry['pathways']['loom']['pre'] *= 2
    elif fault == 'missing_pathway': registry['pathways'] = {}
    elif fault == 'bad_budget': budget = 0
    else: budget = True
    with pytest.raises(ValueError):
        audit_candidate_pathway(client, path, registry, 'loom', max_rows=budget)
    assert not client.materialize.calls


@pytest.mark.parametrize('fault', ['truncate','wrong_direction','duplicate_ids'])
def test_invalid_synapse_response_is_never_published_as_evidence(tmp_path, monkeypatch, fault):
    from generator.audit_candidate_pathway import audit_candidate_pathway
    path, registry, client = candidate_fixture(tmp_path, monkeypatch)
    original = client.materialize.query_table
    def query(table, **kwargs):
        frame = original(table, **kwargs)
        if not kwargs.get('get_counts'):
            if fault == 'truncate': frame = frame.iloc[:-1]
            elif fault == 'wrong_direction': frame.loc[frame.index[0], 'post_pt_root_id'] = 101
            else: frame.loc[frame.index[1], 'id'] = frame.id.iloc[0]
        return frame
    client.materialize.query_table = query
    with pytest.raises(ValueError):
        audit_candidate_pathway(client, path, registry, 'loom')


@pytest.mark.parametrize('failure', [False, True])
def test_cli_restores_request_handler_and_publishes_only_success(tmp_path, monkeypatch, capsys, failure):
    import json
    import sys
    import requests
    from generator.audit_candidate_pathway import main
    path, registry, client = candidate_fixture(tmp_path, monkeypatch)
    registry_path = tmp_path/'registry.json'
    registry_path.write_text(json.dumps(registry), encoding='utf-8')
    output = tmp_path/'report.json'
    original_request = requests.sessions.Session.request
    def factory(datastack, **kwargs):
        assert datastack == 'flywire_fafb_public' and kwargs['version'] == 783
        assert kwargs['write_server_cache'] is False
        if failure:
            raise RuntimeError('SECRET_ERROR_CANARY')
        return client
    monkeypatch.setitem(sys.modules, 'caveclient', SimpleNamespace(CAVEclient=factory))
    monkeypatch.setattr(sys, 'argv', ['audit', '--registry',str(registry_path),
        '--annotations',str(path),'--pathway','loom','--out',str(output)])
    assert main() == (2 if failure else 0)
    assert requests.sessions.Session.request is original_request
    assert output.exists() is not failure
    assert 'SECRET_ERROR_CANARY' not in capsys.readouterr().out
    if not failure:
        assert json.loads(output.read_text())['total_synapses'] == 8
        with pytest.raises(SystemExit):
            main()
