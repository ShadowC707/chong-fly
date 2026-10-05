import json
from types import SimpleNamespace

import pandas as pd
import pytest


def fixture_file(tmp_path, monkeypatch):
    from generator import published_annotations as module
    from generator.source_contract import file_sha256
    frame = pd.DataFrame({
        'root_id': ['720575940600000001', '720575940600000003', '720575940600000005'],
        'cell_type': ['LC4a', 'DNa01', ''], 'hemibrain_type': ['LC4', 'DNa01', 'HSN'],
        'side': ['left', 'right', 'left'], 'super_class': ['visual_projection', 'descending', 'visual_projection'],
        'top_nt': ['acetylcholine', 'acetylcholine', 'acetylcholine'], 'top_nt_conf': ['0.9']*3})
    path = tmp_path/'neurons.tsv'
    frame.to_csv(path, sep='\t', index=False)
    monkeypatch.setattr(module, 'ANNOTATION_SHA256', file_sha256(path))
    return module, path


def test_published_labels_match_exactly_preserving_both_names_and_large_ids(tmp_path, monkeypatch):
    module, path = fixture_file(tmp_path, monkeypatch)
    nodes, coverage = module.select_annotations(path, version=783, target_classes=['LC4','DNa01','HSN','VS10'])
    assert nodes.root_id.tolist() == ['720575940600000001','720575940600000003','720575940600000005']
    assert nodes.cell_type.tolist() == ['LC4','DNa01','HSN']
    assert nodes.source_cell_type.tolist() == ['LC4a','DNa01','']
    assert nodes.source_hemibrain_type.tolist() == ['LC4','DNa01','HSN']
    assert coverage['missing_types'] == ['VS10']
    assert coverage['matched_counts'] == {'DNa01':1,'HSN':1,'LC4':1,'VS10':0}


def test_conflicting_requested_aliases_are_not_silently_assigned(tmp_path, monkeypatch):
    module, path = fixture_file(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='ambiguous'):
        module.select_annotations(path, version=783, target_classes=['LC4','LC4a'])


def test_same_label_in_different_namespaces_requires_explicit_diagnostic_union(tmp_path, monkeypatch):
    module, path = fixture_file(tmp_path, monkeypatch)
    from generator.source_contract import file_sha256
    frame = pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False)
    frame.loc[2, 'hemibrain_type'] = 'DNa01'
    frame.to_csv(path, sep='\t', index=False)
    monkeypatch.setattr(module, 'ANNOTATION_SHA256', file_sha256(path))
    with pytest.raises(ValueError, match='namespace'):
        module.select_annotations(path, version=783, target_classes=['DNa01'])
    nodes, coverage = module.select_annotations(path, version=783, target_classes=['DNa01'],
                                               allow_namespace_union=True)
    assert len(nodes) == 2
    assert coverage['label_namespace_conflicts'] == ['DNa01']
    assert coverage['label_namespace_policy'] == 'explicit_diagnostic_union'


def test_snapshot_and_bytes_are_verified_before_selection(tmp_path, monkeypatch):
    module, path = fixture_file(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='783'):
        module.select_annotations(path, version=630, target_classes=['LC4'])
    path.write_text('corrupted', encoding='utf-8')
    with pytest.raises(ValueError, match='SHA'):
        module.select_annotations(path, version=783, target_classes=['LC4'])


@pytest.mark.parametrize('fault', ['missing_side','duplicate_root'])
def test_selection_rejects_missing_side_and_duplicate_roots(tmp_path, monkeypatch, fault):
    module, path = fixture_file(tmp_path, monkeypatch)
    from generator.source_contract import file_sha256
    frame = pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False)
    if fault == 'missing_side':
        frame.loc[0, 'side'] = 'na'
    else:
        frame.loc[1, 'root_id'] = frame.loc[0, 'root_id']
    frame.to_csv(path, sep='\t', index=False)
    monkeypatch.setattr(module, 'ANNOTATION_SHA256', file_sha256(path))
    with pytest.raises(ValueError):
        module.select_annotations(path, version=783, target_classes=['LC4','DNa01'])


def test_hybrid_source_retains_annotation_origin_and_counted_synapses(tmp_path, monkeypatch):
    module, path = fixture_file(tmp_path, monkeypatch)
    from generator.cave_source import fetch_cave_connectivity
    from generator.source_contract import validate_source
    nodes, coverage = module.select_annotations(path, version=783, target_classes=['LC4','DNa01'])
    class Materialize:
        def query_table(self, table, **kwargs):
            assert table == 'synapses_nt_v1' and kwargs['materialization_version'] == 783
            if kwargs.get('get_counts'):
                assert kwargs['limit'] == 1
                return pd.DataFrame({'count':[2]})
            return pd.DataFrame({'id':[1,2], 'pre_pt_root_id':[int(nodes.root_id.iloc[0])]*2,
                                'post_pt_root_id':[int(nodes.root_id.iloc[1])]*2})
    _, edges, source = fetch_cave_connectivity(SimpleNamespace(materialize=Materialize()), nodes,
        datastack='flywire_fafb_public', version=783, synapse_table='synapses_nt_v1',
        annotation_source=module.annotation_source(), selection=coverage)
    assert edges.weight.tolist() == [2]
    assert source['annotation_table'] is None
    assert source['annotation_source']['sha256'] == module.ANNOTATION_SHA256
    validate_source(source)
    source['annotation_source']['materialization_version'] = 630
    with pytest.raises(ValueError, match='annotation'):
        validate_source(source)


def test_explicit_namespace_avoids_alias_union_and_records_exact_selection(tmp_path, monkeypatch):
    module, path = fixture_file(tmp_path, monkeypatch)
    nodes, report = module.select_annotations(path, version=783, target_classes=['LC4'],
        selectors={'LC4': {'namespace':'cell_type', 'value':'LC4a',
                           'root_ids':['720575940600000001']}})
    assert nodes.root_id.tolist() == ['720575940600000001']
    assert nodes.cell_type.tolist() == ['LC4']
    assert report['resolved_selectors']['LC4']['namespace'] == 'cell_type'
    assert report['resolved_selectors']['LC4']['root_ids'] == nodes.root_id.tolist()
    assert report['label_namespace_policy'] == 'explicit_selectors_with_strict_fallback'


def test_explicit_namespace_resolves_collision_without_disabling_other_guards(tmp_path, monkeypatch):
    module, path = fixture_file(tmp_path, monkeypatch)
    from generator.source_contract import file_sha256
    frame = pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False)
    frame.loc[2, 'hemibrain_type'] = 'DNa01'
    frame.to_csv(path, sep='\t', index=False)
    monkeypatch.setattr(module, 'ANNOTATION_SHA256', file_sha256(path))
    nodes, report = module.select_annotations(path, version=783, target_classes=['DNa01'],
        selectors={'DNa01': {'namespace':'cell_type', 'value':'DNa01'}})
    assert nodes.root_id.tolist() == ['720575940600000003']
    assert report['label_namespace_conflicts'] == ['DNa01']
    with pytest.raises(ValueError, match='ambiguous'):
        module.select_annotations(path, version=783, target_classes=['LC4','LC4a'],
            selectors={'LC4': {'namespace':'hemibrain_type', 'value':'LC4'}})


@pytest.mark.parametrize('selector', [
    {'namespace':'typo','value':'LC4'},
    {'namespace':'hemibrain_type','value':'LC4','root_ids':['720575940600000003']},
    {'namespace':'hemibrain_type','value':'LC4','root_ids':[720575940600000001.0]},
    {'namespace':'hemibrain_type','value':'LC4','root_ids':[]},
    {'namespace':'hemibrain_type','value':'LC4','root_ids':['720575940600000001']*2},
    {'namespace':'hemibrain_type','value':'missing'},
    {'namespace':'hemibrain_type','value':'LC4','typo':True},
])
def test_explicit_selectors_fail_closed_on_invalid_or_changed_identity(tmp_path, monkeypatch, selector):
    module, path = fixture_file(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        module.select_annotations(path, version=783, target_classes=['LC4'], selectors={'LC4':selector})


def test_unused_selector_and_diagnostic_union_cannot_hide_configuration_error(tmp_path, monkeypatch):
    module, path = fixture_file(tmp_path, monkeypatch)
    selectors = {'DNa01': {'namespace':'cell_type','value':'DNa01'}}
    with pytest.raises(ValueError):
        module.select_annotations(path, version=783, target_classes=['LC4'], selectors=selectors)
    with pytest.raises(ValueError):
        module.select_annotations(path, version=783, target_classes=['DNa01'],
                                  selectors=selectors, allow_namespace_union=True)


def root_selector():
    return {'namespace':'root_id', 'root_ids':['720575940600000001'],
            'expected_annotations':{'720575940600000001':{
                'cell_type':'LC4a', 'hemibrain_type':'LC4', 'side':'left'}}}


def test_author_root_selection_preserves_paper_name_and_checks_snapshot_identity(tmp_path, monkeypatch):
    module, path = fixture_file(tmp_path, monkeypatch)
    nodes, report = module.select_annotations(path, version=783, target_classes=['paper_name'],
                                              selectors={'paper_name':root_selector()})
    assert nodes.cell_type.tolist() == ['paper_name']
    assert nodes.source_cell_type.tolist() == ['LC4a']
    assert report['resolved_selectors']['paper_name']['namespace'] == 'root_id'
    assert report['resolved_selectors']['paper_name']['expected_annotations'] == root_selector()['expected_annotations']


@pytest.mark.parametrize('fault', ['missing_root','duplicate','float','missing_lock','incomplete_lock',
                                 'wrong_cell','wrong_side','extra_root','unknown_field'])
def test_author_root_selection_rejects_unverified_or_changed_annotations(tmp_path, monkeypatch, fault):
    module, path = fixture_file(tmp_path, monkeypatch)
    spec = root_selector()
    root = spec['root_ids'][0]
    if fault == 'missing_root': spec['root_ids'] = ['999']
    elif fault == 'duplicate': spec['root_ids'] *= 2
    elif fault == 'float': spec['root_ids'] = [float(root)]
    elif fault == 'missing_lock': del spec['expected_annotations']
    elif fault == 'incomplete_lock': del spec['expected_annotations'][root]['side']
    elif fault == 'wrong_cell': spec['expected_annotations'][root]['cell_type'] = 'wrong'
    elif fault == 'wrong_side': spec['expected_annotations'][root]['side'] = 'right'
    elif fault == 'extra_root': spec['expected_annotations']['999'] = dict(spec['expected_annotations'][root])
    else: spec['value'] = 'unexpected'
    with pytest.raises(ValueError):
        module.select_annotations(path, version=783, target_classes=['paper_name'], selectors={'paper_name':spec})


def test_author_roots_do_not_duplicate_one_neuron_under_two_names(tmp_path, monkeypatch):
    module, path = fixture_file(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='ambiguous'):
        module.select_annotations(path, version=783, target_classes=['paper_name','LC4'],
                                  selectors={'paper_name':root_selector()})
