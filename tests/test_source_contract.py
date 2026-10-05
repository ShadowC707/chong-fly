from types import SimpleNamespace
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest


def tables():
    return (pd.DataFrame({'root_id': ['720575940600000001', '720575940600000003'],
                          'cell_type': ['LC4', 'DNa01'], 'side': ['left', 'left']}),
            pd.DataFrame({'pre_id': ['720575940600000001'],
                          'post_id': ['720575940600000003'], 'weight': [7]}))


@pytest.mark.parametrize('fault', ['float_id', 'duplicate_id', 'dangling', 'zero_weight', 'fractional_weight', 'missing_side'])
def test_source_rejects_ambiguous_ids_and_incomplete_graphs(fault):
    from generator.source_contract import validate_graph
    nodes, edges = tables()
    if fault == 'float_id': nodes['root_id'] = nodes.root_id.astype(float)
    if fault == 'duplicate_id': nodes.loc[1, 'root_id'] = nodes.loc[0, 'root_id']
    if fault == 'dangling': edges.loc[0, 'post_id'] = '42'
    if fault == 'zero_weight': edges.loc[0, 'weight'] = 0
    if fault == 'fractional_weight': edges['weight'] = [1.5]
    if fault == 'missing_side': nodes.loc[0, 'side'] = None
    with pytest.raises(ValueError): validate_graph(nodes, edges)


def test_large_ids_are_preserved_exactly_and_input_is_not_mutated():
    from generator.source_contract import validate_graph
    nodes, edges = tables()
    original = nodes.copy(deep=True)
    checked, _ = validate_graph(nodes, edges)
    assert checked.root_id.tolist() == ['720575940600000001', '720575940600000003']
    pd.testing.assert_frame_equal(nodes, original)


class FakeMaterialize:
    def __init__(self, *, truncate=False, failure=False):
        self.calls = []
        self.truncate, self.failure = truncate, failure
        self.frames = {
            'annotations': pd.DataFrame({'id': [1, 2, 3], 'pt_root_id': [101, 102, 103],
                'cell_type': ['LC4', 'DNa01', 'Mi1'], 'side': ['left']*3,
                'super_class': ['optic', 'descending', 'mushroom_body']}),
            'synapses': pd.DataFrame({'id': [7, 8, 9], 'pre_pt_root_id': [101, 101, 102],
                                      'post_pt_root_id': [102, 102, 101]}),
        }

    def query_table(self, table, **kwargs):
        self.calls.append((table, kwargs))
        if self.failure: raise RuntimeError('network failure')
        frame = self.frames[table]
        for column, values in kwargs.get('filter_in_dict', {}).items():
            frame = frame[frame[column].isin(values)]
        if kwargs.get('get_counts'):
            return pd.DataFrame({'count': [len(frame)]})
        if self.truncate: frame = frame.iloc[:1]
        return frame.copy()


def cave_result(materialize, **kwargs):
    from generator.cave_source import fetch_cave_graph
    return fetch_cave_graph(SimpleNamespace(materialize=materialize),
        datastack='test_stack', version=123, annotation_table='annotations', synapse_table='synapses',
        target_classes=['LC4', 'DNa01', 'Mi1'], excluded_classes={'mushroom_body'},
        excluded_types=set(), max_rows=2, **kwargs)


def test_cave_pins_every_query_splits_by_filter_and_applies_exclusions():
    materialize = FakeMaterialize()
    nodes, edges, source = cave_result(materialize)
    assert set(nodes.root_id) == {'101', '102'}
    assert sorted(edges.weight.tolist()) == [1, 2]
    assert source['materialization_version'] == 123
    assert source['source_kind'] == 'cave'
    assert source['selection']['strategy'] == 'induced_cell_type_whitelist'
    assert all(call[1]['materialization_version'] == 123 for call in materialize.calls)
    assert all(call[1].get('limit', 0) <= 3 for call in materialize.calls)


def test_truncated_cave_response_fails_instead_of_publishing_a_partial_graph():
    from generator.cave_source import fetch_cave_graph
    client = SimpleNamespace(materialize=FakeMaterialize(truncate=True))
    with pytest.raises(ValueError, match='count|truncat'):
        fetch_cave_graph(client, datastack='test', version=123, annotation_table='annotations',
            synapse_table='synapses', target_classes=['LC4', 'DNa01'], excluded_classes=set(),
            excluded_types=set(), max_rows=10)


def test_single_filter_over_budget_fails_without_unbounded_download():
    materialize = FakeMaterialize()
    materialize.frames['annotations'].loc[:, 'cell_type'] = 'LC4'
    with pytest.raises(ValueError, match='budget'):
        cave_result(materialize)
    assert all(kwargs.get('get_counts') for _, kwargs in materialize.calls)


def test_count_probe_is_bounded_when_remote_ignores_count_flag():
    from generator.cave_source import _fetch_complete
    class IgnoresCount:
        def query_table(self, table, **kwargs):
            assert kwargs['get_counts'] is True
            assert kwargs['limit'] == 1
            return pd.DataFrame({'id': [7]})
    with pytest.raises(ValueError, match='count'):
        _fetch_complete(IgnoresCount(), 'annotations', 783, {'cell_type': ['DNa01']}, 'cell_type', 100)


def test_count_frame_must_have_count_column_not_an_arbitrary_single_integer():
    from generator.cave_source import _count
    with pytest.raises(ValueError, match='count'):
        _count(pd.DataFrame({'id': [7]}))


def test_query_order_does_not_change_neuron_indexing_or_edges():
    first, second = FakeMaterialize(), FakeMaterialize()
    second.frames = {key: frame.iloc[::-1].copy() for key, frame in second.frames.items()}
    nodes_a, edges_a, _ = cave_result(first)
    nodes_b, edges_b, _ = cave_result(second)
    pd.testing.assert_frame_equal(nodes_a, nodes_b)
    pd.testing.assert_frame_equal(edges_a, edges_b)


def test_duplicate_synapses_are_rejected_before_count_aggregation():
    materialize = FakeMaterialize()
    materialize.frames['synapses'].loc[1, 'id'] = 7
    with pytest.raises(ValueError, match='unique|Duplicate'):
        cave_result(materialize)


def test_cave_error_never_runs_synthetic_generator_or_touches_existing_output(monkeypatch, tmp_path):
    from generator import circuit_extractor as extractor
    target = tmp_path/'existing'
    target.mkdir()
    marker = target/'circuit_summary.json'
    marker.write_text('original')
    monkeypatch.setattr(extractor, '_build_synthetic_nodes', lambda *a: pytest.fail('Synthetic fallback'))
    monkeypatch.setattr(extractor, '_CAVE_AVAILABLE', True)
    monkeypatch.setattr(extractor, 'CAVEclient', lambda *a, **k:
                        SimpleNamespace(materialize=FakeMaterialize(failure=True)), raising=False)
    with pytest.raises((ValueError, FileExistsError, RuntimeError)):
        extractor.main(source='cave', out_dir=target, version=123,
                       annotation_table='annotations', synapse_table='synapses')
    assert marker.read_text() == 'original'
    # Repeat on a new path so a refused overwrite cannot mask the fallback bug.
    with pytest.raises(RuntimeError, match='network failure'):
        extractor.main(source='cave', out_dir=tmp_path/'new', version=123,
                       annotation_table='annotations', synapse_table='synapses')
    assert not (tmp_path/'new').exists()


def test_extractor_requires_explicit_source(monkeypatch, tmp_path):
    from generator import circuit_extractor as extractor
    monkeypatch.setattr(extractor, '_project_root', lambda: str(tmp_path))
    cfg = extractor.load_cell_mapping(str(Path(__file__).resolve().parents[1]/'configs/cell_mapping.json'))
    monkeypatch.setattr(extractor, 'load_cell_mapping', lambda *args: cfg)
    with pytest.raises(ValueError, match='source'):
        extractor.main()


def test_missing_cave_version_fails_before_network_access(monkeypatch, tmp_path):
    from generator import circuit_extractor as extractor
    monkeypatch.setattr(extractor, '_CAVE_AVAILABLE', True)
    monkeypatch.setattr(extractor, 'CAVEclient', lambda *a, **k: pytest.fail('Unexpected network'), raising=False)
    with pytest.raises(ValueError, match='version'):
        extractor.main(source='cave', out_dir=tmp_path/'new',
                       annotation_table='annotations', synapse_table='synapses')


def test_synthetic_generator_is_seeded_without_changing_global_random_state():
    from generator.circuit_extractor import _build_synthetic_nodes, _build_synthetic_edges, load_cell_mapping
    cfg = load_cell_mapping()
    nodes = _build_synthetic_nodes(cfg, False)
    # Keep only interface neurons so this test remains small.
    nodes = nodes[nodes['column'] == 0].copy()
    np.random.seed(93)
    expected = np.random.random(3)
    np.random.seed(93)
    first = _build_synthetic_edges(nodes, cfg, False, seed=12)
    actual = np.random.random(3)
    second = _build_synthetic_edges(nodes, cfg, False, seed=12)
    np.testing.assert_array_equal(actual, expected)
    pd.testing.assert_frame_equal(first, second)


@pytest.fixture
def csv_parquet_stub(monkeypatch):
    """Exercise publication failures and hashes without requiring a parquet engine."""
    monkeypatch.setattr(pd.DataFrame, 'to_parquet', lambda frame, path, **kw: frame.to_csv(path, index=False))
    monkeypatch.setattr(pd, 'read_parquet', lambda path: pd.read_csv(path, dtype={'pre_id': str, 'post_id': str}))


def test_complete_bundle_roundtrip_hashes_and_large_ids(tmp_path, csv_parquet_stub):
    from generator.source_contract import write_bundle, read_bundle
    nodes, edges = tables()
    manifest = write_bundle(nodes, edges, tmp_path/'bundle', {'source_kind': 'synthetic', 'seed': 42}, {})
    loaded, connections, restored = read_bundle(tmp_path/'bundle')
    assert restored == manifest
    assert loaded.root_id.tolist() == nodes.root_id.tolist()
    assert connections.pre_id.tolist() == edges.pre_id.tolist()
    report = json.loads((tmp_path/'bundle/circuit_summary.json').read_text())
    assert report['dataset'] == 'synthetic'


def test_parquet_failure_never_publishes_half_of_a_bundle(monkeypatch, tmp_path):
    from generator.source_contract import write_bundle
    def broken(*args, **kwargs): raise RuntimeError('serialization failed')
    monkeypatch.setattr(pd.DataFrame, 'to_parquet', broken)
    with pytest.raises(RuntimeError, match='serialization'):
        write_bundle(*tables(), tmp_path/'bundle', {'source_kind': 'synthetic', 'seed': 42}, {})
    assert not (tmp_path/'bundle').exists()
    assert list(tmp_path.iterdir()) == []


def test_bundle_refuses_to_overwrite_existing_data(tmp_path, csv_parquet_stub):
    from generator.source_contract import write_bundle, file_sha256
    output = tmp_path/'bundle'
    write_bundle(*tables(), output, {'source_kind': 'synthetic', 'seed': 42}, {})
    before = {p.name: file_sha256(p) for p in output.iterdir()}
    with pytest.raises(FileExistsError):
        write_bundle(*tables(), output, {'source_kind': 'synthetic', 'seed': 99}, {})
    assert before == {p.name: file_sha256(p) for p in output.iterdir()}


def test_tampered_source_rejected_even_with_allow_legacy(tmp_path, csv_parquet_stub):
    from generator.source_contract import write_bundle, read_bundle
    output = tmp_path/'bundle'
    write_bundle(*tables(), output, {'source_kind': 'synthetic', 'seed': 42}, {})
    with (output/'raw_nodes.csv').open('a') as stream: stream.write('\n')
    with pytest.raises(ValueError, match='integrity'):
        read_bundle(output, allow_legacy=True)


def test_loader_requires_explicit_legacy_optin_and_never_drops_bad_edges(tmp_path, csv_parquet_stub):
    from generator.graph_reducer import load_graph
    nodes, edges = tables()
    nodes.to_csv(tmp_path/'raw_nodes.csv', index=False)
    edges.to_parquet(tmp_path/'raw_edges.parquet')
    with pytest.raises(ValueError, match='manifest'):
        load_graph(str(tmp_path))
    with pytest.warns(UserWarning, match='Legacy'):
        loaded, _, pre, post, weights = load_graph(str(tmp_path), allow_legacy=True)
    assert loaded.root_id.tolist() == nodes.root_id.tolist()
    assert pre.tolist() == [0] and post.tolist() == [1] and weights.tolist() == [7]
    edges['post_id'] = '888'
    edges.to_parquet(tmp_path/'raw_edges.parquet')
    with pytest.warns(UserWarning, match='Legacy'), pytest.raises(ValueError, match='Dangling'):
        load_graph(str(tmp_path), allow_legacy=True)


def test_reducer_inherits_acquisition_and_refuses_a_source_kind_override(tmp_path, csv_parquet_stub):
    from generator.source_contract import write_bundle
    from generator.graph_reducer import load_graph
    from generator.role_reducer import RolePreservingReducer, resolve_source_kind
    from tests.test_role_reduction import graph
    args = graph()
    nodes = args['nodes_df'].drop(columns=['idx'])
    edges = pd.DataFrame({'pre_id': nodes.root_id.iloc[args['pre_idx']].to_numpy(),
                          'post_id': nodes.root_id.iloc[args['post_idx']].to_numpy(),
                          'weight': abs(args['signed_weights']).astype(int)})
    write_bundle(nodes, edges, tmp_path/'bundle', {'source_kind': 'synthetic', 'seed': 42}, {})
    nodes, edges, pre, post, weights = load_graph(str(tmp_path/'bundle'),
        polarity_map={name: 1 for name in nodes.cell_type.unique()})
    reducer = RolePreservingReducer(nodes, edges, args['cell_mapping_cfg'], pre, post, weights)
    model = reducer(reducer.minimum_k)
    assert model.provenance['source_kind'] == 'synthetic'
    assert model.provenance['acquisition']['seed'] == 42
    assert len(model.provenance['source_manifest_sha256']) == 64
    assert resolve_source_kind(model.provenance, None) == 'synthetic'
    with pytest.raises(ValueError, match='source'):
        resolve_source_kind(model.provenance, 'cave')
    with pytest.raises(ValueError, match='source'):
        resolve_source_kind({'source_kind': 'unverified'}, 'cave')


def test_unknown_cell_type_cannot_silently_become_excitatory(tmp_path, csv_parquet_stub):
    from generator.source_contract import write_bundle
    from generator.graph_reducer import load_graph
    nodes, edges = tables()
    nodes.loc[0, 'cell_type'] = 'unmapped_intermediate'
    write_bundle(nodes, edges, tmp_path/'bundle', {'source_kind': 'synthetic', 'seed': 42}, {})
    with pytest.raises(ValueError, match='polarity.*unmapped_intermediate'):
        load_graph(str(tmp_path/'bundle'))
    loaded, _, _, _, weights = load_graph(str(tmp_path/'bundle'),
        polarity_map={'unmapped_intermediate': -1, 'DNa01': 1})
    assert weights.tolist() == [-7]
    assert loaded.attrs['polarity_contract']['method'] == 'explicit_cell_type_map'
    assert loaded.attrs['polarity_contract']['mapping']['unmapped_intermediate'] == -1


@pytest.mark.parametrize('value', [0, float('nan'), 2, True])
def test_invalid_polarity_override_is_rejected(tmp_path, csv_parquet_stub, value):
    from generator.source_contract import write_bundle
    from generator.graph_reducer import load_graph
    write_bundle(*tables(), tmp_path/'bundle', {'source_kind': 'synthetic', 'seed': 42}, {})
    with pytest.raises(ValueError, match='polarity'):
        load_graph(str(tmp_path/'bundle'), polarity_map={'LC4': value, 'DNa01': 1})


def test_publish_retries_transient_windows_rename_lock_without_copying(monkeypatch, tmp_path, csv_parquet_stub):
    from generator import source_contract
    rename = source_contract.os.rename
    attempts = []
    def locked_once(src, dst):
        attempts.append(1)
        if len(attempts) == 1:
            error = PermissionError('temporary Windows sharing lock')
            error.winerror = 5
            raise error
        return rename(src, dst)
    monkeypatch.setattr(source_contract.os, 'rename', locked_once)
    source_contract.write_bundle(*tables(), tmp_path/'bundle', {'source_kind': 'synthetic', 'seed': 42}, {})
    assert len(attempts) == 2
    assert (tmp_path/'bundle/source_manifest.json').is_file()
