"""Real CAVEclient query encoding and Arrow decoding with an offline HTTP peer.

Requires optional acquisition dependencies. No credentials or live service are
used. This tests the library boundary; it does not qualify a remote dataset.
"""
import json
from types import SimpleNamespace

import pandas as pd
import pytest

pytest.importorskip('caveclient', reason='Install requirements-connectome.txt for client integration tests')
pa = pytest.importorskip('pyarrow')
from caveclient.materializationengine import MaterializationClient
from packaging.version import Version
from requests import Response

from generator.cave_source import fetch_cave_graph
from generator.source_contract import write_bundle, read_bundle


class ArrowPeer:
    def __init__(self, truncate=False):
        self.calls = []
        self.truncate = truncate
        self.frames = {
            'annotations': pd.DataFrame({'id': [1, 2],
                'pt_root_id': [720575940600000001, 720575940600000003],
                'cell_type': ['LC4', 'DNa01'], 'side': ['left', 'left'],
                'super_class': ['optic', 'descending']}),
            'synapses': pd.DataFrame({'id': [7, 8],
                'pre_pt_root_id': [720575940600000001]*2,
                'post_pt_root_id': [720575940600000003]*2}),
        }

    def post(self, url, *, data, params, **kwargs):
        body = json.loads(data)
        table = url.rsplit('/', 1)[-1]
        self.calls.append((url, body, params.copy()))
        frame = self.frames[table]
        for column, values in body['filter_in_dict'][table].items():
            frame = frame[frame[column].isin(values)]
        if params.get('count'):
            frame = pd.DataFrame({'count': [len(frame)]})
        elif self.truncate:
            frame = frame.iloc[:1]
        arrow = pa.Table.from_pandas(frame, preserve_index=False)
        stream = pa.BufferOutputStream()
        with pa.ipc.new_stream(stream, arrow.schema) as writer:
            writer.write_table(arrow)
        response = Response()
        response.status_code = 200
        response.url = url
        response.headers['Content-Type'] = 'data.arrow'
        response._content = stream.getvalue().to_pybytes()
        return response


class OfflineMaterialization(MaterializationClient):
    def __init__(self, peer):
        # Replace transport and remote metadata only. Keep real query_table,
        # query construction, response validation and Arrow deserialization.
        self._datastack_name = 'offline_fixture'
        self.session = peer
        self._api_version = 3
        self._endpoints = {'simple_query': 'https://offline.invalid/{datastack_name}/{version}/{table_name}'}
        self.desired_resolution = None

    @property
    def default_url_mapping(self): return {}

    @property
    def server_version(self): return Version('5.13.0')

    def get_table_metadata(self, **kwargs):
        assert kwargs['version'] == 123
        return {'reference_table': None}


def extract(peer):
    return fetch_cave_graph(SimpleNamespace(materialize=OfflineMaterialization(peer)),
        datastack='offline_fixture', version=123, annotation_table='annotations',
        synapse_table='synapses', target_classes=['LC4', 'DNa01'],
        excluded_classes={'mushroom_body'}, excluded_types=set())


def test_actual_client_encodes_pinned_queries_and_decodes_arrow_counts_and_large_ids(tmp_path):
    peer = ArrowPeer()
    nodes, edges, source = extract(peer)
    assert nodes.root_id.tolist() == ['720575940600000001', '720575940600000003']
    assert edges.to_dict('records') == [{'pre_id': '720575940600000001',
                                        'post_id': '720575940600000003', 'weight': 2}]
    assert all('/123/' in url for url, _, _ in peer.calls)
    assert sum(bool(params.get('count')) for _, _, params in peer.calls) == 2
    assert all(params['arrow_format'] and params['direct_sql_pandas'] for _, _, params in peer.calls)
    output = tmp_path/'bundle'
    write_bundle(nodes, edges, output, source, {})
    loaded_nodes, loaded_edges, manifest = read_bundle(output)
    pd.testing.assert_frame_equal(nodes, loaded_nodes)
    pd.testing.assert_frame_equal(edges, loaded_edges)
    assert manifest['source']['materialization_version'] == 123


def test_truncation_survives_actual_client_decoding_and_is_rejected():
    with pytest.raises(ValueError, match='count mismatch|truncated'):
        extract(ArrowPeer(truncate=True))
