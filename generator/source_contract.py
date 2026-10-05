"""Validated, immutable raw graph bundles with explicit acquisition provenance.

Hashes establish file integrity, not biological authenticity. CAVE provenance
records a retrieval from a named snapshot; motor mappings remain hypotheses.
"""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import numbers
import os
import re
from pathlib import Path
import tempfile
import time
import warnings

import numpy as np
import pandas as pd

SOURCE_VERSION = 'raw-source-v1'


def canonical_ids(values, name):
    result = []
    for value in values:
        # A float may already have lost bits before reaching this function.
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (str, numbers.Integral)):
            raise ValueError(f'{name}: root IDs must be exact integers or decimal strings, never floats')
        text = str(value)
        if not text.isascii() or not text.isdecimal() or not 0 < int(text) < 2**64:
            raise ValueError(f'{name}: invalid unsigned 64-bit root ID')
        result.append(str(int(text)))
    return result


def validate_graph(nodes, edges):
    if (not isinstance(nodes, pd.DataFrame) or not isinstance(edges, pd.DataFrame)
            or not {'root_id', 'cell_type', 'side'} <= set(nodes)
            or not {'pre_id', 'post_id', 'weight'} <= set(edges) or nodes.empty or edges.empty):
        raise ValueError('Expected nonempty node/edge tables with root IDs, annotations and synapse counts')
    nodes, edges = nodes.copy(), edges.copy()
    nodes['root_id'] = canonical_ids(nodes.root_id, 'nodes')
    for key in ('pre_id', 'post_id'):
        edges[key] = canonical_ids(edges[key], key)
    if nodes.root_id.duplicated().any():
        raise ValueError('Duplicate root IDs: resolve ambiguous annotations explicitly')
    for key in ('cell_type', 'side'):
        if any(not isinstance(value, str) or not value.strip() for value in nodes[key]):
            raise ValueError(f'Missing or empty node annotation: {key}')
    known = set(nodes.root_id)
    if not set(edges.pre_id) <= known or not set(edges.post_id) <= known:
        raise ValueError('Dangling edge endpoints: refusing to silently drop connections')
    counts = []
    for value in edges.weight:
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real)
                or not np.isfinite(value) or value <= 0 or int(value) != value or value > 2**53):
            raise ValueError('weight must contain positive exact integer synapse counts')
        counts.append(int(value))
    edges['weight'] = counts
    return nodes.reset_index(drop=True), edges.reset_index(drop=True)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def manifest_sha256(manifest):
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, allow_nan=False).encode()).hexdigest()


def validate_source(source):
    if not isinstance(source, dict) or source.get('source_kind') not in {'synthetic', 'cave'}:
        raise ValueError('Source must explicitly identify synthetic or cave acquisition')
    if source['source_kind'] == 'synthetic':
        if type(source.get('seed')) is not int:
            raise ValueError('Synthetic source requires an integer seed')
    else:
        if type(source.get('materialization_version')) is not int or source['materialization_version'] <= 0:
            raise ValueError('CAVE source requires a pinned materialization version')
        for key in ('datastack', 'synapse_table'):
            if not isinstance(source.get(key), str) or not source[key].strip():
                raise ValueError(f'CAVE source requires {key}')
        annotation = source.get('annotation_source')
        if annotation is None:
            if not isinstance(source.get('annotation_table'), str) or not source['annotation_table'].strip():
                raise ValueError('CAVE source requires annotation_table or a published annotation_source')
        elif (not isinstance(annotation, dict) or annotation.get('kind') != 'published_annotations'
              or annotation.get('materialization_version') != source['materialization_version']
              or source.get('annotation_table') is not None
              or not re.fullmatch('[0-9a-f]{64}', str(annotation.get('sha256', '')))
              or not re.fullmatch('[0-9a-f]{40}', str(annotation.get('commit', '')))
              or not isinstance(annotation.get('url'), str) or not annotation['url'].startswith('https://')):
            raise ValueError('Invalid or snapshot-mismatched published annotation source')
        if source.get('completeness') != 'query-count-checked':
            raise ValueError('CAVE source requires count-checked query completeness')


def write_bundle(nodes, edges, output, source, summary):
    """Publish a complete new directory only after both tables serialize successfully."""
    validate_source(source)
    nodes, edges = validate_graph(nodes, edges)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to replace existing graph directory: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.connectome-', dir=output.parent) as temporary:
        stage = Path(temporary)/'bundle'
        stage.mkdir()
        nodes.to_csv(stage/'raw_nodes.csv', index=False)
        edges.to_parquet(stage/'raw_edges.parquet', index=False)
        manifest = {
            'format_version': SOURCE_VERSION,
            'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'source': deepcopy(source),
            'weight_semantics': 'unsigned_synapse_count',
            'num_nodes': len(nodes), 'num_edge_rows': len(edges),
            'files': {name: file_sha256(stage/name) for name in ('raw_nodes.csv', 'raw_edges.parquet')},
        }
        report = {**summary, 'dataset': source['source_kind'], 'source': deepcopy(source),
                  'source_manifest_sha256': manifest_sha256(manifest)}
        (stage/'source_manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding='utf-8')
        (stage/'circuit_summary.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
        # rename, never replace: the existing input dataset is not an update target.
        for attempt in range(4):
            if output.exists():
                raise FileExistsError(f'Output appeared during extraction: {output}')
            try:
                os.rename(stage, output)
                break
            except OSError as exc:
                # Antivirus/indexer handles can briefly deny a Windows rename.
                # Keep publication atomic; permanent failures still propagate.
                if getattr(exc, 'winerror', None) not in (5, 32, 33) or attempt == 3:
                    raise
                time.sleep(.05 * 2**attempt)
    return manifest


def read_bundle(directory, *, allow_legacy=False):
    directory = Path(directory)
    path = directory/'source_manifest.json'
    manifest = None
    if path.exists():
        manifest = json.loads(path.read_text(encoding='utf-8'))
        if manifest.get('format_version') != SOURCE_VERSION or manifest.get('weight_semantics') != 'unsigned_synapse_count':
            raise ValueError('Incompatible raw source contract')
        validate_source(manifest.get('source'))
        for name in ('raw_nodes.csv', 'raw_edges.parquet'):
            if manifest.get('files', {}).get(name) != file_sha256(directory/name):
                raise ValueError(f'Source integrity mismatch: {name}')
    elif not allow_legacy:
        raise ValueError('Missing source manifest; explicit allow_legacy is required for historical data')
    else:
        warnings.warn('Legacy raw graph has unverified acquisition provenance', UserWarning, stacklevel=2)
    nodes = pd.read_csv(directory/'raw_nodes.csv', dtype={'root_id': str})
    edges = pd.read_parquet(directory/'raw_edges.parquet')
    nodes, edges = validate_graph(nodes, edges)
    if manifest and (manifest['num_nodes'] != len(nodes) or manifest['num_edge_rows'] != len(edges)):
        raise ValueError('Source manifest row counts do not match graph')
    nodes.attrs['source_manifest'] = deepcopy(manifest)
    return nodes, edges, manifest
