"""Pinned, count-checked CAVE extraction; no synthetic fallback or offset paging.

Queries are split on disjoint filter values to respect a bounded row budget.
A single filter value over budget fails explicitly; use a prepared export or
a more specific query instead of assuming a server-truncated graph is complete.
"""
import numbers
import numpy as np
import pandas as pd

from generator.source_contract import canonical_ids, validate_graph


def _count(value):
    if isinstance(value, pd.DataFrame):
        if value.shape != (1, 1) or list(value.columns) != ['count']:
            raise ValueError('Unexpected CAVE count response; server may have ignored get_counts')
        value = value.iloc[0, 0]
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral) or value < 0:
        raise ValueError('Unexpected CAVE count response; cannot establish completeness')
    return int(value)


def _fetch_complete(materialize, table, version, filters, split_column, max_rows):
    kwargs = dict(filter_in_dict=filters, materialization_version=version, metadata=False)
    # Some server join endpoints ignore get_counts and return ordinary rows.
    # Bound that response too, and require the actual count schema above.
    expected = _count(materialize.query_table(table, get_counts=True, limit=1, **kwargs))
    values = filters[split_column]
    if expected > max_rows:
        if len(values) < 2:
            raise ValueError(f'CAVE query for {table} exceeds row budget for one {split_column}; use a bounded export')
        midpoint = len(values)//2
        frame = pd.concat([_fetch_complete(materialize, table, version,
                          {**filters, split_column: part}, split_column, max_rows)
                          for part in (values[:midpoint], values[midpoint:])], ignore_index=True)
        if len(frame) != expected:
            raise ValueError(f'CAVE partition count mismatch for {table}')
        return frame
    if expected == 0:
        return pd.DataFrame()
    frame = materialize.query_table(table, limit=expected+1, **kwargs)
    if not isinstance(frame, pd.DataFrame) or len(frame) != expected:
        raise ValueError(f'CAVE count mismatch/truncated response for {table}: expected {expected} rows')
    if 'id' not in frame or frame.id.isna().any() or frame.id.duplicated().any():
        raise ValueError(f'CAVE {table} must return unique annotation/synapse row IDs')
    for column, allowed in filters.items():
        if column not in frame or not frame[column].isin(allowed).all():
            raise ValueError(f'CAVE response violates requested filter: {column}')
    return frame


def fetch_cave_graph(client, *, datastack, version, annotation_table, synapse_table,
                     target_classes, excluded_classes, excluded_types, max_rows=50000):
    if type(version) is not int or version <= 0:
        raise ValueError('A positive materialization version is required')
    if type(max_rows) is not int or max_rows <= 0:
        raise ValueError('max_rows must be a positive integer')
    targets = sorted(set(target_classes) - set(excluded_types))
    if not targets:
        raise ValueError('No target cell types selected')
    annotations = _fetch_complete(client.materialize, annotation_table, version,
                                   {'cell_type': targets}, 'cell_type', max_rows)
    if annotations.empty or not {'pt_root_id', 'cell_type', 'side'} <= set(annotations):
        raise ValueError('CAVE annotations require pt_root_id, cell_type and side; choose an explicit table adapter')
    if annotations.id.duplicated().any():
        raise ValueError('Duplicate annotation IDs across query partitions')
    if excluded_classes:
        if ('super_class' not in annotations
                or any(not isinstance(value, str) or not value.strip() for value in annotations.super_class)):
            raise ValueError('Cannot apply super_class exclusions: annotation field missing')
        annotations = annotations[~annotations.super_class.str.lower().isin(excluded_classes)].copy()
    columns = ['pt_root_id', 'cell_type', 'side'] + [c for c in ('super_class', 'layer', 'column') if c in annotations]
    nodes = annotations[columns].rename(columns={'pt_root_id': 'root_id'}).copy()
    nodes['root_id'] = canonical_ids(nodes.root_id, 'annotations')
    if nodes.empty or nodes.root_id.duplicated().any():
        raise ValueError('Empty selection or duplicate root annotations; resolve explicitly')
    # CAVE query ordering is not a reproducible neuron indexing convention.
    nodes = nodes.sort_values('root_id').reset_index(drop=True)
    return fetch_cave_connectivity(client, nodes, datastack=datastack, version=version,
        synapse_table=synapse_table, annotation_table=annotation_table, max_rows=max_rows,
        selection={'strategy': 'induced_cell_type_whitelist', 'target_classes': targets,
                   'excluded_super_classes': sorted(excluded_classes),
                   'excluded_cell_types': sorted(excluded_types),
                   'intermediate_neurons_outside_whitelist': 'not_included'})


def fetch_cave_connectivity(client, nodes, *, datastack, version, synapse_table,
                            annotation_table=None, annotation_source=None, selection=None,
                            max_rows=50000):
    """Count-checked induced connectivity for explicitly sourced node annotations."""
    from generator.source_contract import validate_source
    if type(max_rows) is not int or max_rows <= 0:
        raise ValueError('max_rows must be a positive integer')
    source = {'source_kind':'cave', 'datastack':datastack, 'materialization_version':version,
              'annotation_table':annotation_table, 'synapse_table':synapse_table,
              'completeness':'query-count-checked', 'selection':selection}
    if annotation_source is not None:
        source['annotation_source'] = dict(annotation_source)
    validate_source(source)
    if nodes.empty or not {'root_id', 'cell_type', 'side'} <= set(nodes):
        raise ValueError('Expected nonempty annotated nodes before querying connectivity')
    nodes = nodes.copy()
    nodes['root_id'] = canonical_ids(nodes.root_id, 'nodes')
    if nodes.root_id.duplicated().any():
        raise ValueError('Duplicate root annotations')
    nodes = nodes.sort_values('root_id').reset_index(drop=True)
    ids = [int(value) for value in nodes.root_id]
    synapses = _fetch_complete(client.materialize, synapse_table, version,
        {'pre_pt_root_id': ids, 'post_pt_root_id': ids}, 'pre_pt_root_id', max_rows)
    if synapses.empty or not {'pre_pt_root_id', 'post_pt_root_id'} <= set(synapses):
        raise ValueError('CAVE selection has no synapses or wrong synapse schema')
    if synapses.id.duplicated().any():
        raise ValueError('Duplicate synapse IDs across query partitions')
    synapses = synapses.copy()
    for key in ('pre_pt_root_id', 'post_pt_root_id'):
        synapses[key] = canonical_ids(synapses[key], key)
    edges = (synapses.groupby(['pre_pt_root_id', 'post_pt_root_id']).size().reset_index(name='weight')
             .rename(columns={'pre_pt_root_id': 'pre_id', 'post_pt_root_id': 'post_id'}))
    nodes, edges = validate_graph(nodes, edges)
    source['synapse_rows'] = len(synapses)
    return nodes, edges, source
