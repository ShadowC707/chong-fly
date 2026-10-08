"""Pinned official FlyWire views and conservative candidate validation.

No synthetic fallback. Transmitter-to-sign conversion is an explicit model
assumption, not a measurement of postsynaptic receptor polarity.
"""
from collections import deque

import numpy as np
import pandas as pd
import scipy.sparse as sp

from generator.cave_source import _count
from generator.source_contract import canonical_ids

SYNAPSE_VIEW = 'valid_synapses_nt_np_v6'
CONNECTION_VIEW = 'valid_connection_v2'
SIGN_ASSUMPTIONS = {'acetylcholine': 1, 'gaba': -1, 'glutamate': -1}


def fetch_view(materialize, view, filters, *, max_rows=50000):
    """Split only disjoint filter sets; reject truncation and duplicate records."""
    if view not in {SYNAPSE_VIEW, CONNECTION_VIEW} or type(max_rows) is not int or max_rows < 1:
        raise ValueError('Unsupported view or row budget')
    if not filters or any(not values for values in filters.values()):
        raise ValueError('Bounded nonempty root filters required')
    filters = {k: sorted(set(int(x) for x in canonical_ids(v, k))) for k, v in filters.items()}
    log = []

    def query(selection):
        kw = dict(filter_in_dict=selection, materialization_version=783, metadata=False)
        expected = _count(materialize.query_view(view, get_counts=True, limit=1, **kw))
        entry = {'view': view, 'filters': {k: [str(x) for x in v] for k,v in selection.items()},
                 'expected_rows': expected, 'version': 783}
        log.append(entry)
        if expected > max_rows:
            key = max(selection, key=lambda k: len(selection[k]))
            values = selection[key]
            if len(values) < 2:
                raise ValueError('Single-root query exceeds row budget; no truncated fallback')
            mid = len(values)//2
            parts = [query({**selection, key: part}) for part in (values[:mid], values[mid:])]
            frame = pd.concat(parts, ignore_index=True)
        elif expected:
            frame = materialize.query_view(view, limit=expected+1, **kw)
        else:
            columns = (['id','cleft_score'] if view == SYNAPSE_VIEW else ['n_syn'])
            frame = pd.DataFrame(columns=['pre_pt_root_id','post_pt_root_id', *columns])
        if not isinstance(frame, pd.DataFrame) or len(frame) != expected:
            raise ValueError('View count mismatch / truncated response')
        entry['returned_rows'] = len(frame)
        for key, values in selection.items():
            if key not in frame or not frame[key].isin(values).all():
                raise ValueError('View response violates root filter')
        return frame

    frame = query(filters)
    required = {'pre_pt_root_id','post_pt_root_id'} | ({'id','cleft_score'} if view == SYNAPSE_VIEW else {'n_syn'})
    if not required <= set(frame):
        raise ValueError('Unexpected view schema')
    keys = ['id'] if view == SYNAPSE_VIEW else ['pre_pt_root_id','post_pt_root_id']
    if frame[keys].isna().any().any() or frame.duplicated(keys).any():
        raise ValueError('Duplicate view IDs/pairs across disjoint partitions')
    frame = frame.copy()
    for key in ['pre_pt_root_id','post_pt_root_id'] + (['id'] if view == SYNAPSE_VIEW else []):
        # Background 0 is retained in the acquisition and excluded explicitly later.
        frame[key] = ['0' if isinstance(v, (str,int,np.integer)) and str(v) == '0'
                      else canonical_ids([v], key)[0] for v in frame[key]]
    if view == CONNECTION_VIEW:
        counts = frame.n_syn.to_numpy(dtype=float)
        if not np.isfinite(counts).all() or np.any(counts <= 0) or np.any(counts != np.floor(counts)):
            raise ValueError('Invalid connection counts')
    return frame.sort_values(keys).reset_index(drop=True), log


def clean_synapses(frame, *, pair_min):
    """Do not redo server deduplication using guessed geometric tie-breaks."""
    if type(pair_min) is not int or pair_min < 1:
        raise ValueError('Positive integer pair threshold required')
    frame = frame.copy()
    if frame.id.duplicated().any() or frame.id.isna().any():
        raise ValueError('Duplicate synapse IDs')
    scores = frame.cleft_score.to_numpy(dtype=float)
    if not np.isfinite(scores).all() or np.any(scores <= 50):
        raise ValueError('Official filtered cleft score contract violated')
    for key in ('pre_pt_root_id','post_pt_root_id'):
        frame[key] = ['0' if str(v) == '0' else canonical_ids([v],key)[0] for v in frame[key]]
    same = frame.pre_pt_root_id == frame.post_pt_root_id
    background = (frame.pre_pt_root_id == '0') | (frame.post_pt_root_id == '0')
    clean = frame[~same & ~background].copy()
    edges = clean.groupby(['pre_pt_root_id','post_pt_root_id']).size().reset_index(name='weight')
    edges = edges.rename(columns={'pre_pt_root_id':'pre_id','post_pt_root_id':'post_id'})
    audit = {'official_view_rows':len(frame), 'same_root_rows':int(same.sum()),
             'background_rows':int(background.sum()), 'inter_neuron_rows':len(clean),
             'pairs_before_threshold':len(edges), 'pair_min':pair_min}
    edges = edges[edges.weight >= pair_min].reset_index(drop=True)
    audit.update(pairs_retained=len(edges), synapses_in_retained_pairs=int(edges.weight.sum()))
    return clean.reset_index(drop=True), edges, audit


def assign_polarities(nodes):
    signs, records = [], []
    for row in nodes.to_dict('records'):
        known = row.get('known_nt', '')
        known = known if isinstance(known,str) else ''
        transmitter = known or row.get('top_nt', '')
        if transmitter not in SIGN_ASSUMPTIONS:
            raise ValueError(f"Unresolved polarity for root {row['root_id']}")
        confidence = float(row['top_nt_conf'])
        if not np.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError('Invalid polarity prediction confidence')
        signs.append(SIGN_ASSUMPTIONS[transmitter])
        records.append({'root_id':str(row['root_id']), 'transmitter':transmitter,
                        'basis':'published_known_nt' if known else 'EM_prediction_top_nt',
                        'prediction_confidence':confidence, 'sign':signs[-1],
                        'known_nt_source':row.get('known_nt_source',''),
                        'synaptic_sign_verified':False,
                        'assumption':'presynaptic Dale-like sign; glutamate inhibitory; receptors unmeasured'})
    return np.asarray(signs, dtype=float), records


def select_bridges(outgoing, incoming, eligible, seeds, *, pair_min, budget):
    """Rank measured S->x->DN bridges by bottleneck count, then exact root ID."""
    outgoing = outgoing[outgoing.n_syn >= pair_min]
    incoming = incoming[incoming.n_syn >= pair_min]
    left = outgoing.groupby('post_pt_root_id').n_syn.sum()
    right = incoming.groupby('pre_pt_root_id').n_syn.sum()
    roots = (set(left.index) & set(right.index) & set(eligible)) - set(seeds)
    rows = [{'root_id':r, 'sensory_input_synapses':int(left[r]), 'dn_output_synapses':int(right[r]),
             'bottleneck':int(min(left[r],right[r]))} for r in roots]
    return sorted(rows, key=lambda r:(-r['bottleneck'], int(r['root_id'])))[:budget]


def compare_reachability(full, reduced, cmap, sources, targets):
    """Every individual interface pair, stronger than population any-to-any."""
    def reachable(matrix, source):
        matrix = sp.csr_matrix(matrix); matrix.eliminate_zeros()
        seen, queue = {source}, deque([source])
        while queue:
            i = queue.popleft()
            for j in matrix.indices[matrix.indptr[i]:matrix.indptr[i+1]]:
                if int(j) not in seen:
                    seen.add(int(j)); queue.append(int(j))
        return seen
    lost, introduced = [], []
    for source in sources:
        before, after = reachable(full,source), reachable(reduced,int(cmap[source]))
        for target in targets:
            a, b = target in before, int(cmap[target]) in after
            if a and not b: lost.append([int(source),int(target)])
            if b and not a: introduced.append([int(source),int(target)])
    return {'lost_pairs':lost, 'introduced_pairs':introduced, 'passed':not lost and not introduced,
            'tested_pairs':len(sources)*len(targets)}
