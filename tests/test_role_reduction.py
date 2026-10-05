"""Behavioral contracts for reduction before any flight training."""
import numpy as np
import pandas as pd
import pytest

from core.routing import motor_mask, structural_rank
from generator.role_reducer import RolePreservingReducer
from generator.graph_reducer import ReducedModel


def graph():
    labels = ([('F', 'left')] * 3 + [('F', 'right')] + [('L', 'left')] * 3
              + [('T', 'left')] * 2 + [('Y', 'right')] + [('PR', 'left')] * 2
              + [('I', 'left')] * 5 + [('I', 'right')] * 3)
    n = len(labels)
    nodes = pd.DataFrame(labels, columns=['cell_type', 'side'])
    nodes['idx'] = np.arange(n)
    nodes['root_id'] = np.arange(1000, 1000 + n)
    nodes['layer'] = 'test'
    pre = np.arange(n, dtype=np.int32)
    post = np.roll(pre, -1)
    weights = np.arange(1, n + 1, dtype=np.float32)
    weights[12] *= -1
    cfg = {'sensor_groups': {'optic_flow': {'cell_types': ['F']},
                             'tof_depth': {'cell_types': ['L']}},
           'motor_groups': {key: {'cell_types': [cell]} for key, cell in
                            [('throttle', 'T'), ('yaw', 'Y'), ('pitch_roll', 'PR')]}}
    return dict(nodes_df=nodes, edges_df=pd.DataFrame({'weight': abs(weights)}),
                cell_mapping_cfg=cfg, pre_idx=pre, post_idx=post, signed_weights=weights)


def test_every_budget_preserves_roles_sides_signs_and_individual_interface_neurons():
    args = graph()
    reducer = RolePreservingReducer(**args)
    assert reducer.minimum_k == 15
    for k in range(reducer.minimum_k, reducer.N + 1):
        model = reducer.reduce(k)
        np.testing.assert_array_equal(np.unique(model.cluster_map), np.arange(k))
        for cluster in range(k):
            members = np.flatnonzero(model.cluster_map == cluster)
            rows = args['nodes_df'].iloc[members]
            assert len(rows[['cell_type', 'side']].drop_duplicates()) == 1
            assert len(set(np.sign(args['signed_weights'][members]))) == 1
            if rows.cell_type.iloc[0] != 'I':
                assert len(members) == 1
        assert structural_rank(motor_mask(k, model.motor_index_map)) == 4
    np.testing.assert_array_equal(reducer.reduce(reducer.N).cluster_map, np.arange(reducer.N))


@pytest.mark.parametrize('k', [0, -1, 14, 21, 15.5, True])
def test_impossible_budget_is_rejected_instead_of_merging_protected_roles(k):
    with pytest.raises(ValueError, match='k|budget'):
        RolePreservingReducer(**graph()).reduce(k)


def test_raw_aggregation_equals_lift_propagate_and_average_with_signed_directed_edges():
    args = graph()
    reducer = RolePreservingReducer(**args)
    model = reducer.reduce(16)
    h = np.linspace(-1, 1, model.k)
    full = np.zeros((reducer.N, reducer.N))
    np.add.at(full, (args['pre_idx'], args['post_idx']), args['signed_weights'])
    propagated = h[model.cluster_map] @ full
    expected = [propagated[model.cluster_map == c].mean() for c in range(model.k)]
    np.testing.assert_allclose(h @ model.W, expected, atol=1e-6)


def test_order_of_dataframe_rows_does_not_relabel_neurons():
    args = graph()
    first = RolePreservingReducer(**args)(16)
    args['nodes_df'] = args['nodes_df'].sample(frac=1, random_state=9)
    second = RolePreservingReducer(**args)(16)
    np.testing.assert_array_equal(first.cluster_map, second.cluster_map)
    np.testing.assert_allclose(first.W.toarray(), second.W.toarray())
    assert first.provenance == second.provenance


def test_normalization_is_recorded_and_does_not_change_edge_directions_or_ratios(tmp_path):
    reducer = RolePreservingReducer(**graph())
    raw, scaled = reducer.reduce(16), reducer(16)
    gain = scaled.provenance['weight_scaling']['divisor']
    assert gain >= 1
    np.testing.assert_allclose(scaled.W.toarray() * gain, raw.W.toarray(), rtol=1e-6)
    assert np.asarray(abs(scaled.W).sum(axis=0)).max() <= 1 + 1e-6
    loaded = ReducedModel.load(scaled.save(str(tmp_path))['meta'], allow_legacy=False)
    assert loaded.provenance == scaled.provenance
    assert len(loaded.provenance['source_root_ids']) == reducer.N
    assert len(loaded.provenance['cluster_signatures']) == 16


@pytest.mark.parametrize('fault', ['duplicate_id', 'duplicate_idx', 'missing_side', 'nan_weight', 'bad_edge'])
def test_malformed_source_graph_is_rejected_before_aggregation(fault):
    args = graph()
    if fault == 'duplicate_id': args['nodes_df'].loc[1, 'root_id'] = 1000
    if fault == 'duplicate_idx': args['nodes_df'].loc[1, 'idx'] = 0
    if fault == 'missing_side': args['nodes_df'] = args['nodes_df'].drop(columns='side')
    if fault == 'nan_weight': args['signed_weights'][0] = np.nan
    if fault == 'bad_edge': args['post_idx'][0] = len(args['nodes_df'])
    with pytest.raises(ValueError):
        RolePreservingReducer(**args)


def test_graph_fingerprint_changes_when_wiring_or_mapping_changes():
    args = graph()
    initial = RolePreservingReducer(**args)(16).provenance['source_sha256']
    args['signed_weights'][0] += .5
    assert RolePreservingReducer(**args)(16).provenance['source_sha256'] != initial
    args = graph()
    args['cell_mapping_cfg']['motor_groups']['yaw']['cell_types'].append('PR')
    assert RolePreservingReducer(**args)(16).provenance['source_sha256'] != initial


def test_policy_factory_can_load_role_preserving_graph_and_report_its_source(tmp_path):
    from optimizer.evaluate import create_model
    model = RolePreservingReducer(**graph())(16)
    model.save(str(tmp_path))
    policy = create_model({'reducer': 'role_degree', 'k_clusters': 16}, base_dir=str(tmp_path))
    assert policy.routing_diagnostics['motor_structural_rank'] == 4
    assert policy.reduction_diagnostics['source_sha256'] == model.provenance['source_sha256']
    assert policy.reduction_diagnostics['source_kind'] == 'unverified'


def test_default_optuna_candidates_do_not_return_to_collapsed_spectral_models(tmp_path):
    from optimizer.evaluate import create_model
    class Trial:
        choices = {}
        def suggest_categorical(self, name, choices):
            self.choices[name] = choices
            return choices[0]
        def suggest_float(self, *args):
            raise AssertionError('Do not tune ineffective all-entry sparsity on these sparse graphs')
    trial = Trial()
    with pytest.raises(FileNotFoundError, match='meta_role_degree_k128'):
        create_model(trial, base_dir=str(tmp_path))
    assert trial.choices['k_clusters'] == [128, 256]
    assert trial.choices['reducer'] == ['role_degree']


def test_default_missing_artifacts_explain_regeneration(tmp_path):
    from optimizer.evaluate import create_model
    with pytest.raises(FileNotFoundError, match='generator.role_reducer'):
        create_model(base_dir=str(tmp_path))
