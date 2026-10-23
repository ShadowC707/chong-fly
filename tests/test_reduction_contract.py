import json

import numpy as np
import pytest
import scipy.sparse as sp

from generator.graph_reducer import ReducedModel, _condense_synapses_jit


def test_reduction_equals_lift_propagate_then_average_target_population():
    # Unequal populations distinguish source averaging from target averaging.
    clusters = np.array([0, 0, 1], dtype=np.int32)
    pre, post = np.array([0, 1, 2]), np.array([2, 2, 0])
    weights = np.array([2., 4., -3.], dtype=np.float32)
    reduced = _condense_synapses_jit(pre, post, weights, clusters, 2)
    h_macro = np.array([.5, .2])
    W_full = np.zeros((3, 3)); W_full[pre, post] = weights
    propagated = h_macro[clusters] @ W_full
    expected = np.array([propagated[:2].mean(), propagated[2]])
    np.testing.assert_allclose(h_macro @ reduced, expected)
    np.testing.assert_allclose(reduced, [[0, 6], [-1.5, 0]])


@pytest.mark.parametrize("sparse", [False, True])
def test_saved_graph_declares_orientation_and_normalization(tmp_path, sparse):
    W = np.array([[0., 6.], [-1.5, 0.]], dtype=np.float32)
    model = ReducedModel(sp.csr_matrix(W) if sparse else W, 2, "test",
                         np.array([0, 0, 1]), {}, {})
    paths = model.save(str(tmp_path))
    meta = json.loads(open(paths["meta"], encoding="utf-8").read())
    assert meta["format_version"] == 2
    assert meta["matrix_orientation"] == "source_target"
    assert meta["reduction_normalization"] == "target_mean"
    loaded = ReducedModel.load(paths["meta"])
    np.testing.assert_array_equal(loaded.W.toarray() if sparse else loaded.W, W)


def write_legacy(tmp_path, sparse=False):
    model = ReducedModel(np.array([[0., 3.], [-3., 0.]], dtype=np.float32), 2, "spectral",
                         np.array([0, 0, 1]), {}, {}, metrics={"rho": 7.})
    if sparse: model.W = sp.csr_matrix(model.W)
    paths = model.save(str(tmp_path))
    with open(paths["meta"], encoding="utf-8") as f: meta = json.load(f)
    for key in ("format_version", "matrix_orientation", "reduction_normalization"):
        meta.pop(key, None)
    with open(paths["meta"], "w", encoding="utf-8") as f: json.dump(meta, f)
    return paths


@pytest.mark.parametrize("sparse", [False, True])
def test_known_legacy_graph_converts_without_rewriting_original(tmp_path, sparse):
    paths = write_legacy(tmp_path, sparse)
    before = open(paths["w"], "rb").read()
    with pytest.warns(UserWarning, match="(?i)legacy"):
        loaded = ReducedModel.load(paths["meta"])
    np.testing.assert_allclose(loaded.W.toarray() if sparse else loaded.W, [[0., 6.], [-1.5, 0.]])
    assert loaded.metrics == {}  # previous benchmark measurements are stale
    assert open(paths["w"], "rb").read() == before


def test_unknown_orientation_is_not_guessed(tmp_path):
    paths = write_legacy(tmp_path)
    with open(paths["meta"], encoding="utf-8") as f: meta = json.load(f)
    meta.update(format_version=2, matrix_orientation="ambiguous", reduction_normalization="target_mean")
    with open(paths["meta"], "w", encoding="utf-8") as f: json.dump(meta, f)
    with pytest.raises(ValueError, match="(?i)orientation|contract"):
        ReducedModel.load(paths["meta"])
