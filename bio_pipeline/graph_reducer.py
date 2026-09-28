"""
graph_reducer.py
================
Multi-scale biological graph reduction pipeline for Chong-Fly.

Three standardised reducers inherit from BaseReducer:
  1. SpectralReducer      – normalised Laplacian + Randomized SVD + MiniBatchKMeans
  2. CentralityReducer    – degree / betweenness centrality condensation
  3. MagnitudePruner      – global magnitude pruning → scipy.sparse.csr_matrix

All reducers share:
  • A common __call__(k) interface → ReducedModel dataclass
  • save() / load() helpers that persist both the matrix and its metadata
    (including sensor / motor index maps into the reduced space)
  • benchmark_matrix() for unified latency + spectral profiling
"""

from __future__ import annotations

import abc
import json
import os
import sys
import time
import dataclasses
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from sklearn.utils.extmath import randomized_svd
from sklearn.cluster import MiniBatchKMeans
from numba import njit, prange

# ---------------------------------------------------------------------------
# Polarity map (excitatory +1 / inhibitory -1)
# ---------------------------------------------------------------------------

POLARITY_MAP: dict[str, float] = {
    # Excitatory
    "T4a": 1, "T4b": 1, "T4c": 1, "T4d": 1,
    "T5a": 1, "T5b": 1, "T5c": 1, "T5d": 1,
    "Tm1": 1, "Tm2": 1, "Tm3": 1, "Tm4": 1, "Tm9": 1,
    "Mi1": 1,
    "HSN": 1, "HSE": 1, "HSS": 1,
    "LC4": 1, "LPLC2": 1,
    "E-PG": 1, "P-EN": 1, "P-FN": 1, "P-FL": 1,
    "DNa01": 1, "DNa02": 1, "DNa03": 1,
    "DNp01": 1, "DNp02": 1, "DNp04": 1, "DNp09": 1, "DNp11": 1, "DNa11": 1,
    "DNb01": 1,
    # Inhibitory
    "Mi4": -1, "Mi9": -1,
    "CH": -1, "dCH": -1, "vCH": -1,
    "LPi1-2": -1, "LPi2-1": -1, "LPi3-4": -1, "LPi4-3": -1,
    "Delta7": -1,
}
for _i in range(1, 11):
    POLARITY_MAP[f"VS{_i}"] = 1


# ---------------------------------------------------------------------------
# ReducedModel – unified output container
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ReducedModel:
    """Holds a reduced weight matrix and all associated metadata."""

    # Core result
    W: np.ndarray | csr_matrix          # (k, k) dense float32 or CSR sparse
    k: int                               # reduced dimension
    reducer_name: str                    # e.g. "spectral", "centrality", "magnitude"

    # Connectivity metadata
    cluster_map: np.ndarray             # (N,) int32 – original node → cluster id
    sensor_index_map: dict[str, list[int]]   # group name → cluster indices
    motor_index_map: dict[str, list[int]]    # group name → cluster indices

    # Benchmark (filled lazily)
    metrics: dict[str, Any] = dataclasses.field(default_factory=dict)
    platform_hint: str = ""

    # ------------------------------------------------------------------ I/O
    def save(self, out_dir: str) -> dict[str, str]:
        """
        Persist matrix + metadata side-car JSON.

        Returns a dict of written file paths.
        """
        os.makedirs(out_dir, exist_ok=True)
        tag  = f"{self.reducer_name}_k{self.k}"
        w_path   = os.path.join(out_dir, f"w_{tag}.npy")
        meta_path = os.path.join(out_dir, f"meta_{tag}.json")
        cmap_path = os.path.join(out_dir, f"cmap_{tag}.npy")

        # Dense or sparse
        if sp.issparse(self.W):
            sp.save_npz(w_path.replace(".npy", ".npz"), self.W.astype(np.float32))
            w_path = w_path.replace(".npy", ".npz")
        else:
            np.save(w_path, self.W.astype(np.float32))

        np.save(cmap_path, self.cluster_map)

        meta = {
            "reducer":  self.reducer_name,
            "k":        self.k,
            "sparse":   sp.issparse(self.W),
            "w_file":   os.path.basename(w_path),
            "cmap_file": os.path.basename(cmap_path),
            "sensor_index_map": self.sensor_index_map,
            "motor_index_map":  self.motor_index_map,
            "metrics":  self.metrics,
            "platform_hint": self.platform_hint,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        return {"w": w_path, "meta": meta_path, "cmap": cmap_path}

    @staticmethod
    def load(meta_path: str) -> "ReducedModel":
        """Reconstruct a ReducedModel from its meta JSON side-car."""
        meta_dir = os.path.dirname(meta_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        w_path = os.path.join(meta_dir, meta["w_file"])
        if meta["sparse"]:
            W = sp.load_npz(w_path)
        else:
            W = np.load(w_path)

        cmap = np.load(os.path.join(meta_dir, meta["cmap_file"]))

        return ReducedModel(
            W=W,
            k=meta["k"],
            reducer_name=meta["reducer"],
            cluster_map=cmap,
            sensor_index_map=meta["sensor_index_map"],
            motor_index_map=meta["motor_index_map"],
            metrics=meta.get("metrics", {}),
            platform_hint=meta.get("platform_hint", ""),
        )


# ---------------------------------------------------------------------------
# Benchmark helper (shared by all reducers)
# ---------------------------------------------------------------------------

def benchmark_matrix(W: np.ndarray | csr_matrix, k: int,
                     num_trials: int = 500) -> dict[str, Any]:
    """
    Pure linear algebraic benchmark: sparsity, spectral radius ρ(W),
    GEMV latency (µs) and MFLOP/s throughput.
    Works on both dense ndarray and scipy CSR.
    """
    is_sparse = sp.issparse(W)
    W_dense   = W.toarray().astype(np.float32) if is_sparse else W.astype(np.float32)

    # Sparsity & footprint
    nnz           = int(np.sum(np.abs(W_dense) > 1e-6))
    total_el      = k * k
    sparsity_pct  = (1.0 - nnz / total_el) * 100.0
    if is_sparse:
        mem_mb = (W.data.nbytes + W.indices.nbytes + W.indptr.nbytes) / 1024 / 1024
    else:
        mem_mb = W_dense.nbytes / 1024 / 1024

    # Spectral radius via power iteration
    v = np.random.default_rng(0).standard_normal(k).astype(np.float32)
    v /= np.linalg.norm(v) + 1e-9
    rho = 0.0
    for _ in range(30):
        v_next = W_dense @ v
        rho    = float(np.linalg.norm(v_next))
        if rho < 1e-9:
            break
        v = v_next / rho

    # GEMV latency
    x = np.random.default_rng(1).standard_normal(k).astype(np.float32)
    for _ in range(20):                        # warm-up
        _ = W_dense @ x
    t0 = time.perf_counter()
    for _ in range(num_trials):
        y  = W_dense @ x
        nx = np.linalg.norm(y) + 1e-9
        x  = y / nx
    elapsed = time.perf_counter() - t0

    step_us = (elapsed / num_trials) * 1e6
    mops    = (2.0 * k * k * num_trials / elapsed) / 1e6

    diverged = np.isnan(y).any() or np.isinf(y).any() or float(np.max(np.abs(y))) > 1e6

    return {
        "mem_mb":       float(mem_mb),
        "nnz":          nnz,
        "sparsity_pct": float(sparsity_pct),
        "spectral_rho": float(rho),
        "step_us":      float(step_us),
        "mops":         float(mops),
        "stable_linear": not diverged,
    }


# ---------------------------------------------------------------------------
# Numba JIT kernels
# ---------------------------------------------------------------------------

@njit(parallel=True, fastmath=True)
def _normalize_rows_jit(matrix: np.ndarray) -> np.ndarray:
    n, m = matrix.shape
    out  = np.empty_like(matrix)
    for i in prange(n):
        norm_sq = 0.0
        for j in range(m):
            norm_sq += matrix[i, j] * matrix[i, j]
        norm = np.sqrt(norm_sq)
        if norm < 1e-9:
            norm = 1e-9
        for j in range(m):
            out[i, j] = matrix[i, j] / norm
    return out


@njit(fastmath=True)
def _condense_synapses_jit(pre_idx: np.ndarray, post_idx: np.ndarray,
                            weights: np.ndarray,
                            clusters: np.ndarray, k: int) -> np.ndarray:
    """Accumulate signed synaptic weights into macro-cluster matrix."""
    W = np.zeros((k, k), dtype=np.float32)
    sz = np.zeros(k, dtype=np.float32)

    for i in range(len(clusters)):
        c = clusters[i]
        if c < k:
            sz[c] += 1.0

    for e in range(len(pre_idx)):
        cp = clusters[pre_idx[e]]
        cq = clusters[post_idx[e]]
        W[cp, cq] += weights[e]

    for p in range(k):
        d = sz[p] if sz[p] >= 1.0 else 1.0
        for q in range(k):
            W[p, q] /= d

    return W


# ---------------------------------------------------------------------------
# BaseReducer – abstract interface
# ---------------------------------------------------------------------------

class BaseReducer(abc.ABC):
    """
    Abstract base for all graph reduction strategies.

    Subclasses must implement:
        reduce(k) → ReducedModel
    """

    #: Human-readable name used in filenames and metadata
    name: str = "base"

    def __init__(self,
                 nodes_df: pd.DataFrame,
                 edges_df: pd.DataFrame,
                 cell_mapping_cfg: dict,
                 pre_idx: np.ndarray,
                 post_idx: np.ndarray,
                 signed_weights: np.ndarray):
        """
        Parameters
        ----------
        nodes_df            : DataFrame with columns root_id, cell_type, idx
        edges_df            : DataFrame with pre_idx, post_idx, weight (signed)
        cell_mapping_cfg    : loaded configs/cell_mapping.json
        pre_idx             : int32 array of pre-synaptic node indices
        post_idx            : int32 array of post-synaptic node indices
        signed_weights      : float32 signed synapse weights
        """
        self.nodes_df       = nodes_df
        self.edges_df       = edges_df
        self.cfg            = cell_mapping_cfg
        self.pre_idx        = pre_idx
        self.post_idx       = post_idx
        self.signed_weights = signed_weights
        self.N              = len(nodes_df)

        # Pre-compute sensor / motor node sets once
        sg = cell_mapping_cfg["sensor_groups"]
        mg = cell_mapping_cfg["motor_groups"]

        self._sensor_node_sets: dict[str, np.ndarray] = {
            "lptc_flow": self._node_mask(sg["optic_flow"]["cell_types"]),
            "lc_looming": self._node_mask(sg["tof_depth"]["cell_types"]),
        }
        self._motor_node_sets: dict[str, np.ndarray] = {
            key: self._node_mask(group["cell_types"])
            for key, group in mg.items()
            if not key.startswith("_")
        }

    # ------------------------------------------------------------------ helpers

    def _node_mask(self, cell_types: list[str]) -> np.ndarray:
        """Return sorted integer node indices for the given cell types."""
        mask = self.nodes_df["cell_type"].isin(cell_types)
        return self.nodes_df.loc[mask, "idx"].values.astype(np.int32)

    def _build_index_maps(self, clusters: np.ndarray) -> tuple[dict, dict]:
        """
        Given a cluster assignment array (shape N,), compute:
            sensor_index_map  – {group_name: sorted list of cluster ids}
            motor_index_map   – {group_name: sorted list of cluster ids}
        """
        def _cluster_ids(node_indices: np.ndarray) -> list[int]:
            valid = node_indices[node_indices < len(clusters)]
            return sorted(set(int(clusters[i]) for i in valid))

        sensor_map = {name: _cluster_ids(idx)
                      for name, idx in self._sensor_node_sets.items()}
        motor_map  = {name: _cluster_ids(idx)
                      for name, idx in self._motor_node_sets.items()}
        return sensor_map, motor_map

    # ------------------------------------------------------------------ public API

    @abc.abstractmethod
    def reduce(self, k: int) -> ReducedModel:
        """Return a ReducedModel of dimension k."""

    def __call__(self, k: int) -> ReducedModel:
        model = self.reduce(k)
        
        # --- Spectral Radius Normalization ---
        # Calculate rho(W) = max|lambda_i| and scale if > 1.0 to prevent explosive chaos
        W = model.W
        is_sparse = sp.issparse(W)
        
        try:
            if is_sparse or W.shape[0] > 2048:
                # Use ARPACK for large or sparse matrices
                # We want the eigenvalue with largest magnitude
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    evals = sp.linalg.eigs(W, k=1, which='LM', return_eigenvectors=False)
                    rho = float(np.max(np.abs(evals)))
            else:
                # Exact calculation for smaller dense matrices
                evals = np.linalg.eigvals(W)
                rho = float(np.max(np.abs(evals)))
        except Exception:
            # Fallback to power iteration if eigs fails to converge
            W_dense = W.toarray().astype(np.float32) if is_sparse else W.astype(np.float32)
            dim = W_dense.shape[0]
            v = np.random.default_rng(42).standard_normal(dim).astype(np.float32)
            v /= np.linalg.norm(v) + 1e-9
            rho = 0.0
            for _ in range(100):
                v_next = W_dense @ v
                r = float(np.linalg.norm(v_next))
                if r < 1e-9:
                    break
                v = v_next / r
                rho = r
                
        if rho > 1.0:
            if is_sparse:
                model.W = model.W / rho
            else:
                model.W = model.W / rho
                
        return model

    @staticmethod
    def _platform_hint(k: int) -> str:
        if k <= 32:
            return "MCU (ESP32 / Cortex-M4)"
        if k <= 128:
            return "SBC (RPi Zero / CM4)"
        if k <= 512:
            return "Edge Embedded (Jetson Nano)"
        if k <= 2048:
            return "High-Perf Edge (VOXL 2)"
        if k <= 8192:
            return "Workstation GPU (RTX 3050 Ti)"
        return "1:1 Full Connectome (Server)"


# ---------------------------------------------------------------------------
# 1. SpectralReducer
# ---------------------------------------------------------------------------

class SpectralReducer(BaseReducer):
    """
    Spectral graph reduction.

    Pipeline:
        1. Symmetric normalised Laplacian  L = I - D^{-1/2} A D^{-1/2}
        2. Randomized SVD  →  64-D spectral embedding
        3. MiniBatchKMeans on embedding  →  k macro-clusters
        4. Numba JIT synapse condensation  →  W_macro ∈ ℝ^{k×k}

    For k ≥ 2048 a spectral hash-sort is used instead of KMeans.
    For k == N the identity mapping (1:1 biological ground truth) is returned.
    """

    name = "spectral"
    SVD_COMPONENTS = 64

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embeddings: np.ndarray | None = None   # computed lazily once

    def _ensure_embedding(self) -> np.ndarray:
        if self._embeddings is not None:
            return self._embeddings

        print("  [Spectral] Computing normalised Laplacian + SVD...", flush=True)
        N  = self.N
        pre, post = self.pre_idx, self.post_idx
        raw_w = self.edges_df["weight"].values.astype(np.float64)

        W_dir  = csr_matrix((raw_w, (pre, post)), shape=(N, N))
        A_sym  = W_dir + W_dir.T
        deg    = np.array(A_sym.sum(axis=1)).flatten()
        deg[deg == 0] = 1e-9
        D_inv_sq = sp.diags(1.0 / np.sqrt(deg))
        L_sym    = sp.eye(N) - D_inv_sq @ A_sym @ D_inv_sq

        U, _, _ = randomized_svd(L_sym, n_components=self.SVD_COMPONENTS,
                                 random_state=42, n_iter=4)
        self._embeddings = _normalize_rows_jit(U.astype(np.float32))
        return self._embeddings

    def reduce(self, k: int) -> ReducedModel:
        N = self.N
        emb = self._ensure_embedding()

        if k >= N:
            clusters  = np.arange(N, dtype=np.int32)
        elif k >= 2048:
            dim  = min(k, self.SVD_COMPONENTS)
            proj = np.ascontiguousarray(emb[:, :dim])
            hv   = proj @ np.arange(1, dim + 1, dtype=np.float32)
            ranks = np.argsort(hv)
            clusters = np.empty(N, dtype=np.int32)
            clusters[ranks] = np.linspace(0, k - 1, N, dtype=np.float32).astype(np.int32)
        else:
            dim = min(32, self.SVD_COMPONENTS)
            km  = MiniBatchKMeans(n_clusters=k, random_state=42,
                                  batch_size=4096, n_init=1, max_iter=20)
            clusters = km.fit_predict(emb[:, :dim]).astype(np.int32)

        W_macro = _condense_synapses_jit(self.pre_idx, self.post_idx,
                                          self.signed_weights, clusters, k)
        sensor_map, motor_map = self._build_index_maps(clusters)

        return ReducedModel(
            W=W_macro, k=k,
            reducer_name=self.name,
            cluster_map=clusters,
            sensor_index_map=sensor_map,
            motor_index_map=motor_map,
            platform_hint=self._platform_hint(k),
        )


# ---------------------------------------------------------------------------
# 2. CentralityReducer
# ---------------------------------------------------------------------------

class CentralityReducer(BaseReducer):
    """
    Degree / centrality condensation.

    Strategy:
        1. Compute per-node weighted degree (in + out) as proxy for
           topological importance.
        2. Sort nodes by descending degree, then greedily assign them
           to k buckets (round-robin over top-ranked nodes) so that
           the most influential neurons anchor each macro-cluster.
        3. Numba JIT synapse condensation  →  W_macro ∈ ℝ^{k×k}

    This ensures that high-degree hub neurons (HS cells, DN premotor
    drivers) are never merged, preserving input/output fidelity.
    """

    name = "centrality"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._degree: np.ndarray | None = None

    def _ensure_degree(self) -> np.ndarray:
        if self._degree is not None:
            return self._degree
        print("  [Centrality] Computing weighted degree...", flush=True)
        N = self.N
        raw_w = np.abs(self.signed_weights)
        out_deg = np.zeros(N, dtype=np.float64)
        in_deg  = np.zeros(N, dtype=np.float64)
        np.add.at(out_deg, self.pre_idx, raw_w)
        np.add.at(in_deg,  self.post_idx, raw_w)
        self._degree = out_deg + in_deg
        return self._degree

    def reduce(self, k: int) -> ReducedModel:
        N      = self.N
        degree = self._ensure_degree()

        if k >= N:
            clusters = np.arange(N, dtype=np.int32)
        else:
            # Rank nodes by degree (descending)
            rank       = np.argsort(-degree)       # highest degree first
            clusters   = np.empty(N, dtype=np.int32)
            # Top-k nodes become anchors (one per cluster)
            clusters[rank[:k]] = np.arange(k, dtype=np.int32)
            # Remaining nodes: assign to cluster of nearest anchor
            # Use a fast approx: mod-k assignment over degree rank
            rest = rank[k:]
            clusters[rest] = np.arange(len(rest), dtype=np.int32) % k

        W_macro = _condense_synapses_jit(self.pre_idx, self.post_idx,
                                          self.signed_weights, clusters, k)
        sensor_map, motor_map = self._build_index_maps(clusters)

        return ReducedModel(
            W=W_macro, k=k,
            reducer_name=self.name,
            cluster_map=clusters,
            sensor_index_map=sensor_map,
            motor_index_map=motor_map,
            platform_hint=self._platform_hint(k),
        )


# ---------------------------------------------------------------------------
# 3. MagnitudePruner
# ---------------------------------------------------------------------------

class MagnitudePruner(BaseReducer):
    """
    Global magnitude pruning.

    Strategy:
        1. Build the full (N × N) signed weight matrix as CSR.
        2. Compute a global absolute-value threshold at the given
           sparsity percentile (50 – 95 %).
        3. Zero entries below threshold  →  retain only strongest synapses.
        4. Return a scipy.sparse.csr_matrix (NOT a dense condensed matrix).

    Because this reducer keeps N = full graph dimension, k is interpreted
    as a *sparsity target* expressed as an integer percentile (50–95).
    To fit the unified k-based interface, a cluster_map = arange(N) is
    returned (identity mapping), and the sensor/motor index maps point
    directly to the original biological node indices.

    Typical usage:
        pruner = MagnitudePruner(...)
        model_80 = pruner(80)   # keep top 20 % of synapses
    """

    name = "magnitude"

    #: Valid sparsity percentile targets
    SPARSITY_TARGETS = [50, 60, 70, 80, 90, 95]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Build full CSR once
        print("  [Magnitude] Building full signed CSR matrix...", flush=True)
        N = self.N
        self._W_full: csr_matrix = csr_matrix(
            (self.signed_weights.astype(np.float32),
             (self.pre_idx, self.post_idx)),
            shape=(N, N), dtype=np.float32
        )

    def reduce(self, k: int) -> ReducedModel:
        """
        k here is the sparsity percentile target (e.g. 80 → keep top 20 %).
        Must be in [0, 100).
        """
        if not (0 <= k < 100):
            raise ValueError(f"MagnitudePruner expects k = sparsity percentile [0,100), got {k}")

        W = self._W_full.copy()
        data = W.data
        if len(data) == 0:
            pruned = W
        else:
            threshold = float(np.percentile(np.abs(data), k))
            # Zero entries below threshold
            mask = np.abs(data) < threshold
            data[mask] = 0.0
            W.eliminate_zeros()
            pruned = W

        N        = self.N
        clusters = np.arange(N, dtype=np.int32)        # identity map
        sensor_map, motor_map = self._build_index_maps(clusters)

        retained_pct = 100.0 - float(k)
        platform     = f"Pruned-{k}%-sparse (direct N={N})"

        return ReducedModel(
            W=pruned, k=N,
            reducer_name=f"{self.name}_p{k}",
            cluster_map=clusters,
            sensor_index_map=sensor_map,
            motor_index_map=motor_map,
            platform_hint=platform,
        )


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def load_graph(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame,
                                        np.ndarray, np.ndarray, np.ndarray]:
    """
    Load nodes + edges, assign integer indices, compute signed weights.

    Returns
    -------
    nodes_df        : DataFrame (root_id, cell_type, idx, ...)
    edges_df        : DataFrame (pre_idx, post_idx, weight)
    pre_idx         : int32 ndarray
    post_idx        : int32 ndarray
    signed_weights  : float32 ndarray
    """
    nodes_file = os.path.join(data_dir, "raw_nodes.csv")
    edges_file = os.path.join(data_dir, "raw_edges.parquet")
    if not os.path.exists(nodes_file) or not os.path.exists(edges_file):
        print(f"[ERROR] Missing data files in {data_dir}. Run circuit_extractor.py first.")
        sys.exit(1)

    nodes_df = pd.read_csv(nodes_file)
    edges_df = pd.read_parquet(edges_file)

    unique_ids   = nodes_df["root_id"].unique()
    node_to_idx  = {nid: i for i, nid in enumerate(unique_ids)}
    nodes_df     = nodes_df.copy()
    nodes_df["idx"] = nodes_df["root_id"].map(node_to_idx)

    edges_df = edges_df.copy()
    edges_df["pre_idx"]  = edges_df["pre_id"].map(node_to_idx)
    edges_df["post_idx"] = edges_df["post_id"].map(node_to_idx)
    edges_df = (edges_df
                .dropna(subset=["pre_idx", "post_idx"])
                .astype({"pre_idx": np.int32, "post_idx": np.int32}))

    pre_idx  = edges_df["pre_idx"].values
    post_idx = edges_df["post_idx"].values
    raw_w    = edges_df["weight"].values.astype(np.float32)

    # Signed weights: excitatory + / inhibitory –
    polarities     = nodes_df["cell_type"].map(POLARITY_MAP).fillna(1.0).values.astype(np.float32)
    signed_weights = raw_w * polarities[pre_idx]

    return nodes_df, edges_df, pre_idx, post_idx, signed_weights


# ---------------------------------------------------------------------------
# Unified run pipeline
# ---------------------------------------------------------------------------

def run_reduction():
    print("=" * 90)
    print("   BIO-GRAPH REDUCTION PIPELINE  (Spectral | Centrality | Magnitude)")
    print("=" * 90, flush=True)
    t0_global = time.time()

    root        = _project_root()
    data_dir    = os.path.join(root, "data", "raw_connectome")
    models_dir  = os.path.join(root, "data", "reduced_models")
    cfg_path    = os.path.join(root, "configs", "cell_mapping.json")
    os.makedirs(models_dir, exist_ok=True)

    # Load config & graph
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    nodes_df, edges_df, pre_idx, post_idx, signed_w = load_graph(data_dir)
    N = len(nodes_df)
    print(f"Graph: N = {N:,} nodes  |E| = {len(edges_df):,}  "
          f"synapses = {edges_df['weight'].sum():,}", flush=True)

    # Shared constructor kwargs
    ctor_kwargs = dict(
        nodes_df=nodes_df,
        edges_df=edges_df,
        cell_mapping_cfg=cfg,
        pre_idx=pre_idx,
        post_idx=post_idx,
        signed_weights=signed_w,
    )

    # ------------------------------------------------------------------ grid
    spectral_ks   = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, N]
    centrality_ks = [16, 64, 256, 1024]
    magnitude_ps  = MagnitudePruner.SPARSITY_TARGETS   # [50,60,70,80,90,95]

    manifest      = {"source_nodes": N, "source_edges": int(len(edges_df)), "models": []}
    all_reports   = []

    # ===== SPECTRAL =====
    print("\n" + "─" * 90)
    print("  SPECTRAL REDUCER")
    print("─" * 90)
    _print_header()

    spectral = SpectralReducer(**ctor_kwargs)
    for k in [kk for kk in spectral_ks if kk <= N]:
        t0 = time.time()
        model   = spectral(k)
        model.metrics = benchmark_matrix(model.W, k)
        model.metrics["reduce_sec"] = round(time.time() - t0, 3)
        paths   = model.save(models_dir)
        _print_row("spectral", k, model)
        _append_manifest(manifest, all_reports, model, paths)

    # ===== CENTRALITY =====
    print("\n" + "─" * 90)
    print("  CENTRALITY REDUCER")
    print("─" * 90)
    _print_header()

    centrality = CentralityReducer(**ctor_kwargs)
    for k in centrality_ks:
        t0 = time.time()
        model   = centrality(k)
        model.metrics = benchmark_matrix(model.W, k)
        model.metrics["reduce_sec"] = round(time.time() - t0, 3)
        paths   = model.save(models_dir)
        _print_row("centrality", k, model)
        _append_manifest(manifest, all_reports, model, paths)

    # ===== MAGNITUDE PRUNER =====
    print("\n" + "─" * 90)
    print("  MAGNITUDE PRUNER  (CSR sparse, identity cluster map)")
    print("─" * 90)
    print(f"  {'PRUNE %':<10} {'NNZ':<10} {'SPARSITY':<10} {'MEM (MB)':<10} "
          f"{'ρ(W)':<10} {'STEP(µs)':<10} {'MFLOP/s':<10} STATUS")
    print("─" * 90)

    pruner = MagnitudePruner(**ctor_kwargs)
    for pct in magnitude_ps:
        t0 = time.time()
        model   = pruner(pct)
        # Benchmark on the dense version for comparability
        W_dense = model.W.toarray().astype(np.float32)
        model.metrics = benchmark_matrix(W_dense, N)
        model.metrics["prune_pct"] = pct
        model.metrics["reduce_sec"] = round(time.time() - t0, 3)
        paths = model.save(models_dir)

        m = model.metrics
        st = "OK" if m["stable_linear"] else "DIVERGED"
        print(f"  p={pct:<8} {m['nnz']:<10,} {m['sparsity_pct']:<9.1f}% "
              f"{m['mem_mb']:<10.3f} {m['spectral_rho']:<10.2f} "
              f"{m['step_us']:<10.2f} {m['mops']:<10.1f} {st}")

        _append_manifest(manifest, all_reports, model, paths)

    # ------------------------------------------------------------------ write reports
    mf_path  = os.path.join(models_dir, "models_grid_manifest.json")
    rpt_path = os.path.join(models_dir, "matrix_benchmark_report.json")
    with open(mf_path,  "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    with open(rpt_path, "w", encoding="utf-8") as f:
        json.dump(all_reports, f, indent=2)

    print("\n" + "=" * 90)
    print(f"Done in {time.time() - t0_global:.1f} s")
    print(f"Manifest : {mf_path}")
    print(f"Report   : {rpt_path}")


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def _print_header():
    print(f"  {'REDUCER':<14} {'K':<7} {'RAM(MB)':<9} {'SPARSITY':<10} "
          f"{'ρ(W)':<10} {'STEP(µs)':<10} {'MFLOP/s':<10} STATUS")
    print("  " + "─" * 86)


def _print_row(name: str, k: int, model: ReducedModel):
    m  = model.metrics
    st = "OK" if m.get("stable_linear", True) else "DIVERGED"
    print(f"  {name:<14} {k:<7} {m['mem_mb']:<9.3f} "
          f"{m['sparsity_pct']:<9.1f}% {m['spectral_rho']:<10.2f} "
          f"{m['step_us']:<10.2f} {m['mops']:<10.1f} {st}")


def _append_manifest(manifest: dict, reports: list,
                     model: ReducedModel, paths: dict):
    entry = {
        "reducer":      model.reducer_name,
        "k":            model.k,
        "platform":     model.platform_hint,
        "w_file":       os.path.basename(paths["w"]),
        "meta_file":    os.path.basename(paths["meta"]),
        "cmap_file":    os.path.basename(paths["cmap"]),
        "sensor_index_map": model.sensor_index_map,
        "motor_index_map":  model.motor_index_map,
        "metrics":      model.metrics,
    }
    manifest["models"].append(entry)
    reports.append(entry)


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_reduction()