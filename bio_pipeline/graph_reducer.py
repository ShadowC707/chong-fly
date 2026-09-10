import os
import sys
import json
import time
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.utils.extmath import randomized_svd
from sklearn.cluster import MiniBatchKMeans
import numba
from numba import njit, prange

POLARITY_MAP = {
    # Excitatory (+1)
    "T4a": 1, "T4b": 1, "T4c": 1, "T4d": 1,
    "T5a": 1, "T5b": 1, "T5c": 1, "T5d": 1,
    "Tm1": 1, "Tm2": 1, "Tm3": 1, "Tm4": 1, "Tm9": 1,
    "Mi1": 1, "HSN": 1, "HSE": 1, "HSS": 1,
    "E-PG": 1, "P-EN": 1, "P-FN": 1, "P-FL": 1,
    "DNa01": 1, "DNa02": 1, "DNa03": 1, "DNp01": 1, "DNp02": 1,
    # Inhibitory (-1)
    "Mi4": -1, "Mi9": -1, "CH": -1, "dCH": -1, "vCH": -1,
    "LPi1-2": -1, "LPi2-1": -1, "LPi3-4": -1, "LPi4-3": -1,
    "Delta7": -1
}
for i in range(1, 11):
    POLARITY_MAP[f"VS{i}"] = 1


# --- NUMBA JIT ACCELERATION ---

@njit(parallel=True, fastmath=True)
def normalize_rows_numba(matrix):
    n, m = matrix.shape
    out = np.empty_like(matrix)
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
def condense_synapses_numba(pre_indices, post_indices, signed_weights, clusters, k):
    w_macro = np.zeros((k, k), dtype=np.float32)
    cluster_sizes = np.zeros(k, dtype=np.float32)

    for i in range(len(clusters)):
        c = clusters[i]
        if c < k:
            cluster_sizes[c] += 1.0

    num_edges = len(pre_indices)
    for e in range(num_edges):
        u = pre_indices[e]
        v = post_indices[e]
        c_pre = clusters[u]
        c_post = clusters[v]
        w_macro[c_pre, c_post] += signed_weights[e]

    for p in range(k):
        denom = cluster_sizes[p]
        if denom < 1.0:
            denom = 1.0
        for q in range(k):
            w_macro[p, q] /= denom

    return w_macro


# --- PURE GRAPH & MATRIX BENCHMARK (NO CfC) ---

def benchmark_matrix_properties(W, k, num_trials=500):
    """
    Pure linear algebraic & topological benchmark on synthetic signal patterns.
    Measures matrix sparsity, spectral properties, and pure GEMV throughput.
    """
    # 1. Sparsity & Footprint
    total_elements = k * k
    zero_elements = np.sum(np.abs(W) < 1e-6)
    sparsity_pct = (zero_elements / total_elements) * 100.0
    mem_mb = W.nbytes / (1024 * 1024)

    # 2. Spectral Radius (Power Iteration)
    v = np.random.randn(k).astype(np.float32)
    v /= np.linalg.norm(v) + 1e-9
    for _ in range(25):
        v_next = W.dot(v)
        norm = np.linalg.norm(v_next)
        if norm < 1e-9:
            break
        v = v_next / norm
    spectral_rho = float(norm)

    # 3. Pure GEMV Latency Benchmark (Synthetic Optical Flow Injection)
    # Synthetic inputs: Left bias, Looming, Right bias, Noise
    x_test = np.random.randn(k).astype(np.float32)
    
    # Warmup
    for _ in range(20):
        _ = W.dot(x_test)

    t0 = time.perf_counter()
    for _ in range(num_trials):
        y = W.dot(x_test)
        # Recurrent feedback loop simulation (pure linear)
        x_test = y / (np.linalg.norm(y) + 1e-9)
    elapsed = time.perf_counter() - t0

    step_us = (elapsed / num_trials) * 1e6  # microseconds per pass
    throughput_mops = (2.0 * k * k * num_trials / elapsed) / 1e6  # MegaFLOPs/s

    # Dynamic stability check: does linear recursion diverge?
    diverged = np.isnan(y).any() or np.isinf(y).any() or float(np.max(np.abs(y))) > 1e6

    return {
        "mem_mb": float(mem_mb),
        "sparsity_pct": float(sparsity_pct),
        "spectral_rho": float(spectral_rho),
        "step_us": float(step_us),
        "mops": float(throughput_mops),
        "stable_linear": not diverged
    }


def run_reduction():
    print("=" * 85)
    print("   BIO-GRAPH REDUCTION & LINEAR MATRIX BENCHMARK (CfC EXCLUDED)")
    print("=" * 85, flush=True)
    t_global_start = time.time()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    data_dir = os.path.join(project_root, "data", "raw_connectome")
    models_dir = os.path.join(project_root, "data", "reduced_models")
    os.makedirs(models_dir, exist_ok=True)

    nodes_file = os.path.join(data_dir, "raw_nodes.csv")
    edges_file = os.path.join(data_dir, "raw_edges.parquet")

    if not os.path.exists(nodes_file) or not os.path.exists(edges_file):
        print(f"Error: Files missing in {data_dir}", flush=True)
        sys.exit(1)

    print("Ingesting bilateral connectome (Optomotor + CX)...", flush=True)
    nodes_df = pd.read_csv(nodes_file)
    edges_df = pd.read_parquet(edges_file)

    unique_nodes = nodes_df["root_id"].unique()
    node_to_idx = {nid: i for i, nid in enumerate(unique_nodes)}
    N = len(unique_nodes)

    nodes_df["idx"] = nodes_df["root_id"].map(node_to_idx)
    edges_df["pre_idx"] = edges_df["pre_id"].map(node_to_idx)
    edges_df["post_idx"] = edges_df["post_id"].map(node_to_idx)
    edges_df = edges_df.dropna(subset=["pre_idx", "post_idx"]).astype({"pre_idx": np.int32, "post_idx": np.int32})

    print(f"Graph Order: N = {N:,} nodes, |E| = {len(edges_df):,} edges, Synapses = {edges_df['weight'].sum():,}", flush=True)

    pre_indices = edges_df["pre_idx"].values
    post_indices = edges_df["post_idx"].values
    raw_weights = edges_df["weight"].values.astype(np.float32)

    polarities = nodes_df["cell_type"].map(POLARITY_MAP).fillna(1.0).values.astype(np.float32)
    signed_weights = raw_weights * polarities[pre_indices]

    # Graph Laplacian
    t0 = time.time()
    W_dir = sparse.csr_matrix((raw_weights.astype(np.float64), (pre_indices, post_indices)), shape=(N, N))
    A_sym = W_dir + W_dir.T
    degrees = np.array(A_sym.sum(axis=1)).flatten()
    degrees[degrees == 0] = 1e-9

    d_inv_sqrt = sparse.diags(1.0 / np.sqrt(degrees))
    L_sym = sparse.eye(N) - d_inv_sqrt.dot(A_sym).dot(d_inv_sqrt)

    # Spectral manifold via Randomized SVD
    svd_components = 64
    U, _, _ = randomized_svd(L_sym, n_components=svd_components, random_state=42, n_iter=4)
    embeddings_norm = normalize_rows_numba(U.astype(np.float32))

    requested_k = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    valid_k = [k for k in requested_k if k < N]
    valid_k.append(N)

    print("\n" + "=" * 90)
    print(f"{'MODEL':<8} | {'RAM (MB)':<9} | {'SPARSITY':<9} | {'RHO (EIG)':<10} | {'STEP (us)':<10} | {'MFLOP/s':<10} | {'STABLE'}")
    print("=" * 90)

    manifest = {"source_nodes": N, "source_edges": int(len(edges_df)), "models": []}
    benchmark_report = []

    for k in valid_k:
        if k == N:
            clusters = np.arange(N, dtype=np.int32)
            hw = "1:1 Full Connectome (Server)"
        elif k >= 2048:
            proj = np.ascontiguousarray(embeddings_norm[:, :min(k, svd_components)])
            hash_vals = np.sum(proj * np.arange(1, proj.shape[1] + 1, dtype=np.float32), axis=1)
            ranks = np.argsort(hash_vals)
            clusters = np.empty(N, dtype=np.int32)
            clusters[ranks] = np.linspace(0, k - 1, N).astype(np.int32)
            hw = "High-Perf Edge (VOXL 2 / GPU)" if k <= 4096 else "Workstation GPU (3050 Ti)"
        else:
            kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=4096, n_init=1, max_iter=20)
            clusters = kmeans.fit_predict(embeddings_norm[:, :min(32, svd_components)]).astype(np.int32)
            if k <= 32:
                hw = "MCU (ESP32 / Cortex-M4)"
            elif k <= 128:
                hw = "SBC (RPi Zero / CM3)"
            elif k <= 512:
                hw = "Edge Embedded (Jetson Nano)"
            else:
                hw = "High-Perf Edge (VOXL 2)"

        # Numba condensation
        W_macro = condense_synapses_numba(pre_indices, post_indices, signed_weights, clusters, k)

        out_path = os.path.join(models_dir, f"w_macro_k{k}.npy")
        np.save(out_path, W_macro)

        # Pure Linear Matrix Benchmark (No ODE / No CfC)
        bench = benchmark_matrix_properties(W_macro, k)
        status = "OK" if bench["stable_linear"] else "DIVERGED"

        print(f"K = {k:<4} | {bench['mem_mb']:<9.2f} | {bench['sparsity_pct']:<8.1f}% | {bench['spectral_rho']:<10.2f} | {bench['step_us']:<10.2f} | {bench['mops']:<10.1f} | {status}")

        model_entry = {
            "k": k,
            "shape": [k, k],
            "size_mb": bench["mem_mb"],
            "platform": hw,
            "filename": f"w_macro_k{k}.npy",
            "metrics": bench
        }
        manifest["models"].append(model_entry)
        benchmark_report.append(model_entry)

    with open(os.path.join(models_dir, "models_grid_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    report_path = os.path.join(models_dir, "matrix_benchmark_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_report, f, indent=2)

    print("=" * 90)
    print(f"Done in {time.time() - t_global_start:.2f} s. Report: {report_path}")


if __name__ == "__main__":
    run_reduction()