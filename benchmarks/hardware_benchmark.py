import os
import time
import json
import numpy as np
import scipy.sparse as sp

def cfc_step_numpy(x, u, W_macro, W_in, b, A, tau, dt=0.004):
    """Numpy implementation of CfC forward pass for accurate sparse benchmarking."""
    # f_bb is roughly a 2-layer MLP. For proxy benchmark, we simulate its latency
    # with two dense matrix multiplications (units=64).
    k = x.shape[0]
    input_size = u.shape[0]
    
    # Simulate backbone MLP (2 layers of 64 units)
    # W1: (k+input_size) x 64
    # W2: 64 x 64
    # W3: 64 x k
    # We'll just do dummy operations to match the FLOPs
    cat_in = np.concatenate([x, u])
    # MLP latency is dominated by matrix mults. We use random matrices.
    # To avoid generating random matrices inside the loop, we'll pre-generate them.
    pass

class ProxyBenchmark:
    def __init__(self, k, input_size=66, output_dim=4):
        self.k = k
        self.input_size = input_size
        self.bb_units = 64
        self.W1 = np.random.randn(self.bb_units, k + input_size).astype(np.float32)
        self.W2 = np.random.randn(self.bb_units, self.bb_units).astype(np.float32)
        self.W3 = np.random.randn(k, self.bb_units).astype(np.float32)
        self.W_in = np.random.randn(k, input_size).astype(np.float32)
        self.b = np.random.randn(k).astype(np.float32)
        self.A = np.random.randn(k).astype(np.float32)
        self.tau = np.abs(np.random.randn(k).astype(np.float32)) + 0.1
        self.dt = 0.004

    def simulate_forward(self, x, u, W_macro):
        # Backbone (simulated)
        cat_in = np.concatenate([x, u])
        h1 = np.tanh(self.W1 @ cat_in)
        h2 = np.tanh(self.W2 @ h1)
        f_bb = self.W3 @ h2
        
        # Recurrent + Input
        if sp.issparse(W_macro):
            f_rec = W_macro.dot(x)
        else:
            f_rec = W_macro @ x
            
        f_in = self.W_in @ u
        f = f_bb + f_rec + f_in + self.b
        
        # Gate & State update
        gate = 1.0 / (1.0 + np.exp((f + 1.0 / self.tau) * self.dt)) # sigmoid(- (f + 1/tau)*dt)
        h_inf = self.A * np.tanh(f)
        x_new = gate * x + (1.0 - gate) * h_inf
        return x_new

def run_benchmark():
    models_dir = "../data/reduced_models"
    manifest_path = os.path.join(models_dir, "models_grid_manifest.json")
    
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
        
    print(f"{'Model':<25} {'Static RAM':<12} {'Dyn RAM':<10} {'Latency(ms)':<12} {'Sparsity':<10} {'FPS'}")
    print("-" * 80)
    
    for m in manifest["models"]:
        w_path = os.path.join(models_dir, m["w_file"])
        if w_path.endswith('.npz'):
            W = sp.load_npz(w_path).astype(np.float32)
            is_sparse = True
        else:
            W = np.load(w_path).astype(np.float32)
            is_sparse = False
            
        k = m["k"]
        input_size = 66
        
        # 1. Sparsity Ratio
        total_elements = k * k
        if is_sparse:
            nnz = W.nnz
        else:
            nnz = np.count_nonzero(W)
        sparsity_ratio = (1.0 - nnz / total_elements) * 100
        
        # 2. Static Memory Calculation (Weights + Parameters)
        # Backbone weights
        bb_mem = (k + input_size) * 64 + 64 * 64 + 64 * k
        # W_in, b, A, tau
        param_mem = k * input_size + 3 * k
        
        if is_sparse:
            w_macro_mem = (W.data.nbytes + W.indices.nbytes + W.indptr.nbytes) / 4
        else:
            w_macro_mem = k * k
            
        static_mem_bytes = (bb_mem + param_mem + w_macro_mem) * 4
        static_mem_mb = static_mem_bytes / (1024 * 1024)
        
        # 3. Dynamic Memory (Activations)
        # x, u, cat_in, h1, h2, f_bb, f_rec, f_in, f, gate, h_inf, x_new
        activations_elements = k + input_size + (k + input_size) + 64 + 64 + k + k + k + k + k + k + k
        dynamic_mem_bytes = activations_elements * 4
        dynamic_mem_mb = dynamic_mem_bytes / (1024 * 1024)
        
        # 4. CPU Latency (Single Threaded)
        bench = ProxyBenchmark(k, input_size)
        x = np.random.randn(k).astype(np.float32)
        u = np.random.randn(input_size).astype(np.float32)
        
        # Warmup
        for _ in range(10):
            x = bench.simulate_forward(x, u, W)
            
        # Benchmark
        num_trials = 100 if k > 1000 else 1000
        t0 = time.perf_counter()
        for _ in range(num_trials):
            x = bench.simulate_forward(x, u, W)
        t1 = time.perf_counter()
        
        latency_ms = ((t1 - t0) / num_trials) * 1000.0
        fps = 1000.0 / latency_ms if latency_ms > 0 else 0
        
        tag = m["reducer"]
        print(f"{tag:<25} {static_mem_mb:<10.3f}MB {dynamic_mem_mb:<8.3f}MB {latency_ms:<10.3f}ms {sparsity_ratio:<8.2f}% {fps:.0f}")

if __name__ == "__main__":
    run_benchmark()
