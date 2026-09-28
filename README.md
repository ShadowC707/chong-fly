# Chong Fly: Biologically Grounded Drone Autopilot (Drosophila Connectome & CfC)

Autonomous drone flight control architecture based on the biological connectome of the *Drosophila melanogaster* fruit fly.
The project translates whole-brain connectomic pathways into high-frequency closed-loop flight reflex controllers deployable across edge devices, microcontrollers, and neural accelerators.

---

## 🌟 Architecture Overview

The neural architecture integrates two functional subsystems extracted directly from the FlyWire whole-brain connectome dataset:

1. **Optomotor Reflex Pathway (Lobula Plate & Medulla)**:
   - **400 visual columns per eye** (800 bilateral columns) providing spatial-temporal filtering and elementary motion detection.
   - **Columnar Filters**: $Mi1, Mi4, Mi9$ (ON-pathway fast delay / GABAergic / glutamatergic filters) and $Tm1, Tm2, Tm3, Tm4, Tm9$ (OFF-pathway transmedullary channels).
   - **Directional Motion**: Elementary Motion Detectors ($T4a-d$ for light edges, $T5a-d$ for dark edges).
   - **Spatial Flow Integration**: Horizontal System ($HSN, HSE, HSS$) and Vertical System ($VS1..VS10$) Tangential Cells.
   - **Commissural & Stabilizing Interneurons**: Centrifugal Horizontal ($CH, dCH, vCH$) for bilateral inhibition and $LPi$ cross-layer directional gating.

2. **Central Complex (CX) Navigation & Steering Core**:
   - **Ellipsoid Body (EB)**: 16-wedge ring attractor compass ($E	ext{-}PG$) and phase shifters ($P	ext{-}EN$) maintaining continuous 360° allocentric heading estimation.
   - **Protocerebral Bridge (PB)**: Glomerular lateral inhibition matrix ($\Delta 7$).
   - **Fan-shaped Body (FB)**: Allocentric goal-heading translation units ($P	ext{-}FN$) driving premotor steering drivers ($P	ext{-}FL$).
   - **Descending Premotor Drivers**: Direct projections onto descending command neurons ($DNa01..03, DNb01, DNp01..02$) for dynamic yaw correction, bank/roll stabilization, and looming escape thrust.

---

## 🔬 Multi-Scale Graph Reduction & Benchmarks

To deploy the biological connectome across constrained embedded hardware (from ultra-low-power MCUs to onboard workstation GPUs), we implemented a multi-scale spectral reduction pipeline accelerated via **Numba JIT** and **Randomized SVD**.

### Extraction Metrics
- **Total Neurons ($|V|$)**: 12,934 (Optic Tract: 12,852, Central Complex: 82)
- **Synaptic Edges ($|E|$)**: 32,392
- **Total Synapses**: 686,087
- **Symmetry**: 100% Bilateral Retained Topology

### Multi-Scale Grid (11 Target Models)

The reduction collapses the 12.9k biological graph into 11 macro-cluster interaction matrices ($W^{\text{macro}} \in \mathbb{R}^{K \times K}$):

| Model ($K$) | Sparsity (%) | RAM Footprint | Spectral Radius $\rho(W)$ | Step Latency | GEMV Throughput | Target Deployment Platform |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$K = 16$** | 3.9% | 1.0 KB | 33.60 | **3.94 µs** (~250 kHz) | 130.1 MFLOP/s | Ultra-Low Power MCU (ESP32, Cortex-M4) |
| **$K = 32$** | 19.3% | 4.0 KB | 33.69 | **4.38 µs** (~228 kHz) | 467.4 MFLOP/s | MCU / Flight Control Interposer |
| **$K = 64$** | 53.0% | 16.0 KB | 33.47 | **4.25 µs** (~235 kHz) | 1,927.0 MFLOP/s | Single Board Computer (RPi Zero 2W) |
| **$K = 128$** | 83.5% | 64.0 KB | 34.09 | **6.14 µs** (~162 kHz) | 5,335.1 MFLOP/s | SBC (Raspberry Pi 4/5, CM4) |
| **$K = 256$** | 94.9% | 256.0 KB | 35.63 | **8.37 µs** (~119 kHz) | 15,656.4 MFLOP/s | Embedded Edge (NVIDIA Jetson Nano) |
| **$K = 512$** | 98.3% | 1.0 MB | 44.12 | **21.30 µs** (~47 kHz) | 24,618.9 MFLOP/s | Embedded Edge (Jetson Orin Nano) |
| **$K = 1024$** | 99.5% | 4.0 MB | 43.99 | **217.39 µs** (~4.6 kHz) | 9,647.0 MFLOP/s | High-Perf Autonomous Edge (ModalAI VOXL 2) |
| **$K = 2048$** | 99.3% | 16.0 MB | 42.50 | **1.14 ms** (~870 Hz) | 7,347.8 MFLOP/s | High-Perf Edge / GPU Compute Unit |
| **$K = 4096$** | 99.8% | 64.0 MB | 41.23 | **4.75 ms** (~210 Hz) | 7,057.2 MFLOP/s | Workstation GPU Flight Companion |
| **$K = 8192$** | 100.0% | 256.0 MB | 66.98 | **18.07 ms** (~55 Hz) | 7,426.8 MFLOP/s | Onboard High-End GPU (RTX 3050 Ti / 4060) |
| **$K = 12934$** | 100.0% | 638.1 MB | 173.47 | **46.38 ms** (~21 Hz) | 7,213.1 MFLOP/s | 1:1 Biological Ground Truth (Sparse CUDA) |

---

## 📂 Repository Structure

```text
chong-fly/
├── benchmarks/                  # Hardware profiling and spectral benchmarks
│   ├── check_spectral.py
│   └── hardware_benchmark.py
├── configs/                     # Hyperparameters and biological mappings
│   └── cell_mapping.json
├── core/                        # Biological Neural Network logic
│   └── models.py                # Biological Closed-Form Continuous-Time (CfC) neural core
├── data/
│   ├── raw_connectome/          # Extracted topology, metadata, and cell class registries
│   │   ├── raw_nodes.csv
│   │   ├── raw_edges.parquet
│   │   └── circuit_summary.json
│   └── reduced_models/          # Compressed multi-scale interaction matrices & manifests
│       ├── w_macro_k16.npy ... w_macro_k12934.npy
│       ├── models_grid_manifest.json
│       └── matrix_benchmark_report.json
├── edge/                        # Deployment and real-time execution
│   ├── embedded/                # Embedded real-time execution kernels (C/C++ & MicroPython)
│   └── exporter/                # Target firmware cross-compilation & ONNX/C header exporters
├── generator/
│   ├── circuit_extractor.py     # Connectome query & synthetic bilateral generator (Optomotor + CX)
│   └── graph_reducer.py         # Numba JIT spectral Laplacian reduction & linear benchmark
├── optimizer/                   # Evolutionary Optuna tuner & metrics evaluator
│   ├── evaluate.py              # Evaluator & O(N) tuning environment wrapper
│   └── optuna_tuner.py
├── simulation/                  # Sensor→PWM translation & Isaac Gym / MSP Drone Environment
│   ├── avionics_filter.py
│   ├── drone_env.py
│   ├── drone_interface.py
│   ├── isaac_hud.py
│   ├── memory.py
│   ├── metrics.py
│   ├── pmw3901_emulator.py
│   └── policy.py
└── tests/                       # Infrastructure health checks and PyTest suites
    ├── test_cfc_dynamics.py
    ├── test_drone_env.py
    ├── test_optuna_search_space.py
    └── test_policy.py
```

---

## ⚡ Quick Start

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/ShadowC707/chong-fly.git
cd chong-fly

# Create virtual environment and install dependencies
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install numpy pandas scipy scikit-learn torch numba pyarrow
```

### 2. Extract Connectome Topology
```bash
python generator/circuit_extractor.py
```

### 3. Generate Multi-Scale Model Grid & Benchmark
Runs parallel spectral embedding and synaptic condensation with high-performance linear throughput profiling:
```bash
python generator/graph_reducer.py
```

---

## 🛣️ Roadmap: Two-Phase Co-Evolutionary Strategy

The next phase deploys the generated matrices into a closed-loop 16-room spatial navigation environment:

- **Phase A (Locomotion & Reflex Coupling)**:
  - Freeze internal membrane dynamics.
  - Optimize premotor synaptic projections ($W_{\text{out}}$) using CMA-ES for stable attitude stabilization, drift rejection, and barrier repulsion.
- **Phase B (Neuromorphic & Computational Efficiency)**:
  - Freeze motor couplings.
  - Optimize Closed-Form Continuous-time (CfC) membrane parameters (time constants $\boldsymbol{\tau}$, liquid modulation $\boldsymbol{\alpha}$, and gating sparsity) to minimize computational overhead and maximize event-driven sparsity without sacrificing control authority.
