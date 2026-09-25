# Chong Fly: Biologically Grounded Drone Autopilot (Drosophila Connectome & CfC)

Autonomous drone flight control architecture based on the biological connectome of the *Drosophila melanogaster* fruit fly. The project translates whole-brain connectomic pathways into high-frequency closed-loop flight reflex controllers deployable across edge devices, microcontrollers, and neural accelerators.

---

## 🌟 Architecture Overview

The neural architecture integrates two functional subsystems extracted directly from the FlyWire whole-brain connectome dataset:

1. **Optomotor Reflex Pathway (Lobula Plate & Medulla)**:
   - **400 visual columns per eye** (800 bilateral columns) providing spatial-temporal filtering and elementary motion detection.
   - **Columnar Filters**: $Mi1, Mi4, Mi9$ (ON-pathway fast delay / GABAergic / glutamatergic filters) and $Tm1, Tm2, Tm3, Tm4, Tm9$ (OFF-pathway transmedullary channels).
   - **Directional Motion**: Elementary Motion Detectors ($T4a$–$d$ for light edges, $T5a$–$d$ for dark edges).
   - **Spatial Flow Integration**: Horizontal System ($HSN, HSE, HSS$) and Vertical System ($VS1$–$VS10$) Tangential Cells (LPTC).
   - **Looming Threat Detection**: Lobula columnar $LC4$ and $LPLC2$ neurons driven by ToF 8×8 depth grids.
   - **Commissural & Stabilizing Interneurons**: Centrifugal Horizontal ($CH, dCH, vCH$) for bilateral inhibition and $LPi$ cross-layer directional gating.

2. **Central Complex (CX) Navigation & Steering Core** *(optional, flag-controlled)*:
   - **Ellipsoid Body (EB)**: 16-wedge ring attractor compass ($E\text{-}PG$) and phase shifters ($P\text{-}EN$) maintaining continuous 360° allocentric heading estimation.
   - **Protocerebral Bridge (PB)**: Glomerular lateral inhibition matrix ($\Delta 7$).
   - **Fan-shaped Body (FB)**: Allocentric goal-heading translation units ($P\text{-}FN$) driving premotor steering drivers ($P\text{-}FL$).
   - **Descending Premotor Drivers**: Direct projections onto descending command neurons for dynamic yaw correction, bank/roll stabilization, and looming escape thrust.

---

## ⚙️ Configurable Extraction Pipeline

Cell-type selection, sensor/motor group definitions, and biological exclusions are fully externalised to **`configs/cell_mapping.json`** — no code changes required to adapt the circuit topology.

### Sensor Input Groups

| Sensor Channel | Biological Target | Cell Types |
| :---: | :---: | :--- |
| `FlowX`, `FlowY` | Large-Field Tangential Cells (LPTC) | `HSN, HSE, HSS, VS1`–`VS10` |
| `ToF_8x8` (64 px) | Looming Detectors (LC) | `LC4, LPLC2` |

### Motor Output Groups (Descending Neurons)

| Actuator Channel | Wiring Mode | Cell Types |
| :---: | :---: | :--- |
| **Throttle** | Bilateral | `DNp09, DNp01` |
| **Yaw** | Differential L/R | `DNa01, DNa02` |
| **Pitch / Roll** | Bilateral | `DNp02, DNp04, DNp11, DNa11` |

### Biological Exclusion Filter

The pipeline automatically drops neurons belonging to **Mushroom Body** (KCab, MBON, APL), **Neuroendocrine** (DAN, OAN, Am), **Courtship** (fru-M/F, dsx, P1a, pC1/pC2), and **primary Lamina** (L1–L5) super-classes — preserving only sensorimotor-relevant pathways.

### Central Complex Toggle

```bash
# Default — include CX (set in configs/cell_mapping.json)
python bio_pipeline/circuit_extractor.py

# Force-disable CX (hypothesis: reflex-only controller)
python bio_pipeline/circuit_extractor.py --no-cx

# Force-enable CX regardless of config
python bio_pipeline/circuit_extractor.py --include-cx

# Custom config file
python bio_pipeline/circuit_extractor.py --config path/to/my_mapping.json
```

---

## 🔬 Multi-Scale Graph Reduction & Benchmarks

Three **object-oriented compression strategies** inherit from a common `BaseReducer` interface and output standardised `ReducedModel` objects containing both the weight matrix and sensor/motor index maps for seamless CfC integration.

### Reduction Strategies

| Strategy | Class | Output Format | Key Parameter |
| :--- | :--- | :--- | :--- |
| Spectral Clustering + SVD | `SpectralReducer` | dense `float32` `.npy` | `k` — macro-cluster count |
| Degree / Centrality Condensation | `CentralityReducer` | dense `float32` `.npy` | `k` — macro-cluster count |
| Global Magnitude Pruning | `MagnitudePruner` | sparse `csr_matrix` `.npz` | sparsity percentile 50–95 % |

### Standardised Output per Model (3 files)

```
data/reduced_models/
 ├── w_{reducer}_{tag}.npy / .npz   ← weight matrix (dense or CSR sparse)
 ├── cmap_{reducer}_{tag}.npy       ← cluster_map[N] — node → cluster id
 └── meta_{reducer}_{tag}.json      ← sensor_index_map, motor_index_map,
                                        metrics, platform_hint
```

The `sensor_index_map` and `motor_index_map` inside each `meta_*.json` provide the **exact cluster indices** of LPTC/LC sensor inputs and DN motor outputs in the reduced space — directly consumable by the CfC controller as `input_indices` / `output_indices`.

### Extraction Metrics (latest run)
- **Total Neurons ($|V|$)**: 12,942 (Optic Tract: 12,860, Central Complex: 82)
- **Synaptic Edges ($|E|$)**: 32,400
- **Total Synapses**: 702,655
- **Symmetry**: 100% Bilateral Retained Topology

### SpectralReducer Grid (11 Models)

The Laplacian spectral embedding collapses the 12.9 k biological graph into macro-cluster interaction matrices $W^{\text{macro}} \in \mathbb{R}^{K \times K}$:

| Model ($K$) | Sparsity | RAM | $\rho(W)$ | Step Latency | GEMV | Target Platform |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **16** | 2.0 % | 1.2 KB | 34.48 | **3.93 µs** (~254 kHz) | 130 MFLOP/s | Ultra-Low Power MCU (ESP32, Cortex-M4) |
| **32** | 22.9 % | 4.2 KB | 33.71 | **4.02 µs** (~249 kHz) | 509 MFLOP/s | MCU / Flight Control Interposer |
| **64** | 57.1 % | 17 KB | 34.43 | **4.19 µs** (~239 kHz) | 1,954 MFLOP/s | SBC (RPi Zero 2W) |
| **128** | 83.4 % | 65 KB | 36.42 | **4.90 µs** (~204 kHz) | 6,683 MFLOP/s | SBC (Raspberry Pi 4/5, CM4) |
| **256** | 94.2 % | 257 KB | 42.80 | **8.44 µs** (~119 kHz) | 15,536 MFLOP/s | Embedded Edge (NVIDIA Jetson Nano) |
| **512** | 98.3 % | 1.1 MB | 42.79 | **17.27 µs** (~58 kHz) | 30,357 MFLOP/s | Embedded Edge (Jetson Orin Nano) |
| **1024** | 99.5 % | 4.1 MB | 48.54 | **24.68 µs** (~41 kHz) | 84,976 MFLOP/s | High-Perf Edge (ModalAI VOXL 2) |
| **2048** | 99.3 % | 16 MB | 44.04 | **1.59 ms** (~630 Hz) | 5,279 MFLOP/s | High-Perf Edge / GPU Compute Unit |
| **4096** | 99.8 % | 64 MB | 29.74 | **7.02 ms** (~142 Hz) | 4,777 MFLOP/s | Workstation GPU Flight Companion |
| **8192** | 100.0 % | 256 MB | 100.95 | **23.04 ms** (~43 Hz) | 5,827 MFLOP/s | Onboard High-End GPU (RTX 3050 Ti) |
| **12942** | 100.0 % | 639 MB | 155.36 | **56.62 ms** (~18 Hz) | 5,917 MFLOP/s | 1:1 Biological Ground Truth (Sparse CUDA) |

### MagnitudePruner Grid (6 Sparsity Levels, CSR)

| Prune % | Retained NNZ | `.npz` size | $\rho(W)$ |
| :---: | :---: | :---: | :---: |
| 50 % | 17,416 | 55 KB | 150.10 |
| 60 % | 13,964 | 46 KB | 150.36 |
| 70 % | 10,441 | 36 KB | 150.65 |
| 80 % | 6,883 | 25 KB | 153.88 |
| 90 % | 3,305 | 11 KB | 155.01 |
| **95 %** | **1,685** | **6.3 KB** | 153.46 |

---

## 📂 Repository Structure

```text
chong-fly/
├── configs/
│   └── cell_mapping.json        # External cell-type ontology, sensor/motor groups,
│                                #   exclusion rules, CX toggle, synapse weight ranges
├── bio_pipeline/
│   ├── circuit_extractor.py     # Configurable connectome query + synthetic bilateral
│   │                            #   generator; --no-cx / --include-cx CLI flags
│   ├── graph_reducer.py         # BaseReducer + SpectralReducer + CentralityReducer
│   │                            #   + MagnitudePruner; ReducedModel dataclass w/ save/load
│   ├── models.py                # BiologicalCfCCell (ODE-free CfC) + BiologicalCfCNetwork
│   │                            #   + build_network_from_meta(); W_macro fixed/masked/free modes
│   ├── test_cfc_dynamics.py     # Pre-evolutionary dynamical validation (33/33 ✓)
├── data/
│   ├── raw_connectome/          # Extracted topology, metadata, and cell class registries
│   │   ├── raw_nodes.csv
│   │   ├── raw_edges.parquet
│   │   └── circuit_summary.json
│   └── reduced_models/          # Compressed multi-scale matrices + metadata side-cars
│       ├── w_{reducer}_{tag}.npy / .npz
│       ├── cmap_{reducer}_{tag}.npy
│       ├── meta_{reducer}_{tag}.json
│       ├── models_grid_manifest.json
│       └── matrix_benchmark_report.json
├── deployment/                  # Target firmware cross-compilation & ONNX/C exporters
├── firmware/                    # Embedded real-time execution kernels (C/C++ & MicroPython)
│   ├── esp32_main/
│   └── cm3_daemon/
├── simulation/                  # Sensor→PWM translation, SITL & Drone Environment
│   ├── drone_env.py             # 6-DOF DroneSimulationEnv w/ Betaflight Cascaded PID & ToF/Flow Sensors
│   ├── avionics_filter.py       # BetaflightCascadedPID (Angle P, Rate PID, PT1 filter, Quad-X mixer)
│   ├── pmw3901_emulator.py      # PMW3901 optical flow sensor (surface velocity & gyro derotation)
│   ├── policy.py                # ChongFlyMSPPolicy: SensorInputLayer + CfC + DNProjectionHead + PWM
│   ├── test_drone_env.py        # Environment & avionics validation suite (18/18 ✓)
│   └── test_policy.py           # Policy validation suite (44/44 ✓)
└── training/                    # Optuna Search Space, Multi-Level Funnel & Simulation
    ├── env.py                   # 6-DOF DroneSimulationEnv & simulate_policy_rollout
    ├── evaluate.py              # Level 1 math_screening + Level 2 evaluate_simulation_behavior
    ├── optuna_tuner.py          # Multi-objective Optuna search & Pareto manifest exporter
    └── test_optuna_search_space.py # Search space & funnel test suite (50/50 ✓)
```

---

## 🧠 Biological CfC Neural Core

[`bio_pipeline/models.py`](bio_pipeline/models.py) implements the **Closed-Form Continuous-time** (CfC) recurrent cell derived from the Liquid Time-constant (LTC) ODE, with no numerical ODE solver required.

### Closed-Form Update (no Runge-Kutta)

The full LTC dynamics reduce to a single analytical step:

$$x(t{+}\Delta t) = \underbrace{\sigma\!\bigl[-(f + \tfrac{1}{\tau})\,\Delta t\bigr]}_{\text{time-aware decay gate}} \cdot x(t) \;+\; \bigl(1 - \sigma[\ldots]\bigr) \cdot A \cdot \tanh(f)$$

where $f = \text{MLP}([x;\,u]) + W_{\text{macro}}\cdot x + W_{\text{in}}\cdot u + b$ combines the non-linear backbone, the biological recurrent matrix $W_{\text{macro}}$, and the sensory projection.

- $\tau = \text{softplus}(\tau_{\text{raw}}) > 0$ — per-neuron membrane time constants (always positive)
- $A$ — learnable asymptotic attractor amplitude
- Default $\Delta t = 4\,\text{ms}$ → **250 Hz** closed-loop rate

### W_macro Topology Control (three modes)

| Mode | Behaviour | Use Case |
| :--- | :--- | :--- |
| `fixed` | Frozen buffer — zero gradient, maximum inference speed | Hardware deployment, ablation |
| `masked` | Trainable + backward hook zeros gradients outside the pruning mask; `apply_topology_mask()` hard-zeros after every `optimizer.step()` | Training with biological wiring constraints |
| `free` | Fully trainable — no structural prior | Baseline comparison |

### Usage

```python
from bio_pipeline.models import BiologicalCfCCell, build_network_from_meta

# One-call factory from a ReducedModel meta file
net = build_network_from_meta(
    meta_path       = "data/reduced_models/meta_spectral_k64.json",
    input_size      = 66,          # 2 flow + 64 ToF
    output_dim      = 4,
    mode            = "masked",    # enforce pruned topology
    backbone_units  = 64,
    backbone_layers = 2,
    dt              = 0.004,       # 250 Hz
)

# Single step (250 Hz flight loop)
h_new = net.cell(sensor_tensor, hx, dt=0.004)

# After optimizer.step() — re-enforce sparsity
net.post_step()
```

---

## 🎮 ChongFlyMSPPolicy — Sensor → RC PWM

[`simulation/policy.py`](simulation/policy.py) is the full sensor-to-actuator translation layer that maps raw sensor data to standard RC PWM microsecond commands.

### Pipeline

```
SensorInputLayer          66 inputs: [FlowX, FlowY] + [ToF_00 … ToF_63]
      ↓                   learnable per-channel gain & bias
BiologicalCfCCell         recurrent CfC core (W_macro biological topology)
      ↓
DNProjectionHead          sparse readout — only descending neuron clusters active
      ↓
PWMOutputLayer            affine transform to RC PWM microseconds
```

### Affine PWM Transform

| Channel | Activation | Formula | Range |
| :---: | :---: | :---: | :---: |
| Throttle | `sigmoid` | $1000 + \sigma(x)\times 1000$ | 1000–2000 µs |
| Roll | `tanh` | $1500 + \tanh(x)\times 500$ | 1000–2000 µs |
| Pitch | `tanh` | $1500 + \tanh(x)\times 500$ | 1000–2000 µs |
| Yaw | `tanh` | $1500 + \tanh(x)\times 500$ | 1000–2000 µs |

Neutral (uninitialised network): throttle ≈ 1500 µs (50 %), attitude channels ≈ 1500 µs.

### DNProjectionHead — Biologically Constrained Readout

Motor commands are projected exclusively from the descending-neuron (DN) cluster indices stored in `motor_index_map`. All other hidden units are masked to zero in the output projection — mirroring the biological pathway from DN populations to thoracic motor circuits.

```
throttle  ← cluster [50]
yaw       ← clusters [42, 50]
pitch/roll← cluster [50]
```

### Flight-Loop API

```python
from simulation.policy import ChongFlyMSPPolicy
import numpy as np

policy = ChongFlyMSPPolicy.from_meta(
    "data/reduced_models/meta_spectral_k64.json",
    mode="masked", dt=0.004,
)

policy.reset_state()       # call at episode / flight start

# 250 Hz loop
while flying:
    pwm = policy.step_np(
        flow_xy = pmw3901.read(),           # (2,)  ∈ [-1, 1]
        tof_8x8 = vl53l5cx.read_frame(),    # (64,) ∈ [0, 1]
    )
    # pwm → [throttle_µs, roll_µs, pitch_µs, yaw_µs]
    msp.send_rc(pwm)
```

---

## 🎯 Optuna Architecture Search & Multi-Level Pareto Funnel

[`training/optuna_tuner.py`](training/optuna_tuner.py) implements the automated architecture search and multi-objective Pareto optimization pipeline designed to discover energy-efficient, robust biological neural controllers.

### 1. Search Space (Section 4.1)

| Parameter | Type | Domain / Values | Description |
| :--- | :--- | :--- | :--- |
| `k_clusters` | Categorical | `[32, 64, 128, 256]` | Spectral reduction cluster count ($W^{\text{macro}} \in \mathbb{R}^{K \times K}$) |
| `pruning_sparsity` | Float | `[0.50, 0.95]` | Dynamic biological synapse magnitude pruning |
| `solver_type` | Categorical | `['CfC', 'Euler_dt_0.02']` | Biological closed-form decay gate vs explicit Euler baseline ($\Delta t = 0.02\,\text{s}$) |
| `ablate_cx` | Categorical | `[True, False]` | Central Complex ablation (EB, PB, FB) testing reflex-only flight |

### 2. Multi-Level Filtering Funnel (Objective Function)

Candidate architectures undergo a two-tier evaluation to maximize compute throughput and discard unstable configurations early:

```
                  ┌──────────────────────────────┐
                  │   Sample Trial Parameters    │
                  └──────────────┬───────────────┘
                                 │
                                 ▼
         ┌────────────────────────────────────────────────┐
         │ Level 1: Mathematical Screening O(1)/O(NNZ)    │
         │  • NaN/Inf weight and time constant check      │
         │  • Sparsity tolerance (|s_actual - s_target|<=5%)│
         │  • Euler stability check (tau_min >= dt / 10)  │
         │  • Gershgorin circle spectral bound (rho<=1.5) │
         └───────────────────────┬────────────────────────┘
                                 │
                     ┌───────────┴───────────┐
                  Pass                      Fail (< 1 ms)
                     │                         │
                     ▼                         ▼
         ┌───────────────────────┐   ┌────────────────────┐
         │ Level 2: 6-DOF Drone  │   │ raise              │
         │ Simulation O(N)       │   │ optuna.            │
         │  • FlowX/Y + 8x8 ToF  │   │ TrialPruned()      │
         │  • Obstacle avoidance │   └────────────────────┘
         │  • Altitude & tilt    │
         │  • Energy integration │
         └───────────┬───────────┘
                     │
                     ▼
         ┌────────────────────────────────────────────────┐
         │ Multi-Objective Pareto Frontier Evaluator      │
         │  1. Maximize: survival_time_s                  │
         │  2. Minimize: energy_cost_j                    │
         └────────────────────────────────────────────────┘
```

- **Level 1 (`math_screening`, $\mathcal{O}(1) / \mathcal{O}(\text{NNZ})$)**: Instant check executing in $< 1\,\text{ms}$. If numerical instability, spectral divergence, or invalid sparsity is detected, the trial calls `raise optuna.TrialPruned()` without touching the physics simulator.
- **Level 2 (`evaluate_simulation_behavior`, $\mathcal{O}(N)$)**: Deploys the neural policy inside [`training/env.py`](training/env.py) (`DroneSimulationEnv`). Evaluates flight duration $T_{\text{surv}}$, actuator energy consumption $E_{\text{flight}} = \int P(t)\,dt$, altitude error, and PWM jitter.

### 3. Pareto Frontier & Manifest Export

The multi-objective study identifies non-dominated trade-offs between flight endurance and electrical power demand. Pareto-optimal models are exported to:
📄 **`data/filtered_models_manifest.json`**

Each entry includes exact model filenames, sparsity, solver configuration, spectral radius, survival time, energy cost, and platform hardware targets (e.g. ESP32, Cortex-M4, Jetson Nano).

---

## ⚡ Quick Start

### 1. Environment Setup
```bash
git clone https://github.com/ShadowC707/chong-fly.git
cd chong-fly

python -m venv venv
# Linux/macOS:
source venv/bin/activate
# Windows:
.\venv\Scripts\activate

pip install numpy pandas scipy scikit-learn torch numba pyarrow caveclient optuna
```

### 2. Extract Connectome Topology
```bash
# Full circuit (Optomotor + CX)
python bio_pipeline/circuit_extractor.py

# Optomotor-only (disable CX navigation core)
python bio_pipeline/circuit_extractor.py --no-cx
```

### 3. Run Multi-Scale Reduction & Benchmark
Executes all three reduction strategies (Spectral, Centrality, Magnitude) and writes
standardised matrices with sensor/motor connectivity metadata:
```bash
python bio_pipeline/graph_reducer.py
```

### 4. Load a Reduced Model Programmatically
```python
from bio_pipeline.graph_reducer import ReducedModel

# Load spectral K=64 model
model = ReducedModel.load("data/reduced_models/meta_spectral_k64.json")

print(model.W.shape)                   # (64, 64)
print(model.sensor_index_map)          # {'lptc_flow': [1,4,7,...], 'lc_looming': [7,29]}
print(model.motor_index_map)           # {'throttle': [50], 'yaw': [42,50], 'pitch_roll': [50]}

# Load magnitude-pruned sparse model (p=90)
sparse_model = ReducedModel.load("data/reduced_models/meta_magnitude_p90_k12942.json")
# sparse_model.W is a scipy.sparse.csr_matrix with 3,305 NNZ
```

### 5. Validate Biological CfC Dynamics
Runs 33 dynamical tests validating analytical non-ODE integration, gradient propagation, and topological synaptic masking:
```bash
python bio_pipeline/test_cfc_dynamics.py
```

### 6. Run Policy Pipeline & PWM Verification
Runs 44 validation tests across the full sensor-to-actuator pipeline (66 sensor inputs, DN sparse readout, and [1000, 2000] µs RC PWM output):
```bash
python simulation/test_policy.py
```

### 7. Run Drone Simulation & Avionics Validation
Runs 18 validation tests for the 6-DOF simulation environment, Betaflight cascaded PID (Angle P + Rate PID + Quad-X mixer), 8x8 ToF raycaster, and PMW3901 optical flow sensor:
```bash
python simulation/test_drone_env.py
```

### 8. Run Optuna Search Space & Funnel Validation
Runs 50 validation tests covering the Optuna Search Space (`k_clusters` $\in [32, 64, 128, 256]$, `pruning_sparsity` $\in [0.50, 0.95]$, `solver_type` $\in [\text{'CfC'}, \text{'Euler\_dt\_0.02'}]$, and `ablate_cx` $\in [\text{True}, \text{False}]$), Level 1 math screening pruning, and simulation evaluation:
```bash
python training/test_optuna_search_space.py
```

### 9. Run Optuna Multi-Objective Tuning & Export Pareto Manifest
Executes multi-objective optimization (survival time vs energy expenditure) across the 2-level filtering funnel and exports the non-dominated Pareto front:
```bash
# Run 50 trials and export Pareto manifest
python training/optuna_tuner.py --n-trials 50 --pareto --export data/filtered_models_manifest.json

# Quick sanity run (5 trials)
python training/optuna_tuner.py --n-trials 5 --pareto
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
