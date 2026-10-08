# Chong Fly: Drosophila Connectome Navigation Research

The [brake-first and sensor geometry report](TEACHER_GEOMETRY_UK.md) covers
teacher v4, correct downward cylinder-cap hits and box-face normals, stable
scene seeds, and a shared comparison runner. Across 128 ten-second scenes,
v4 had no collisions and three telemetry stops, including one new regression.
Close-clearance cases and range surface transitions still prevent admission.

The [teacher braking report](TEACHER_BRAKING_UK.md) describes the latest
versioned teacher fix and matched 96-scene comparison: collisions fell from
8 to 1, with two telemetry faults still unresolved. Historical v2 demonstrations
remain available for exact replay but are incompatible with current pretraining.
No replacement training dataset or newly admitted model was produced.

The [spatial calibration and teacher replay report](SPATIAL_CALIBRATION_UK.md)
documents the latest twelve-case admission gate, measured width dependence,
an experimental divisive encoding and exact reproduction of ten terminated
teacher episodes. Simple input normalization did not resolve the longer-run
amplitude failures; the default encoding remains unchanged.

The [current registry selection guide](REGISTRY_SELECTION_UK.md) provides the
executable research pipeline, configuration, reproducibility checks and measured
results for k301/k306. It exports each trial's exact weights and graph, and
excludes candidates that fail functional admission. The first comparison
completed all seven short physical scenes for both graphs, but neither passed
the distance/amplitude gate. No candidate has scientific or flight admission.

The [navigation learning report](NAVIGATION_LEARNING_UK.md) documents explicit
threat encoding, a neutral-origin option, channel-specific yaw balancing,
temporal response checks, two trained seeds and named physical challenge scenes.
Its runs provide the historical comparison for the current guide above.

The [k306 learning diagnostic](CANDIDATE_DIAGNOSTICS_UK.md) now includes
held-out metrics, explicit research-only pretraining, optional preservation of
initial synaptic signs, and a verified checkpoint. Candidates remain unapproved
for a large search; short-run improvement is not flight validation.

Flight evaluation now uses `flight-benchmark-v8`, with physical yaw-burst metrics, navigation-only learning,
a shared altitude supervisor, corrected coordinates,
versioned geometric reflex demonstrations, explicit input/output routes
and no dense recurrent bypass in structured mode.
Read the [current navigation training contract](NAVIGATION_TRAINING_UK.md),
[CAVE acquisition and source integrity contract](CONNECTOME_SOURCE_UK.md),
[coordinate and pretraining history](COORDINATES_AND_PRETRAIN_UK.md),
[reduction contract](REDUCTION_CONTRACT_UK.md),
[routing contract](ROUTING_CONTRACT_UK.md), [neural contract](NEURAL_CONTRACT_UK.md), and the
[simulation benchmark contract](BENCHMARK_V2_UK.md) before continuing Optuna.
Earlier scores and trained checkpoints are incompatible with the new dynamics.
The historical synthetic graph, reduced candidates, v1/v2 demonstrations and
Optuna database have been moved to a reversible local archive. New measured
candidates are isolated under `data/reduced_models/flywire783_filtered_v2/`. See the
[artifact cleanup record](ARTIFACT_CLEANUP_UK.md) for the inventory and recovery details.
Tests now generate tiny explicitly synthetic graphs in temporary directories;
they do not load active flight candidates. Real FlyWire sources and v3
geometric demonstrations remain available. Historical architecture claims and
benchmark tables below are not validation of flight readiness.
The [filtered FlyWire 783 preparation report](FILTERED_CANDIDATES_783_UK.md)
documents official filtered acquisition, four versioned candidates and offline
verification. The k301/k306 graphs pass required directed-route checks; k242 is
rejected for missing ToF→yaw. All remain `training_ready=false`, with explicit
source-quality and engineering-mapping questions. Use the commands in that
report for this measured candidate set; older grids below are historical.
The archived structured candidates failed the teacher-route check (ToF→yaw is
absent in the source graph as well as k128/k256). The [route audit](data/sensor_route_audit_v1.json)
distinguishes source/mapping gaps from routes lost or introduced by reduction.
The [range altitude controller](ALTITUDE_CONTROL_UK.md) is now included in
demonstration collection and the standard objective. Historical raw v6 and
assisted v6 results are excluded from v7 studies. Resolve the source/mapping
gap before a large Optuna search; no biological edges have been invented.

The [flight candidate registry and directed audit](FLIGHT_CANDIDATES_UK.md)
now verify a separate real FlyWire 783 LPLC2→DNp06 projection: 1,442 synapses
across 201 directed neuron pairs. This is anatomical evidence for a candidate
pathway; its ToF encoding and RC decoder remain unvalidated. It does not replace
the archived synthetic model artifacts or certify their routes.

The [saccade circuit audit](SACCADE_CIRCUIT_UK.md) resolves DNae014/DNb01
from author root IDs and identifies substantial same-root counts in raw CAVE
data. A verified synapse quality policy is required before training on these
real graphs. The [GL5528 sensor plan](LIGHT_SENSORS_UK.md) records the proposed
light channels; they are not yet part of the policy observation contract.

Autonomous drone flight control architecture based on the biological connectome of the *Drosophila melanogaster* fruit fly.
The project translates whole-brain connectomic pathways into high-frequency closed-loop flight reflex controllers deployable across edge devices, microcontrollers, and neural accelerators.

---

## Historical Architecture Proposal

The architecture claims, grid sizes, benchmark numbers and roadmap below are
historical material. The active controller is a graph-constrained leaky-rate
model; the current measured candidates and launch commands are documented in
[the registry selection guide](REGISTRY_SELECTION_UK.md).

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
Acquisition now requires an explicit source and writes a new directory. For
measured CAVE data, use a pinned version and verified table names as described
in [the source contract](CONNECTOME_SOURCE_UK.md). For a synthetic pipeline check:
```bash
python -m generator.circuit_extractor --source synthetic --seed 42
```

### 3. Generate the Current Candidate Grid
Preserves individual sensor/motor neurons and separates other cells by type,
side, layer and outgoing sign profile. For the current synthetic source:
```bash
python -m generator.role_reducer --data-dir data/raw_connectome_synthetic_seed42 --out-dir data/reduced_models/synthetic_source_v1 --k 128 256
```
Acquisition provenance is inherited from the source manifest. Historical raw
files require explicit `--allow-legacy-source`; a source flag cannot promote
them to CAVE data. Existing default candidates are kept for historical comparison.

### 4. Collect and Check Reflex Demonstrations
```bash
python -m generator.generate_reflex_dataset --episodes 64 --seq_len 100 --seed 42
python -m pytest tests -q -p no:cacheprovider
```
This creates `data/reflex_dataset_v3.pt` plus a JSON collection report. Old
unversioned demonstrations and trained checkpoints require regeneration/retraining.

---

## 🛣️ Roadmap: Two-Phase Co-Evolutionary Strategy

The next phase deploys the generated matrices into a closed-loop 16-room spatial navigation environment:

- **Phase A (Locomotion & Reflex Coupling)**:
  - Freeze internal membrane dynamics.
  - Optimize premotor synaptic projections ($W_{\text{out}}$) using CMA-ES for stable attitude stabilization, drift rejection, and barrier repulsion.
- **Phase B (Neuromorphic & Computational Efficiency)**:
  - Freeze motor couplings.
  - Optimize Closed-Form Continuous-time (CfC) membrane parameters (time constants $\boldsymbol{\tau}$, liquid modulation $\boldsymbol{\alpha}$, and gating sparsity) to minimize computational overhead and maximize event-driven sparsity without sacrificing control authority.
