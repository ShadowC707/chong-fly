# Chong-Fly

Graph-constrained fruit-fly connectome navigation research, with a frozen k306
rate model, standalone simulation and a verified NumPy offline runtime.
**This repository does not yet provide a hardware flight controller or flight qualification.**

## Current MVP

- Exact research model: `data/selection/mvp_frozen_core_v1/trial_00003/after`.
- Exact NumPy bundle: `data/edge/k306_tau025_integrated_v1`.
- Configuration: [`configs/mvp_runtime_v3.json`](configs/mvp_runtime_v3.json).
- Portable offline release: `data/releases/k306_tau025_offline_v1`.
- XY anti-rollback + turn latch, STOP enabled, Levy disabled; weights unchanged.

Read the [project map](docs/README.md), [integration contracts and results](docs/MODULE_INTEGRATION_UK.md),
[reasons for MVP decisions](docs/MVP_DECISIONS_UK.md), and [cleanup/recovery record](docs/REPOSITORY_CLEANUP_UK.md), and [migration history](docs/GIT_HISTORY_UK.md).

## Offline self-check (NumPy only)

Use Python 3.10+ in a dedicated environment; the current host validation uses
Python 3.12. No Torch, simulator or repository imports are needed for this command:

```powershell
python -m pip install -r data/releases/k306_tau025_offline_v1/bundle/requirements.txt
python -I data/releases/k306_tau025_offline_v1/run.py --self-check
```

It checks synthetic telemetry through the receiver, expiry and explicit session
reset. It does not open a network connection, arm motors or simulate flight.
The packaged bytes and checksums must remain unchanged; `.gitattributes` protects
them from Git newline conversion.

## Research and physical simulation

Use the project's research environment, or install the host dependencies into
an environment of your choice. CUDA training needs a compatible PyTorch build;
this cleanup does not replace or reconfigure your existing installation.

```powershell
python -m pip install -r requirements.txt -r requirements-experiments.txt
python -m pytest tests -q
python -m edge.system --config configs/mvp_runtime_v3.json --manifest-out resolved_mvp.json
python -m evaluate.validate_transport_loop --config configs/mvp_runtime_v3.json --out-dir data/diagnostics/transport_clean_check
```

The last command runs the 14 receiver/transport scenarios in standalone physics;
choose a new output directory each time. It uses the committed NumPy bundle,
not local training datasets. To package another offline release:

```powershell
python -m edge.release --config configs/mvp_runtime_v3.json --out-dir data/releases/offline_repeat
```

`evaluate.watch_mvp` visualizes the exact research checkpoint. Its direct rollout
omits FlightBridge and the receiver; it is not a substitute for the transport test.
FlyWire acquisition has additional dependencies in `requirements-connectome.txt`;
CAVE credentials and full source/training datasets are not included in Git.

## Evidence and limitations

The previous module refactor reproduced 8000 control ticks bit-for-bit and all
14 reference physical trajectories. See the [recorded report](data/diagnostics/module_boundaries_v1/report.json).
The frozen model's wide-stimulus joint criterion was **3/4**, not 4/4; it remains
a research candidate. Real acquisition, scheduler timing, clock synchronization,
Wi-Fi/FC adapters, firmware fallback and hardware tests are still outstanding.

Only selected small artifacts and summary reports are committed. New bulk
experiment outputs are ignored. The reconstructed migration history omits 231
historical output files, preserved in local recovery archives. Full retraining
and exact historical trace comparison require
additional local artifacts, as explained in the cleanup record.

The [previous README](docs/history/README_BEFORE_CLEANUP.md) is preserved for history.
Its old architecture claims and benchmark tables do not describe current flight readiness.
