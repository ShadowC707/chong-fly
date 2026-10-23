# Active reduced candidates

Measured FlyWire 783 research candidates are installed in `flywire783_filtered_v2/`.
Its `registry.json` is the explicit candidate inventory: anatomical k6, rejected
interface k242, and structurally passing bridge k301/k306. All have
`training_ready=false`; source-quality and engineering assumptions remain open.
For preparation and validation see `generator.prepare_flywire_candidates` and
`generator.validate_flywire_candidates` (`--help`). Full validation requires
local measured source data; the checkout does not contain all bulk source files.

The repeated `flywire783_filtered_repeat/` build was archived during cleanup.
All four weight matrices, four cluster maps and six raw source tables were
byte-identical to v2. JSON differences were timestamps and dependent provenance
hashes. Do not mix metadata from those builds when recovering the repeat.

The chosen trained checkpoint is separately preserved at
`../selection/mvp_frozen_core_v1/trial_00003/after/`. Its graph is part of its
hash-checked bundle and is deliberately retained.

Historical synthetic models were moved to `archive/legacy-artifacts-20261006/data/reduced_models/`.
The recovery inventory is `data/artifact_archive_20261006.json`.

Only install deliberately selected, versioned candidates here after validating
their source and routes. The tiny synthetic test examples are generated in
temporary directories by `tests/synthetic_graphs.py`; do not copy them here.

Missing candidate metadata raises an error by default. Cleanup does not mean
that the real FlyWire source is already ready for training or flight. See the
[current project entrypoints](../../README.md) and
[cleanup classification and recovery](../../docs/CLEANUP_COMPLETION_UK.md).
