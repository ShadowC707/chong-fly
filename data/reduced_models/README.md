# Active reduced candidates

Measured FlyWire 783 research candidates are installed in `flywire783_filtered_v2/`.
Its `registry.json` is the explicit candidate inventory: anatomical k6, rejected
interface k242, and structurally passing bridge k301/k306. All have
`training_ready=false`; source-quality and engineering assumptions remain open.
See `FILTERED_CANDIDATES_783_UK.md` and the offline validation command there.

Historical synthetic models were moved to `archive/legacy-artifacts-20261006/data/reduced_models/`.
The recovery inventory is `data/artifact_archive_20261006.json`.

Only install deliberately selected, versioned candidates here after validating
their source and routes. The tiny synthetic test examples are generated in
temporary directories by `tests/synthetic_graphs.py`; do not copy them here.

Missing candidate metadata raises an error by default. Cleanup does not mean
that the real FlyWire source is already ready for training. See
`ARTIFACT_CLEANUP_UK.md` and `CONNECTOME_SOURCE_UK.md`.
