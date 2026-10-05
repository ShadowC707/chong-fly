"""Shared artifact semantics, independent of neural/graph dependencies."""
MATRIX_ORIENTATION = "source_target"  # W[source, target], batched row states h @ W
REDUCTION_NORMALIZATION = "target_mean"
REDUCED_FORMAT_VERSION = 2
DYNAMICS_VERSION = "leaky-rate-v1"
TAU_FLOOR_S = 1e-6


def canonical_solver(name):
    aliases = {"cfc": "exponential_euler", "exponential_euler": "exponential_euler",
               "euler": "euler", "euler_dt_0.02": "euler"}
    try:
        return aliases[name.lower()]
    except (KeyError, AttributeError) as exc:
        raise ValueError(f"Unknown solver {name!r}; use exponential_euler or euler") from exc
