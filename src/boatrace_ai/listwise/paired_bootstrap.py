from __future__ import annotations

from typing import Sequence

import numpy as np


def paired_mean_bootstrap(
    differences: Sequence[float],
    *,
    samples: int = 20_000,
    seed: int = 20260722,
    chunk_size: int = 1_000,
) -> dict[str, float | int]:
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("differences must be a non-empty vector")
    if not np.all(np.isfinite(values)):
        raise ValueError("differences must be finite")
    if samples < 100:
        raise ValueError("samples must be at least 100")
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    step = max(1, int(chunk_size))
    for start in range(0, samples, step):
        stop = min(samples, start + step)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    lower, upper = np.quantile(means, (0.025, 0.975))
    return {
        "observations": len(values),
        "samples": int(samples),
        "mean_difference": float(values.mean()),
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
        "probability_less_than_zero": float(np.mean(means < 0.0)),
        "probability_greater_than_zero": float(np.mean(means > 0.0)),
    }
