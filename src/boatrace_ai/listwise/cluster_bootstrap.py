from __future__ import annotations

from typing import Hashable, Sequence

import numpy as np


def paired_cluster_mean_bootstrap(
    differences: Sequence[float],
    cluster_labels: Sequence[Hashable],
    *,
    samples: int = 20_000,
    seed: int = 20260726,
    chunk_size: int = 2_000,
) -> dict[str, float | int]:
    values = np.asarray(differences, dtype=np.float64)
    labels = list(cluster_labels)
    if values.ndim != 1 or not len(values):
        raise ValueError("differences must be a non-empty vector")
    if len(labels) != len(values):
        raise ValueError("cluster labels must match differences")
    if not np.all(np.isfinite(values)):
        raise ValueError("differences must be finite")
    if samples < 100:
        raise ValueError("samples must be at least 100")

    cluster_index: dict[Hashable, int] = {}
    indices = np.empty(len(labels), dtype=np.int64)
    for row_index, label in enumerate(labels):
        try:
            index = cluster_index.setdefault(label, len(cluster_index))
        except TypeError as exc:
            raise ValueError("cluster labels must be hashable") from exc
        indices[row_index] = index
    cluster_count = len(cluster_index)
    sums = np.bincount(indices, weights=values, minlength=cluster_count)
    counts = np.bincount(indices, minlength=cluster_count).astype(np.float64)

    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    step = max(1, int(chunk_size))
    for start in range(0, samples, step):
        stop = min(samples, start + step)
        sampled = rng.integers(
            0,
            cluster_count,
            size=(stop - start, cluster_count),
        )
        sampled_sums = sums[sampled].sum(axis=1)
        sampled_counts = counts[sampled].sum(axis=1)
        means[start:stop] = sampled_sums / sampled_counts
    lower, upper = np.quantile(means, (0.025, 0.975))
    return {
        "observations": len(values),
        "clusters": cluster_count,
        "samples": int(samples),
        "mean_difference": float(values.mean()),
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
        "probability_less_than_zero": float(np.mean(means < 0.0)),
        "probability_greater_than_zero": float(np.mean(means > 0.0)),
    }
