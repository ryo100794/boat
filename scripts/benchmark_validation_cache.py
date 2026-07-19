#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time

from sklearn.feature_extraction import FeatureHasher


def synthetic_features(races: int, width: int) -> list[dict[str, float]]:
    return [
        {
            **{f"numeric_{index}": float((race + lane + index) % 37) / 37.0 for index in range(width)},
            f"lane={lane}": 1.0,
            f"venue={race % 24 + 1:02d}": 1.0,
        }
        for race in range(races)
        for lane in range(1, 7)
    ]


def benchmark(*, races: int, width: int, folds: int, n_features: int) -> dict:
    rows = synthetic_features(races, width)
    hasher = FeatureHasher(
        n_features=n_features,
        input_type="dict",
        alternate_sign=False,
    )

    started = time.perf_counter()
    repeated_nnz = 0
    for _ in range(folds):
        repeated_nnz += hasher.transform(rows).nnz
    repeated_seconds = time.perf_counter() - started

    started = time.perf_counter()
    cached = hasher.transform(rows).tocsr()
    cache_build_seconds = time.perf_counter() - started
    started = time.perf_counter()
    cached_nnz = 0
    window = max(1, cached.shape[0] // folds)
    for fold in range(folds):
        end = cached.shape[0] if fold == folds - 1 else min(
            cached.shape[0],
            (fold + 1) * window,
        )
        cached_nnz += cached[:end].nnz
    cache_reuse_seconds = time.perf_counter() - started
    cached_total = cache_build_seconds + cache_reuse_seconds

    return {
        "races": races,
        "rows": len(rows),
        "features_per_row": width + 2,
        "folds": folds,
        "n_features": n_features,
        "repeated_hash_seconds": round(repeated_seconds, 6),
        "cache_build_seconds": round(cache_build_seconds, 6),
        "cache_reuse_seconds": round(cache_reuse_seconds, 6),
        "cached_total_seconds": round(cached_total, 6),
        "end_to_end_speedup": round(repeated_seconds / cached_total, 3),
        "reuse_only_speedup": round(
            repeated_seconds / max(cache_reuse_seconds, 1e-12),
            3,
        ),
        "repeated_nnz": repeated_nnz,
        "cached_fold_nnz": cached_nnz,
        "cached_csr_bytes": int(
            cached.data.nbytes + cached.indices.nbytes + cached.indptr.nbytes
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark fold feature hashing vs CSR reuse.")
    parser.add_argument("--races", type=int, default=2_000)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--n-features", type=int, default=1 << 14)
    args = parser.parse_args()
    print(json.dumps(benchmark(**vars(args)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
