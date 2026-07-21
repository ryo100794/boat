from __future__ import annotations

import gc
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
from scipy import sparse
from sklearn.feature_extraction import FeatureHasher


CACHE_VERSION = 1


@dataclass(frozen=True)
class HashedRaceDataset:
    matrix: sparse.csr_matrix
    race_keys: list[tuple[str, str, str, int]]
    ranks: np.ndarray
    n_features: int
    drop_feature_groups: tuple[str, ...]

    @property
    def race_count(self) -> int:
        return len(self.race_keys)

    @property
    def example_count(self) -> int:
        return int(self.matrix.shape[0])

    def row_slice(self, race_start: int, race_end: int) -> slice:
        return slice(max(0, race_start) * 6, max(0, race_end) * 6)


def load_or_build_hashed_dataset(
    *,
    cache_prefix: Path | None,
    race_keys: list[tuple[str, str, str, int]],
    race_rows: Callable[[], Iterable[list[dict[str, Any]]]],
    hasher: FeatureHasher,
    to_hashable: Callable[[dict[str, Any]], dict[str, float]],
    ensure_sparse_index32: Callable[[Any], sparse.csr_matrix],
    drop_feature_groups: tuple[str, ...],
    batch_size: int,
    write_cache: bool = True,
) -> tuple[HashedRaceDataset, str]:
    if cache_prefix is not None:
        loaded = load_hashed_dataset(
            cache_prefix,
            race_keys=race_keys,
            n_features=int(hasher.n_features),
            drop_feature_groups=drop_feature_groups,
        )
        if loaded is not None:
            return loaded, "disk"

    dataset = build_hashed_dataset(
        race_keys=race_keys,
        race_rows=race_rows(),
        hasher=hasher,
        to_hashable=to_hashable,
        ensure_sparse_index32=ensure_sparse_index32,
        drop_feature_groups=drop_feature_groups,
        batch_size=batch_size,
    )
    if cache_prefix is not None and write_cache:
        save_hashed_dataset(cache_prefix, dataset)
    return dataset, "built"


def build_hashed_dataset(
    *,
    race_keys: list[tuple[str, str, str, int]],
    race_rows: Iterable[list[dict[str, Any]]],
    hasher: FeatureHasher,
    to_hashable: Callable[[dict[str, Any]], dict[str, float]],
    ensure_sparse_index32: Callable[[Any], sparse.csr_matrix],
    drop_feature_groups: tuple[str, ...],
    batch_size: int,
) -> HashedRaceDataset:
    matrices: list[sparse.csr_matrix] = []
    batch: list[dict[str, float]] = []
    ranks: list[list[int]] = []
    observed_race_ids: list[str] = []

    def flush() -> None:
        if not batch:
            return
        matrices.append(ensure_sparse_index32(hasher.transform(batch)))
        batch.clear()

    for rows in race_rows:
        ordered = sorted(rows, key=lambda item: int(item["meta"]["lane"]))
        if len(ordered) != 6:
            continue
        race_id = str(ordered[0]["meta"]["race_id"])
        observed_race_ids.append(race_id)
        ranks.append([int(item["meta"]["rank"]) for item in ordered])
        batch.extend(to_hashable(item["features"]) for item in ordered)
        if len(batch) >= max(6, int(batch_size)):
            flush()
    flush()

    expected_ids = [race_id for race_id, *_ in race_keys]
    if observed_race_ids != expected_ids:
        raise ValueError(
            "hashed feature race order mismatch: "
            f"observed={len(observed_race_ids)} expected={len(expected_ids)}"
        )
    if not matrices:
        raise ValueError("no hashed feature rows")
    matrix = _stack_csr_balanced(
        matrices,
        ensure_sparse_index32=ensure_sparse_index32,
    )
    rank_matrix = np.asarray(ranks, dtype=np.int8)
    if matrix.shape[0] != len(race_keys) * 6 or rank_matrix.shape != (
        len(race_keys),
        6,
    ):
        raise ValueError(
            f"invalid hashed dataset shape: matrix={matrix.shape} ranks={rank_matrix.shape}"
        )
    return HashedRaceDataset(
        matrix=matrix,
        race_keys=race_keys,
        ranks=rank_matrix,
        n_features=int(hasher.n_features),
        drop_feature_groups=drop_feature_groups,
    )


def _stack_csr_balanced(
    matrices: list[sparse.csr_matrix],
    *,
    ensure_sparse_index32: Callable[[Any], sparse.csr_matrix],
) -> sparse.csr_matrix:
    """Stack and consume CSR batches without a large multi-input temporary."""
    current: list[sparse.csr_matrix | None] = list(matrices)
    matrices.clear()
    while len(current) > 1:
        merged: list[sparse.csr_matrix | None] = []
        for index in range(0, len(current), 2):
            left = current[index]
            current[index] = None
            if left is None:
                raise ValueError("missing CSR batch")
            if index + 1 >= len(current):
                merged.append(left)
                continue
            right = current[index + 1]
            current[index + 1] = None
            if right is None:
                raise ValueError("missing CSR batch")
            merged.append(
                ensure_sparse_index32(
                    sparse.vstack((left, right), format="csr"),
                )
            )
            del left, right
        current = merged
        gc.collect()
    result = current[0]
    if result is None:
        raise ValueError("no CSR batches")
    return ensure_sparse_index32(result)


def save_hashed_dataset(prefix: Path, dataset: HashedRaceDataset) -> None:
    paths = cache_paths(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(paths["matrix"], dataset.matrix, compressed=False)
    np.save(paths["ranks"], dataset.ranks, allow_pickle=False)
    first = dataset.race_keys[0][0] if dataset.race_keys else None
    last = dataset.race_keys[-1][0] if dataset.race_keys else None
    manifest = {
        "cache_version": CACHE_VERSION,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "race_count": dataset.race_count,
        "example_count": dataset.example_count,
        "first_race_id": first,
        "last_race_id": last,
        "n_features": dataset.n_features,
        "drop_feature_groups": list(dataset.drop_feature_groups),
        "matrix_shape": list(dataset.matrix.shape),
        "matrix_nnz": int(dataset.matrix.nnz),
    }
    paths["manifest"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_hashed_dataset(
    prefix: Path,
    *,
    race_keys: list[tuple[str, str, str, int]],
    n_features: int,
    drop_feature_groups: tuple[str, ...],
) -> HashedRaceDataset | None:
    paths = cache_paths(prefix)
    if not all(path.exists() for path in paths.values()):
        return None
    try:
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    expected_first = race_keys[0][0] if race_keys else None
    expected_last = race_keys[-1][0] if race_keys else None
    expected = {
        "cache_version": CACHE_VERSION,
        "race_count": len(race_keys),
        "example_count": len(race_keys) * 6,
        "first_race_id": expected_first,
        "last_race_id": expected_last,
        "n_features": int(n_features),
        "drop_feature_groups": list(drop_feature_groups),
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        return None
    try:
        matrix = sparse.load_npz(paths["matrix"]).tocsr(copy=False)
        ranks = np.load(paths["ranks"], allow_pickle=False)
    except (OSError, ValueError):
        return None
    if matrix.shape != (len(race_keys) * 6, int(n_features)):
        return None
    if ranks.shape != (len(race_keys), 6):
        return None
    return HashedRaceDataset(
        matrix=matrix,
        race_keys=race_keys,
        ranks=np.asarray(ranks, dtype=np.int8),
        n_features=int(n_features),
        drop_feature_groups=drop_feature_groups,
    )


def cache_paths(prefix: Path) -> dict[str, Path]:
    base = str(prefix)
    return {
        "matrix": Path(base + ".matrix.npz"),
        "ranks": Path(base + ".ranks.npy"),
        "manifest": Path(base + ".manifest.json"),
    }
