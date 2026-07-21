from __future__ import annotations

import json
from pathlib import Path

from scipy import sparse
from sklearn.feature_extraction import FeatureHasher

from boatrace_ai.hashed_feature_dataset import (
    CACHE_VERSION,
    _stack_csr_balanced,
    build_hashed_dataset,
    load_hashed_dataset,
    save_hashed_dataset,
)


def _ensure_index32(matrix):
    result = matrix.tocsr(copy=False)
    result.indices = result.indices.astype("int32", copy=False)
    result.indptr = result.indptr.astype("int32", copy=False)
    return result


def _race_rows():
    for day in range(1, 3):
        race_id = f"2026-01-0{day}-01-01"
        yield [
            {
                "features": {"lane": float(lane), f"lane_{lane}": 1.0},
                "meta": {"race_id": race_id, "lane": lane, "rank": lane},
            }
            for lane in range(1, 7)
        ]


def test_hashed_dataset_build_save_and_reload(tmp_path: Path) -> None:
    race_keys = [
        ("2026-01-01-01-01", "2026-01-01", "01", 1),
        ("2026-01-02-01-01", "2026-01-02", "01", 1),
    ]
    hasher = FeatureHasher(
        n_features=256,
        input_type="dict",
        alternate_sign=False,
    )
    dataset = build_hashed_dataset(
        race_keys=race_keys,
        race_rows=_race_rows(),
        hasher=hasher,
        to_hashable=lambda row: row,
        ensure_sparse_index32=_ensure_index32,
        drop_feature_groups=(),
        batch_size=7,
    )
    assert dataset.matrix.shape == (12, 256)
    assert dataset.ranks.tolist() == [list(range(1, 7)), list(range(1, 7))]
    assert dataset.row_slice(1, 2) == slice(6, 12)

    prefix = tmp_path / "features"
    save_hashed_dataset(prefix, dataset)
    loaded = load_hashed_dataset(
        prefix,
        race_keys=race_keys,
        n_features=256,
        drop_feature_groups=(),
    )
    assert loaded is not None
    assert loaded.matrix.shape == dataset.matrix.shape
    assert loaded.matrix.nnz == dataset.matrix.nnz
    assert loaded.ranks.tolist() == dataset.ranks.tolist()
    manifest = json.loads((tmp_path / "features.manifest.json").read_text())
    assert manifest["cache_version"] == CACHE_VERSION


def test_hashed_dataset_rejects_stale_race_identity(tmp_path: Path) -> None:
    race_keys = [("2026-01-01-01-01", "2026-01-01", "01", 1)]
    hasher = FeatureHasher(n_features=64, input_type="dict", alternate_sign=False)
    dataset = build_hashed_dataset(
        race_keys=race_keys,
        race_rows=[next(_race_rows())],
        hasher=hasher,
        to_hashable=lambda row: row,
        ensure_sparse_index32=_ensure_index32,
        drop_feature_groups=(),
        batch_size=6,
    )
    prefix = tmp_path / "features"
    save_hashed_dataset(prefix, dataset)

    changed = [("2026-01-02-01-01", "2026-01-02", "01", 1)]
    assert (
        load_hashed_dataset(
            prefix,
            race_keys=changed,
            n_features=64,
            drop_feature_groups=(),
        )
        is None
    )


def test_balanced_csr_stack_preserves_row_order_and_values() -> None:
    batches = [
        sparse.csr_matrix([[float(index), 1.0, float(index % 2)]])
        for index in range(17)
    ]
    result = _stack_csr_balanced(
        batches,
        ensure_sparse_index32=_ensure_index32,
    )
    expected = sparse.vstack(
        [
            sparse.csr_matrix([[float(index), 1.0, float(index % 2)]])
            for index in range(17)
        ],
        format="csr",
    )
    assert batches == []
    assert result.shape == expected.shape
    assert (result != expected).nnz == 0
    assert result.indices.dtype.name == "int32"
    assert result.indptr.dtype.name == "int32"
