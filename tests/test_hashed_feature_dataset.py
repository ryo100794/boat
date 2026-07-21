from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse
from sklearn.feature_extraction import FeatureHasher

from boatrace_ai.hashed_feature_dataset import (
    CACHE_VERSION,
    FEATURE_SCHEMA_VERSION,
    _stack_csr_balanced,
    build_hashed_dataset,
    load_or_build_hashed_dataset,
    load_hashed_dataset,
    race_ids_sha256,
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
    assert manifest["feature_schema_version"] == FEATURE_SCHEMA_VERSION
    assert manifest["race_ids_sha256"] == race_ids_sha256(race_keys)
    assert manifest["ranks_sha256"]
    assert manifest["matrix_file_sha256"]
    assert manifest["hasher"] == {
        "class": "sklearn.feature_extraction.FeatureHasher",
        "n_features": 256,
        "input_type": "dict",
        "alternate_sign": False,
        "dtype": "float64",
    }


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


def test_hashed_dataset_can_build_without_writing_cache(tmp_path: Path) -> None:
    race_keys = [
        ("2026-01-01-01-01", "2026-01-01", "01", 1),
        ("2026-01-02-01-01", "2026-01-02", "01", 1),
    ]
    dataset, source = load_or_build_hashed_dataset(
        cache_prefix=tmp_path / "features",
        race_keys=race_keys,
        race_rows=_race_rows,
        hasher=FeatureHasher(
            n_features=256, input_type="dict", alternate_sign=False
        ),
        to_hashable=lambda row: row,
        ensure_sparse_index32=_ensure_index32,
        drop_feature_groups=(),
        batch_size=7,
        write_cache=False,
    )
    assert source == "built"
    assert dataset.race_count == 2
    assert not list(tmp_path.iterdir())


def _saved_dataset(tmp_path: Path, *, n_features: int = 64):
    race_keys = [
        (f"2026-01-0{day}-01-01", f"2026-01-0{day}", "01", 1)
        for day in range(1, 4)
    ]
    rows = [
        [
            {
                "features": {"lane": float(lane), f"day_{day}": 1.0},
                "meta": {
                    "race_id": race_keys[day - 1][0],
                    "lane": lane,
                    "rank": lane,
                },
            }
            for lane in range(1, 7)
        ]
        for day in range(1, 4)
    ]
    hasher = FeatureHasher(
        n_features=n_features,
        input_type="dict",
        alternate_sign=False,
    )
    dataset = build_hashed_dataset(
        race_keys=race_keys,
        race_rows=rows,
        hasher=hasher,
        to_hashable=lambda row: row,
        ensure_sparse_index32=_ensure_index32,
        drop_feature_groups=(),
        batch_size=7,
    )
    prefix = tmp_path / "features"
    save_hashed_dataset(prefix, dataset)
    return prefix, race_keys, hasher


def _load(prefix: Path, race_keys, *, hasher=None):
    return load_hashed_dataset(
        prefix,
        race_keys=race_keys,
        n_features=64,
        drop_feature_groups=(),
        hasher=hasher,
    )


def test_hashed_dataset_rejects_middle_race_identity_change(tmp_path: Path) -> None:
    prefix, race_keys, hasher = _saved_dataset(tmp_path)
    changed = list(race_keys)
    changed[1] = ("different-race", race_keys[1][1], "01", 1)
    assert _load(prefix, changed, hasher=hasher) is None


def test_hashed_dataset_rejects_truncated_npz_without_raising(tmp_path: Path) -> None:
    prefix, race_keys, hasher = _saved_dataset(tmp_path)
    matrix_path = Path(f"{prefix}.matrix.npz")
    matrix_path.write_bytes(matrix_path.read_bytes()[:32])
    assert _load(prefix, race_keys, hasher=hasher) is None


def test_hashed_dataset_rejects_modified_ranks(tmp_path: Path) -> None:
    prefix, race_keys, hasher = _saved_dataset(tmp_path)
    ranks_path = Path(f"{prefix}.ranks.npy")
    ranks = np.load(ranks_path, allow_pickle=False)
    ranks[1] = np.roll(ranks[1], 1)
    np.save(ranks_path, ranks, allow_pickle=False)
    assert _load(prefix, race_keys, hasher=hasher) is None


@pytest.mark.parametrize(
    ("manifest_key", "replacement"),
    (("matrix_shape", [17, 64]), ("matrix_nnz", -1), ("ranks_shape", [2, 6])),
)
def test_hashed_dataset_rejects_manifest_shape_or_nnz_mismatch(
    tmp_path: Path,
    manifest_key: str,
    replacement,
) -> None:
    prefix, race_keys, hasher = _saved_dataset(tmp_path)
    manifest_path = Path(f"{prefix}.manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[manifest_key] = replacement
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert _load(prefix, race_keys, hasher=hasher) is None


def test_hashed_dataset_rejects_matrix_nnz_different_from_manifest(
    tmp_path: Path,
) -> None:
    prefix, race_keys, hasher = _saved_dataset(tmp_path)
    matrix_path = Path(f"{prefix}.matrix.npz")
    manifest_path = Path(f"{prefix}.manifest.json")
    matrix = sparse.load_npz(matrix_path).tocsr()
    matrix.data = matrix.data[:-1]
    matrix.indices = matrix.indices[:-1]
    matrix.indptr[-1] -= 1
    sparse.save_npz(matrix_path, matrix, compressed=False)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["matrix_file_sha256"] = hashlib.sha256(matrix_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert _load(prefix, race_keys, hasher=hasher) is None


def test_hashed_dataset_rejects_hasher_schema_and_legacy_version(tmp_path: Path) -> None:
    prefix, race_keys, _hasher = _saved_dataset(tmp_path)
    different_hasher = FeatureHasher(
        n_features=64,
        input_type="dict",
        alternate_sign=True,
    )
    assert _load(prefix, race_keys, hasher=different_hasher) is None
    assert load_hashed_dataset(
        prefix,
        race_keys=race_keys,
        n_features=64,
        drop_feature_groups=(),
        feature_schema_version="different-schema",
    ) is None

    manifest_path = Path(f"{prefix}.manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["cache_version"] = 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert _load(prefix, race_keys) is None


def test_hashed_dataset_atomic_replace_commits_manifest_last(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import boatrace_ai.hashed_feature_dataset as module

    prefix, race_keys, hasher = _saved_dataset(tmp_path)
    dataset = _load(prefix, race_keys, hasher=hasher)
    assert dataset is not None
    destinations: list[str] = []
    real_replace = module.os.replace

    def recording_replace(source, destination):
        destinations.append(Path(destination).name)
        real_replace(source, destination)

    monkeypatch.setattr(module.os, "replace", recording_replace)
    save_hashed_dataset(prefix, dataset)

    assert destinations == [
        "features.matrix.npz",
        "features.ranks.npy",
        "features.manifest.json",
    ]
    assert not list(tmp_path.glob(".features.*.tmp"))


def test_hashed_dataset_partial_replace_is_rejected_by_old_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import boatrace_ai.hashed_feature_dataset as module

    prefix, race_keys, hasher = _saved_dataset(tmp_path)
    dataset = _load(prefix, race_keys, hasher=hasher)
    assert dataset is not None
    changed_matrix = dataset.matrix.copy()
    changed_matrix.data[0] += 1.0
    changed = replace(dataset, matrix=changed_matrix)
    real_replace = module.os.replace

    def fail_before_ranks_commit(source, destination):
        if Path(destination).name == "features.ranks.npy":
            raise OSError("simulated interruption")
        real_replace(source, destination)

    monkeypatch.setattr(module.os, "replace", fail_before_ranks_commit)
    with pytest.raises(OSError, match="simulated interruption"):
        save_hashed_dataset(prefix, changed)

    assert _load(prefix, race_keys, hasher=hasher) is None
    assert not list(tmp_path.glob(".features.*.tmp"))
