from __future__ import annotations

import gc
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
from scipy import sparse
from sklearn.feature_extraction import FeatureHasher

from .feature_schema import FEATURE_SCHEMA_VERSION


# Version 1 files are left in place and rejected per prefix; callers lazily rebuild
# them as version 2 instead of globally invalidating caches used by running jobs.
CACHE_VERSION = 2


@dataclass(frozen=True)
class HashedRaceDataset:
    matrix: sparse.csr_matrix
    race_keys: list[tuple[str, str, str, int]]
    ranks: np.ndarray
    n_features: int
    drop_feature_groups: tuple[str, ...]
    hasher_settings: dict[str, Any] | None = None
    feature_schema_version: str = FEATURE_SCHEMA_VERSION

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
    feature_schema_version: str = FEATURE_SCHEMA_VERSION,
) -> tuple[HashedRaceDataset, str]:
    if cache_prefix is not None:
        loaded = load_hashed_dataset(
            cache_prefix,
            race_keys=race_keys,
            n_features=int(hasher.n_features),
            drop_feature_groups=drop_feature_groups,
            hasher=hasher,
            feature_schema_version=feature_schema_version,
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
        feature_schema_version=feature_schema_version,
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
    feature_schema_version: str = FEATURE_SCHEMA_VERSION,
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
        hasher_settings=feature_hasher_settings(hasher),
        feature_schema_version=feature_schema_version,
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
    matrix = dataset.matrix.tocsr(copy=False)
    ranks = np.asarray(dataset.ranks)
    _validate_dataset_arrays(
        matrix,
        ranks,
        race_count=dataset.race_count,
        n_features=dataset.n_features,
    )
    temporary_paths: list[Path] = []
    matrix_temporary = _temporary_path(prefix, "matrix.npz")
    ranks_temporary = _temporary_path(prefix, "ranks.npy")
    manifest_temporary = _temporary_path(prefix, "manifest.json")
    temporary_paths.extend((matrix_temporary, ranks_temporary, manifest_temporary))
    try:
        with matrix_temporary.open("wb") as handle:
            sparse.save_npz(handle, matrix, compressed=False)
            _flush_and_fsync(handle)
        with ranks_temporary.open("wb") as handle:
            np.save(handle, ranks, allow_pickle=False)
            _flush_and_fsync(handle)

        first = dataset.race_keys[0][0] if dataset.race_keys else None
        last = dataset.race_keys[-1][0] if dataset.race_keys else None
        manifest = {
            "cache_version": CACHE_VERSION,
            "feature_schema_version": dataset.feature_schema_version,
            "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "race_count": dataset.race_count,
            "race_ids_sha256": race_ids_sha256(dataset.race_keys),
            "example_count": dataset.example_count,
            "first_race_id": first,
            "last_race_id": last,
            "n_features": dataset.n_features,
            "hasher": dataset.hasher_settings
            or default_feature_hasher_settings(dataset.n_features),
            "drop_feature_groups": list(dataset.drop_feature_groups),
            "matrix_shape": list(matrix.shape),
            "matrix_nnz": int(matrix.nnz),
            "matrix_dtype": str(matrix.dtype),
            "matrix_file_sha256": _file_sha256(matrix_temporary),
            "ranks_shape": list(ranks.shape),
            "ranks_dtype": str(ranks.dtype),
            "ranks_sha256": ranks_sha256(ranks),
            "ranks_file_sha256": _file_sha256(ranks_temporary),
        }
        with manifest_temporary.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            _flush_and_fsync(handle)

        os.replace(matrix_temporary, paths["matrix"])
        os.replace(ranks_temporary, paths["ranks"])
        _fsync_directory(prefix.parent)
        # The manifest is the commit marker and must become visible last.
        os.replace(manifest_temporary, paths["manifest"])
        _fsync_directory(prefix.parent)
    finally:
        for temporary_path in temporary_paths:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def load_hashed_dataset(
    prefix: Path,
    *,
    race_keys: list[tuple[str, str, str, int]],
    n_features: int,
    drop_feature_groups: tuple[str, ...],
    hasher: FeatureHasher | None = None,
    feature_schema_version: str = FEATURE_SCHEMA_VERSION,
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
        "feature_schema_version": feature_schema_version,
        "race_count": len(race_keys),
        "race_ids_sha256": race_ids_sha256(race_keys),
        "example_count": len(race_keys) * 6,
        "first_race_id": expected_first,
        "last_race_id": expected_last,
        "n_features": int(n_features),
        "hasher": feature_hasher_settings(hasher)
        if hasher is not None
        else default_feature_hasher_settings(n_features),
        "drop_feature_groups": list(drop_feature_groups),
        "matrix_shape": [len(race_keys) * 6, int(n_features)],
        "ranks_shape": [len(race_keys), 6],
        "ranks_dtype": "int8",
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        return None
    if not isinstance(manifest.get("matrix_nnz"), int) or manifest["matrix_nnz"] < 0:
        return None
    required_hashes = (
        "matrix_file_sha256",
        "ranks_file_sha256",
        "ranks_sha256",
    )
    if any(not _is_sha256(manifest.get(key)) for key in required_hashes):
        return None
    try:
        if _file_sha256(paths["matrix"]) != manifest["matrix_file_sha256"]:
            return None
        if _file_sha256(paths["ranks"]) != manifest["ranks_file_sha256"]:
            return None
        matrix = sparse.load_npz(paths["matrix"]).tocsr(copy=False)
        ranks = np.load(paths["ranks"], allow_pickle=False)
    except Exception:
        return None
    if list(matrix.shape) != manifest["matrix_shape"]:
        return None
    if int(matrix.nnz) != manifest["matrix_nnz"]:
        return None
    if str(matrix.dtype) != manifest.get("matrix_dtype"):
        return None
    if list(ranks.shape) != manifest["ranks_shape"]:
        return None
    if str(ranks.dtype) != manifest["ranks_dtype"]:
        return None
    if ranks_sha256(ranks) != manifest["ranks_sha256"]:
        return None
    try:
        _validate_dataset_arrays(
            matrix,
            ranks,
            race_count=len(race_keys),
            n_features=n_features,
        )
    except ValueError:
        return None
    return HashedRaceDataset(
        matrix=matrix,
        race_keys=race_keys,
        ranks=np.asarray(ranks, dtype=np.int8),
        n_features=int(n_features),
        drop_feature_groups=drop_feature_groups,
        hasher_settings=expected["hasher"],
        feature_schema_version=feature_schema_version,
    )


def promote_legacy_hashed_dataset(
    prefix: Path,
    *,
    race_keys: list[tuple[str, str, str, int]],
    n_features: int,
    drop_feature_groups: tuple[str, ...],
    hasher: FeatureHasher | None = None,
    feature_schema_version: str = FEATURE_SCHEMA_VERSION,
) -> bool:
    """Explicitly validate and promote one version 1 cache manifest to version 2."""
    paths = cache_paths(prefix)
    if not all(path.exists() for path in paths.values()):
        return False

    try:
        original_manifest = paths["manifest"].read_bytes()
        manifest = json.loads(original_manifest.decode("utf-8"))
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError):
        return False
    if not isinstance(manifest, dict) or manifest.get("cache_version") != 1:
        return False

    expected_race_count = len(race_keys)
    expected_example_count = expected_race_count * 6
    expected_first = race_keys[0][0] if race_keys else None
    expected_last = race_keys[-1][0] if race_keys else None
    expected_hasher = (
        feature_hasher_settings(hasher)
        if hasher is not None
        else default_feature_hasher_settings(n_features)
    )
    if expected_hasher["n_features"] != int(n_features):
        return False
    expected_legacy = {
        "cache_version": 1,
        "race_count": expected_race_count,
        "example_count": expected_example_count,
        "first_race_id": expected_first,
        "last_race_id": expected_last,
        "n_features": int(n_features),
        "drop_feature_groups": list(drop_feature_groups),
        "matrix_shape": [expected_example_count, int(n_features)],
    }
    if any(
        key not in manifest or not _strict_json_equal(manifest[key], value)
        for key, value in expected_legacy.items()
    ):
        return False
    legacy_nnz = manifest.get("matrix_nnz")
    if (
        not isinstance(legacy_nnz, int)
        or isinstance(legacy_nnz, bool)
        or legacy_nnz < 0
    ):
        return False

    try:
        matrix = _load_csr_npz_without_pickle(paths["matrix"])
        ranks = np.load(paths["ranks"], allow_pickle=False)
        _validate_dataset_arrays(
            matrix,
            ranks,
            race_count=expected_race_count,
            n_features=n_features,
        )
        if int(matrix.nnz) != legacy_nnz:
            return False
        matrix_file_sha256 = _file_sha256(paths["matrix"])
        ranks_file_sha256 = _file_sha256(paths["ranks"])
    except Exception:
        return False

    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    promoted_manifest = {
        "cache_version": CACHE_VERSION,
        "feature_schema_version": feature_schema_version,
        "generated_at": manifest.get("generated_at", now),
        "promoted_at": now,
        "race_count": expected_race_count,
        "race_ids_sha256": race_ids_sha256(race_keys),
        "example_count": expected_example_count,
        "first_race_id": expected_first,
        "last_race_id": expected_last,
        "n_features": int(n_features),
        "hasher": expected_hasher,
        "drop_feature_groups": list(drop_feature_groups),
        "matrix_shape": list(matrix.shape),
        "matrix_nnz": int(matrix.nnz),
        "matrix_dtype": str(matrix.dtype),
        "matrix_file_sha256": matrix_file_sha256,
        "ranks_shape": list(ranks.shape),
        "ranks_dtype": str(ranks.dtype),
        "ranks_sha256": ranks_sha256(ranks),
        "ranks_file_sha256": ranks_file_sha256,
    }

    manifest_temporary: Path | None = None
    try:
        manifest_temporary = _temporary_path(prefix, "manifest.json")
        with manifest_temporary.open("w", encoding="utf-8") as handle:
            json.dump(promoted_manifest, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            _flush_and_fsync(handle)
        # Refuse to overwrite a cache changed while the large files were checked.
        if paths["manifest"].read_bytes() != original_manifest:
            return False
        os.replace(manifest_temporary, paths["manifest"])
        manifest_temporary = None
        _fsync_directory(prefix.parent)
    except OSError:
        return False
    finally:
        if manifest_temporary is not None:
            try:
                manifest_temporary.unlink()
            except FileNotFoundError:
                pass
    return True


def race_ids_sha256(race_keys: Iterable[tuple[str, str, str, int]]) -> str:
    digest = hashlib.sha256()
    for race_id, *_rest in race_keys:
        encoded = str(race_id).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def ranks_sha256(ranks: np.ndarray) -> str:
    array = np.ascontiguousarray(ranks)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def feature_hasher_settings(hasher: FeatureHasher) -> dict[str, Any]:
    return {
        "class": "sklearn.feature_extraction.FeatureHasher",
        "n_features": int(hasher.n_features),
        "input_type": str(hasher.input_type),
        "alternate_sign": bool(hasher.alternate_sign),
        "dtype": str(np.dtype(hasher.dtype)),
    }


def default_feature_hasher_settings(n_features: int) -> dict[str, Any]:
    return feature_hasher_settings(
        FeatureHasher(
            n_features=int(n_features),
            input_type="dict",
            alternate_sign=False,
        )
    )


def _validate_dataset_arrays(
    matrix: sparse.csr_matrix,
    ranks: np.ndarray,
    *,
    race_count: int,
    n_features: int,
) -> None:
    expected_matrix_shape = (int(race_count) * 6, int(n_features))
    expected_ranks_shape = (int(race_count), 6)
    if matrix.shape != expected_matrix_shape or ranks.shape != expected_ranks_shape:
        raise ValueError(
            f"invalid hashed dataset shape: matrix={matrix.shape} ranks={ranks.shape}"
        )
    if ranks.dtype != np.dtype(np.int8):
        raise ValueError(f"invalid ranks dtype: {ranks.dtype}")
    if matrix.indptr.shape != (matrix.shape[0] + 1,):
        raise ValueError("invalid CSR indptr shape")
    if matrix.data.size != matrix.nnz or matrix.indices.size != matrix.nnz:
        raise ValueError("invalid CSR nnz arrays")
    if matrix.indptr[0] != 0 or matrix.indptr[-1] != matrix.nnz:
        raise ValueError("invalid CSR indptr bounds")
    if np.any(matrix.indptr[1:] < matrix.indptr[:-1]):
        raise ValueError("invalid CSR indptr ordering")
    if matrix.nnz and (
        np.any(matrix.indices < 0)
        or np.any(matrix.indices >= matrix.shape[1])
        or not np.all(np.isfinite(matrix.data))
    ):
        raise ValueError("invalid CSR values")
    if race_count:
        sorted_ranks = np.sort(ranks, axis=1)
        valid_competition_ranks = (sorted_ranks[:, 0] == 1) & np.all(
            (sorted_ranks[:, 1:] == sorted_ranks[:, :-1])
            | (sorted_ranks[:, 1:] == np.arange(2, 7)),
            axis=1,
        )
        if not np.all(valid_competition_ranks):
            raise ValueError("each race ranks row must use valid competition ranking")


def _load_csr_npz_without_pickle(path: Path) -> sparse.csr_matrix:
    with np.load(path, allow_pickle=False) as archive:
        required = {"format", "shape", "data", "indices", "indptr"}
        if not required.issubset(archive.files):
            raise ValueError("invalid sparse NPZ members")
        stored_format = np.asarray(archive["format"])
        if stored_format.shape != ():
            raise ValueError("invalid sparse NPZ format")
        format_value = stored_format.item()
        if isinstance(format_value, bytes):
            format_value = format_value.decode("ascii")
        if format_value != "csr":
            raise ValueError(f"expected CSR sparse NPZ, got {format_value!r}")
        shape_values = np.asarray(archive["shape"])
        if shape_values.shape != (2,):
            raise ValueError("invalid sparse NPZ shape")
        shape = tuple(int(value) for value in shape_values)
        data = np.asarray(archive["data"])
        indices = np.asarray(archive["indices"])
        indptr = np.asarray(archive["indptr"])
    return sparse.csr_matrix((data, indices, indptr), shape=shape, copy=False)


def _strict_json_equal(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _strict_json_equal(left, right)
            for left, right in zip(actual, expected, strict=True)
        )
    return actual == expected


def _temporary_path(prefix: Path, kind: str) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{prefix.name}.",
        suffix=f".{kind}.tmp",
        dir=prefix.parent,
    )
    os.close(descriptor)
    return Path(name)


def _flush_and_fsync(handle: Any) -> None:
    handle.flush()
    os.fsync(handle.fileno())


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def cache_paths(prefix: Path) -> dict[str, Path]:
    base = str(prefix)
    return {
        "matrix": Path(base + ".matrix.npz"),
        "ranks": Path(base + ".ranks.npy"),
        "manifest": Path(base + ".manifest.json"),
    }
