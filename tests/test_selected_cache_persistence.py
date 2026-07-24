from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
from scipy import sparse

from scripts import persist_selected_feature_cache as persistence


def _write_source_cache(root: Path) -> tuple[Path, Path]:
    source_dir = root / "transient"
    source_dir.mkdir()
    prefix = source_dir / "listwise_search_8_drop_base_pastlog"
    matrix_path = Path(f"{prefix}.matrix.npz")
    ranks_path = Path(f"{prefix}.ranks.npy")
    manifest_path = Path(f"{prefix}.manifest.json")
    matrix = sparse.csr_matrix(
        np.asarray(
            [
                [0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0],
            ],
            dtype=np.float64,
        )
    )
    sparse.save_npz(matrix_path, matrix, compressed=False)
    ranks = np.asarray([[1, 2, 3, 4, 5, 6]], dtype=np.int8)
    np.save(ranks_path, ranks, allow_pickle=False)
    manifest_path.write_text(
        json.dumps(
            {
                "n_features": 8,
                "drop_feature_groups": ["base_pastlog"],
                "matrix_file_sha256": persistence._sha256(matrix_path),
                "ranks_file_sha256": persistence._sha256(ranks_path),
            }
        ),
        encoding="utf-8",
    )
    artifact_path = root / "listwise_feature_teacher.json"
    artifact_path.write_text(
        json.dumps(
            {
                "n_features": 8,
                "selected": {
                    "feature_variant": "drop_base_pastlog",
                    "drop_feature_groups": ["base_pastlog"],
                },
                "selected_cache_prefix": str(prefix),
                "selected_cache_dir": str(source_dir),
                "selected_cache_persistent": False,
            }
        ),
        encoding="utf-8",
    )
    return artifact_path, prefix


def test_persist_selected_cache_recompresses_and_updates_contract(
    tmp_path: Path,
    monkeypatch,
) -> None:
    artifact_path, source_prefix = _write_source_cache(tmp_path)
    monkeypatch.setattr(
        persistence,
        "TRANSIENT_SELECTED_CACHE_DIR",
        source_prefix.parent,
    )
    destination_dir = tmp_path / "persistent"

    result = persistence.persist_selected_cache(artifact_path, destination_dir)

    destination_prefix = destination_dir / source_prefix.name
    destination_matrix = Path(f"{destination_prefix}.matrix.npz")
    destination_ranks = Path(f"{destination_prefix}.ranks.npy")
    destination_manifest = json.loads(
        Path(f"{destination_prefix}.manifest.json").read_text(encoding="utf-8")
    )
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    np.testing.assert_array_equal(
        sparse.load_npz(destination_matrix).toarray(),
        sparse.load_npz(Path(f"{source_prefix}.matrix.npz")).toarray(),
    )
    np.testing.assert_array_equal(
        np.load(destination_ranks, allow_pickle=False),
        np.load(Path(f"{source_prefix}.ranks.npy"), allow_pickle=False),
    )
    with zipfile.ZipFile(destination_matrix) as archive:
        assert all(
            entry.compress_type == zipfile.ZIP_DEFLATED
            for entry in archive.infolist()
        )
    assert destination_manifest["storage_compression"] == "zip-deflate-level-1"
    assert destination_manifest["matrix_file_sha256"] == persistence._sha256(
        destination_matrix
    )
    assert artifact["selected_cache_prefix"] == str(destination_prefix)
    assert artifact["selected_cache_dir"] == str(destination_dir)
    assert artifact["selected_cache_persistent"] is True
    assert result["status"] == "persisted"
    assert persistence.persist_selected_cache(
        artifact_path, destination_dir
    )["status"] == "already_persistent"


def test_persist_selected_cache_rejects_unapproved_source(
    tmp_path: Path,
) -> None:
    artifact_path, _source_prefix = _write_source_cache(tmp_path)

    try:
        persistence.persist_selected_cache(
            artifact_path,
            tmp_path / "persistent",
        )
    except ValueError as exc:
        assert "approved transient directory" in str(exc)
    else:
        raise AssertionError("unapproved source must be rejected")
