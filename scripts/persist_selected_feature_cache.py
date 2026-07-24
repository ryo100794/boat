from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any


TRANSIENT_SELECTED_CACHE_DIR = Path("/tmp/boatrace-standardized-365d-v2")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _recompress_npz(source: Path, destination: Path) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with (
            zipfile.ZipFile(source, "r") as input_archive,
            zipfile.ZipFile(
                temporary,
                "w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=1,
                allowZip64=True,
            ) as output_archive,
        ):
            for entry in input_archive.infolist():
                if entry.is_dir():
                    continue
                with (
                    input_archive.open(entry, "r") as input_handle,
                    output_archive.open(entry.filename, "w", force_zip64=True) as output_handle,
                ):
                    shutil.copyfileobj(input_handle, output_handle, length=1 << 20)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def wait_for_new_artifact(
    artifact_path: Path,
    *,
    newer_than: float,
    timeout_seconds: float,
    poll_seconds: float = 10.0,
) -> None:
    if timeout_seconds <= 0.0 or poll_seconds <= 0.0:
        raise ValueError("cache wait intervals must be positive")
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            modified = artifact_path.stat().st_mtime
        except FileNotFoundError:
            modified = 0.0
        if modified > newer_than:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "timed out waiting for a newly generated selected cache artifact"
            )
        time.sleep(min(poll_seconds, max(0.01, deadline - time.monotonic())))


def persist_selected_cache(
    artifact_path: Path,
    destination_dir: Path,
) -> dict[str, Any]:
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    source_prefix = Path(str(artifact["selected_cache_prefix"]))
    selected = artifact["selected"]
    n_features = int(artifact["n_features"])
    variant = str(selected["feature_variant"])
    expected_name = f"listwise_search_{n_features}_{variant}"
    if source_prefix.name != expected_name:
        raise ValueError("selected cache prefix does not match the selected variant")
    if source_prefix.parent == destination_dir:
        return {
            "status": "already_persistent",
            "cache_prefix": str(source_prefix),
        }
    if source_prefix.parent != TRANSIENT_SELECTED_CACHE_DIR:
        raise ValueError("selected cache source is outside the approved transient directory")

    source_matrix = Path(f"{source_prefix}.matrix.npz")
    source_ranks = Path(f"{source_prefix}.ranks.npy")
    source_manifest = Path(f"{source_prefix}.manifest.json")
    for source in (source_matrix, source_ranks, source_manifest):
        if not source.is_file() or source.stat().st_size <= 0:
            raise FileNotFoundError(f"selected cache component is missing: {source}")

    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    if int(manifest.get("n_features") or 0) != n_features:
        raise ValueError("selected cache manifest feature count mismatch")
    if tuple(manifest.get("drop_feature_groups") or ()) != tuple(
        selected.get("drop_feature_groups") or ()
    ):
        raise ValueError("selected cache manifest feature groups mismatch")
    if _sha256(source_matrix) != str(manifest.get("matrix_file_sha256") or ""):
        raise ValueError("selected cache matrix checksum mismatch")
    if _sha256(source_ranks) != str(manifest.get("ranks_file_sha256") or ""):
        raise ValueError("selected cache ranks checksum mismatch")

    destination_dir.mkdir(parents=True, exist_ok=True)
    destination_prefix = destination_dir / expected_name
    destination_matrix = Path(f"{destination_prefix}.matrix.npz")
    destination_ranks = Path(f"{destination_prefix}.ranks.npy")
    destination_manifest = Path(f"{destination_prefix}.manifest.json")
    _recompress_npz(source_matrix, destination_matrix)
    shutil.copyfile(source_ranks, destination_ranks)
    manifest["matrix_file_sha256"] = _sha256(destination_matrix)
    manifest["ranks_file_sha256"] = _sha256(destination_ranks)
    manifest["storage_compression"] = "zip-deflate-level-1"
    _write_json_atomic(destination_manifest, manifest)

    persisted = dict(artifact)
    persisted["selected_cache_prefix"] = str(destination_prefix)
    persisted["selected_cache_dir"] = str(destination_dir)
    persisted["selected_cache_persistent"] = True
    _write_json_atomic(artifact_path, persisted)
    return {
        "status": "persisted",
        "source_bytes": source_matrix.stat().st_size,
        "destination_bytes": destination_matrix.stat().st_size,
        "cache_prefix": str(destination_prefix),
        "artifact": str(artifact_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Persist the standardized selected feature cache with compression"
    )
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--destination-dir", type=Path, required=True)
    parser.add_argument("--wait-for-mtime-after", type=float)
    parser.add_argument("--wait-timeout-seconds", type=float, default=21_300.0)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.wait_for_mtime_after is not None:
        wait_for_new_artifact(
            args.artifact,
            newer_than=args.wait_for_mtime_after,
            timeout_seconds=args.wait_timeout_seconds,
            poll_seconds=args.poll_seconds,
        )
    result = persist_selected_cache(args.artifact, args.destination_dir)
    if args.output is not None:
        _write_json_atomic(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
