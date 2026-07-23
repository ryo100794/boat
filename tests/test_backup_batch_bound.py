from pathlib import Path


def test_queued_backup_is_bounded_to_one_snapshot_batch() -> None:
    script = Path(
        "scripts/deployment/run-boatrace-raw-archive.sh"
    ).read_text(encoding="utf-8")
    task = Path("src/boatrace_ai/maintenance_tasks.py").read_text(
        encoding="utf-8"
    )

    assert "BOATRACE_RAW_ARCHIVE_MAX_BATCHES" in script
    assert 'batches=$((batches + 1))' in script
    assert 'env["BOATRACE_RAW_ARCHIVE_MAX_BATCHES"] = "1"' in task
