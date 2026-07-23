from pathlib import Path


def test_raw_archive_is_queue_driven_and_locked() -> None:
    script = Path(
        "scripts/deployment/run-boatrace-raw-archive.sh"
    ).read_text(encoding="utf-8")
    supervisor = Path(
        "scripts/deployment/supervisor-boatrace-raw-archive.ini"
    ).read_text(encoding="utf-8")

    assert 'exec 9>"$lock_dir/raw-archive.lock"' in script
    assert "flock -w 300 9" in script
    assert "autostart=false" in supervisor
    assert "autorestart=false" in supervisor
