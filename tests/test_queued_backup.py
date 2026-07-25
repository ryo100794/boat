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
    assert '"$rclone_bin" md5sum "$archive"' in script
    assert '"$rclone_bin" md5sum "$remote/$(basename "$archive")"' in script
    assert "remote archive verification failed" in script
    assert "autostart=false" in supervisor
    assert "autorestart=false" in supervisor


def test_model_cache_archive_verifies_remote_before_removal() -> None:
    script = Path(
        "scripts/deployment/run-boatrace-model-cache-archive.sh"
    ).read_text(encoding="utf-8")

    assert 'exec 9>"$lock_dir/model-cache-archive.lock"' in script
    assert "flock -w 300 9" in script
    assert '"$rclone_bin" md5sum "$source"' in script
    assert '"$rclone_bin" md5sum "$remote"' in script
    assert "remote model cache verification failed" in script
    assert 'mv "$marker_tmp" "$marker"' in script
    assert script.index('mv "$marker_tmp" "$marker"') < script.index(
        'rm -f -- "$source"'
    )
