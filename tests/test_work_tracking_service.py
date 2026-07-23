from pathlib import Path


def test_work_tracking_sync_service_is_periodic_and_secret_free() -> None:
    config = Path(
        "scripts/deployment/supervisor-boatrace-work-tracking-sync.ini"
    ).read_text(encoding="utf-8")

    assert "boatrace_ai.work_tracking_sync" in config
    assert "--repo ryo100794/boat" in config
    assert "--direction both" in config
    assert "--apply" in config
    assert "--interval-seconds 900" in config
    assert "PGPASSFILE=" in config
    assert "GITHUB_TOKEN=" not in config
    assert "github_pat_" not in config
