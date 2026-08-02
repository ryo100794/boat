from datetime import datetime, timezone
from pathlib import Path

from scripts.restore_missing_odds_snapshots import page_identity


def test_page_identity_uses_race_path_and_utc_capture_time() -> None:
    race_id, captured_at = page_identity(
        Path("pages/20260731/13/10/odds3t-20260731T060108Z.html")
    )

    assert race_id == "2026-07-31-13-10"
    assert captured_at == datetime(2026, 7, 31, 6, 1, 8, tzinfo=timezone.utc)
