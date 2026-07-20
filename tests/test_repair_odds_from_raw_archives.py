from __future__ import annotations

from datetime import datetime, timezone

from scripts.repair_odds_from_raw_archives import _closest_snapshot, _race_cutoff


def test_race_cutoff_is_ten_minutes_before_stored_start() -> None:
    cutoff = _race_cutoff("2026-07-20T12:00:00+09:00")

    assert cutoff == datetime(2026, 7, 20, 2, 50, tzinfo=timezone.utc)


def test_closest_snapshot_accepts_only_nearby_capture() -> None:
    snapshots = [
        (10, datetime(2026, 7, 20, 2, 49, 20, tzinfo=timezone.utc)),
        (11, datetime(2026, 7, 20, 2, 49, 55, tzinfo=timezone.utc)),
    ]
    capture = datetime(2026, 7, 20, 2, 50, tzinfo=timezone.utc)

    assert _closest_snapshot(snapshots, capture) == 11
    assert (
        _closest_snapshot(
            snapshots,
            datetime(2026, 7, 20, 2, 53, tzinfo=timezone.utc),
        )
        is None
    )
