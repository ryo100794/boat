from datetime import datetime, timedelta

from boatrace_ai.runtime.collector import (
    beforeinfo_interval,
    odds_interval,
    t5_guard_rows,
    t5_priority_due,
)
from boatrace_ai.runtime.time_semantics import JST


def test_beforeinfo_polling_targets_model_decision_window() -> None:
    assert beforeinfo_interval(31 * 60, has_rows=False) is None
    assert beforeinfo_interval(20 * 60, has_rows=False) == 30.0
    assert beforeinfo_interval(20 * 60, has_rows=True) == 90.0
    assert beforeinfo_interval(10 * 60, has_rows=True) == 30.0
    assert beforeinfo_interval(4 * 60, has_rows=False) is None


def test_odds_polling_does_not_probe_before_the_collection_window() -> None:
    assert odds_interval(61 * 60) is None
    assert odds_interval(60 * 60) == 90.0
    assert odds_interval(15 * 60) == 45.0
    assert odds_interval(5 * 60) == 20.0
    assert odds_interval(90) == 10.0
    assert odds_interval(-1) is None


def test_t5_priority_is_due_during_last_minute_without_fresh_odds() -> None:
    start_at = datetime(2026, 7, 23, 12, 5, tzinfo=JST)
    model_cutoff = start_at - timedelta(minutes=10)

    assert t5_priority_due(
        start_at=start_at,
        now=model_cutoff - timedelta(seconds=60),
        latest_odds=None,
    )
    assert t5_priority_due(
        start_at=start_at,
        now=model_cutoff - timedelta(seconds=10),
        latest_odds=model_cutoff - timedelta(seconds=75),
    )


def test_t5_priority_accepts_fresh_snapshot_and_stays_inside_window() -> None:
    start_at = datetime(2026, 7, 23, 12, 5, tzinfo=JST)
    model_cutoff = start_at - timedelta(minutes=10)

    assert not t5_priority_due(
        start_at=start_at,
        now=model_cutoff - timedelta(seconds=10),
        latest_odds=model_cutoff - timedelta(seconds=30),
    )
    assert not t5_priority_due(
        start_at=start_at, now=model_cutoff - timedelta(seconds=61), latest_odds=None
    )
    assert not t5_priority_due(
        start_at=start_at, now=model_cutoff + timedelta(seconds=1), latest_odds=None
    )


def test_t5_guard_reserves_imminent_window_until_snapshot_is_fresh() -> None:
    start_at = datetime(2026, 7, 23, 12, 5, tzinfo=JST)
    model_cutoff = start_at - timedelta(minutes=10)
    row = {
        "race_id": "20260723-01-01",
        "deadline_at": start_at.isoformat(),
        "latest_odds_at": None,
    }

    guarded = t5_guard_rows([row], now=model_cutoff - timedelta(seconds=90))
    assert len(guarded) == 1
    assert guarded[0][0] == 90.0
    assert t5_guard_rows([row], now=model_cutoff - timedelta(seconds=91)) == []
    assert t5_guard_rows([row], now=model_cutoff + timedelta(seconds=1)) == []

    row["latest_odds_at"] = (model_cutoff - timedelta(seconds=30)).isoformat()
    assert t5_guard_rows([row], now=model_cutoff - timedelta(seconds=10)) == []


def test_t5_guard_skips_capture_completed_in_current_loop() -> None:
    start_at = datetime(2026, 7, 23, 12, 5, tzinfo=JST)
    model_cutoff = start_at - timedelta(minutes=10)
    row = {
        "race_id": "20260723-01-01",
        "deadline_at": start_at.isoformat(),
        "latest_odds_at": None,
    }
    assert t5_guard_rows(
        [row],
        now=model_cutoff - timedelta(seconds=10),
        satisfied_race_ids={row["race_id"]},
    ) == []
