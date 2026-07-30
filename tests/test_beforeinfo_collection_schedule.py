from datetime import date, datetime, timedelta
from pathlib import Path

import boatrace_ai.runtime.collector as collector

from boatrace_ai.runtime.collector import (
    beforeinfo_interval,
    closing_guard_rows,
    closing_priority_rows,
    closing_snapshot_is_fresh,
    odds_interval,
    schedule_refresh_blocked,
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
    assert odds_interval(5 * 60) == 15.0
    assert odds_interval(181) == 15.0
    assert odds_interval(180) == 10.0
    assert odds_interval(76) == 10.0
    assert odds_interval(75) == 5.0
    assert odds_interval(-1) is None


def test_closing_priority_orders_cutoffs_and_enforces_snapshot_freshness() -> None:
    now = datetime(2026, 7, 23, 12, 0, tzinfo=JST)
    rows = [
        {
            "race_id": "later",
            "deadline_at": (now + timedelta(minutes=6, seconds=10)).isoformat(),
            "latest_odds_at": (now - timedelta(seconds=13)).isoformat(),
        },
        {
            "race_id": "sooner",
            "deadline_at": (now + timedelta(minutes=5, seconds=20)).isoformat(),
            "latest_odds_at": None,
        },
        {
            "race_id": "fresh",
            "deadline_at": (now + timedelta(minutes=5, seconds=30)).isoformat(),
            "latest_odds_at": (now - timedelta(seconds=12)).isoformat(),
        },
    ]

    priority = closing_priority_rows(rows, now=now)
    assert [(seconds, row["race_id"]) for seconds, row in priority] == [
        (20.0, "sooner"),
        (70.0, "later"),
    ]
    assert closing_guard_rows(rows, now=now) == priority
    assert closing_snapshot_is_fresh(
        now=now, latest_odds=now - timedelta(seconds=12)
    )
    assert not closing_snapshot_is_fresh(
        now=now, latest_odds=now - timedelta(seconds=13)
    )


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

    guarded = t5_guard_rows([row], now=model_cutoff - timedelta(seconds=300))
    assert len(guarded) == 1
    assert guarded[0][0] == 300.0
    assert t5_guard_rows([row], now=model_cutoff - timedelta(seconds=301)) == []
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


def test_schedule_refresh_is_deferred_near_a_betting_cutoff() -> None:
    now = datetime(2026, 7, 28, 14, 0, tzinfo=JST)
    imminent = {
        "deadline_at": (now + timedelta(minutes=24, seconds=59)).isoformat()
    }
    outside_guard = {
        "deadline_at": (now + timedelta(minutes=25, seconds=1)).isoformat()
    }

    assert schedule_refresh_blocked([imminent], now=now)
    assert not schedule_refresh_blocked([outside_guard], now=now)
    assert not schedule_refresh_blocked([], now=now)


def test_priority_odds_uses_short_timeout_without_retry(monkeypatch, tmp_path) -> None:
    observed = {}

    def fake_collect_odds(*args, **kwargs):
        observed.update(kwargs)
        return True

    monkeypatch.setattr(collector, "collect_odds", fake_collect_odds)
    row = {"jcd": "01", "rno": 3}
    assert collector.collect_priority_odds(
        object(),
        race_date=date(2026, 7, 28),
        row=row,
        raw_dir=Path(tmp_path),
        cache_bust=True,
    )
    assert observed["timeout"] == 5.0
    assert observed["retries"] == 0
    assert observed["cache_bust"] is True


def test_priority_odds_failure_is_isolated_to_one_race(monkeypatch, tmp_path) -> None:
    class Connection:
        rolled_back = False

        def rollback(self):
            self.rolled_back = True

    def fail(*args, **kwargs):
        raise TimeoutError("official endpoint stalled")

    monkeypatch.setattr(collector, "collect_odds", fail)
    conn = Connection()
    assert not collector.collect_priority_odds(
        conn,
        race_date=date(2026, 7, 28),
        row={"jcd": "01", "rno": 3},
        raw_dir=Path(tmp_path),
    )
    assert conn.rolled_back
