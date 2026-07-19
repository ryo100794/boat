from __future__ import annotations

from datetime import datetime

from boatrace_ai.result_polling import due_result_rows, result_interval


def test_result_polling_starts_at_five_minutes_and_is_aggressive() -> None:
    assert result_interval(-299) is None
    assert result_interval(-300) == 10.0
    assert result_interval(-30 * 60) == 10.0
    assert result_interval(-31 * 60) == 30.0


def test_due_results_prioritize_latest_completed_race() -> None:
    rows = [
        {
            "race_id": "older",
            "deadline_at": "2026-07-19T09:00:00+09:00",
            "latest_result_attempt_at": None,
        },
        {
            "race_id": "latest",
            "deadline_at": "2026-07-19T09:05:00+09:00",
            "latest_result_attempt_at": None,
        },
    ]
    due = due_result_rows(
        rows,
        now=datetime.fromisoformat("2026-07-19T09:10:00+09:00"),
    )
    assert [row["race_id"] for row in due] == ["latest", "older"]


def test_recent_result_attempt_is_not_retried_early() -> None:
    rows = [
        {
            "race_id": "race",
            "deadline_at": "2026-07-19T09:05:00+09:00",
            "latest_result_attempt_at": "2026-07-19T00:09:55+00:00",
        }
    ]
    due = due_result_rows(
        rows,
        now=datetime.fromisoformat("2026-07-19T09:10:00+09:00"),
    )
    assert due == []
