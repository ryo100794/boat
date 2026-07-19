from datetime import datetime
from pathlib import Path

from boatrace_ai.web.dashboard import JST, _venue_display_row


NOW = datetime(2026, 7, 19, 12, 3, tzinfo=JST)


def test_venue_keeps_latest_unfinished_started_race() -> None:
    rows = [
        {"race_id": "old-final", "deadline_at": "2026-07-19T11:55:00+09:00", "result_rows": 3},
        {"race_id": "current", "deadline_at": "2026-07-19T12:00:00+09:00", "result_rows": 0},
        {"race_id": "next", "deadline_at": "2026-07-19T12:20:00+09:00", "result_rows": 0},
    ]

    selected = _venue_display_row(rows, now=NOW)

    assert selected is not None
    assert selected[2]["race_id"] == "current"


def test_venue_advances_after_current_race_is_final() -> None:
    rows = [
        {"race_id": "current", "deadline_at": "2026-07-19T12:00:00+09:00", "result_rows": 3},
        {"race_id": "next", "deadline_at": "2026-07-19T12:20:00+09:00", "result_rows": 0},
    ]

    selected = _venue_display_row(rows, now=NOW)

    assert selected is not None
    assert selected[2]["race_id"] == "next"


def test_dashboard_highlights_racing_venue() -> None:
    html = Path("src/boatrace_ai/templates/dashboard.html").read_text(encoding="utf-8")
    assert ".venue.s-racing" in html
    assert 'v.next_time_status==="出走"' in html
    assert "${venueRaceLabel(v)}" in html
    assert "venueFilter" not in html
    assert "? \"\" : el.dataset.jcd" in html
