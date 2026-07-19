from __future__ import annotations

from datetime import datetime
from pathlib import Path

from boatrace_ai.web.dashboard import _live_window_rows


def test_live_window_returns_latest_four_started_races() -> None:
    rows = [
        {"race_id": f"r{minute}", "deadline_at": f"2026-07-19T09:{minute:02d}:00+09:00"}
        for minute in range(0, 8, 2)
    ]
    selected = _live_window_rows(
        rows,
        now=datetime.fromisoformat("2026-07-19T09:06:00+09:00"),
    )
    assert [row["race_id"] for row in selected] == ["r6", "r4", "r2", "r0"]


def test_dashboard_uses_four_slot_live_grid() -> None:
    html = Path("src/boatrace_ai/templates/dashboard.html").read_text(encoding="utf-8")
    assert 'id="liveWipeGrid"' in html
    assert "payload.races" in html
    assert "while(slots.length < (multi ? 4 : 1))" in html
    assert "sort((a,b) => new Date(b.race_time_at || 0).getTime()" in html
    assert 'id="liveWipeFrame"' not in html
