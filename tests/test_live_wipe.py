from __future__ import annotations

import unittest
from datetime import datetime

from boatrace_ai.web.dashboard import _latest_live_window_row, boatcast_live_player_url


class LiveWipeTest(unittest.TestCase):
    def test_selects_latest_started_race_within_seven_minutes(self) -> None:
        rows = [
            {"race_id": "old", "deadline_at": "2026-07-19T09:00:00+09:00"},
            {"race_id": "latest", "deadline_at": "2026-07-19T09:03:00+09:00"},
            {"race_id": "future", "deadline_at": "2026-07-19T09:07:00+09:00"},
        ]
        selected = _latest_live_window_row(
            rows,
            now=datetime.fromisoformat("2026-07-19T09:04:00+09:00"),
        )
        self.assertEqual(selected["race_id"], "latest")

    def test_stops_at_exactly_seven_minutes(self) -> None:
        rows = [{"race_id": "race", "deadline_at": "2026-07-19T09:00:00+09:00"}]
        selected = _latest_live_window_row(
            rows,
            now=datetime.fromisoformat("2026-07-19T09:07:00+09:00"),
        )
        self.assertIsNone(selected)


if __name__ == "__main__":
    unittest.main()
