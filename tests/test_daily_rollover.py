from __future__ import annotations

import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import patch

from boatrace_ai import adaptive_odds_loop_safe5_no_odds_v8_modelrank as adaptive_loop
from boatrace_ai import web_dashboard
from boatrace_ai.db import connection, init_db, upsert_race
from boatrace_ai.time_semantics import operational_race_date


class DailyRolloverTest(unittest.TestCase):
    def test_operational_date_rolls_at_jst_midnight(self) -> None:
        before = datetime(2026, 7, 18, 14, 59, tzinfo=timezone.utc)
        after = datetime(2026, 7, 18, 15, 0, tzinfo=timezone.utc)
        self.assertEqual(operational_race_date(at=before), date(2026, 7, 18))
        self.assertEqual(operational_race_date(at=after), date(2026, 7, 19))
        self.assertEqual(
            operational_race_date(date(2020, 1, 2), at=after),
            date(2020, 1, 2),
        )

    def test_schedule_fallback_only_expands_confirmed_active_venues(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "schedule.sqlite"
            raw_dir = Path(directory) / "raw"
            init_db(db_path)
            calls: list[tuple[str, int]] = []

            def fake_collect(conn, *, race_date, jcd, rno, raw_dir):
                calls.append((jcd, rno))
                return jcd in {"01", "03"}

            with connection(db_path) as conn, patch.object(adaptive_loop, "discover_races", return_value=[]), patch.object(adaptive_loop, "collect_racelist", side_effect=fake_collect):
                result = adaptive_loop.refresh_daily_schedule(
                    conn,
                    race_date=date(2026, 7, 19),
                    raw_dir=raw_dir,
                    sleep_seconds=0,
                )

            expanded = {(jcd, rno) for jcd, rno in calls if rno != 1}
            self.assertEqual(result["schedule_discovery"], "venue_probe")
            self.assertEqual(result["schedule_targets"], 24)
            self.assertTrue(expanded)
            self.assertEqual({jcd for jcd, _ in expanded}, {"01", "03"})

    def test_web_default_does_not_keep_previous_day_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "rollover.sqlite"
            init_db(db_path)
            with connection(db_path) as conn:
                upsert_race(
                    conn,
                    {
                        "race_id": "2026-07-18-01-01",
                        "race_date": "2026-07-18",
                        "jcd": "01",
                        "venue_name": "桐生",
                        "rno": 1,
                        "deadline_at": "2026-07-18T10:00:00+09:00",
                    },
                )
            original_now = web_dashboard.now_jst
            try:
                web_dashboard._DEFAULT_DATE_CACHE.pop(db_path, None)
                web_dashboard.now_jst = lambda: datetime.fromisoformat("2026-07-18T00:01:00+09:00")
                self.assertEqual(web_dashboard.default_race_date(db_path), "2026-07-18")
                with connection(db_path) as conn:
                    upsert_race(
                        conn,
                        {
                            "race_id": "2026-07-19-01-01",
                            "race_date": "2026-07-19",
                            "jcd": "01",
                            "venue_name": "桐生",
                            "rno": 1,
                            "deadline_at": "2026-07-19T10:00:00+09:00",
                        },
                    )
                web_dashboard.now_jst = lambda: datetime.fromisoformat("2026-07-19T00:01:00+09:00")
                self.assertEqual(web_dashboard.default_race_date(db_path), "2026-07-19")
            finally:
                web_dashboard.now_jst = original_now
                web_dashboard._DEFAULT_DATE_CACHE.pop(db_path, None)


if __name__ == "__main__":
    unittest.main()
