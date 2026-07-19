from __future__ import annotations

import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from boatrace_ai.runtime import collector as adaptive_loop
from boatrace_ai.web import dashboard as web_dashboard
from boatrace_ai.db import connection, init_db, upsert_race
from boatrace_ai.runtime.time_semantics import operational_race_date


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

    def test_schedule_targets_prioritize_imminent_cutoffs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "priority.sqlite"
            init_db(db_path)
            now = datetime(2026, 7, 19, 8, 0, tzinfo=timezone(timedelta(hours=9)))
            with connection(db_path) as conn:
                for jcd, start in (("01", "08:50"), ("02", "08:20"), ("03", "07:50")):
                    upsert_race(
                        conn,
                        {
                            "race_date": "2026-07-19",
                            "jcd": jcd,
                            "venue_name": jcd,
                            "rno": 1,
                            "deadline_at": f"2026-07-19T{start}:00+09:00",
                        },
                    )
                ordered = adaptive_loop._prioritize_schedule_targets(
                    conn,
                    date(2026, 7, 19),
                    [("01", 1), ("03", 1), ("04", 1), ("02", 1)],
                    now=now,
                )

            self.assertEqual(ordered, [("02", 1), ("01", 1), ("03", 1), ("04", 1)])

    def test_complete_program_entries_do_not_wait_for_racelist_html(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "program.sqlite"
            raw_dir = Path(directory) / "raw"
            init_db(db_path)
            with connection(db_path) as conn:
                race_id_value = upsert_race(
                    conn,
                    {
                        "race_date": "2026-07-19",
                        "jcd": "01",
                        "venue_name": "桐生",
                        "rno": 1,
                        "deadline_at": "2026-07-19T09:00:00+09:00",
                    },
                )
                conn.executemany(
                    "INSERT INTO entries(race_id, lane) VALUES (?, ?)",
                    [(race_id_value, lane) for lane in range(1, 7)],
                )
                with patch.object(adaptive_loop, "discover_races", return_value=[("01", 1)]), patch.object(
                    adaptive_loop, "collect_racelist"
                ) as collect:
                    result = adaptive_loop.refresh_daily_schedule(
                        conn,
                        race_date=date(2026, 7, 19),
                        raw_dir=raw_dir,
                        sleep_seconds=0,
                    )

            collect.assert_not_called()
            self.assertEqual(result["schedule_failed"], 0)

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
