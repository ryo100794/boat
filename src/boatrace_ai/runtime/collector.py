from __future__ import annotations

import argparse
import json
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from ..constants import RACES_PER_DAY, VENUES
from ..ingestion.program import load_daily_program
from ..db import connection, init_db
from ..operational_model import predict_open_races
from .result_polling import due_result_rows, result_interval
from .time_semantics import JST, estimated_deadline_from_start, now_jst, operational_race_date, stored_start_time

from ..ingestion.live import collect_odds, collect_racelist, collect_result, discover_races


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Adaptive odds/results loop using stored race-start times and no-odds v8.")
    parser.add_argument("--db", default="data/boatrace.sqlite")
    parser.add_argument("--model", default="data/models/win_model_no_odds_v8.joblib")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--date", help="Fix one race date; omit to follow the current JST date automatically.")
    parser.add_argument("--sleep-loop", type=float, default=10.0)
    parser.add_argument("--sleep-page", type=float, default=0.4)
    parser.add_argument("--max-loops", type=int)
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--collect-results", action="store_true")
    args = parser.parse_args(argv)

    init_db(args.db)
    fixed_date = date.fromisoformat(args.date) if args.date else None
    raw_dir = Path(args.raw_dir)
    model_path = Path(args.model)
    loop = 0
    schedule_date: date | None = None
    next_schedule_refresh = 0.0

    while True:
        now = now_jst()
        target_date = operational_race_date(fixed_date, at=now)
        counters = {
            "loop": loop,
            "odds_targets": 0,
            "odds_ok": 0,
            "odds_failed": 0,
            "result_targets": 0,
            "result_rows": 0,
            "result_empty": 0,
            "predicted": 0,
            "prediction_failed": 0,
            "time_basis": "stored_deadline_at_is_race_start",
            "race_date": target_date.isoformat(),
            "date_mode": "fixed" if fixed_date else "jst_auto",
            "schedule_targets": 0,
            "schedule_loaded": 0,
            "schedule_failed": 0,
            "program_status": "not_due",
            "program_races": 0,
            "program_entries": 0,
        }
        with connection(args.db) as conn:
            refresh_due = fixed_date is None and (
                schedule_date != target_date or time.monotonic() >= next_schedule_refresh
            )
            if refresh_due:
                try:
                    counters.update(load_daily_program(conn, race_date=target_date, raw_dir=raw_dir))
                except Exception as exc:
                    counters["program_status"] = f"error:{type(exc).__name__}"
                schedule = refresh_daily_schedule(
                    conn,
                    race_date=target_date,
                    raw_dir=raw_dir,
                    sleep_seconds=args.sleep_page,
                )
                counters.update(schedule)
                schedule_date = target_date
                next_schedule_refresh = time.monotonic() + 15 * 60
            rows = scheduled_races(conn, target_date)
            if args.collect_results:
                for result_row in due_result_rows(rows, now=now):
                    counters["result_targets"] += 1
                    count = collect_result(
                        conn,
                        race_date=target_date,
                        jcd=result_row["jcd"],
                        rno=int(result_row["rno"]),
                        raw_dir=raw_dir,
                    )
                    conn.commit()
                    if count:
                        counters["result_rows"] += count
                    else:
                        counters["result_empty"] += 1
                    time.sleep(args.sleep_page)
            for row in rows:
                start_at = stored_start_time(row["deadline_at"])
                cutoff_at = estimated_deadline_from_start(start_at)
                latest_odds = parse_time(row["latest_odds_at"], default_tz=timezone.utc)
                latest_result_attempt = parse_time(row["latest_result_attempt_at"], default_tz=timezone.utc)
                if not start_at or not cutoff_at:
                    continue
                seconds_to_cutoff = (cutoff_at - now).total_seconds()
                seconds_to_start = (start_at - now).total_seconds()

                interval = odds_interval(seconds_to_cutoff)
                if interval is None:
                    continue
                age = (now - latest_odds).total_seconds() if latest_odds else None
                if latest_odds and age is not None and age < interval:
                    continue
                counters["odds_targets"] += 1
                ok = collect_odds(
                    conn,
                    race_date=target_date,
                    jcd=row["jcd"],
                    rno=int(row["rno"]),
                    raw_dir=raw_dir,
                )
                conn.commit()
                if ok:
                    counters["odds_ok"] += 1
                    if args.predict and model_path.exists():
                        result = predict_open_races(
                            conn,
                            model_path=model_path,
                            race_date=target_date,
                            jcd=row["jcd"],
                            rno=int(row["rno"]),
                        )
                        counters["predicted"] += result["predicted"]
                        counters["prediction_failed"] += result["failed"]
                        conn.commit()
                else:
                    counters["odds_failed"] += 1
                time.sleep(args.sleep_page)
        counters["now_jst"] = now.isoformat(timespec="seconds")
        print(json.dumps(counters, ensure_ascii=False), flush=True)
        loop += 1
        if args.max_loops is not None and loop >= args.max_loops:
            return 0
        time.sleep(args.sleep_loop)


def refresh_daily_schedule(
    conn,
    *,
    race_date: date,
    raw_dir: Path,
    sleep_seconds: float,
) -> dict[str, int]:
    targets = discover_races(race_date, sleep_seconds=sleep_seconds, fallback_all=False)
    existing = {
        (str(row["jcd"]).zfill(2), int(row["rno"])): {
            "entries": int(row["entries"] or 0),
            "html": bool(row["has_html"]),
        }
        for row in conn.execute(
            """
            SELECT r.jcd, r.rno, COUNT(e.lane) AS entries,
                   EXISTS(SELECT 1 FROM raw_pages rp WHERE rp.race_id = r.race_id AND rp.page_type = "racelist") AS has_html
            FROM races r
            LEFT JOIN entries e ON e.race_id = r.race_id
            WHERE r.race_date = ?
            GROUP BY r.race_id, r.jcd, r.rno
            """,
            (race_date.isoformat(),),
        )
    }
    loaded = 0
    failed = 0
    discovery_mode = "official_index"

    if not targets:
        discovery_mode = "venue_probe"
        active_venues: set[str] = set()
        for venue in VENUES:
            key = (venue.code, 1)
            if existing.get(key, {}).get("entries", 0) >= 6:
                active_venues.add(venue.code)
                continue
            try:
                available = collect_racelist(
                    conn,
                    race_date=race_date,
                    jcd=venue.code,
                    rno=1,
                    raw_dir=raw_dir,
                )
                if available:
                    active_venues.add(venue.code)
                    existing[key] = {"entries": 6, "html": True}
                    loaded += 1
                    conn.commit()
            except Exception:
                failed += 1
                conn.rollback()
            if sleep_seconds:
                time.sleep(sleep_seconds)
        targets = [
            (venue.code, int(rno))
            for venue in VENUES
            if venue.code in active_venues
            for rno in RACES_PER_DAY
        ]

    targets = _prioritize_schedule_targets(conn, race_date, targets, now=now_jst())
    for jcd, rno in targets:
        # The official program is already a complete, persisted racelist source.
        # Re-fetching every HTML page here can delay imminent odds collection by minutes.
        if existing.get((jcd, rno), {}).get("entries", 0) >= 6:
            continue
        try:
            if collect_racelist(conn, race_date=race_date, jcd=jcd, rno=rno, raw_dir=raw_dir):
                loaded += 1
                conn.commit()
            else:
                failed += 1
        except Exception:
            failed += 1
            conn.rollback()
        if sleep_seconds:
            time.sleep(sleep_seconds)
    return {
        "schedule_targets": len(targets),
        "schedule_loaded": loaded,
        "schedule_failed": failed,
        "schedule_discovery": discovery_mode,
    }


def _prioritize_schedule_targets(
    conn,
    race_date: date,
    targets: list[tuple[str, int]],
    *,
    now: datetime,
) -> list[tuple[str, int]]:
    starts = {
        (str(row["jcd"]).zfill(2), int(row["rno"])): stored_start_time(row["deadline_at"])
        for row in conn.execute(
            "SELECT jcd, rno, deadline_at FROM races WHERE race_date = ?",
            (race_date.isoformat(),),
        )
    }

    def priority(target: tuple[str, int]) -> tuple[int, float, str, int]:
        start = starts.get(target)
        cutoff = estimated_deadline_from_start(start)
        if cutoff is None:
            return (2, 0.0, target[0], target[1])
        if cutoff >= now:
            return (0, cutoff.timestamp(), target[0], target[1])
        return (1, -start.timestamp(), target[0], target[1])

    return sorted(targets, key=priority)


def scheduled_races(conn, race_date: date) -> list[Any]:
    return conn.execute(
        """
        SELECT r.race_id, r.jcd, r.rno, r.deadline_at,
               (SELECT MAX(captured_at) FROM odds_snapshots os WHERE os.race_id = r.race_id) AS latest_odds_at,
               (SELECT MAX(fetched_at) FROM raw_pages rp WHERE rp.race_id = r.race_id AND rp.page_type = 'result') AS latest_result_attempt_at,
               (SELECT COUNT(*) FROM race_results rr WHERE rr.race_id = r.race_id AND rr.rank IS NOT NULL) AS result_rows,
               EXISTS(
                 SELECT 1
                 FROM race_result_status rs
                 WHERE rs.race_id = r.race_id
                   AND rs.status = 'final'
                   AND rs.trifecta_evaluable = 0
               ) AS result_not_evaluable
        FROM races r
        WHERE r.race_date = ?
          AND r.deadline_at IS NOT NULL
          AND (
            (SELECT COUNT(*) FROM entries e WHERE e.race_id = r.race_id) = 6
            OR (SELECT COUNT(*) FROM odds_snapshots os WHERE os.race_id = r.race_id) > 0
          )
          AND NOT (
            (SELECT COUNT(*) FROM race_results rr WHERE rr.race_id = r.race_id AND rr.rank IS NOT NULL) >= 3
            OR EXISTS(
              SELECT 1
              FROM race_result_status rs
              WHERE rs.race_id = r.race_id
                AND rs.status = 'final'
                AND rs.trifecta_evaluable = 0
            )
          )
        ORDER BY r.deadline_at, r.jcd, r.rno
        """,
        (race_date.isoformat(),),
    ).fetchall()


def odds_interval(seconds_to_cutoff: float) -> float | None:
    if seconds_to_cutoff < 0:
        return None
    if seconds_to_cutoff <= 90:
        return 10.0
    if seconds_to_cutoff <= 5 * 60:
        return 20.0
    if seconds_to_cutoff <= 15 * 60:
        return 45.0
    if seconds_to_cutoff <= 60 * 60:
        return 90.0
    return 300.0


def parse_time(value: str | None, *, default_tz: timezone) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=default_tz)
    return parsed.astimezone(JST)


if __name__ == "__main__":
    raise SystemExit(main())
