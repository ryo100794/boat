from __future__ import annotations

import argparse
import json
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .db import connection, init_db
from .live_safe_patch4 import install
from .modeling_no_odds_v8_modelrank import predict_open_races
from .time_semantics import estimated_deadline_from_start, stored_start_time

install()

from .live import collect_odds, collect_result


JST = timezone(timedelta(hours=9))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Adaptive odds/results loop using stored race-start times and no-odds v8.")
    parser.add_argument("--db", default="data/boatrace.sqlite")
    parser.add_argument("--model", default="data/models/win_model_no_odds_v8.joblib")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--sleep-loop", type=float, default=10.0)
    parser.add_argument("--sleep-page", type=float, default=0.4)
    parser.add_argument("--max-loops", type=int)
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--collect-results", action="store_true")
    args = parser.parse_args(argv)

    init_db(args.db)
    target_date = date.fromisoformat(args.date)
    raw_dir = Path(args.raw_dir)
    model_path = Path(args.model)
    loop = 0

    while True:
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
        }
        now = datetime.now(timezone.utc).astimezone(JST)
        with connection(args.db) as conn:
            for row in scheduled_races(conn, target_date):
                start_at = stored_start_time(row["deadline_at"])
                cutoff_at = estimated_deadline_from_start(start_at)
                latest_odds = parse_time(row["latest_odds_at"], default_tz=timezone.utc)
                latest_result_attempt = parse_time(row["latest_result_attempt_at"], default_tz=timezone.utc)
                if not start_at or not cutoff_at:
                    continue
                seconds_to_cutoff = (cutoff_at - now).total_seconds()
                seconds_to_start = (start_at - now).total_seconds()

                if args.collect_results:
                    result_wait = int(row["result_rows"] or 0) < 3
                    result_due = result_interval(seconds_to_start)
                    result_age = (
                        (now - latest_result_attempt).total_seconds()
                        if latest_result_attempt
                        else None
                    )
                    if result_wait and result_due is not None and (result_age is None or result_age >= result_due):
                        counters["result_targets"] += 1
                        count = collect_result(
                            conn,
                            race_date=target_date,
                            jcd=row["jcd"],
                            rno=int(row["rno"]),
                            raw_dir=raw_dir,
                        )
                        conn.commit()
                        if count:
                            counters["result_rows"] += count
                        else:
                            counters["result_empty"] += 1
                        time.sleep(args.sleep_page)
                        continue

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


def scheduled_races(conn, race_date: date) -> list[Any]:
    return conn.execute(
        """
        SELECT r.race_id, r.jcd, r.rno, r.deadline_at,
               (SELECT MAX(captured_at) FROM odds_snapshots os WHERE os.race_id = r.race_id) AS latest_odds_at,
               (SELECT MAX(fetched_at) FROM raw_pages rp WHERE rp.race_id = r.race_id AND rp.page_type = 'result') AS latest_result_attempt_at,
               (SELECT COUNT(*) FROM race_results rr WHERE rr.race_id = r.race_id AND rr.rank IS NOT NULL) AS result_rows
        FROM races r
        WHERE r.race_date = ?
          AND r.deadline_at IS NOT NULL
          AND (
            (SELECT COUNT(*) FROM entries e WHERE e.race_id = r.race_id) = 6
            OR (SELECT COUNT(*) FROM odds_snapshots os WHERE os.race_id = r.race_id) > 0
          )
          AND (SELECT COUNT(*) FROM race_results rr WHERE rr.race_id = r.race_id AND rr.rank IS NOT NULL) < 3
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


def result_interval(seconds_to_start: float) -> float | None:
    if seconds_to_start >= 0:
        return None
    elapsed = -seconds_to_start
    if elapsed <= 90 * 60:
        return 60.0
    if elapsed <= 6 * 60 * 60:
        return 300.0
    return None


def parse_time(value: str | None, *, default_tz: timezone) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=default_tz)
    return parsed.astimezone(JST)


if __name__ == "__main__":
    raise SystemExit(main())

