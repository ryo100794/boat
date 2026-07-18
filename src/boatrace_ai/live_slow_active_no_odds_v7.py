from __future__ import annotations

import argparse
import json
import re
import time
from datetime import date
from pathlib import Path

from .db import connection, init_db
from .http import fetch_text
from .live_safe_patch4 import install
from .modeling_no_odds_v7 import predict_open_races
from .official import race_index_url

install()

from .live import collect_beforeinfo, collect_odds, collect_racelist, collect_result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Slow active-venue live collection and v7 prediction loop.")
    parser.add_argument("--db", default="data/boatrace.sqlite")
    parser.add_argument("--model", default="data/models/win_model_no_odds_v7.joblib")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--sleep-page", type=float, default=2.0)
    parser.add_argument("--sleep-loop", type=float, default=120.0)
    parser.add_argument("--max-loops", type=int)
    parser.add_argument("--predict", action="store_true")
    args = parser.parse_args(argv)

    init_db(args.db)
    target_date = date.fromisoformat(args.date)
    raw_dir = Path(args.raw_dir)
    model_path = Path(args.model)
    loop = 0
    while True:
        counters = {
            "loop": loop,
            "targets": 0,
            "racelist": 0,
            "beforeinfo": 0,
            "odds": 0,
            "result_rows": 0,
            "predicted": 0,
            "prediction_failed": 0,
        }
        with connection(args.db) as conn:
            targets = active_targets(conn, target_date)
            counters["targets"] = len(targets)
            for jcd, rno in targets:
                if collect_racelist(conn, race_date=target_date, jcd=jcd, rno=rno, raw_dir=raw_dir):
                    counters["racelist"] += 1
                conn.commit()
                time.sleep(args.sleep_page)

                if collect_beforeinfo(conn, race_date=target_date, jcd=jcd, rno=rno, raw_dir=raw_dir):
                    counters["beforeinfo"] += 1
                conn.commit()
                time.sleep(args.sleep_page)

                if collect_odds(conn, race_date=target_date, jcd=jcd, rno=rno, raw_dir=raw_dir):
                    counters["odds"] += 1
                conn.commit()
                time.sleep(args.sleep_page)

                counters["result_rows"] += collect_result(
                    conn,
                    race_date=target_date,
                    jcd=jcd,
                    rno=rno,
                    raw_dir=raw_dir,
                )
                conn.commit()
                time.sleep(args.sleep_page)

            if args.predict and model_path.exists():
                predicted = predict_open_races(conn, model_path=model_path, race_date=target_date)
                counters["predicted"] = predicted["predicted"]
                counters["prediction_failed"] = predicted["failed"]
                conn.commit()
        print(json.dumps(counters, ensure_ascii=False), flush=True)
        loop += 1
        if args.max_loops is not None and loop >= args.max_loops:
            return 0
        time.sleep(args.sleep_loop)


def active_targets(conn, race_date: date) -> list[tuple[str, int]]:
    discovered = discover_active_targets(race_date)
    if discovered:
        return discovered
    rows = conn.execute(
        """
        SELECT jcd, rno
        FROM races
        WHERE race_date = ?
          AND (
            deadline_at IS NOT NULL
            OR (SELECT COUNT(*) FROM entries e WHERE e.race_id = races.race_id) = 6
            OR (SELECT COUNT(*) FROM odds_snapshots os WHERE os.race_id = races.race_id) > 0
          )
        ORDER BY jcd, rno
        """,
        (race_date.isoformat(),),
    ).fetchall()
    return [(row["jcd"], int(row["rno"])) for row in rows]


def discover_active_targets(race_date: date) -> list[tuple[str, int]]:
    status_code, html, _ = fetch_text(race_index_url(race_date), sleep_seconds=0.0)
    if status_code != 200:
        return []
    found = set()
    for match in re.finditer(r"[?&]jcd=(\d{2}).*?[?&]rno=(\d{1,2})", html):
        found.add((match.group(1), int(match.group(2))))
    for match in re.finditer(r"[?&]rno=(\d{1,2}).*?[?&]jcd=(\d{2})", html):
        found.add((match.group(2), int(match.group(1))))
    return sorted(found)


if __name__ == "__main__":
    raise SystemExit(main())
