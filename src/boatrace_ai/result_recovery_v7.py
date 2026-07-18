from __future__ import annotations

import argparse
import json
import time
from datetime import date
from pathlib import Path
from typing import Any

from .db import connection, init_db
from .live_safe_patch4 import install
from .webserver_operational2 import now_jst, parse_jst

install()

from .live import collect_result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Recover missing same-day race results.")
    parser.add_argument("--db", default="data/boatrace.sqlite")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--sleep-page", type=float, default=0.4)
    parser.add_argument("--max-races", type=int)
    args = parser.parse_args(argv)

    init_db(args.db)
    target_date = date.fromisoformat(args.date)
    raw_dir = Path(args.raw_dir)
    now = now_jst()
    counters = {"targets": 0, "result_rows": 0, "result_empty": 0, "skipped_future": 0}
    events: list[dict[str, Any]] = []

    with connection(args.db) as conn:
        rows = conn.execute(
            """
            SELECT r.race_id, r.jcd, r.venue_name, r.rno, r.deadline_at,
                   (SELECT COUNT(*) FROM entries e WHERE e.race_id = r.race_id) AS entries,
                   (SELECT COUNT(*) FROM odds_snapshots os WHERE os.race_id = r.race_id) AS odds_rows,
                   (SELECT COUNT(*) FROM race_results rr WHERE rr.race_id = r.race_id AND rr.rank IS NOT NULL) AS result_rows
            FROM races r
            WHERE r.race_date = ?
              AND r.deadline_at IS NOT NULL
              AND (SELECT COUNT(*) FROM race_results rr WHERE rr.race_id = r.race_id AND rr.rank IS NOT NULL) < 3
              AND (
                (SELECT COUNT(*) FROM entries e WHERE e.race_id = r.race_id) = 6
                OR (SELECT COUNT(*) FROM odds_snapshots os WHERE os.race_id = r.race_id) > 0
              )
            ORDER BY r.deadline_at DESC, r.jcd DESC, r.rno DESC
            """,
            (target_date.isoformat(),),
        ).fetchall()
        for row in rows:
            deadline = parse_jst(row["deadline_at"])
            if not deadline or deadline > now:
                counters["skipped_future"] += 1
                continue
            if args.max_races is not None and counters["targets"] >= args.max_races:
                break
            counters["targets"] += 1
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
            events.append(
                {
                    "race_id": row["race_id"],
                    "venue": row["venue_name"],
                    "rno": int(row["rno"]),
                    "rows": count,
                }
            )
            if args.sleep_page:
                time.sleep(args.sleep_page)

    print(json.dumps({"counters": counters, "events": events[:50]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
