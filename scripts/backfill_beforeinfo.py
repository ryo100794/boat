from __future__ import annotations

import argparse
import json
import time
from datetime import date
from pathlib import Path

from boatrace_ai.db import connection, init_db
from boatrace_ai.ingestion.live import collect_beforeinfo


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Backfill official beforeinfo pages from newest race to oldest."
    )
    parser.add_argument("--db", default="data/boatrace.sqlite")
    parser.add_argument("--date", required=True)
    parser.add_argument("--raw-dir", default="data/raw", type=Path)
    parser.add_argument("--retries", default=3, type=int)
    parser.add_argument("--sleep", default=0.4, type=float)
    args = parser.parse_args(argv)

    race_date = date.fromisoformat(args.date)
    init_db(args.db)
    with connection(args.db) as conn:
        rows = conn.execute(
            """
            SELECT r.race_id, r.jcd, r.rno,
                   (SELECT COUNT(DISTINCT b.lane)
                    FROM beforeinfo b WHERE b.race_id = r.race_id) AS lanes
            FROM races r
            WHERE r.race_date = ? AND r.deadline_at IS NOT NULL
            ORDER BY r.deadline_at DESC, r.jcd DESC, r.rno DESC
            """,
            (race_date.isoformat(),),
        ).fetchall()
        pending = [row for row in rows if int(row["lanes"] or 0) < 6]
        counters = {
            "race_date": race_date.isoformat(),
            "targets": len(pending),
            "ok": 0,
            "failed": 0,
            "attempts": 0,
            "order": "newest_to_oldest",
        }
        for index, row in enumerate(pending, start=1):
            ok = False
            for attempt in range(1, max(1, args.retries) + 1):
                counters["attempts"] += 1
                ok = collect_beforeinfo(
                    conn,
                    race_date=race_date,
                    jcd=str(row["jcd"]),
                    rno=int(row["rno"]),
                    raw_dir=args.raw_dir,
                )
                conn.commit()
                if ok:
                    break
                time.sleep(args.sleep * attempt)
            counters["ok" if ok else "failed"] += 1
            if index % 12 == 0 or index == len(pending):
                print(
                    json.dumps(
                        {**counters, "processed": index},
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            time.sleep(args.sleep)
    return 0 if counters["failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
