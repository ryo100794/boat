#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from boatrace_ai.db import connection, init_db, upsert_entry
from boatrace_ai.parsers import parse_racelist_html
from boatrace_ai.time_semantics import operational_race_date


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reparse the latest saved racelist HTML for one race date.")
    parser.add_argument("--db", default="data/boatrace.sqlite")
    parser.add_argument("--date", help="Fix one date; omit to use the current JST date.")
    args = parser.parse_args(argv)

    race_date = date.fromisoformat(args.date) if args.date else operational_race_date()
    init_db(args.db)
    counters = {"race_date": race_date.isoformat(), "targets": 0, "repaired": 0, "incomplete": 0, "missing_file": 0}
    with connection(args.db) as conn:
        rows = conn.execute(
            """
            SELECT r.race_id, r.jcd, r.rno, rp.local_path, rp.source_url
            FROM races r
            JOIN raw_pages rp ON rp.race_id = r.race_id AND rp.page_type = 'racelist'
            JOIN (
              SELECT race_id, MAX(fetched_at) AS fetched_at
              FROM raw_pages
              WHERE page_type = 'racelist'
              GROUP BY race_id
            ) latest ON latest.race_id = rp.race_id AND latest.fetched_at = rp.fetched_at
            WHERE r.race_date = ?
            GROUP BY r.race_id
            ORDER BY r.jcd, r.rno
            """,
            (race_date.isoformat(),),
        ).fetchall()
        counters["targets"] = len(rows)
        for row in rows:
            path = Path(row["local_path"])
            if not path.exists():
                counters["missing_file"] += 1
                continue
            _, entries = parse_racelist_html(
                path.read_text(encoding="utf-8", errors="replace"),
                race_date=race_date,
                jcd=str(row["jcd"]),
                rno=int(row["rno"]),
                source_url=str(row["source_url"] or "saved_racelist"),
            )
            if len(entries) != 6:
                counters["incomplete"] += 1
                continue
            for entry in entries:
                upsert_entry(conn, str(row["race_id"]), entry)
            counters["repaired"] += 1
        conn.commit()
    print(json.dumps(counters, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
