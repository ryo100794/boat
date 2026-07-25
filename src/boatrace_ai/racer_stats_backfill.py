from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from .db import connection, init_db
from .ingestion.historical import fetch_racer_stats


def run_backfill(
    conn: Any,
    *,
    from_year: int,
    to_year: int,
    raw_dir: Path,
    sleep_seconds: float,
) -> dict[str, Any]:
    if from_year < 2000 or to_year > 2100 or from_year > to_year:
        raise ValueError("invalid racer statistics year range")
    if to_year - from_year > 20:
        raise ValueError("racer statistics range must not exceed 21 years")
    if sleep_seconds < 0 or sleep_seconds > 10:
        raise ValueError("sleep_seconds must be between 0 and 10")
    stored = fetch_racer_stats(
        conn,
        from_year=from_year,
        to_year=to_year,
        raw_dir=raw_dir,
        sleep_seconds=sleep_seconds,
    )
    row = conn.execute(
        """
        SELECT COUNT(*) AS row_count,
               COUNT(DISTINCT CAST(year AS TEXT) || ':' || CAST(half AS TEXT))
                 AS period_count,
               MIN(year) AS first_year,
               MAX(year) AS last_year
        FROM racer_period_stats
        """
    ).fetchone()
    return {
        "status": "completed",
        "task": "racer_stats_backfill",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "from_year": from_year,
        "to_year": to_year,
        "stored_rows": int(stored),
        "database_rows": int(row["row_count"] or 0),
        "period_count": int(row["period_count"] or 0),
        "first_year": row["first_year"],
        "last_year": row["last_year"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fetch official half-year racer statistics"
    )
    parser.add_argument("--db", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--from-year", type=int, required=True)
    parser.add_argument("--to-year", type=int, required=True)
    parser.add_argument("--sleep-seconds", type=float, default=1.5)
    args = parser.parse_args(argv)
    init_db(args.db)
    with connection(args.db) as conn:
        result = run_backfill(
            conn,
            from_year=args.from_year,
            to_year=args.to_year,
            raw_dir=args.raw_dir,
            sleep_seconds=args.sleep_seconds,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
