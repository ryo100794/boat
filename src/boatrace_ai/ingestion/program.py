from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any

from ..db import connection, init_db
from .archive import parse_official_archive_v6
from ..http import fetch_bytes, save_payload
from ..official import historical_download_url
from ..storage import raw_file_cache_valid, record_raw_file
from ..runtime.time_semantics import operational_race_date


def load_daily_program(conn, *, race_date: date, raw_dir: Path) -> dict[str, Any]:
    url = historical_download_url("program", race_date)
    local_path = raw_dir / "program" / f"{race_date:%Y}" / f"{race_date:%Y%m%d}.lzh"
    complete_races = conn.execute(
        """
        SELECT COUNT(*) FROM (
          SELECT r.race_id
          FROM races r
          JOIN entries e ON e.race_id = r.race_id
          WHERE r.race_date = ?
          GROUP BY r.race_id
          HAVING COUNT(e.lane) = 6
        )
        """,
        (race_date.isoformat(),),
    ).fetchone()[0]
    scheduled_races = conn.execute(
        "SELECT COUNT(*) FROM races WHERE race_date = ? AND deadline_at IS NOT NULL",
        (race_date.isoformat(),),
    ).fetchone()[0]
    cached = raw_file_cache_valid(conn, kind="program", source_url=url, local_path=local_path)
    if (
        cached
        and int(scheduled_races or 0) > 0
        and int(complete_races or 0) >= int(scheduled_races or 0)
    ):
        return {
            "program_status": "cached_db",
            "program_url": url,
            "program_races": int(complete_races),
            "program_entries": int(complete_races) * 6,
        }

    if not cached:
        status_code, payload = fetch_bytes(url, retries=2, sleep_seconds=1.0)
        saved = save_payload(local_path, payload)
        record_raw_file(
            conn,
            kind="program",
            source_url=url,
            local_path=saved["local_path"],
            race_date=race_date.isoformat(),
            status_code=status_code,
            sha256=saved["sha256"],
            bytes_count=saved["bytes"],
        )
        conn.commit()
        if status_code != 200 or not payload:
            return {"program_status": f"http_{status_code}", "program_url": url, "program_races": 0, "program_entries": 0}

    parsed = parse_official_archive_v6(conn, path=local_path, kind="program", race_date=race_date)
    conn.commit()
    return {
        "program_status": "parsed",
        "program_url": url,
        "program_races": int(parsed.get("races") or 0),
        "program_entries": int(parsed.get("entries") or 0),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Load the official daily program for the current JST date.")
    parser.add_argument("--db", default="data/boatrace.sqlite")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--date", help="Fix one date; omit to use the current JST date.")
    args = parser.parse_args(argv)
    race_date = date.fromisoformat(args.date) if args.date else operational_race_date()
    init_db(args.db)
    with connection(args.db) as conn:
        result = load_daily_program(conn, race_date=race_date, raw_dir=Path(args.raw_dir))
    print(json.dumps({"race_date": race_date.isoformat(), **result}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
