from __future__ import annotations

import argparse
import json
import time
from datetime import date, timedelta
from pathlib import Path

from .db import connection, init_db
from .historical_official import parse_official_archive
from .http import fetch_bytes, save_payload
from .official import historical_download_url
from .storage import raw_file_exists, record_raw_file


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Official B/K LZH backfill from newest dates to older dates."
    )
    parser.add_argument("--db", default="data/boatrace.sqlite")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--end", default=date.today().isoformat())
    parser.add_argument("--days", type=int, default=3650)
    parser.add_argument("--kind", choices=["program", "result", "both"], default="both")
    parser.add_argument("--sleep", type=float, default=4.0)
    parser.add_argument("--limit-files", type=int)
    args = parser.parse_args(argv)

    init_db(args.db)
    end = date.fromisoformat(args.end)
    start = end - timedelta(days=max(0, args.days - 1))
    kinds = ["program", "result"] if args.kind == "both" else [args.kind]
    stats = {
        "direction": "newest_to_oldest",
        "parser": "official_bk_lzh",
        "start": start.isoformat(),
        "end": end.isoformat(),
        "requested": 0,
        "downloaded": 0,
        "skipped": 0,
        "parsed_races": 0,
        "parsed_entries": 0,
        "parsed_results": 0,
        "parsed_payouts": 0,
    }
    processed = 0
    with connection(args.db) as conn:
        current = end
        while current >= start:
            for kind in kinds:
                if args.limit_files is not None and processed >= args.limit_files:
                    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)
                    return 0
                processed += 1
                stats["requested"] += 1
                url = historical_download_url(kind, current)
                local_path = (
                    Path(args.raw_dir)
                    / kind
                    / f"{current:%Y}"
                    / f"{current:%Y%m%d}.lzh"
                )
                if raw_file_exists(conn, kind=kind, source_url=url) and local_path.exists():
                    stats["skipped"] += 1
                    status_code = "cached"
                else:
                    status_code, payload = fetch_bytes(url, sleep_seconds=args.sleep)
                    saved = save_payload(local_path, payload)
                    record_raw_file(
                        conn,
                        kind=kind,
                        source_url=url,
                        local_path=saved["local_path"],
                        race_date=current.isoformat(),
                        status_code=status_code,
                        sha256=saved["sha256"],
                        bytes_count=saved["bytes"],
                    )
                    if status_code == 200 and payload:
                        stats["downloaded"] += 1
                if local_path.exists():
                    parsed = parse_official_archive(
                        conn, path=local_path, kind=kind, race_date=current
                    )
                    stats["parsed_races"] += parsed.get("races", 0)
                    stats["parsed_entries"] += parsed.get("entries", 0)
                    stats["parsed_results"] += parsed.get("results", 0)
                    stats["parsed_payouts"] += parsed.get("payouts", 0)
                conn.commit()
                print(
                    json.dumps(
                        {
                            "date": current.isoformat(),
                            "kind": kind,
                            "status_code": status_code,
                            "stats": stats,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                if args.sleep:
                    time.sleep(args.sleep)
            current -= timedelta(days=1)
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
