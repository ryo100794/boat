from __future__ import annotations

import argparse
import json
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from .db import connection, init_db
from .historical_official4 import parse_official_archive
from .http import fetch_bytes, save_payload
from .official import historical_download_url
from .storage import raw_file_exists, record_raw_file


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Safe official B/K LZH backfill from newest dates to older dates."
    )
    parser.add_argument("--db", default="data/boatrace.sqlite")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--end", default=date.today().isoformat())
    parser.add_argument("--days", type=int, default=3650)
    parser.add_argument("--kind", choices=["program", "result", "both"], default="both")
    parser.add_argument("--sleep", type=float, default=4.0)
    parser.add_argument("--limit-files", type=int)
    parser.add_argument("--retry-parse-failed", action="store_true")
    args = parser.parse_args(argv)

    init_db(args.db)
    end = date.fromisoformat(args.end)
    start = end - timedelta(days=max(0, args.days - 1))
    kinds = ["program", "result"] if args.kind == "both" else [args.kind]
    stats = {
        "direction": "newest_to_oldest",
        "parser": "official_bk_lzh_v4_safe",
        "start": start.isoformat(),
        "end": end.isoformat(),
        "requested": 0,
        "downloaded": 0,
        "skipped": 0,
        "fetch_failed": 0,
        "parse_failed": 0,
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
                event = process_file(
                    conn,
                    raw_dir=Path(args.raw_dir),
                    current=current,
                    kind=kind,
                    sleep=args.sleep,
                    retry_parse_failed=args.retry_parse_failed,
                )
                _merge_stats(stats, event)
                conn.commit()
                print(
                    json.dumps(
                        {
                            "date": current.isoformat(),
                            "kind": kind,
                            **event,
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


def process_file(
    conn,
    *,
    raw_dir: Path,
    current: date,
    kind: str,
    sleep: float,
    retry_parse_failed: bool,
) -> dict[str, Any]:
    url = historical_download_url(kind, current)
    local_path = raw_dir / kind / f"{current:%Y}" / f"{current:%Y%m%d}.lzh"
    event: dict[str, Any] = {
        "status_code": "cached",
        "downloaded": 0,
        "skipped": 0,
        "fetch_failed": 0,
        "parse_failed": 0,
        "parsed_races": 0,
        "parsed_entries": 0,
        "parsed_results": 0,
        "parsed_payouts": 0,
    }
    if local_path.exists() or raw_file_exists(conn, kind=kind, source_url=url):
        event["skipped"] = 1
    else:
        fetched = _fetch_and_record(conn, url=url, local_path=local_path, kind=kind, current=current, sleep=sleep)
        event["status_code"] = fetched["status_code"]
        if fetched["status_code"] == 200 and fetched["bytes"]:
            event["downloaded"] = 1
        else:
            event["fetch_failed"] = 1
            return event

    parsed = _try_parse(conn, local_path=local_path, kind=kind, current=current)
    if parsed["ok"]:
        _add_parsed(event, parsed)
        return event

    if retry_parse_failed:
        fetched = _fetch_and_record(conn, url=url, local_path=local_path, kind=kind, current=current, sleep=sleep)
        event["status_code"] = fetched["status_code"]
        if fetched["status_code"] != 200 or not fetched["bytes"]:
            event["fetch_failed"] = 1
            event["parse_failed"] = 1
            event["parse_error"] = parsed["error"]
            return event
        event["downloaded"] = max(1, int(event["downloaded"]))
        parsed = _try_parse(conn, local_path=local_path, kind=kind, current=current)
        if parsed["ok"]:
            _add_parsed(event, parsed)
            event["retried_parse"] = True
            return event

    event["parse_failed"] = 1
    event["parse_error"] = parsed["error"]
    return event


def _fetch_and_record(conn, *, url: str, local_path: Path, kind: str, current: date, sleep: float) -> dict[str, Any]:
    status_code, payload = fetch_bytes(url, sleep_seconds=sleep)
    saved = save_payload(local_path, payload)
    record_raw_file(
        conn,
        kind=kind,
        source_url=url,
        local_path=saved["local_path"],
        race_date=current.isoformat(),
        status_code=int(status_code),
        sha256=saved["sha256"],
        bytes_count=saved["bytes"],
    )
    return {"status_code": int(status_code), "bytes": int(saved["bytes"])}


def _try_parse(conn, *, local_path: Path, kind: str, current: date) -> dict[str, Any]:
    try:
        parsed = parse_official_archive(conn, path=local_path, kind=kind, race_date=current)
        return {"ok": True, **parsed}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _add_parsed(event: dict[str, Any], parsed: dict[str, Any]) -> None:
    event["parsed_races"] += int(parsed.get("races", 0))
    event["parsed_entries"] += int(parsed.get("entries", 0))
    event["parsed_results"] += int(parsed.get("results", 0))
    event["parsed_payouts"] += int(parsed.get("payouts", 0))


def _merge_stats(stats: dict[str, Any], event: dict[str, Any]) -> None:
    for key in (
        "downloaded",
        "skipped",
        "fetch_failed",
        "parse_failed",
        "parsed_races",
        "parsed_entries",
        "parsed_results",
        "parsed_payouts",
    ):
        stats[key] += int(event.get(key) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
