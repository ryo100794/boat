#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from boatrace_ai.db import insert_odds_snapshot
from boatrace_ai.ingestion.parsers import parse_odds3t_html
from boatrace_ai.odds_quality import TRIFECTA_PARSER_VERSION, describe_trifecta_market
from boatrace_ai.postgresql import connection


PAGE_PATTERN = re.compile(
    r"(?:^|/)pages/(?P<date>\d{8})/(?P<jcd>\d{2})/(?P<rno>\d{2})/"
    r"odds3t-(?P<stamp>\d{8}T\d{6}Z)\.html$"
)


def page_identity(path: Path) -> tuple[str, datetime]:
    match = PAGE_PATTERN.search(path.as_posix())
    if not match:
        raise ValueError(f"not an archived odds3t page: {path}")
    day = match.group("date")
    race_id = (
        f"{day[:4]}-{day[4:6]}-{day[6:8]}-"
        f"{match.group('jcd')}-{int(match.group('rno')):02d}"
    )
    captured_at = datetime.strptime(
        match.group("stamp"), "%Y%m%dT%H%M%SZ"
    ).replace(tzinfo=timezone.utc)
    return race_id, captured_at


def restore_pages(
    conn,
    paths: Iterable[Path],
    *,
    apply: bool,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "examined": 0,
        "valid": 0,
        "restored": 0,
        "already_present": 0,
        "parse_failed": 0,
        "raw_metadata_missing": 0,
        "races": {},
    }
    for path in sorted(paths):
        summary["examined"] += 1
        race_id, filename_time = page_identity(path)
        parsed = parse_odds3t_html(path.read_text(encoding="utf-8", errors="replace"))
        odds = parsed.get("odds") or {}
        market_shape = describe_trifecta_market(odds, allow_zero=True)
        if (
            parsed.get("parser_version") != TRIFECTA_PARSER_VERSION
            or int(parsed.get("parsed_count") or 0) != 120
            or market_shape is None
        ):
            summary["parse_failed"] += 1
            continue
        parsed["market_shape"] = market_shape

        metadata = conn.execute(
            """
            SELECT source_url, fetched_at
            FROM raw_pages
            WHERE race_id = ? AND page_type = 'odds3t'
              AND local_path LIKE ?
            ORDER BY ABS(EXTRACT(EPOCH FROM (fetched_at - ?::timestamptz)))
            LIMIT 1
            """,
            (race_id, f"%/{path.name}", filename_time.isoformat()),
        ).fetchone()
        if metadata is None:
            summary["raw_metadata_missing"] += 1
            continue
        captured_at = str(metadata["fetched_at"])
        exists = conn.execute(
            """
            SELECT 1 FROM odds_snapshots
            WHERE race_id = ? AND bet_type = 'trifecta'
              AND captured_at = ?::timestamptz
            LIMIT 1
            """,
            (race_id, captured_at),
        ).fetchone()
        if exists is not None:
            summary["already_present"] += 1
            continue

        summary["valid"] += 1
        race_summary = summary["races"].setdefault(
            race_id, {"valid": 0, "restored": 0}
        )
        race_summary["valid"] += 1
        if not apply:
            continue
        insert_odds_snapshot(
            conn,
            race_id,
            captured_at,
            parsed.get("source_update_time"),
            {
                combination: value
                for combination, value in odds.items()
                if value is not None
            },
            str(metadata["source_url"]),
            parsed,
        )
        summary["restored"] += 1
        race_summary["restored"] += 1
    if apply:
        conn.commit()
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Restore missing PostgreSQL odds snapshots from extracted raw pages."
    )
    parser.add_argument("--postgres-dsn", required=True)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument(
        "--race-id",
        action="append",
        default=[],
        help="Limit restoration to one race ID; may be repeated.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write verified missing snapshots. The default is a dry run.",
    )
    args = parser.parse_args()
    paths = list(args.raw_root.rglob("odds3t-*.html"))
    if args.race_id:
        wanted = set(args.race_id)
        paths = [path for path in paths if page_identity(path)[0] in wanted]
    with connection(args.postgres_dsn) as conn:
        result = restore_pages(conn, paths, apply=args.apply)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["parse_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
