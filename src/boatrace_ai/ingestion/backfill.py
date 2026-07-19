from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

from ..http import fetch_bytes, save_payload
from ..official import historical_download_url
from ..storage import raw_file_exists, record_raw_file


@dataclass(frozen=True)
class BackfillStats:
    requested: int
    downloaded: int
    skipped: int
    parsed_races: int
    parsed_entries: int
    parsed_results: int


def date_range(start: date, end: date) -> Iterable[date]:
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def backfill_historical(
    conn,
    *,
    start: date,
    end: date,
    kind: str,
    raw_dir: Path,
    sleep_seconds: float,
    skip_existing: bool = True,
) -> BackfillStats:
    kinds = ["program", "result"] if kind == "both" else [kind]
    stats = {
        "requested": 0,
        "downloaded": 0,
        "skipped": 0,
        "parsed_races": 0,
        "parsed_entries": 0,
        "parsed_results": 0,
    }
    for target_date in date_range(start, end):
        for item_kind in kinds:
            stats["requested"] += 1
            url = historical_download_url(item_kind, target_date)
            if skip_existing and raw_file_exists(conn, kind=item_kind, source_url=url):
                stats["skipped"] += 1
                continue
            status_code, payload = fetch_bytes(url, sleep_seconds=sleep_seconds)
            local_path = (
                raw_dir
                / item_kind
                / f"{target_date:%Y}"
                / f"{target_date:%Y%m%d}.lzh"
            )
            saved = save_payload(local_path, payload)
            record_raw_file(
                conn,
                kind=item_kind,
                source_url=url,
                local_path=saved["local_path"],
                race_date=target_date.isoformat(),
                status_code=status_code,
                sha256=saved["sha256"],
                bytes_count=saved["bytes"],
            )
            if status_code == 200 and payload:
                stats["downloaded"] += 1
                parsed = parse_archive(conn, path=local_path, kind=item_kind, race_date=target_date)
                stats["parsed_races"] += parsed["races"]
                stats["parsed_entries"] += parsed["entries"]
                stats["parsed_results"] += parsed["results"]
            conn.commit()
            if sleep_seconds:
                time.sleep(sleep_seconds)
    return BackfillStats(**stats)


def parse_archive(conn, *, path: Path, kind: str, race_date: date) -> dict[str, int]:
    from .archive import parse_official_archive_v6

    return parse_official_archive_v6(
        conn,
        path=path,
        kind=kind,
        race_date=race_date,
    )
