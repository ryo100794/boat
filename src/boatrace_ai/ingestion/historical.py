from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

from .archives import decode_japanese_text, extract_lzh
from ..constants import VENUE_BY_CODE
from ..db import race_id, upsert_entry, upsert_race
from ..http import fetch_bytes, save_payload
from ..official import historical_download_url, racer_stats_url
from .parsers import parse_historical_result_text, parse_racer_stats_bytes, to_float, to_int
from ..storage import (
    raw_file_exists,
    record_raw_file,
    upsert_racer_period_stats,
    upsert_result_row,
)


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
            conn.commit()
            if status_code == 200 and payload:
                stats["downloaded"] += 1
                parsed = parse_historical_archive(
                    conn, path=local_path, kind=item_kind, race_date=target_date
                )
                stats["parsed_races"] += parsed.get("races", 0)
                stats["parsed_entries"] += parsed.get("entries", 0)
                stats["parsed_results"] += parsed.get("results", 0)
                conn.commit()
            if sleep_seconds:
                time.sleep(sleep_seconds)
    return BackfillStats(**stats)


def parse_historical_archive(conn, *, path: Path, kind: str, race_date: date) -> dict[str, int]:
    members = extract_lzh(path)
    counters = {"races": 0, "entries": 0, "results": 0}
    for filename, payload in members:
        text = decode_japanese_text(payload)
        extracted_path = path.with_suffix("") / filename
        extracted_path.parent.mkdir(parents=True, exist_ok=True)
        extracted_path.write_text(text, encoding="utf-8")
        if kind == "program":
            rows = parse_historical_program_text(text, race_date=race_date)
            for race_payload, entries in rows:
                rid = upsert_race(conn, race_payload)
                counters["races"] += 1
                for entry in entries:
                    upsert_entry(conn, rid, entry)
                    counters["entries"] += 1
        elif kind == "result":
            rows = parse_historical_result_text(text, race_date=race_date)
            for row in rows:
                rid = row["race_id"]
                race_date_part, jcd, rno = rid.split("-")
                venue = VENUE_BY_CODE.get(jcd)
                upsert_race(
                    conn,
                    {
                        "race_id": rid,
                        "race_date": race_date_part,
                        "jcd": jcd,
                        "venue_name": venue.name if venue else jcd,
                        "rno": int(rno),
                        "status": "final",
                    },
                )
                upsert_result_row(conn, race_id=rid, row=row)
                counters["results"] += 1
    return counters


def parse_historical_program_text(
    text: str,
    *,
    race_date: date,
) -> list[tuple[dict[str, object], list[dict[str, object]]]]:
    rows: list[tuple[dict[str, object], list[dict[str, object]]]] = []
    current_jcd: str | None = None
    current_rno: int | None = None
    current_entries: list[dict[str, object]] = []
    current_meta: dict[str, object] | None = None

    def flush() -> None:
        nonlocal current_meta, current_entries
        if current_meta and current_entries:
            rows.append((current_meta, current_entries))
        current_entries = []

    for raw_line in text.splitlines():
        line = _normalize(raw_line)
        for code, venue in VENUE_BY_CODE.items():
            if venue.name in line:
                current_jcd = code
                break
        race_match = re.search(r"(^|\s)(?P<rno>\d{1,2})R(\s|$)", line)
        if race_match and current_jcd:
            flush()
            current_rno = int(race_match.group("rno"))
            venue = VENUE_BY_CODE.get(current_jcd)
            current_meta = {
                "race_id": race_id(race_date.isoformat(), current_jcd, current_rno),
                "race_date": race_date.isoformat(),
                "jcd": current_jcd,
                "venue_name": venue.name if venue else current_jcd,
                "rno": current_rno,
                "status": "scheduled",
                "source_url": "historical_program",
            }
            continue
        if not current_meta:
            continue
        entry = _parse_program_entry_line(line)
        if entry:
            current_entries.append(entry)
    flush()
    return rows


def _normalize(value: str) -> str:
    return (
        value.translate(str.maketrans("０１２３４５６７８９．：－　", "0123456789.:- "))
        .replace("\xa0", " ")
        .strip()
    )


def _parse_program_entry_line(line: str) -> dict[str, object] | None:
    match = re.match(
        r"^\s*(?P<lane>[1-6])\s+(?P<racer_no>\d{4})\s+"
        r"(?P<name>.+?)\s+(?P<class>[AB]\d)(?:\s+|$)(?P<tail>.*)$",
        line,
    )
    if not match:
        match = re.match(
            r"^\s*(?P<lane>[1-6])\s+(?P<racer_no>\d{4})\s+"
            r"(?P<name>[^\d]+?)\s+(?P<tail>.*)$",
            line,
        )
    if not match:
        return None
    tail = match.groupdict().get("tail") or ""
    numbers = [to_float(token) for token in re.findall(r"\d+\.\d+|\d+", tail)]
    numbers = [value for value in numbers if value is not None]
    racer_class = match.groupdict().get("class")
    if not racer_class:
        class_match = re.search(r"\b([AB]\d)\b", tail)
        racer_class = class_match.group(1) if class_match else None
    return {
        "lane": int(match.group("lane")),
        "racer_no": int(match.group("racer_no")),
        "racer_name": match.group("name").strip(),
        "racer_class": racer_class,
        "age": _first_int_in_range(numbers, 16, 90),
        "weight_kg": _first_float_in_range(numbers, 35, 80),
        "avg_st": _first_float_in_range(numbers, 0.01, 0.5),
        "national_win_rate": _first_float_in_range(numbers, 1.0, 10.0),
        "motor_no": _first_int_in_range(numbers, 1, 99),
        "boat_no": None,
        "source": "historical_program",
        "raw_text": line,
    }


def _first_float_in_range(
    values: list[float],
    low: float,
    high: float,
) -> float | None:
    return next((value for value in values if low <= value <= high), None)


def _first_int_in_range(values: list[float], low: int, high: int) -> int | None:
    value = _first_float_in_range(values, float(low), float(high))
    if value is None:
        return None
    return int(value)


def fetch_racer_stats(
    conn,
    *,
    from_year: int,
    to_year: int,
    raw_dir: Path,
    sleep_seconds: float,
    skip_existing: bool = True,
) -> int:
    stored = 0
    for year in range(from_year, to_year + 1):
        for half in (1, 2):
            url = racer_stats_url(year, half)
            kind = "racer_stats"
            if skip_existing and raw_file_exists(conn, kind=kind, source_url=url):
                continue
            status_code, payload = fetch_bytes(url, sleep_seconds=sleep_seconds)
            local_path = raw_dir / "racer_stats" / str(year) / f"{year}-{half}.lzh"
            saved = save_payload(local_path, payload)
            record_raw_file(
                conn,
                kind=kind,
                source_url=url,
                local_path=saved["local_path"],
                year=year,
                half=half,
                status_code=status_code,
                sha256=saved["sha256"],
                bytes_count=saved["bytes"],
            )
            if status_code == 200:
                for filename, member in extract_lzh(local_path):
                    rows = parse_racer_stats_bytes(member, year=year, half=half)
                    upsert_racer_period_stats(conn, year=year, half=half, rows=rows)
                    stored += len(rows)
                    text_path = local_path.with_suffix("") / filename
                    text_path.parent.mkdir(parents=True, exist_ok=True)
                    text_path.write_bytes(member)
            conn.commit()
            if sleep_seconds:
                time.sleep(sleep_seconds)
    return stored
