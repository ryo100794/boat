from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .historical_official4 import (
    ENTRY_DEFAULTS,
    normalize,
    parse_official_archive,
    parse_program_text,
    parse_result_text,
    read_lzh,
)
from .series_form import extract_fixed_series_fields


def parse_program_entry(line: str) -> dict[str, Any] | None:
    match = re.match(
        r"^\s*(?P<lane>[1-6])\s*(?P<racer_no>\d{4})(?P<head>.+?)(?P<class>[AB]\d)\s+"
        r"(?P<nwin>\d+\.\d{2})\s+(?P<n2>\d+\.\d{2})\s+"
        r"(?P<lwin>\d+\.\d{2})\s+(?P<l2>\d+\.\d{2})\s+"
        r"(?P<motor>\d+)\s*(?P<m2>\d+\.\d{2})\s*"
        r"(?P<boat>\d{2,3})\s*(?P<b2>\d+\.\d{2})",
        line,
    )
    if not match:
        return None
    head_match = re.match(
        r"(?P<name>.*?)(?P<age>\d{2})(?P<branch>.{2})(?P<weight>\d{2})$",
        match.group("head"),
    )
    if not head_match:
        return None
    return {
        **ENTRY_DEFAULTS,
        "lane": int(match.group("lane")),
        "racer_no": int(match.group("racer_no")),
        "racer_name": head_match.group("name").strip(),
        "racer_class": match.group("class"),
        "branch": head_match.group("branch").strip(),
        "age": int(head_match.group("age")),
        "weight_kg": float(match.group("weight")),
        "national_win_rate": float(match.group("nwin")),
        "national_2_rate": float(match.group("n2")),
        "local_win_rate": float(match.group("lwin")),
        "local_2_rate": float(match.group("l2")),
        "motor_no": int(match.group("motor")),
        "motor_2_rate": float(match.group("m2")),
        "boat_no": int(match.group("boat")),
        "boat_2_rate": float(match.group("b2")),
        "source": "official_program_lzh_v5",
        "raw_text": line,
        **extract_fixed_series_fields(line, data_end=match.end()),
    }


def parse_official_archive_v5(conn, *, path: Path, kind: str, race_date) -> dict[str, int]:
    counters = {"races": 0, "entries": 0, "results": 0, "payouts": 0}
    for filename, payload in read_lzh(path):
        text = normalize_archive_text(payload)
        extracted_path = path.with_suffix("") / filename
        extracted_path.parent.mkdir(parents=True, exist_ok=True)
        extracted_path.write_text(text, encoding="utf-8")
        parsed = (
            parse_program_text_v5(conn, text=text, race_date=race_date)
            if kind == "program"
            else parse_result_text(conn, text=text, race_date=race_date)
        )
        for key, value in parsed.items():
            counters[key] = counters.get(key, 0) + value
    return counters


def parse_program_text_v5(conn, *, text: str, race_date) -> dict[str, int]:
    from .constants import VENUE_BY_CODE
    from .db import race_id, upsert_entry, upsert_race

    counters = {"races": 0, "entries": 0}
    current_jcd: str | None = None
    current_race_id: str | None = None
    for raw_line in text.splitlines():
        line = normalize(raw_line)
        section = re.match(r"^(?P<jcd>\d{2})BBGN", line)
        if section:
            current_jcd = section.group("jcd")
            current_race_id = None
            continue
        if not current_jcd:
            continue
        race_match = re.match(
            r"^\s*(?P<rno>\d{1,2})R\s+(?P<title>.*?)\s+H(?P<distance>\d{3,4})m.*?(?P<deadline>\d{1,2}:\d{2})",
            line,
        )
        if race_match:
            rno = int(race_match.group("rno"))
            current_race_id = race_id(race_date.isoformat(), current_jcd, rno)
            venue = VENUE_BY_CODE.get(current_jcd)
            upsert_race(
                conn,
                {
                    "race_id": current_race_id,
                    "race_date": race_date.isoformat(),
                    "jcd": current_jcd,
                    "venue_name": venue.name if venue else current_jcd,
                    "rno": rno,
                    "title": race_match.group("title").strip(),
                    "distance_m": int(race_match.group("distance")),
                    "deadline_at": f"{race_date.isoformat()}T{race_match.group('deadline')}:00",
                    "status": "scheduled",
                    "source_url": "official_program_lzh",
                },
            )
            counters["races"] += 1
            continue
        if current_race_id:
            entry = parse_program_entry(line)
            if entry:
                upsert_entry(conn, current_race_id, entry)
                counters["entries"] += 1
    return counters


def normalize_archive_text(payload: bytes) -> str:
    from .historical_official3 import decode_text

    return decode_text(payload)
