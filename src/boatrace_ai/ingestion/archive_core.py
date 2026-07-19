from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Any

from ..constants import VENUE_BY_CODE
from ..db import race_id, upsert_entry, upsert_race
from ..storage import upsert_payout, upsert_result_row


TRANS = str.maketrans("０１２３４５６７８９．：－　ＲＨｍ", "0123456789.:- RHm")


def parse_official_archive(conn, *, path: Path, kind: str, race_date: date) -> dict[str, int]:
    counters = {"races": 0, "entries": 0, "results": 0, "payouts": 0}
    for filename, payload in read_lzh(path):
        text = decode_text(payload)
        extracted_path = path.with_suffix("") / filename
        extracted_path.parent.mkdir(parents=True, exist_ok=True)
        extracted_path.write_text(text, encoding="utf-8")
        parsed = (
            parse_program_text(conn, text=text, race_date=race_date)
            if kind == "program"
            else parse_result_text(conn, text=text, race_date=race_date)
        )
        for key, value in parsed.items():
            counters[key] = counters.get(key, 0) + value
    return counters


def read_lzh(path: Path) -> list[tuple[str, bytes]]:
    import lhafile

    archive = lhafile.Lhafile(str(path))
    return [
        (info.filename, archive.read(info.filename))
        for info in archive.infolist()
        if not info.filename.endswith("/")
    ]


def decode_text(payload: bytes) -> str:
    for encoding in ("cp932", "shift_jis", "utf-8"):
        try:
            return payload.decode(encoding)
        except UnicodeDecodeError:
            continue
    return payload.decode("cp932", errors="replace")


def parse_program_text(conn, *, text: str, race_date: date) -> dict[str, int]:
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
        "lane": int(match.group("lane")),
        "racer_no": int(match.group("racer_no")),
        "racer_name": head_match.group("name").strip(),
        "racer_class": match.group("class"),
        "branch": head_match.group("branch").strip(),
        "age": int(head_match.group("age")),
        "weight_kg": float(head_match.group("weight")),
        "national_win_rate": float(match.group("nwin")),
        "national_2_rate": float(match.group("n2")),
        "local_win_rate": float(match.group("lwin")),
        "local_2_rate": float(match.group("l2")),
        "motor_no": int(match.group("motor")),
        "motor_2_rate": float(match.group("m2")),
        "boat_no": int(match.group("boat")),
        "boat_2_rate": float(match.group("b2")),
        "source": "official_program_lzh_v3",
        "raw_text": line,
    }


def parse_result_text(conn, *, text: str, race_date: date) -> dict[str, int]:
    counters = {"races": 0, "results": 0, "payouts": 0}
    current_jcd: str | None = None
    current_race_id: str | None = None
    for raw_line in text.splitlines():
        line = normalize(raw_line)
        section = re.match(r"^(?P<jcd>\d{2})KBGN", line)
        if section:
            current_jcd = section.group("jcd")
            current_race_id = None
            continue
        if not current_jcd:
            continue
        race_match = re.match(
            r"^\s*(?P<rno>\d{1,2})R\s+(?P<title>.*?)\s+H(?P<distance>\d{3,4})m",
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
                    "status": "final",
                    "source_url": "official_result_lzh",
                },
            )
            counters["races"] += 1
            continue
        if not current_race_id:
            continue
        result = parse_result_row(line)
        if result:
            upsert_result_row(conn, race_id=current_race_id, row=result)
            counters["results"] += 1
            continue
        payout = parse_payout_row(line)
        if payout:
            upsert_payout(conn, race_id=current_race_id, row=payout)
            counters["payouts"] += 1
    return counters


def parse_result_row(line: str) -> dict[str, Any] | None:
    match = re.match(
        r"^\s*(?P<rank>\d{2})\s+(?P<lane>[1-6])\s+(?P<racer_no>\d{4})\s+"
        r"(?P<name>.*?)\s+(?P<motor>\d+)\s+(?P<boat>\d{2,3})\s+"
        r"(?P<exhibition>\d\.\d{2})\s+(?P<course>[1-6])\s+"
        r"(?P<st>0?\.\d{2}|[0-9.]+)",
        line,
    )
    if not match:
        return None
    return {
        "lane": int(match.group("lane")),
        "rank": int(match.group("rank")),
        "course": int(match.group("course")),
        "start_timing": parse_float(match.group("st")),
        "racer_no": int(match.group("racer_no")),
        "racer_name": match.group("name").strip(),
        "motor_no": int(match.group("motor")),
        "boat_no": int(match.group("boat")),
        "exhibition_time": float(match.group("exhibition")),
        "source": "official_result_lzh_v3",
        "raw_text": line,
    }


def parse_payout_row(line: str) -> dict[str, Any] | None:
    match = re.match(
        r"^\s*(?P<bet>3連単|3連複|2連単|2連複|拡連複|単勝|複勝)\s+"
        r"(?P<combo>[1-6-]+)\s+(?P<payout>\d+)(?:\s+人気\s+(?P<popularity>\d+))?",
        line,
    )
    if not match:
        return None
    return {
        "bet_type": match.group("bet"),
        "combination": match.group("combo"),
        "payout_yen": int(match.group("payout")),
        "popularity": int(match.group("popularity")) if match.group("popularity") else None,
        "source": "official_result_lzh_v3",
        "raw_text": line,
    }


def normalize(value: str) -> str:
    return value.translate(TRANS).replace("\xa0", " ").strip()


def parse_float(value: str) -> float | None:
    if value.startswith("."):
        value = "0" + value
    try:
        return float(value)
    except ValueError:
        return None
