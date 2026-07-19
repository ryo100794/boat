from __future__ import annotations

import re
import time
from datetime import date, datetime, timezone
from pathlib import Path

from .constants import RACES_PER_DAY, VENUE_BY_CODE, VENUES
from .db import (
    insert_odds_snapshot,
    race_id,
    upsert_entry,
    upsert_race,
)
from .http import fetch_text, save_payload
from .official import race_index_url, race_page_url
from .result_parser import parse_result_html_v2
from .parsers import (
    parse_beforeinfo_html,
    parse_odds3t_html,
    parse_racelist_html,
    parse_result_html,
)
from .storage import (
    insert_beforeinfo_rows,
    record_raw_page,
    upsert_payout,
    upsert_result_row,
    upsert_result_status,
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def discover_races(
    race_date: date, *, sleep_seconds: float = 0.0, fallback_all: bool = True
) -> list[tuple[str, int]]:
    url = race_index_url(race_date)
    status_code, html, _ = fetch_text(url, sleep_seconds=sleep_seconds)
    fallback = [(venue.code, rno) for venue in VENUES for rno in RACES_PER_DAY] if fallback_all else []
    if status_code != 200:
        return fallback
    found = set()
    for match in re.finditer(r"[?&]jcd=(\d{2}).*?[?&]rno=(\d{1,2})", html):
        found.add((match.group(1), int(match.group(2))))
    for match in re.finditer(r"[?&]rno=(\d{1,2}).*?[?&]jcd=(\d{2})", html):
        found.add((match.group(2), int(match.group(1))))
    return sorted(found) or fallback


def collect_live_once(
    conn,
    *,
    race_date: date,
    raw_dir: Path,
    sleep_seconds: float = 0.4,
    jcd: str | None = None,
    rno: int | None = None,
) -> dict[str, int]:
    targets = [(jcd.zfill(2), int(rno))] if jcd and rno else discover_races(race_date)
    counters = {
        "targets": len(targets),
        "racelist": 0,
        "beforeinfo": 0,
        "odds": 0,
        "results": 0,
        "skipped": 0,
    }
    for race_jcd, race_rno in targets:
        rid = race_id(race_date.isoformat(), race_jcd, race_rno)
        if collect_racelist(conn, race_date=race_date, jcd=race_jcd, rno=race_rno, raw_dir=raw_dir):
            counters["racelist"] += 1
        else:
            counters["skipped"] += 1
        if collect_beforeinfo(conn, race_date=race_date, jcd=race_jcd, rno=race_rno, raw_dir=raw_dir):
            counters["beforeinfo"] += 1
        if collect_odds(conn, race_date=race_date, jcd=race_jcd, rno=race_rno, raw_dir=raw_dir):
            counters["odds"] += 1
        result_count = collect_result(conn, race_date=race_date, jcd=race_jcd, rno=race_rno, raw_dir=raw_dir)
        counters["results"] += result_count
        conn.commit()
        _ensure_minimal_race(conn, race_date=race_date, jcd=race_jcd, rno=race_rno, status="scheduled")
        if sleep_seconds:
            time.sleep(sleep_seconds)
    return counters


def collect_racelist(conn, *, race_date: date, jcd: str, rno: int, raw_dir: Path) -> bool:
    url = race_page_url("racelist", race_date, jcd, rno)
    html = _fetch_page(conn, page_type="racelist", race_date=race_date, jcd=jcd, rno=rno, url=url, raw_dir=raw_dir)
    if not html:
        return False
    meta, entries = parse_racelist_html(html, race_date=race_date, jcd=jcd, rno=rno, source_url=url)
    if meta.get("status") == "no_data" or len(entries) != 6:
        return False
    rid = upsert_race(conn, meta)
    for entry in entries:
        upsert_entry(conn, rid, entry)
    return True


def collect_beforeinfo(conn, *, race_date: date, jcd: str, rno: int, raw_dir: Path) -> bool:
    rid = race_id(race_date.isoformat(), jcd, rno)
    url = race_page_url("beforeinfo", race_date, jcd, rno)
    html = _fetch_page(conn, page_type="beforeinfo", race_date=race_date, jcd=jcd, rno=rno, url=url, raw_dir=raw_dir)
    if not html:
        return False
    parsed = parse_beforeinfo_html(html)
    if not parsed["rows"]:
        return False
    _ensure_minimal_race(conn, race_date=race_date, jcd=jcd, rno=rno, status="scheduled")
    insert_beforeinfo_rows(conn, race_id=rid, captured_at=utc_now_iso(), rows=parsed["rows"])
    return True


def collect_odds(conn, *, race_date: date, jcd: str, rno: int, raw_dir: Path) -> bool:
    rid = race_id(race_date.isoformat(), jcd, rno)
    url = race_page_url("odds3t", race_date, jcd, rno)
    html = _fetch_page(conn, page_type="odds3t", race_date=race_date, jcd=jcd, rno=rno, url=url, raw_dir=raw_dir)
    if not html:
        return False
    parsed = parse_odds3t_html(html)
    if parsed["parsed_count"] < 20:
        return False
    _ensure_minimal_race(conn, race_date=race_date, jcd=jcd, rno=rno, status="scheduled")
    insert_odds_snapshot(
        conn,
        rid,
        utc_now_iso(),
        parsed.get("source_update_time"),
        parsed["odds"],
        url,
        parsed,
    )
    return True


def collect_result(conn, *, race_date: date, jcd: str, rno: int, raw_dir: Path) -> int:
    rid = race_id(race_date.isoformat(), jcd, rno)
    url = race_page_url("result", race_date, jcd, rno)
    html = _fetch_page(conn, page_type="result", race_date=race_date, jcd=jcd, rno=rno, url=url, raw_dir=raw_dir)
    if not html:
        return 0
    parsed = parse_result_html_v2(html)
    if parsed["status"] == "unknown":
        parsed = parse_result_html(html)
    if parsed["status"] == "no_data":
        return 0
    _ensure_minimal_race(conn, race_date=race_date, jcd=jcd, rno=rno, status=parsed["status"])
    upsert_result_status(conn, race_id=rid, row=parsed)
    for row in parsed["rows"]:
        upsert_result_row(conn, race_id=rid, row=row)
    for payout in parsed["payouts"]:
        upsert_payout(conn, race_id=rid, row=payout)
    return len(parsed["rows"])


def monitor_live(
    conn,
    *,
    race_date: date,
    raw_dir: Path,
    model_path: Path,
    interval_seconds: int,
    max_loops: int | None = None,
    jcd: str | None = None,
    rno: int | None = None,
) -> None:
    loops = 0
    while True:
        collect_live_once(conn, race_date=race_date, raw_dir=raw_dir, jcd=jcd, rno=rno)
        if model_path.exists():
            from .modeling import predict_open_races

            predict_open_races(conn, model_path=model_path, race_date=race_date, jcd=jcd, rno=rno)
            conn.commit()
        loops += 1
        if max_loops is not None and loops >= max_loops:
            return
        time.sleep(interval_seconds)


def _fetch_page(
    conn,
    *,
    page_type: str,
    race_date: date,
    jcd: str,
    rno: int,
    url: str,
    raw_dir: Path,
) -> str | None:
    status_code, html, payload = fetch_text(url)
    if status_code != 200:
        return None
    captured = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = raw_dir / "pages" / f"{race_date:%Y%m%d}" / jcd.zfill(2) / f"{int(rno):02d}" / f"{page_type}-{captured}.html"
    saved = save_payload(path, payload)
    record_raw_page(
        conn,
        page_type=page_type,
        race_id=race_id(race_date.isoformat(), jcd, rno),
        source_url=url,
        local_path=saved["local_path"],
        sha256=saved["sha256"],
        bytes_count=saved["bytes"],
    )
    return html


def _ensure_minimal_race(conn, *, race_date: date, jcd: str, rno: int, status: str) -> None:
    venue = VENUE_BY_CODE.get(jcd.zfill(2))
    upsert_race(
        conn,
        {
            "race_id": race_id(race_date.isoformat(), jcd, rno),
            "race_date": race_date.isoformat(),
            "jcd": jcd.zfill(2),
            "venue_name": venue.name if venue else jcd.zfill(2),
            "rno": int(rno),
            "status": status,
        },
    )
