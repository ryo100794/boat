from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


def record_raw_file(
    conn: sqlite3.Connection,
    *,
    kind: str,
    source_url: str,
    local_path: str,
    race_date: str | None = None,
    year: int | None = None,
    half: int | None = None,
    status_code: int | None = None,
    sha256: str | None = None,
    bytes_count: int | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO raw_files (
          kind, source_url, local_path, race_date, year, half,
          status_code, sha256, bytes
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(kind, source_url) DO UPDATE SET
          local_path=excluded.local_path,
          race_date=excluded.race_date,
          year=excluded.year,
          half=excluded.half,
          status_code=excluded.status_code,
          sha256=excluded.sha256,
          bytes=excluded.bytes,
          fetched_at=CURRENT_TIMESTAMP
        """,
        (
            kind,
            source_url,
            local_path,
            race_date,
            year,
            half,
            status_code,
            sha256,
            bytes_count,
        ),
    )


def raw_file_exists(conn: sqlite3.Connection, *, kind: str, source_url: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM raw_files WHERE kind = ? AND source_url = ? LIMIT 1",
        (kind, source_url),
    ).fetchone()
    return row is not None


def raw_file_cache_valid(
    conn: sqlite3.Connection, *, kind: str, source_url: str, local_path: str | Path
) -> bool:
    row = conn.execute(
        """
        SELECT status_code, bytes
        FROM raw_files
        WHERE kind = ? AND source_url = ?
        LIMIT 1
        """,
        (kind, source_url),
    ).fetchone()
    if row is None:
        return False

    status_code, bytes_count = row
    if status_code != 200 or not bytes_count or int(bytes_count) <= 0:
        return False

    path = Path(local_path)
    try:
        return path.exists() and path.stat().st_size > 0
    except OSError:
        return False


def record_raw_page(
    conn: sqlite3.Connection,
    *,
    page_type: str,
    race_id: str | None,
    source_url: str,
    local_path: str,
    sha256: str,
    bytes_count: int,
) -> None:
    conn.execute(
        """
        INSERT INTO raw_pages (
          page_type, race_id, source_url, local_path, sha256, bytes
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (page_type, race_id, source_url, local_path, sha256, bytes_count),
    )


def upsert_result_row(
    conn: sqlite3.Connection,
    *,
    race_id: str,
    row: dict[str, Any],
) -> None:
    payload = {
        "race_id": race_id,
        "lane": row["lane"],
        "rank": row.get("rank"),
        "course": row.get("course"),
        "start_timing": row.get("start_timing"),
        "race_time": row.get("race_time"),
        "decision": row.get("decision"),
        "raw_json": json.dumps(row, ensure_ascii=False, sort_keys=True),
    }
    conn.execute(
        """
        INSERT INTO race_results (
          race_id, lane, rank, course, start_timing, race_time, decision, raw_json
        )
        VALUES (
          :race_id, :lane, :rank, :course, :start_timing,
          :race_time, :decision, :raw_json
        )
        ON CONFLICT(race_id, lane) DO UPDATE SET
          rank=excluded.rank,
          course=excluded.course,
          start_timing=excluded.start_timing,
          race_time=excluded.race_time,
          decision=excluded.decision,
          raw_json=excluded.raw_json,
          updated_at=CURRENT_TIMESTAMP
        """,
        payload,
    )


def upsert_result_status(
    conn: sqlite3.Connection,
    *,
    race_id: str,
    row: dict[str, Any],
) -> None:
    payload = {
        "race_id": race_id,
        "status": row.get("status") or "unknown",
        "trifecta_evaluable": 1 if row.get("trifecta_evaluable", True) else 0,
        "reason": row.get("result_reason") or row.get("reason"),
        "finish_rows": len(row.get("rows") or []),
        "payout_rows": len(row.get("payouts") or []),
        "raw_json": json.dumps(row, ensure_ascii=False, sort_keys=True),
    }
    conn.execute(
        """
        INSERT INTO race_result_status (
          race_id, status, trifecta_evaluable, reason, finish_rows, payout_rows, raw_json
        )
        VALUES (
          :race_id, :status, :trifecta_evaluable, :reason, :finish_rows, :payout_rows, :raw_json
        )
        ON CONFLICT(race_id) DO UPDATE SET
          status=excluded.status,
          trifecta_evaluable=excluded.trifecta_evaluable,
          reason=excluded.reason,
          finish_rows=excluded.finish_rows,
          payout_rows=excluded.payout_rows,
          raw_json=excluded.raw_json,
          updated_at=CURRENT_TIMESTAMP
        """,
        payload,
    )


def upsert_payout(
    conn: sqlite3.Connection,
    *,
    race_id: str,
    row: dict[str, Any],
) -> None:
    conn.execute(
        """
        INSERT INTO payouts (
          race_id, bet_type, combination, payout_yen, popularity, raw_json
        )
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(race_id, bet_type, combination) DO UPDATE SET
          payout_yen=excluded.payout_yen,
          popularity=excluded.popularity,
          raw_json=excluded.raw_json
        """,
        (
            race_id,
            row["bet_type"],
            row["combination"],
            row.get("payout_yen"),
            row.get("popularity"),
            json.dumps(row, ensure_ascii=False, sort_keys=True),
        ),
    )


def insert_beforeinfo_rows(
    conn: sqlite3.Connection,
    *,
    race_id: str,
    captured_at: str,
    rows: list[dict[str, Any]],
) -> None:
    conn.executemany(
        """
        INSERT OR REPLACE INTO beforeinfo (
          race_id, captured_at, lane, weight_kg, exhibition_time, tilt,
          adjusted_weight, propeller, parts_exchange, course, start_timing,
          weather, wind_direction, wind_speed_m, air_temp_c, water_temp_c,
          wave_cm, raw_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                race_id,
                captured_at,
                row["lane"],
                row.get("weight_kg"),
                row.get("exhibition_time"),
                row.get("tilt"),
                row.get("adjusted_weight"),
                row.get("propeller"),
                row.get("parts_exchange"),
                row.get("course"),
                row.get("start_timing"),
                row.get("weather"),
                row.get("wind_direction"),
                row.get("wind_speed_m"),
                row.get("air_temp_c"),
                row.get("water_temp_c"),
                row.get("wave_cm"),
                json.dumps(row, ensure_ascii=False, sort_keys=True),
            )
            for row in rows
        ],
    )


def upsert_racer_period_stats(
    conn: sqlite3.Connection,
    *,
    year: int,
    half: int,
    rows: list[dict[str, Any]],
) -> None:
    conn.executemany(
        """
        INSERT INTO racer_period_stats (
          year, half, racer_no, racer_name, racer_class, raw_json
        )
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(year, half, racer_no) DO UPDATE SET
          racer_name=excluded.racer_name,
          racer_class=excluded.racer_class,
          raw_json=excluded.raw_json
        """,
        [
            (
                year,
                half,
                row["racer_no"],
                row.get("racer_name"),
                row.get("racer_class"),
                json.dumps(row, ensure_ascii=False, sort_keys=True),
            )
            for row in rows
        ],
    )
