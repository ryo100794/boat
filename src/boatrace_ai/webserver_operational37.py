from __future__ import annotations

import sqlite3
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from . import webserver_operational36 as base
from .db import connect


HTML = base.HTML

_archive_base = base.base

_DEFAULT_DAYS = 90
_EQUIPMENT_DAYS = 90
_DEFAULT_LIMIT = 120
_MAX_LIMIT = 500
_SCOPES = {"lane", "venue", "rno", "class", "motor", "boat"}


def archive_stats_fast(db_path: Path, query: dict[str, list[str]]) -> dict[str, Any]:
    scope = (_archive_base._one(query, "scope", "lane") or "lane").lower()
    if scope not in _SCOPES:
        scope = "lane"

    default_days = _EQUIPMENT_DAYS if scope in {"motor", "boat"} else _DEFAULT_DAYS
    days = _bounded_int(_archive_base._one(query, "days", str(default_days)), default_days, 1, 3650)
    limit = _bounded_int(_archive_base._one(query, "limit", str(_DEFAULT_LIMIT)), _DEFAULT_LIMIT, 1, _MAX_LIMIT)
    min_starts = _bounded_int(
        _archive_base._one(query, "min_starts", str(_default_min_starts(scope))),
        _default_min_starts(scope),
        1,
        1000,
    )

    with connect(db_path) as conn:
        cutoff_date, latest_date = _recent_cutoff(conn, days)
        rows = _stat_rows_fast(conn, scope, cutoff_date, limit, min_starts)

    return {
        "scope": scope,
        "generated_at": _archive_base._now(),
        "period_days": days,
        "cutoff_date": cutoff_date,
        "latest_date": latest_date,
        "rows": rows,
    }


def _stat_rows_fast(
    conn: sqlite3.Connection,
    scope: str,
    cutoff_date: str,
    limit: int,
    min_starts: int,
) -> list[dict[str, Any]]:
    select_key, group_sql, where_sql, order_sql = _scope_sql(scope)
    return _archive_base._rows(
        conn,
        f"""
        WITH recent_races AS MATERIALIZED (
          SELECT race_id, jcd, venue_name, rno
          FROM races
          WHERE race_date >= ?
        )
        SELECT
          {select_key},
          COUNT(*) AS starts,
          SUM(CASE WHEN rr.rank = 1 THEN 1 ELSE 0 END) AS wins,
          SUM(CASE WHEN rr.rank <= 3 THEN 1 ELSE 0 END) AS top3,
          SUM(CASE WHEN rr.rank = 1 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS win_rate,
          SUM(CASE WHEN rr.rank <= 3 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS top3_rate,
          AVG(rr.rank) AS avg_rank,
          AVG(rr.start_timing) AS avg_start,
          AVG(e.national_win_rate) AS avg_national_win_rate,
          AVG(e.local_win_rate) AS avg_local_win_rate,
          AVG(e.motor_2_rate) AS avg_motor_2_rate,
          AVG(e.boat_2_rate) AS avg_boat_2_rate
        FROM recent_races r
        JOIN race_results rr ON rr.race_id = r.race_id AND rr.rank IS NOT NULL
        LEFT JOIN entries e ON e.race_id = rr.race_id AND e.lane = rr.lane
        WHERE {where_sql}
        GROUP BY {group_sql}
        HAVING COUNT(*) >= ?
        ORDER BY {order_sql}
        LIMIT ?
        """,
        (cutoff_date, min_starts, limit),
    )


def _scope_sql(scope: str) -> tuple[str, str, str, str]:
    if scope == "venue":
        return (
            "r.jcd AS key, MAX(r.venue_name) AS label",
            "r.jcd",
            "1 = 1",
            "win_rate DESC, starts DESC, r.jcd",
        )
    if scope == "rno":
        return (
            "r.rno AS key, printf('%02dR', r.rno) AS label",
            "r.rno",
            "1 = 1",
            "win_rate DESC, starts DESC, r.rno",
        )
    if scope == "class":
        class_expr = "COALESCE(NULLIF(e.racer_class, ''), '-')"
        return (
            f"{class_expr} AS key, {class_expr} AS label",
            class_expr,
            "1 = 1",
            "win_rate DESC, starts DESC, label",
        )
    if scope == "motor":
        return (
            "printf('%s-M%s', r.jcd, e.motor_no) AS key, printf('%s M%s', MAX(r.venue_name), e.motor_no) AS label",
            "r.jcd, e.motor_no",
            "e.motor_no IS NOT NULL",
            "starts DESC, win_rate DESC, label",
        )
    if scope == "boat":
        return (
            "printf('%s-B%s', r.jcd, e.boat_no) AS key, printf('%s B%s', MAX(r.venue_name), e.boat_no) AS label",
            "r.jcd, e.boat_no",
            "e.boat_no IS NOT NULL",
            "starts DESC, win_rate DESC, label",
        )
    return (
        "rr.lane AS key, printf('%d号艇', rr.lane) AS label",
        "rr.lane",
        "1 = 1",
        "rr.lane",
    )


def _recent_cutoff(conn: sqlite3.Connection, days: int) -> tuple[str, str | None]:
    row = conn.execute("SELECT MAX(race_date) FROM races").fetchone()
    latest = row[0] if row else None
    try:
        latest_date = date.fromisoformat(str(latest)) if latest else date.today()
    except ValueError:
        latest_date = date.today()
    cutoff = latest_date - timedelta(days=days - 1)
    return cutoff.isoformat(), latest


def _default_min_starts(scope: str) -> int:
    return 5 if scope in {"motor", "boat"} else 20


def _bounded_int(raw: str | None, default: int, min_value: int, max_value: int) -> int:
    try:
        value = int(raw or default)
    except (TypeError, ValueError):
        value = default
    return min(max(value, min_value), max_value)


archive_stats = archive_stats_fast
base.archive_stats = archive_stats_fast
_archive_base.archive_stats = archive_stats_fast


def main(argv: list[str] | None = None) -> int:
    return base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
