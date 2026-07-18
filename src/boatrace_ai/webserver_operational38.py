from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from . import webserver_operational37 as base
from .db import connect


HTML = base.HTML

_archive_base = base._archive_base
_DEFAULT_HISTORY_DAYS = 365
_MAX_DAYS = 3650


def archive_history_fast(db_path: Path, query: dict[str, list[str]]) -> dict[str, Any]:
    kind = (_archive_base._one(query, "kind", "racer") or "racer").lower()
    days = base._bounded_int(_archive_base._one(query, "days", str(_DEFAULT_HISTORY_DAYS)), _DEFAULT_HISTORY_DAYS, 1, _MAX_DAYS)
    with connect(db_path) as conn:
        cutoff_date, latest_date = base._recent_cutoff(conn, days)
        if kind == "race":
            race_id = _archive_base._required(query, "race_id")
            return {"kind": kind, "generated_at": _archive_base._now(), **_archive_base._race_archive(conn, race_id)}
        if kind == "racer":
            payload = _history_racer(conn, _archive_base._required(query, "racer_no"), cutoff_date)
        elif kind == "venue":
            payload = _history_venue(conn, _archive_base._required(query, "jcd"), cutoff_date)
        elif kind == "motor":
            payload = _history_equipment(conn, "motor", _archive_base._required(query, "motor_no"), _archive_base._one(query, "jcd"), cutoff_date)
        elif kind == "boat":
            payload = _history_equipment(conn, "boat", _archive_base._required(query, "boat_no"), _archive_base._one(query, "jcd"), cutoff_date)
        elif kind == "lane":
            payload = _history_lane(conn, _archive_base._required(query, "lane"), _archive_base._one(query, "jcd"), _archive_base._one(query, "rno"), cutoff_date)
        elif kind == "combo":
            payload = _history_combo(conn, _archive_base._required(query, "combination"), cutoff_date)
        else:
            raise ValueError(f"unsupported history kind: {kind}")
    payload["period_days"] = days
    payload["cutoff_date"] = cutoff_date
    payload["latest_date"] = latest_date
    payload.setdefault("summary", {})["period_days"] = days
    payload.setdefault("summary", {})["cutoff_date"] = cutoff_date
    return payload


def _history_racer(conn: sqlite3.Connection, racer_no: str, cutoff_date: str) -> dict[str, Any]:
    summary = _add_rates(
        _archive_base._row(
            conn,
            """
            WITH recent AS MATERIALIZED (
              SELECT race_id, race_date, jcd, venue_name, rno, title
              FROM races
              WHERE race_date >= ?
            )
            SELECT
              e.racer_no,
              MAX(e.racer_name) AS racer_name,
              MAX(e.racer_class) AS latest_class,
              MAX(e.branch) AS branch,
              MAX(e.origin) AS origin,
              COUNT(*) AS starts,
              SUM(CASE WHEN rr.rank IS NOT NULL THEN 1 ELSE 0 END) AS result_rows,
              SUM(CASE WHEN rr.rank = 1 THEN 1 ELSE 0 END) AS wins,
              SUM(CASE WHEN rr.rank <= 3 THEN 1 ELSE 0 END) AS top3,
              AVG(CASE WHEN rr.rank IS NOT NULL THEN rr.rank END) AS avg_rank,
              AVG(rr.start_timing) AS avg_start,
              AVG(e.national_win_rate) AS avg_national_win_rate,
              AVG(e.local_win_rate) AS avg_local_win_rate
            FROM entries e
            JOIN recent r ON r.race_id = e.race_id
            LEFT JOIN race_results rr ON rr.race_id = e.race_id AND rr.lane = e.lane
            WHERE e.racer_no = ?
            """,
            (cutoff_date, racer_no),
        )
    )
    rows = _archive_base._rows(
        conn,
        """
        WITH recent AS MATERIALIZED (
          SELECT race_id, race_date, jcd, venue_name, rno, title
          FROM races
          WHERE race_date >= ?
        )
        SELECT
          r.race_id, r.race_date, r.jcd, r.venue_name, r.rno, r.title,
          e.lane, e.racer_no, e.racer_name, e.racer_class,
          e.motor_no, e.boat_no, rr.rank, rr.course, rr.start_timing,
          p.combination AS result_combination, p.payout_yen
        FROM entries e
        JOIN recent r ON r.race_id = e.race_id
        LEFT JOIN race_results rr ON rr.race_id = e.race_id AND rr.lane = e.lane
        LEFT JOIN payouts p ON p.race_id = e.race_id AND p.bet_type = '3連単'
        WHERE e.racer_no = ?
        ORDER BY r.race_date DESC, r.jcd DESC, r.rno DESC
        LIMIT 80
        """,
        (cutoff_date, racer_no),
    )
    return {"kind": "racer", "generated_at": _archive_base._now(), "summary": summary, "rows": rows}


def _history_venue(conn: sqlite3.Connection, jcd: str, cutoff_date: str) -> dict[str, Any]:
    jcd = jcd.zfill(2)
    summary = _archive_base._row(
        conn,
        """
        WITH recent AS MATERIALIZED (
          SELECT race_id, race_date, jcd, venue_name, rno, title, race_type, distance_m
          FROM races
          WHERE race_date >= ? AND jcd = ?
        )
        SELECT
          jcd,
          MAX(venue_name) AS venue_name,
          COUNT(*) AS races,
          SUM(CASE WHEN (SELECT COUNT(*) FROM race_results rr WHERE rr.race_id = recent.race_id AND rr.rank IS NOT NULL) >= 3 THEN 1 ELSE 0 END) AS result_races,
          AVG(distance_m) AS avg_distance_m
        FROM recent
        """,
        (cutoff_date, jcd),
    )
    facets = _archive_base._rows(
        conn,
        """
        WITH recent AS MATERIALIZED (
          SELECT race_id FROM races WHERE race_date >= ? AND jcd = ?
        )
        SELECT
          rr.lane,
          COUNT(*) AS starts,
          SUM(CASE WHEN rr.rank = 1 THEN 1 ELSE 0 END) AS wins,
          SUM(CASE WHEN rr.rank <= 3 THEN 1 ELSE 0 END) AS top3,
          AVG(rr.start_timing) AS avg_start
        FROM recent r
        JOIN race_results rr ON rr.race_id = r.race_id AND rr.rank IS NOT NULL
        GROUP BY rr.lane
        ORDER BY rr.lane
        """,
        (cutoff_date, jcd),
    )
    rows = _recent_races(conn, "race_date >= ? AND jcd = ?", (cutoff_date, jcd))
    return {"kind": "venue", "generated_at": _archive_base._now(), "summary": summary or {"jcd": jcd}, "facets": facets, "rows": rows}


def _history_equipment(conn: sqlite3.Connection, kind: str, number: str, jcd: str | None, cutoff_date: str) -> dict[str, Any]:
    column = "motor_no" if kind == "motor" else "boat_no"
    rate2 = "motor_2_rate" if kind == "motor" else "boat_2_rate"
    rate3 = "motor_3_rate" if kind == "motor" else "boat_3_rate"
    params: list[Any] = [cutoff_date, number]
    filters = [f"e.{column} = ?"]
    if jcd:
        filters.append("r.jcd = ?")
        params.append(jcd.zfill(2))
    where = " AND ".join(filters)
    summary = _add_rates(
        _archive_base._row(
            conn,
            f"""
            WITH recent AS MATERIALIZED (
              SELECT race_id, race_date, jcd, venue_name, rno
              FROM races
              WHERE race_date >= ?
            )
            SELECT
              MAX(r.jcd) AS jcd,
              MAX(r.venue_name) AS venue_name,
              e.{column} AS number,
              COUNT(*) AS starts,
              SUM(CASE WHEN rr.rank IS NOT NULL THEN 1 ELSE 0 END) AS result_rows,
              SUM(CASE WHEN rr.rank = 1 THEN 1 ELSE 0 END) AS wins,
              SUM(CASE WHEN rr.rank <= 3 THEN 1 ELSE 0 END) AS top3,
              AVG(rr.rank) AS avg_rank,
              AVG(rr.start_timing) AS avg_start,
              AVG(e.{rate2}) AS avg_2_rate,
              AVG(e.{rate3}) AS avg_3_rate
            FROM recent r
            JOIN entries e ON e.race_id = r.race_id
            LEFT JOIN race_results rr ON rr.race_id = e.race_id AND rr.lane = e.lane
            WHERE {where}
            """,
            tuple(params),
        )
    )
    rows = _archive_base._rows(
        conn,
        f"""
        WITH recent AS MATERIALIZED (
          SELECT race_id, race_date, jcd, venue_name, rno
          FROM races
          WHERE race_date >= ?
        )
        SELECT
          r.race_id, r.race_date, r.jcd, r.venue_name, r.rno,
          e.lane, e.racer_no, e.racer_name, e.racer_class,
          e.motor_no, e.boat_no, rr.rank, rr.course, rr.start_timing
        FROM recent r
        JOIN entries e ON e.race_id = r.race_id
        LEFT JOIN race_results rr ON rr.race_id = e.race_id AND rr.lane = e.lane
        WHERE {where}
        ORDER BY r.race_date DESC, r.jcd DESC, r.rno DESC
        LIMIT 80
        """,
        tuple(params),
    )
    return {"kind": kind, "generated_at": _archive_base._now(), "summary": summary, "rows": rows}


def _history_lane(conn: sqlite3.Connection, lane: str, jcd: str | None, rno: str | None, cutoff_date: str) -> dict[str, Any]:
    params: list[Any] = [cutoff_date, lane]
    filters = ["e.lane = ?"]
    if jcd:
        filters.append("r.jcd = ?")
        params.append(jcd.zfill(2))
    if rno:
        filters.append("r.rno = ?")
        params.append(int(rno))
    where = " AND ".join(filters)
    summary = _add_rates(
        _archive_base._row(
            conn,
            f"""
            WITH recent AS MATERIALIZED (
              SELECT race_id, race_date, jcd, venue_name, rno
              FROM races
              WHERE race_date >= ?
            )
            SELECT
              e.lane,
              COUNT(*) AS starts,
              SUM(CASE WHEN rr.rank IS NOT NULL THEN 1 ELSE 0 END) AS result_rows,
              SUM(CASE WHEN rr.rank = 1 THEN 1 ELSE 0 END) AS wins,
              SUM(CASE WHEN rr.rank <= 3 THEN 1 ELSE 0 END) AS top3,
              AVG(rr.rank) AS avg_rank,
              AVG(rr.start_timing) AS avg_start,
              AVG(e.national_win_rate) AS avg_national_win_rate,
              AVG(e.motor_2_rate) AS avg_motor_2_rate,
              AVG(e.boat_2_rate) AS avg_boat_2_rate
            FROM recent r
            JOIN entries e ON e.race_id = r.race_id
            LEFT JOIN race_results rr ON rr.race_id = e.race_id AND rr.lane = e.lane
            WHERE {where}
            """,
            tuple(params),
        )
    )
    rows = _archive_base._rows(
        conn,
        f"""
        WITH recent AS MATERIALIZED (
          SELECT race_id, race_date, jcd, venue_name, rno
          FROM races
          WHERE race_date >= ?
        )
        SELECT
          r.race_id, r.race_date, r.jcd, r.venue_name, r.rno,
          e.lane, e.racer_no, e.racer_name, e.racer_class,
          e.motor_no, e.boat_no, rr.rank, rr.course, rr.start_timing
        FROM recent r
        JOIN entries e ON e.race_id = r.race_id
        LEFT JOIN race_results rr ON rr.race_id = e.race_id AND rr.lane = e.lane
        WHERE {where}
        ORDER BY r.race_date DESC, r.jcd DESC, r.rno DESC
        LIMIT 80
        """,
        tuple(params),
    )
    return {"kind": "lane", "generated_at": _archive_base._now(), "summary": summary, "rows": rows}


def _history_combo(conn: sqlite3.Connection, combination: str, cutoff_date: str) -> dict[str, Any]:
    summary = _archive_base._row(
        conn,
        """
        WITH recent AS MATERIALIZED (
          SELECT race_id FROM races WHERE race_date >= ?
        )
        SELECT
          ? AS combination,
          COUNT(*) AS hits,
          AVG(p.payout_yen) AS avg_payout_yen,
          MIN(p.payout_yen) AS min_payout_yen,
          MAX(p.payout_yen) AS max_payout_yen,
          AVG(p.popularity) AS avg_popularity
        FROM recent r
        JOIN payouts p ON p.race_id = r.race_id
        WHERE p.bet_type = '3連単' AND p.combination = ?
        """,
        (cutoff_date, combination, combination),
    )
    rows = _archive_base._rows(
        conn,
        """
        WITH recent AS MATERIALIZED (
          SELECT race_id, race_date, jcd, venue_name, rno, title
          FROM races
          WHERE race_date >= ?
        )
        SELECT
          r.race_id, r.race_date, r.jcd, r.venue_name, r.rno, r.title,
          p.combination, p.payout_yen, p.popularity
        FROM recent r
        JOIN payouts p ON p.race_id = r.race_id
        WHERE p.bet_type = '3連単' AND p.combination = ?
        ORDER BY r.race_date DESC, r.jcd DESC, r.rno DESC
        LIMIT 80
        """,
        (cutoff_date, combination),
    )
    return {"kind": "combo", "generated_at": _archive_base._now(), "summary": summary, "rows": rows}


def _recent_races(conn: sqlite3.Connection, where_sql: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
    return _archive_base._rows(
        conn,
        f"""
        SELECT
          r.race_id, r.race_date, r.jcd, r.venue_name, r.rno, r.title,
          r.race_type, r.distance_m,
          (SELECT COUNT(*) FROM entries e WHERE e.race_id = r.race_id) AS entries,
          (SELECT COUNT(*) FROM race_results rr WHERE rr.race_id = r.race_id AND rr.rank IS NOT NULL) AS result_rows,
          (SELECT combination FROM payouts p WHERE p.race_id = r.race_id AND p.bet_type = '3連単' LIMIT 1) AS result_combination,
          (SELECT payout_yen FROM payouts p WHERE p.race_id = r.race_id AND p.bet_type = '3連単' LIMIT 1) AS payout_yen
        FROM races r
        WHERE {where_sql}
        ORDER BY r.race_date DESC, r.jcd DESC, r.rno DESC
        LIMIT 80
        """,
        params,
    )


def _add_rates(summary: dict[str, Any] | None) -> dict[str, Any]:
    out = dict(summary or {})
    results = float(out.get("result_rows") or 0)
    if results:
        out["win_rate"] = float(out.get("wins") or 0) / results
        out["top3_rate"] = float(out.get("top3") or 0) / results
    return out


archive_history = archive_history_fast
base.archive_history = archive_history_fast
_archive_base.archive_history = archive_history_fast


def main(argv: list[str] | None = None) -> int:
    return base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
