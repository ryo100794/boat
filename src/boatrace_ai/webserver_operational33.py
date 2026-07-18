from __future__ import annotations

import argparse
import sqlite3
from datetime import date, datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .constants import VENUES
from .db import connect, init_db
from .webserver_all import backtest, odds, rowdict, send_html, send_json, summary
from .webserver_model_rank import accuracy_model_rank
from . import webserver_operational25 as ops_base
from . import webserver_operational28 as prediction_base
from . import webserver_operational32 as ui_base


HTML = None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="BOAT RACE AI dashboard with archive drilldown pages.")
    parser.add_argument("--db", default="data/boatrace.sqlite")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10001)
    parser.add_argument("--backtest", default="data/models/backtest_no_odds_v8.json")
    args = parser.parse_args(argv)
    init_db(args.db)
    handler = make_handler(Path(args.db), Path(args.backtest) if args.backtest else None)
    print(f"Serving BOAT RACE AI Archive Drilldown Monitor on http://{args.host}:{args.port}", flush=True)
    ThreadingHTTPServer((args.host, args.port), handler).serve_forever()
    return 0


def make_handler(db_path: Path, backtest_path: Path | None):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            try:
                if parsed.path in ("/", "/archive"):
                    send_html(self, HTML)
                elif parsed.path == "/api/summary":
                    send_json(self, summary(db_path))
                elif parsed.path == "/api/venues":
                    send_json(self, ops_base.venue_cards_start_time(db_path, query))
                elif parsed.path == "/api/day":
                    send_json(self, ops_base.day_overview_start_time(db_path, query))
                elif parsed.path == "/api/guide":
                    send_json(self, ops_base.purchase_guide_start_time(db_path, query))
                elif parsed.path == "/api/live-wipe":
                    send_json(self, ops_base.live_wipe_start_time(db_path, query))
                elif parsed.path == "/api/progress":
                    send_json(self, ops_base.progress_active(db_path, query))
                elif parsed.path == "/api/predictions":
                    send_json(self, prediction_base.predictions_with_names(db_path, query))
                elif parsed.path == "/api/odds":
                    send_json(self, odds(db_path, query))
                elif parsed.path == "/api/backtest":
                    send_json(self, backtest(backtest_path))
                elif parsed.path == "/api/accuracy":
                    send_json(self, accuracy_model_rank(db_path, query))
                elif parsed.path == "/api/archive/overview":
                    send_json(self, archive_overview(db_path, query))
                elif parsed.path == "/api/archive/today":
                    send_json(self, archive_today(db_path, query))
                elif parsed.path == "/api/archive/history":
                    send_json(self, archive_history(db_path, query))
                elif parsed.path == "/api/archive/stats":
                    send_json(self, archive_stats(db_path, query))
                else:
                    self.send_error(404)
            except Exception as exc:
                send_json(self, {"error": str(exc)}, status=500)

        def log_message(self, fmt: str, *args: Any) -> None:
            return

    return Handler


def archive_overview(db_path: Path, query: dict[str, list[str]]) -> dict[str, Any]:
    race_date = _one(query, "date", date.today().isoformat())
    with connect(db_path) as conn:
        totals = _row(
            conn,
            """
            SELECT
              COUNT(*) AS races,
              MIN(race_date) AS first_date,
              MAX(race_date) AS last_date,
              (SELECT COUNT(*) FROM entries) AS entries,
              (SELECT COUNT(DISTINCT race_id) FROM race_results WHERE rank IS NOT NULL) AS result_races,
              (SELECT COUNT(*) FROM odds_snapshots) AS odds_snapshots,
              (SELECT COUNT(DISTINCT race_id) FROM predictions) AS prediction_races,
              (SELECT COUNT(DISTINCT race_id) FROM beforeinfo) AS beforeinfo_races
            FROM races
            """,
        )
        today = _row(
            conn,
            """
            SELECT
              COUNT(*) AS races,
              SUM(CASE WHEN (SELECT COUNT(*) FROM entries e WHERE e.race_id = r.race_id) = 6 THEN 1 ELSE 0 END) AS entry_races,
              SUM(CASE WHEN (SELECT COUNT(*) FROM race_results rr WHERE rr.race_id = r.race_id AND rr.rank IS NOT NULL) >= 3 THEN 1 ELSE 0 END) AS result_races,
              SUM(CASE WHEN EXISTS (SELECT 1 FROM odds_snapshots os WHERE os.race_id = r.race_id) THEN 1 ELSE 0 END) AS odds_races,
              SUM(CASE WHEN EXISTS (SELECT 1 FROM predictions p WHERE p.race_id = r.race_id) THEN 1 ELSE 0 END) AS prediction_races
            FROM races r
            WHERE r.race_date = ?
            """,
            (race_date,),
        )
        years = _rows(
            conn,
            """
            SELECT
              substr(r.race_date, 1, 4) AS year,
              COUNT(*) AS races,
              SUM(CASE WHEN (SELECT COUNT(*) FROM entries e WHERE e.race_id = r.race_id) = 6 THEN 1 ELSE 0 END) AS entry_races,
              SUM(CASE WHEN (SELECT COUNT(*) FROM race_results rr WHERE rr.race_id = r.race_id AND rr.rank IS NOT NULL) >= 3 THEN 1 ELSE 0 END) AS result_races,
              SUM(CASE WHEN EXISTS (SELECT 1 FROM predictions p WHERE p.race_id = r.race_id) THEN 1 ELSE 0 END) AS prediction_races
            FROM races r
            GROUP BY year
            ORDER BY year DESC
            LIMIT 14
            """,
        )
        venues = _rows(
            conn,
            """
            SELECT
              r.jcd,
              MAX(r.venue_name) AS venue_name,
              COUNT(*) AS races,
              SUM(CASE WHEN (SELECT COUNT(*) FROM entries e WHERE e.race_id = r.race_id) = 6 THEN 1 ELSE 0 END) AS entry_races,
              SUM(CASE WHEN (SELECT COUNT(*) FROM race_results rr WHERE rr.race_id = r.race_id AND rr.rank IS NOT NULL) >= 3 THEN 1 ELSE 0 END) AS result_races,
              SUM(CASE WHEN EXISTS (SELECT 1 FROM odds_snapshots os WHERE os.race_id = r.race_id) THEN 1 ELSE 0 END) AS odds_races
            FROM races r
            GROUP BY r.jcd
            ORDER BY r.jcd
            """,
        )
    return {
        "date": race_date,
        "generated_at": _now(),
        "totals": totals,
        "today": today,
        "years": years,
        "venues": venues,
    }


def archive_today(db_path: Path, query: dict[str, list[str]]) -> dict[str, Any]:
    race_date = _one(query, "date", date.today().isoformat())
    race_id = _one(query, "race_id")
    jcd = _one(query, "jcd")
    with connect(db_path) as conn:
        if race_id:
            return {
                "date": race_date,
                "generated_at": _now(),
                "mode": "race",
                **_race_archive(conn, race_id),
            }
        params: list[Any] = [race_date]
        jcd_sql = ""
        if jcd:
            jcd_sql = "AND r.jcd = ?"
            params.append(jcd.zfill(2))
        races = _rows(
            conn,
            f"""
            SELECT
              r.race_id, r.race_date, r.jcd, r.venue_name, r.rno, r.title, r.race_type,
              r.distance_m, r.deadline_at, r.status, r.updated_at,
              (SELECT COUNT(*) FROM entries e WHERE e.race_id = r.race_id) AS entries,
              (SELECT COUNT(*) FROM odds_snapshots os WHERE os.race_id = r.race_id) AS odds_snapshots,
              (SELECT MAX(captured_at) FROM odds_snapshots os WHERE os.race_id = r.race_id) AS latest_odds_at,
              (SELECT COUNT(*) FROM beforeinfo b WHERE b.race_id = r.race_id) AS beforeinfo_rows,
              (SELECT COUNT(*) FROM race_results rr WHERE rr.race_id = r.race_id AND rr.rank IS NOT NULL) AS result_rows,
              (SELECT MAX(generated_at) FROM predictions p WHERE p.race_id = r.race_id) AS latest_prediction
            FROM races r
            WHERE r.race_date = ? {jcd_sql}
            ORDER BY r.deadline_at IS NULL, r.deadline_at, r.jcd, r.rno
            """,
            tuple(params),
        )
    return {
        "date": race_date,
        "jcd": jcd,
        "generated_at": _now(),
        "mode": "day",
        "races": races,
    }


def archive_history(db_path: Path, query: dict[str, list[str]]) -> dict[str, Any]:
    kind = (_one(query, "kind", "racer") or "racer").lower()
    with connect(db_path) as conn:
        if kind == "race":
            race_id = _required(query, "race_id")
            return {"kind": kind, "generated_at": _now(), **_race_archive(conn, race_id)}
        if kind == "racer":
            return _history_racer(conn, _required(query, "racer_no"))
        if kind == "venue":
            return _history_venue(conn, _required(query, "jcd"))
        if kind == "motor":
            return _history_equipment(conn, "motor", _required(query, "motor_no"), _one(query, "jcd"))
        if kind == "boat":
            return _history_equipment(conn, "boat", _required(query, "boat_no"), _one(query, "jcd"))
        if kind == "lane":
            return _history_lane(conn, _required(query, "lane"), _one(query, "jcd"), _one(query, "rno"))
        if kind == "combo":
            return _history_combo(conn, _required(query, "combination"))
    raise ValueError(f"unsupported history kind: {kind}")


def archive_stats(db_path: Path, query: dict[str, list[str]]) -> dict[str, Any]:
    scope = (_one(query, "scope", "lane") or "lane").lower()
    limit = min(500, max(1, int(_one(query, "limit", "120") or "120")))
    with connect(db_path) as conn:
        if scope == "venue":
            rows = _stat_rows(conn, "r.jcd AS key, MAX(r.venue_name) AS label", "r.jcd", limit)
        elif scope == "rno":
            rows = _stat_rows(conn, "r.rno AS key, printf('%02dR', r.rno) AS label", "r.rno", limit)
        elif scope == "class":
            rows = _stat_rows(conn, "COALESCE(e.racer_class, '-') AS key, COALESCE(e.racer_class, '-') AS label", "COALESCE(e.racer_class, '-')", limit)
        elif scope == "motor":
            rows = _stat_rows(conn, "printf('%s-M%s', r.jcd, e.motor_no) AS key, printf('%s M%s', MAX(r.venue_name), e.motor_no) AS label", "r.jcd, e.motor_no", limit, extra_where="e.motor_no IS NOT NULL")
        elif scope == "boat":
            rows = _stat_rows(conn, "printf('%s-B%s', r.jcd, e.boat_no) AS key, printf('%s B%s', MAX(r.venue_name), e.boat_no) AS label", "r.jcd, e.boat_no", limit, extra_where="e.boat_no IS NOT NULL")
        else:
            scope = "lane"
            rows = _stat_rows(conn, "e.lane AS key, printf('%d号艇', e.lane) AS label", "e.lane", limit)
    return {"scope": scope, "generated_at": _now(), "rows": rows}


def _race_archive(conn: sqlite3.Connection, race_id: str) -> dict[str, Any]:
    race = _row(
        conn,
        """
        SELECT
          r.*,
          (SELECT COUNT(*) FROM entries e WHERE e.race_id = r.race_id) AS entries,
          (SELECT COUNT(*) FROM odds_snapshots os WHERE os.race_id = r.race_id) AS odds_snapshots,
          (SELECT MIN(captured_at) FROM odds_snapshots os WHERE os.race_id = r.race_id) AS first_odds_at,
          (SELECT MAX(captured_at) FROM odds_snapshots os WHERE os.race_id = r.race_id) AS latest_odds_at,
          (SELECT COUNT(*) FROM beforeinfo b WHERE b.race_id = r.race_id) AS beforeinfo_rows,
          (SELECT COUNT(*) FROM race_results rr WHERE rr.race_id = r.race_id AND rr.rank IS NOT NULL) AS result_rows,
          (SELECT MAX(generated_at) FROM predictions p WHERE p.race_id = r.race_id) AS latest_prediction
        FROM races r
        WHERE r.race_id = ?
        """,
        (race_id,),
    )
    entries = _rows(
        conn,
        """
        WITH latest_before AS (
          SELECT MAX(captured_at) AS captured_at FROM beforeinfo WHERE race_id = ?
        )
        SELECT
          e.lane, e.racer_no, e.racer_name, e.racer_class, e.branch, e.origin,
          e.age, e.weight_kg, e.f_count, e.l_count, e.avg_st,
          e.national_win_rate, e.national_2_rate, e.national_3_rate,
          e.local_win_rate, e.local_2_rate, e.local_3_rate,
          e.motor_no, e.motor_2_rate, e.motor_3_rate,
          e.boat_no, e.boat_2_rate, e.boat_3_rate,
          rr.rank, rr.course AS result_course, rr.start_timing AS result_start_timing,
          b.captured_at AS beforeinfo_at, b.exhibition_time, b.course AS exhibition_course,
          b.start_timing AS exhibition_start_timing, b.weather, b.wind_direction, b.wind_speed_m,
          b.air_temp_c, b.water_temp_c, b.wave_cm
        FROM entries e
        LEFT JOIN race_results rr ON rr.race_id = e.race_id AND rr.lane = e.lane
        LEFT JOIN latest_before lb
        LEFT JOIN beforeinfo b ON b.race_id = e.race_id AND b.lane = e.lane AND b.captured_at = lb.captured_at
        WHERE e.race_id = ?
        ORDER BY e.lane
        """,
        (race_id, race_id),
    )
    predictions = _latest_prediction_rows(conn, race_id, limit=30)
    payouts = _rows(
        conn,
        """
        SELECT bet_type, combination, payout_yen, popularity
        FROM payouts
        WHERE race_id = ?
        ORDER BY bet_type, popularity IS NULL, popularity
        """,
        (race_id,),
    )
    return {"race": race, "entries": entries, "predictions": predictions, "payouts": payouts}


def _history_racer(conn: sqlite3.Connection, racer_no: str) -> dict[str, Any]:
    summary = _row(
        conn,
        """
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
          AVG(CASE WHEN rr.start_timing IS NOT NULL THEN rr.start_timing END) AS avg_start,
          AVG(e.national_win_rate) AS avg_national_win_rate,
          AVG(e.local_win_rate) AS avg_local_win_rate
        FROM entries e
        JOIN races r ON r.race_id = e.race_id
        LEFT JOIN race_results rr ON rr.race_id = e.race_id AND rr.lane = e.lane
        WHERE e.racer_no = ?
        """,
        (racer_no,),
    )
    rows = _rows(
        conn,
        """
        SELECT
          r.race_id, r.race_date, r.jcd, r.venue_name, r.rno, r.title,
          e.lane, e.racer_class, e.motor_no, e.motor_2_rate, e.boat_no, e.boat_2_rate,
          rr.rank, rr.course, rr.start_timing,
          p.combination AS result_combination, p.payout_yen
        FROM entries e
        JOIN races r ON r.race_id = e.race_id
        LEFT JOIN race_results rr ON rr.race_id = e.race_id AND rr.lane = e.lane
        LEFT JOIN payouts p ON p.race_id = e.race_id AND p.bet_type = '3連単'
        WHERE e.racer_no = ?
        ORDER BY r.race_date DESC, r.jcd DESC, r.rno DESC
        LIMIT 80
        """,
        (racer_no,),
    )
    return _history_payload("racer", summary, rows)


def _history_venue(conn: sqlite3.Connection, jcd: str) -> dict[str, Any]:
    jcd = jcd.zfill(2)
    venue_name = next((venue.name for venue in VENUES if venue.code == jcd), jcd)
    summary = _row(
        conn,
        """
        SELECT
          r.jcd,
          MAX(r.venue_name) AS venue_name,
          COUNT(*) AS races,
          SUM(CASE WHEN (SELECT COUNT(*) FROM race_results rr WHERE rr.race_id = r.race_id AND rr.rank IS NOT NULL) >= 3 THEN 1 ELSE 0 END) AS result_races,
          AVG(r.distance_m) AS avg_distance_m
        FROM races r
        WHERE r.jcd = ?
        """,
        (jcd,),
    )
    lanes = _rows(
        conn,
        """
        SELECT
          e.lane,
          COUNT(*) AS starts,
          SUM(CASE WHEN rr.rank IS NOT NULL THEN 1 ELSE 0 END) AS result_rows,
          SUM(CASE WHEN rr.rank = 1 THEN 1 ELSE 0 END) AS wins,
          SUM(CASE WHEN rr.rank <= 3 THEN 1 ELSE 0 END) AS top3,
          AVG(CASE WHEN rr.start_timing IS NOT NULL THEN rr.start_timing END) AS avg_start
        FROM entries e
        JOIN races r ON r.race_id = e.race_id
        LEFT JOIN race_results rr ON rr.race_id = e.race_id AND rr.lane = e.lane
        WHERE r.jcd = ?
        GROUP BY e.lane
        ORDER BY e.lane
        """,
        (jcd,),
    )
    recent = _recent_races(conn, "r.jcd = ?", (jcd,))
    return {"kind": "venue", "generated_at": _now(), "summary": summary or {"jcd": jcd, "venue_name": venue_name}, "facets": lanes, "rows": recent}


def _history_equipment(conn: sqlite3.Connection, kind: str, number: str, jcd: str | None) -> dict[str, Any]:
    column = "motor_no" if kind == "motor" else "boat_no"
    params: list[Any] = [number]
    filters = [f"e.{column} = ?"]
    if jcd:
        filters.append("r.jcd = ?")
        params.append(jcd.zfill(2))
    where = " AND ".join(filters)
    summary = _row(
        conn,
        f"""
        SELECT
          MAX(r.jcd) AS jcd,
          MAX(r.venue_name) AS venue_name,
          e.{column} AS number,
          COUNT(*) AS starts,
          SUM(CASE WHEN rr.rank IS NOT NULL THEN 1 ELSE 0 END) AS result_rows,
          SUM(CASE WHEN rr.rank = 1 THEN 1 ELSE 0 END) AS wins,
          SUM(CASE WHEN rr.rank <= 3 THEN 1 ELSE 0 END) AS top3,
          AVG(CASE WHEN rr.rank IS NOT NULL THEN rr.rank END) AS avg_rank,
          AVG(CASE WHEN rr.start_timing IS NOT NULL THEN rr.start_timing END) AS avg_start,
          AVG(e.{column.replace('_no', '_2_rate')}) AS avg_2_rate,
          AVG(e.{column.replace('_no', '_3_rate')}) AS avg_3_rate
        FROM entries e
        JOIN races r ON r.race_id = e.race_id
        LEFT JOIN race_results rr ON rr.race_id = e.race_id AND rr.lane = e.lane
        WHERE {where}
        """,
        tuple(params),
    )
    rows = _rows(
        conn,
        f"""
        SELECT
          r.race_id, r.race_date, r.jcd, r.venue_name, r.rno,
          e.lane, e.racer_no, e.racer_name, e.racer_class,
          e.motor_no, e.motor_2_rate, e.boat_no, e.boat_2_rate,
          rr.rank, rr.course, rr.start_timing
        FROM entries e
        JOIN races r ON r.race_id = e.race_id
        LEFT JOIN race_results rr ON rr.race_id = e.race_id AND rr.lane = e.lane
        WHERE {where}
        ORDER BY r.race_date DESC, r.jcd DESC, r.rno DESC
        LIMIT 80
        """,
        tuple(params),
    )
    return _history_payload(kind, summary, rows)


def _history_lane(conn: sqlite3.Connection, lane: str, jcd: str | None, rno: str | None) -> dict[str, Any]:
    params: list[Any] = [lane]
    filters = ["e.lane = ?"]
    if jcd:
        filters.append("r.jcd = ?")
        params.append(jcd.zfill(2))
    if rno:
        filters.append("r.rno = ?")
        params.append(int(rno))
    where = " AND ".join(filters)
    summary = _row(
        conn,
        f"""
        SELECT
          e.lane,
          COUNT(*) AS starts,
          SUM(CASE WHEN rr.rank IS NOT NULL THEN 1 ELSE 0 END) AS result_rows,
          SUM(CASE WHEN rr.rank = 1 THEN 1 ELSE 0 END) AS wins,
          SUM(CASE WHEN rr.rank <= 3 THEN 1 ELSE 0 END) AS top3,
          AVG(CASE WHEN rr.rank IS NOT NULL THEN rr.rank END) AS avg_rank,
          AVG(CASE WHEN rr.start_timing IS NOT NULL THEN rr.start_timing END) AS avg_start,
          AVG(e.national_win_rate) AS avg_national_win_rate,
          AVG(e.motor_2_rate) AS avg_motor_2_rate,
          AVG(e.boat_2_rate) AS avg_boat_2_rate
        FROM entries e
        JOIN races r ON r.race_id = e.race_id
        LEFT JOIN race_results rr ON rr.race_id = e.race_id AND rr.lane = e.lane
        WHERE {where}
        """,
        tuple(params),
    )
    rows = _rows(
        conn,
        f"""
        SELECT
          r.race_id, r.race_date, r.jcd, r.venue_name, r.rno,
          e.lane, e.racer_no, e.racer_name, e.racer_class,
          e.motor_no, e.boat_no, rr.rank, rr.course, rr.start_timing
        FROM entries e
        JOIN races r ON r.race_id = e.race_id
        LEFT JOIN race_results rr ON rr.race_id = e.race_id AND rr.lane = e.lane
        WHERE {where}
        ORDER BY r.race_date DESC, r.jcd DESC, r.rno DESC
        LIMIT 80
        """,
        tuple(params),
    )
    return _history_payload("lane", summary, rows)


def _history_combo(conn: sqlite3.Connection, combination: str) -> dict[str, Any]:
    summary = _row(
        conn,
        """
        SELECT
          ? AS combination,
          COUNT(*) AS hits,
          AVG(payout_yen) AS avg_payout_yen,
          MIN(payout_yen) AS min_payout_yen,
          MAX(payout_yen) AS max_payout_yen,
          AVG(popularity) AS avg_popularity
        FROM payouts
        WHERE bet_type = '3連単' AND combination = ?
        """,
        (combination, combination),
    )
    rows = _rows(
        conn,
        """
        SELECT
          r.race_id, r.race_date, r.jcd, r.venue_name, r.rno, r.title,
          p.combination, p.payout_yen, p.popularity
        FROM payouts p
        JOIN races r ON r.race_id = p.race_id
        WHERE p.bet_type = '3連単' AND p.combination = ?
        ORDER BY r.race_date DESC, r.jcd DESC, r.rno DESC
        LIMIT 80
        """,
        (combination,),
    )
    return _history_payload("combo", summary, rows)


def _history_payload(kind: str, summary_row: dict[str, Any] | None, rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = dict(summary_row or {})
    results = float(summary.get("result_rows") or 0)
    if results:
        summary["win_rate"] = float(summary.get("wins") or 0) / results
        summary["top3_rate"] = float(summary.get("top3") or 0) / results
    return {"kind": kind, "generated_at": _now(), "summary": summary, "rows": rows}


def _recent_races(conn: sqlite3.Connection, where_sql: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
    return _rows(
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


def _latest_prediction_rows(conn: sqlite3.Connection, race_id: str, *, limit: int) -> list[dict[str, Any]]:
    latest = conn.execute(
        "SELECT generated_at FROM predictions WHERE race_id = ? ORDER BY generated_at DESC LIMIT 1",
        (race_id,),
    ).fetchone()
    if not latest:
        return []
    return _rows(
        conn,
        """
        SELECT combination, probability, odds, expected_value, generated_at
        FROM predictions
        WHERE race_id = ? AND generated_at = ?
        ORDER BY probability DESC, COALESCE(expected_value, 0) DESC, combination
        LIMIT ?
        """,
        (race_id, latest["generated_at"], limit),
    )


def _stat_rows(
    conn: sqlite3.Connection,
    select_key: str,
    group_sql: str,
    limit: int,
    *,
    extra_where: str | None = None,
) -> list[dict[str, Any]]:
    where = "rr.rank IS NOT NULL"
    if extra_where:
        where += f" AND {extra_where}"
    return _rows(
        conn,
        f"""
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
        FROM entries e
        JOIN races r ON r.race_id = e.race_id
        JOIN race_results rr ON rr.race_id = e.race_id AND rr.lane = e.lane
        WHERE {where}
        GROUP BY {group_sql}
        HAVING COUNT(*) >= 20
        ORDER BY win_rate DESC, starts DESC
        LIMIT ?
        """,
        (limit,),
    )


def _row(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
    row = conn.execute(sql, params).fetchone()
    return rowdict(row) if row else None


def _rows(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    return [rowdict(row) for row in conn.execute(sql, params).fetchall()]


def _one(query: dict[str, list[str]], key: str, default: str | None = None) -> str | None:
    values = query.get(key)
    if not values or not values[0]:
        return default
    return values[0]


def _required(query: dict[str, list[str]], key: str) -> str:
    value = _one(query, key)
    if value is None:
        raise ValueError(f"missing query parameter: {key}")
    return value


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


ARCHIVE_CSS = """
    .navbtn { height:22px; padding:0 6px; border-radius:4px; font-size:10px; background:#fff; color:var(--accent); }
    .archive-panel { min-width:0; }
    .archive-head { display:flex; align-items:center; justify-content:space-between; gap:6px; margin-bottom:4px; }
    .archive-head h2 { margin:0; font-size:12px; }
    .archive-tabs { display:flex; gap:3px; align-items:center; flex-wrap:wrap; }
    .archive-tabs button,.entry-actions button,.race-title-actions button { height:18px; padding:0 5px; border-radius:3px; font-size:9px; background:#fff; color:var(--accent); border-color:#9bbdc1; }
    .archive-meta { color:var(--muted); font-size:10px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    .archive-content { max-height:430px; overflow:auto; border-top:1px solid var(--line); }
    .archive-kv { display:grid; grid-template-columns:repeat(4,minmax(82px,1fr)); gap:1px; background:var(--line); border:1px solid var(--line); margin:4px 0; }
    .archive-kv div { background:#fff; padding:3px 4px; min-width:0; }
    .archive-kv b { display:block; font-size:12px; line-height:1.1; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    .archive-kv span { color:var(--muted); font-size:9px; }
    .archive-table { margin-top:4px; table-layout:auto; }
    .archive-table th,.archive-table td { padding:2px 4px; font-size:10px; line-height:1.15; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    .archive-table tr.clickable,.pred-link { cursor:pointer; }
    .archive-table tr.clickable:hover,.pred-link:hover { background:#eef8f7; }
    .entry-actions { display:flex; gap:2px; margin-top:2px; }
    .entry-actions button { flex:1 1 auto; min-width:0; }
    .race-title-line { display:flex; align-items:center; gap:5px; min-width:0; }
    .race-title-name { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .race-title-actions { display:flex; gap:3px; }
    .venue-archive { height:15px; min-width:18px; padding:0 3px; margin-left:3px; font-size:9px; border-radius:3px; background:#fff; color:var(--accent); border-color:#9bbdc1; }
    @media (max-width:720px) { .archive-kv { grid-template-columns:repeat(2,minmax(86px,1fr)); } .archive-content { max-height:320px; } }
"""


ARCHIVE_MARKUP = """
        <div class="panel archive-panel">
          <div class="archive-head">
            <h2 id="archiveTitle">データ参照</h2>
            <div class="archive-tabs">
              <button id="archivePast" type="button">過去総合</button>
              <button id="archiveToday" type="button">当日</button>
              <button id="archiveStats" type="button">統計</button>
            </div>
          </div>
          <div id="archiveMeta" class="archive-meta">選択したレース/選手/場/モーター/ボート/枠の蓄積データを表示</div>
          <div id="archiveContent" class="archive-content"></div>
        </div>
"""


ARCHIVE_ENTRY_JS = """function escapeHtml(v){
  return String(v ?? "").replace(/[&<>"']/g, ch => ({ "&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;" }[ch]));
}
function racerDisplayName(e){
  const raw = String(e.racer_name || "").trim();
  const no = String(e.racer_no || "").trim();
  if(raw && raw !== no && !/^\\d+$/.test(raw)) return raw;
  return "選手名未取得";
}
function renderEntryCard(e){
  const lane = Number(e.lane || 0);
  const name = racerDisplayName(e);
  const no = e.racer_no ? `#${e.racer_no}` : "#-";
  const meta = `${e.racer_class || "-"} M${e.motor_no || "-"} B${e.boat_no || "-"}`;
  return `<div class="entry lane${lane}" title="${escapeHtml(`${lane}号艇 ${name} ${no} ${meta}`)}">
    <div class="entry-main"><span class="lane">${lane || "-"}</span><span class="racer-name">${escapeHtml(name)}</span></div>
    <div class="entry-meta"><span>${escapeHtml(no)}</span><span>${escapeHtml(meta)}</span></div>
    <div class="entry-actions">
      <button type="button" data-archive-kind="racer" data-racer-no="${escapeHtml(e.racer_no || "")}">選手</button>
      <button type="button" data-archive-kind="lane" data-lane="${lane || ""}">枠</button>
      <button type="button" data-archive-kind="motor" data-motor-no="${escapeHtml(e.motor_no || "")}">M</button>
      <button type="button" data-archive-kind="boat" data-boat-no="${escapeHtml(e.boat_no || "")}">B</button>
    </div>
  </div>`;
}
async function selectRace(raceId){
  state.raceId = raceId;
  const data = await getJson(`/api/predictions?race_id=${encodeURIComponent(raceId)}`);
  const race = data.race || {};
  state.currentRace = race;
  $("raceTitle").innerHTML = `<span class="race-title-line"><span class="race-title-name">${escapeHtml(`${race.venue_name || ""} ${race.rno || ""}R ${race.title || ""}`)}</span><span class="race-title-actions"><button id="raceArchiveBtn" type="button">R</button><button id="venueArchiveBtn" type="button">場</button></span></span>`;
  $("entries").innerHTML = data.entries.map(renderEntryCard).join("");
  $("predictions").innerHTML = data.predictions.map(p => `<tr class="pred-link" data-combo="${escapeHtml(p.combination)}"><td class="mono">${p.combination}</td><td>${pct(p.probability)}</td><td>${num(p.odds)}</td><td>${num(p.expected_value)}</td></tr>`).join("") || `<tr><td colspan="4" class="empty">予測はまだありません。</td></tr>`;
  $("combo").innerHTML = data.predictions.slice(0,20).map(p => `<option>${p.combination}</option>`).join("") || `<option>1-2-3</option>`;
  $("raceArchiveBtn").onclick = () => loadArchiveView("today", { race_id: raceId });
  $("venueArchiveBtn").onclick = () => loadArchiveView("history", { kind:"venue", jcd: race.jcd || "" });
  document.querySelectorAll("#entries button[data-archive-kind]").forEach(btn => btn.onclick = ev => {
    ev.stopPropagation();
    const kind = btn.dataset.archiveKind;
    const payload = { kind, jcd: race.jcd || "", rno: race.rno || "" };
    if(kind === "racer") payload.racer_no = btn.dataset.racerNo || "";
    if(kind === "lane") payload.lane = btn.dataset.lane || "";
    if(kind === "motor") payload.motor_no = btn.dataset.motorNo || "";
    if(kind === "boat") payload.boat_no = btn.dataset.boatNo || "";
    loadArchiveView("history", payload);
  });
  document.querySelectorAll("#predictions tr[data-combo]").forEach(row => row.onclick = () => loadArchiveView("history", { kind:"combo", combination: row.dataset.combo || "" }));
  state.combo = $("combo").value;
  await loadOdds();
  await loadArchiveView("today", { race_id: raceId }, { silent:true });
}
"""


ARCHIVE_VENUES_JS = """function renderVenues(items){
  $("venueFilter").innerHTML = `<option value="">全場</option>` + items.map(v => `<option value="${v.code}">${v.name}</option>`).join("");
  $("venueFilter").value = state.jcd;
  $("venueGrid").innerHTML = items.map(v => `<div class="venue ${v.code === state.jcd ? "active" : ""} ${venueTone(v)}" data-jcd="${v.code}">
    <b><span>${v.code} ${v.name}</span><span><button class="venue-archive" type="button" data-jcd="${v.code}" title="場の履歴">履</button><span class="badge ${statusClass(v.status)}" title="${statusTitle(v.status)}">${v.status}</span></span></b>
    <div class="next"><strong>${v.next_rno ? `${v.next_rno}R ${hm(v.next_deadline_at)}` : "-"}</strong><span>${minLabel(v.minutes_to_next_deadline)}</span><span class="od">od ${hm(v.latest_odds_at)}</span></div>
  </div>`).join("");
  document.querySelectorAll(".venue").forEach(el => el.onclick = () => { state.jcd = el.dataset.jcd; state.raceId = null; loadAll(); });
  document.querySelectorAll(".venue-archive").forEach(btn => btn.onclick = ev => { ev.stopPropagation(); loadArchiveView("history", { kind:"venue", jcd:btn.dataset.jcd || "" }); });
}
"""


ARCHIVE_JS = """function queryString(params){
  return Object.entries(params || {}).filter(([,v]) => v !== undefined && v !== null && String(v) !== "").map(([k,v]) => `${encodeURIComponent(k)}=${encodeURIComponent(v)}`).join("&");
}
function fmtCell(v){
  if(v === null || v === undefined || v === "") return "-";
  if(typeof v === "number"){
    if(Math.abs(v) < 1 && v !== 0) return v.toFixed(4);
    if(Math.abs(v) >= 1000) return Math.round(v).toLocaleString("ja-JP");
    return Number.isInteger(v) ? String(v) : v.toFixed(3);
  }
  return escapeHtml(String(v));
}
function pctCell(v){ return v == null ? "-" : `${(Number(v)*100).toFixed(2)}%`; }
function yenCell(v){ return v == null ? "-" : `${Math.round(Number(v)).toLocaleString("ja-JP")}円`; }
function archiveKvs(items){
  return `<div class="archive-kv">${items.map(([label,value,mode]) => `<div><b>${mode==="pct" ? pctCell(value) : mode==="yen" ? yenCell(value) : fmtCell(value)}</b><span>${escapeHtml(label)}</span></div>`).join("")}</div>`;
}
function archiveTable(rows, columns, options={}){
  if(!rows || !rows.length) return `<div class="empty">表示できるデータがありません。</div>`;
  const body = rows.map(row => {
    const cls = options.raceLink && row.race_id ? "clickable" : "";
    const attr = options.raceLink && row.race_id ? ` data-race="${escapeHtml(row.race_id)}"` : "";
    return `<tr class="${cls}"${attr}>${columns.map(c => `<td class="${c.mono ? "mono" : ""}" title="${escapeHtml(row[c.key] ?? "")}">${c.format ? c.format(row[c.key], row) : fmtCell(row[c.key])}</td>`).join("")}</tr>`;
  }).join("");
  const html = `<table class="archive-table"><thead><tr>${columns.map(c => `<th>${escapeHtml(c.label)}</th>`).join("")}</tr></thead><tbody>${body}</tbody></table>`;
  setTimeout(() => document.querySelectorAll("#archiveContent tr[data-race]").forEach(row => row.onclick = () => selectRace(row.dataset.race)), 0);
  return html;
}
async function loadArchiveView(view, params={}, opts={}){
  const d = $("raceDate").value || today;
  const merged = { date:d, ...params };
  const endpoint = view === "overview" ? "/api/archive/overview" : view === "today" ? "/api/archive/today" : view === "stats" ? "/api/archive/stats" : "/api/archive/history";
  $("archiveTitle").textContent = archiveTitle(view, merged);
  if(!opts.silent) $("archiveMeta").textContent = "読み込み中";
  const data = await getJson(`${endpoint}?${queryString(merged)}`);
  renderArchive(view, merged, data);
}
function archiveTitle(view, params){
  if(view === "overview") return "過去データ総合";
  if(view === "today") return params.race_id ? "当日レース詳細" : "当日データ";
  if(view === "stats") return "統計データ";
  const names = { racer:"選手履歴", venue:"場履歴", motor:"モーター履歴", boat:"ボート履歴", lane:"枠番履歴", combo:"組番履歴", race:"レース履歴" };
  return names[params.kind] || "履歴データ";
}
function renderArchive(view, params, data){
  $("archiveMeta").textContent = `更新 ${String(data.generated_at || "").replace("T"," ").slice(0,19)}`;
  if(view === "overview") return renderArchiveOverview(data);
  if(view === "today") return renderArchiveToday(data);
  if(view === "stats") return renderArchiveStats(data);
  return renderArchiveHistory(data);
}
function renderArchiveOverview(data){
  const t = data.totals || {}, td = data.today || {};
  $("archiveContent").innerHTML =
    archiveKvs([["総レース",t.races],["期間",`${t.first_date || "-"} - ${t.last_date || "-"}`],["出走行",t.entries],["結果R",t.result_races],["odds",t.odds_snapshots],["予測R",t.prediction_races],["展示R",t.beforeinfo_races],["当日R",td.races]]) +
    `<h3>年別</h3>` + archiveTable(data.years || [], [{key:"year",label:"年"},{key:"races",label:"R"},{key:"entry_races",label:"出走"},{key:"result_races",label:"結果"},{key:"prediction_races",label:"予測"}]) +
    `<h3>場別</h3>` + archiveTable(data.venues || [], [{key:"jcd",label:"場"},{key:"venue_name",label:"名称"},{key:"races",label:"R"},{key:"entry_races",label:"出走"},{key:"result_races",label:"結果"},{key:"odds_races",label:"od"}]);
}
function renderArchiveToday(data){
  if(data.mode === "race"){
    const r = data.race || {};
    $("archiveContent").innerHTML =
      archiveKvs([["場/R",`${r.venue_name || "-"} ${r.rno || "-"}R`],["日付",r.race_date],["締切/出",timePair(r)],["出走",r.entries],["odds",r.odds_snapshots],["展示",r.beforeinfo_rows],["結果",r.result_rows],["予測",r.latest_prediction]]) +
      `<h3>出走・展示・結果</h3>` + archiveTable(data.entries || [], [
        {key:"lane",label:"枠",mono:true},{key:"racer_name",label:"選手"},{key:"racer_no",label:"登番",mono:true},{key:"racer_class",label:"級"},{key:"branch",label:"支部"},{key:"origin",label:"出身"},
        {key:"national_win_rate",label:"全国"},{key:"local_win_rate",label:"当地"},{key:"motor_no",label:"M",mono:true},{key:"motor_2_rate",label:"M2"},{key:"boat_no",label:"B",mono:true},{key:"boat_2_rate",label:"B2"},
        {key:"exhibition_time",label:"展示"},{key:"exhibition_course",label:"展進"},{key:"exhibition_start_timing",label:"展ST"},{key:"rank",label:"着"},{key:"result_start_timing",label:"ST"}
      ]) +
      `<h3>モデル予測</h3>` + archiveTable(data.predictions || [], [{key:"combination",label:"3連単",mono:true},{key:"probability",label:"確率",format:pctCell},{key:"odds",label:"od"},{key:"expected_value",label:"EV"},{key:"generated_at",label:"生成"}]) +
      `<h3>払戻</h3>` + archiveTable(data.payouts || [], [{key:"bet_type",label:"式別"},{key:"combination",label:"組番",mono:true},{key:"payout_yen",label:"払戻",format:yenCell},{key:"popularity",label:"人気"}]);
    return;
  }
  $("archiveContent").innerHTML = archiveTable(data.races || [], [
    {key:"race_date",label:"日"},{key:"venue_name",label:"場"},{key:"rno",label:"R",mono:true},{key:"title",label:"タイトル"},{key:"race_type",label:"種別"},
    {key:"deadline_at",label:"締切/出",format:hm},{key:"entries",label:"出走"},{key:"odds_snapshots",label:"od"},{key:"beforeinfo_rows",label:"展示"},{key:"result_rows",label:"結果"},{key:"latest_prediction",label:"予測"}
  ], { raceLink:true });
}
function renderArchiveHistory(data){
  const s = data.summary || {};
  $("archiveContent").innerHTML =
    archiveKvs([["対象",s.racer_name || s.venue_name || s.number || s.combination || s.lane || "-"],["出走/件数",s.starts || s.races || s.hits],["結果",s.result_rows || s.result_races],["1着率",s.win_rate,"pct"],["3着内",s.top3_rate,"pct"],["平均着",s.avg_rank],["平均ST",s.avg_start],["平均払戻",s.avg_payout_yen,"yen"]]) +
    (data.facets ? `<h3>内訳</h3>` + archiveTable(data.facets, [{key:"lane",label:"枠"},{key:"starts",label:"出走"},{key:"wins",label:"1着"},{key:"top3",label:"3内"},{key:"avg_start",label:"ST"}]) : "") +
    `<h3>履歴</h3>` + archiveTable(data.rows || [], [
      {key:"race_date",label:"日"},{key:"venue_name",label:"場"},{key:"rno",label:"R",mono:true},{key:"lane",label:"枠",mono:true},{key:"racer_name",label:"選手"},{key:"racer_class",label:"級"},
      {key:"motor_no",label:"M",mono:true},{key:"boat_no",label:"B",mono:true},{key:"rank",label:"着"},{key:"start_timing",label:"ST"},{key:"result_combination",label:"結果",mono:true},{key:"payout_yen",label:"払戻",format:yenCell}
    ], { raceLink:true });
}
function renderArchiveStats(data){
  $("archiveContent").innerHTML =
    `<div class="archive-tabs"><button type="button" onclick="loadArchiveView('stats',{scope:'lane'})">枠</button><button type="button" onclick="loadArchiveView('stats',{scope:'venue'})">場</button><button type="button" onclick="loadArchiveView('stats',{scope:'rno'})">R</button><button type="button" onclick="loadArchiveView('stats',{scope:'class'})">級</button><button type="button" onclick="loadArchiveView('stats',{scope:'motor'})">M</button><button type="button" onclick="loadArchiveView('stats',{scope:'boat'})">B</button></div>` +
    archiveTable(data.rows || [], [
      {key:"label",label:"対象"},{key:"starts",label:"件数"},{key:"wins",label:"1着"},{key:"top3",label:"3内"},{key:"win_rate",label:"1着率",format:pctCell},{key:"top3_rate",label:"3内率",format:pctCell},
      {key:"avg_rank",label:"平均着"},{key:"avg_start",label:"ST"},{key:"avg_national_win_rate",label:"全国"},{key:"avg_local_win_rate",label:"当地"},{key:"avg_motor_2_rate",label:"M2"},{key:"avg_boat_2_rate",label:"B2"}
    ]);
}
function wireArchiveNav(){
  $("navPast").onclick = () => loadArchiveView("overview");
  $("navToday").onclick = () => loadArchiveView("today", { jcd: state.jcd || "" });
  $("navStats").onclick = () => loadArchiveView("stats", { scope:"lane" });
  $("archivePast").onclick = () => loadArchiveView("overview");
  $("archiveToday").onclick = () => loadArchiveView("today", { jcd: state.jcd || "" });
  $("archiveStats").onclick = () => loadArchiveView("stats", { scope:"lane" });
}
wireArchiveNav();
"""


def build_html() -> str:
    html = ui_base.HTML
    html = _replace_once(html, "</style>", ARCHIVE_CSS + "\n  </style>")
    html = _replace_once(
        html,
        '<button id="reload">更新</button><span id="clock"',
        '<button id="reload">更新</button><button id="navPast" class="navbtn" type="button">過去</button><button id="navToday" class="navbtn" type="button">当日</button><button id="navStats" class="navbtn" type="button">統計</button><span id="clock"',
    )
    html = _replace_once(
        html,
        '<canvas id="oddsChart" width="720" height="200"></canvas><div id="backtest" class="empty"></div></div>\n      </div>',
        '<canvas id="oddsChart" width="720" height="200"></canvas><div id="backtest" class="empty"></div></div>\n'
        f'{ARCHIVE_MARKUP}\n      </div>',
    )
    html = _replace_block(html, "function renderVenues(", "\nfunction renderActionTable", ARCHIVE_VENUES_JS)
    html = _replace_block(html, "function escapeHtml(", "\nasync function loadOdds", ARCHIVE_ENTRY_JS)
    html = _replace_once(html, "loadAll(); setInterval(loadAll,30000);", ARCHIVE_JS + "\nloadAll(); setInterval(loadAll,30000);")
    return html


def _replace_once(text: str, old: str, new: str) -> str:
    if old not in text:
        raise ValueError(f"HTML anchor not found: {old[:80]}")
    return text.replace(old, new, 1)


def _replace_block(text: str, start_marker: str, end_marker: str, replacement: str) -> str:
    start = text.index(start_marker)
    end = text.index(end_marker, start)
    return text[:start] + replacement + text[end:]


HTML = build_html()


if __name__ == "__main__":
    raise SystemExit(main())
