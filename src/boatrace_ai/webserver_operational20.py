from __future__ import annotations

import argparse
import sqlite3
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .constants import VENUES
from .db import connect, init_db
from .webserver_all import backtest, odds, predictions, send_html, send_json, summary
from .webserver_operational2 import dict_row, iso, minutes_between, parse_any_time, race_payload
from .webserver_operational12 import progress as base_progress
from .webserver_operational19 import HTML as BASE_HTML, purchase_guide_with_recent_closed
from .webserver_operational2 import now_jst, parse_jst
from .webserver_realtime import accuracy


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="BOAT RACE AI active-venue dashboard.")
    parser.add_argument("--db", default="data/boatrace.sqlite")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10001)
    parser.add_argument("--backtest", default="data/models/backtest_no_odds_v7.json")
    args = parser.parse_args(argv)
    init_db(args.db)
    handler = make_handler(Path(args.db), Path(args.backtest) if args.backtest else None)
    print(f"Serving BOAT RACE AI Active Venue Monitor on http://{args.host}:{args.port}", flush=True)
    ThreadingHTTPServer((args.host, args.port), handler).serve_forever()
    return 0


def make_handler(db_path: Path, backtest_path: Path | None):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            try:
                if parsed.path == "/":
                    send_html(self, HTML)
                elif parsed.path == "/api/summary":
                    send_json(self, summary(db_path))
                elif parsed.path == "/api/venues":
                    send_json(self, venue_cards_active(db_path, query))
                elif parsed.path == "/api/day":
                    send_json(self, day_overview_t5_active(db_path, query))
                elif parsed.path == "/api/guide":
                    send_json(self, purchase_guide_with_recent_closed(db_path, query))
                elif parsed.path == "/api/progress":
                    send_json(self, progress_active(db_path, query))
                elif parsed.path == "/api/predictions":
                    send_json(self, predictions(db_path, query))
                elif parsed.path == "/api/odds":
                    send_json(self, odds(db_path, query))
                elif parsed.path == "/api/backtest":
                    send_json(self, backtest(backtest_path))
                elif parsed.path == "/api/accuracy":
                    send_json(self, accuracy(db_path, query))
                else:
                    self.send_error(404)
            except Exception as exc:
                send_json(self, {"error": str(exc)}, status=500)

        def log_message(self, fmt: str, *args: Any) -> None:
            return

    return Handler


def venue_cards_active(db_path: Path, query: dict[str, list[str]]) -> dict[str, Any]:
    race_date = query.get("date", [date.today().isoformat()])[0]
    now = now_jst()
    with connect(db_path) as conn:
        grouped = conn.execute(
            f"""
            WITH per_race AS ({PER_RACE_SQL})
            SELECT
              jcd,
              COUNT(*) AS raw_races,
              SUM(is_active) AS races,
              SUM(CASE WHEN is_active = 1 AND entries = 6 THEN 1 ELSE 0 END) AS racelists,
              COALESCE(SUM(CASE WHEN is_active = 1 THEN odds_snapshots ELSE 0 END), 0) AS odds_snapshots,
              SUM(CASE WHEN is_active = 1 AND result_rows >= 3 THEN 1 ELSE 0 END) AS finals,
              MAX(CASE WHEN is_active = 1 THEN latest_prediction ELSE NULL END) AS latest_prediction,
              MAX(CASE WHEN is_active = 1 THEN latest_odds_at ELSE NULL END) AS latest_odds_at
            FROM per_race
            GROUP BY jcd
            """,
            (race_date,),
        ).fetchall()
        upcoming = conn.execute(
            f"""
            WITH per_race AS ({PER_RACE_SQL})
            SELECT jcd, deadline_at, rno, result_rows
            FROM per_race
            WHERE is_active = 1 AND deadline_at IS NOT NULL
            ORDER BY deadline_at, jcd, rno
            """,
            (race_date,),
        ).fetchall()

    by_code = {row["jcd"]: dict_row(row) for row in grouped}
    next_by_code: dict[str, tuple[Any, int]] = {}
    for row in upcoming:
        if int(row["result_rows"] or 0) >= 3:
            continue
        deadline = parse_jst(row["deadline_at"])
        if deadline and deadline >= now and row["jcd"] not in next_by_code:
            next_by_code[row["jcd"]] = (deadline, int(row["rno"]))

    cards = []
    for venue in VENUES:
        stats = by_code.get(venue.code, {})
        active_races = int(stats.get("races") or 0)
        racelists = int(stats.get("racelists") or 0)
        odds_count = int(stats.get("odds_snapshots") or 0)
        finals = int(stats.get("finals") or 0)
        if active_races == 0:
            status = "開催なし"
        elif finals >= active_races:
            status = "終了"
        elif odds_count > 0:
            status = "監視中"
        elif racelists > 0:
            status = "出走表"
        else:
            status = "取得中"
        next_deadline, next_rno = next_by_code.get(venue.code, (None, None))
        latest_odds = parse_any_time(stats.get("latest_odds_at"))
        cards.append(
            {
                "code": venue.code,
                "name": venue.name,
                "status": status,
                "races": active_races,
                "raw_races": int(stats.get("raw_races") or 0),
                "racelists": racelists,
                "odds_snapshots": odds_count,
                "finals": finals,
                "latest_prediction": stats.get("latest_prediction"),
                "latest_odds_at": iso(latest_odds),
                "next_rno": next_rno,
                "next_deadline_at": iso(next_deadline),
                "minutes_to_next_deadline": minutes_between(now, next_deadline),
            }
        )
    return {"date": race_date, "now_jst": iso(now), "venues": cards}


def day_overview_t5_active(db_path: Path, query: dict[str, list[str]]) -> dict[str, Any]:
    race_date = query.get("date", [date.today().isoformat()])[0]
    jcd = query.get("jcd", [None])[0]
    now = now_jst()
    params: list[Any] = [race_date]
    jcd_filter = ""
    if jcd:
        jcd_filter = "AND jcd = ?"
        params.append(jcd.zfill(2))
    with connect(db_path) as conn:
        rows = conn.execute(
            f"""
            WITH per_race AS ({PER_RACE_SQL})
            SELECT race_id, race_date, jcd, venue_name, rno, title, status, deadline_at,
                   entries, odds_snapshots, latest_odds_at, result_rows, latest_prediction
            FROM per_race
            WHERE is_active = 1 {jcd_filter}
            ORDER BY deadline_at IS NULL, deadline_at, jcd, rno
            """,
            params,
        ).fetchall()
        races = [_race_payload_t5(conn, row, now=now) for row in rows]
    return {"date": race_date, "now_jst": iso(now), "races": races}


def progress_active(db_path: Path, query: dict[str, list[str]]) -> dict[str, Any]:
    payload = base_progress(db_path, query)
    race_date = query.get("date", [date.today().isoformat()])[0]
    with connect(db_path) as conn:
        row = conn.execute(
            f"""
            WITH per_race AS ({PER_RACE_SQL})
            SELECT
              SUM(is_active) AS races,
              SUM(CASE WHEN is_active = 1 AND entries = 6 THEN 1 ELSE 0 END) AS racelists,
              SUM(CASE WHEN is_active = 1 AND odds_snapshots > 0 THEN 1 ELSE 0 END) AS odds_races,
              SUM(CASE WHEN is_active = 1 AND result_rows >= 3 THEN 1 ELSE 0 END) AS finals
            FROM per_race
            """,
            (race_date,),
        ).fetchone()
    active = {
        "races": int(row["races"] or 0),
        "racelists": int(row["racelists"] or 0),
        "odds_races": int(row["odds_races"] or 0),
        "finals": int(row["finals"] or 0),
    }
    payload["today"].update(
        {
            "target_races": active["races"],
            **active,
            "race_remaining": 0,
            "racelist_remaining": max(0, active["races"] - active["racelists"]),
            "odds_remaining": max(0, active["races"] - active["odds_races"]),
            "final_remaining": max(0, active["races"] - active["finals"]),
        }
    )
    return payload


def _race_payload_t5(conn: sqlite3.Connection, row: sqlite3.Row, *, now) -> dict[str, Any]:
    item = race_payload(conn, row, now=now, before_minutes=5)
    if item.get("time_status") == "T-10超過":
        item["time_status"] = "T-5超過"
    return item


PER_RACE_SQL = """
SELECT
  r.race_id,
  r.race_date,
  r.jcd,
  r.venue_name,
  r.rno,
  r.title,
  r.status,
  r.deadline_at,
  COALESCE(entry_counts.entries, 0) AS entries,
  COALESCE(odds_counts.snapshots, 0) AS odds_snapshots,
  odds_counts.latest_odds_at AS latest_odds_at,
  COALESCE(result_counts.results, 0) AS result_rows,
  pred_counts.generated_at AS latest_prediction,
  CASE
    WHEN r.deadline_at IS NOT NULL
      OR COALESCE(entry_counts.entries, 0) = 6
      OR COALESCE(odds_counts.snapshots, 0) > 0
      OR COALESCE(result_counts.results, 0) >= 3
      OR pred_counts.generated_at IS NOT NULL
    THEN 1 ELSE 0
  END AS is_active
FROM races r
LEFT JOIN (
  SELECT race_id, COUNT(*) AS entries FROM entries GROUP BY race_id
) entry_counts ON entry_counts.race_id = r.race_id
LEFT JOIN (
  SELECT race_id, COUNT(*) AS snapshots, MAX(captured_at) AS latest_odds_at
  FROM odds_snapshots GROUP BY race_id
) odds_counts ON odds_counts.race_id = r.race_id
LEFT JOIN (
  SELECT race_id, COUNT(*) AS results FROM race_results WHERE rank IS NOT NULL GROUP BY race_id
) result_counts ON result_counts.race_id = r.race_id
LEFT JOIN (
  SELECT race_id, MAX(generated_at) AS generated_at FROM predictions GROUP BY race_id
) pred_counts ON pred_counts.race_id = r.race_id
WHERE r.race_date = ?
"""


def build_html() -> str:
    html = BASE_HTML
    html = html.replace(
        'function statusClass(v){ return v==="監視中" ? "live" : v==="終了" ? "done" : v==="未取得" ? "" : "wait"; }',
        'function statusClass(v){ return v==="監視中" ? "live" : v==="終了" ? "done" : (v==="未取得"||v==="開催なし") ? "" : "wait"; }',
    )
    html = html.replace(
        'function statusTitle(v){ return v==="監視中" ? "出走表とオッズを取得済みで、ライブ更新対象です。" : v==="出走表" ? "出走表は取得済み、オッズは未取得です。" : v==="取得中" ? "当日レース情報を取得中です。" : v==="終了" ? "全レースの結果が入っています。" : "当日データはまだ未取得です。"; }',
        'function statusTitle(v){ return v==="監視中" ? "出走表とオッズを取得済みで、ライブ更新対象です。" : v==="出走表" ? "出走表は取得済み、オッズは未取得です。" : v==="開催なし" ? "本日はこの場の開催が確認されていません。" : v==="取得中" ? "当日レース情報を取得中です。" : v==="終了" ? "全レースの結果が入っています。" : "当日データはまだ未取得です。"; }',
    )
    html = html.replace(
        'if(v.status==="未取得") return "s-none"; return "s-wait"; }',
        'if(v.status==="未取得" || v.status==="開催なし") return "s-none"; return "s-wait"; }',
    )
    return html


HTML = build_html()


if __name__ == "__main__":
    raise SystemExit(main())
