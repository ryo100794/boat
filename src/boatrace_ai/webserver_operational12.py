from __future__ import annotations

import argparse
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .constants import RACES_PER_DAY, VENUES
from .db import connect, init_db
from .webserver_all import backtest, odds, predictions, send_html, send_json, summary
from .webserver_operational11 import (
    HTML as BASE_HTML,
    day_overview_t5,
    purchase_guide_with_finished,
    venue_cards,
)
from .webserver_realtime import accuracy


HISTORICAL_TARGET_DAYS = 3650
TODAY_TARGET_RACES = len(VENUES) * len(RACES_PER_DAY)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="BOAT RACE AI T-5 dashboard with progress.")
    parser.add_argument("--db", default="data/boatrace.sqlite")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10001)
    parser.add_argument("--backtest", default="data/models/backtest.json")
    args = parser.parse_args(argv)
    init_db(args.db)
    handler = make_handler(Path(args.db), Path(args.backtest) if args.backtest else None)
    print(f"Serving BOAT RACE AI T-5 Progress Monitor on http://{args.host}:{args.port}", flush=True)
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
                    send_json(self, venue_cards(db_path, query))
                elif parsed.path == "/api/day":
                    send_json(self, day_overview_t5(db_path, query))
                elif parsed.path == "/api/guide":
                    send_json(self, purchase_guide_with_finished(db_path, query))
                elif parsed.path == "/api/progress":
                    send_json(self, progress(db_path, query))
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


def progress(db_path: Path, query: dict[str, list[str]]) -> dict[str, Any]:
    race_date = query.get("date", [date.today().isoformat()])[0]
    with connect(db_path) as conn:
        program_days = conn.execute(
            "SELECT COUNT(DISTINCT race_date) FROM raw_files WHERE kind = 'program' AND race_date < ?",
            (race_date,),
        ).fetchone()[0]
        result_days = conn.execute(
            "SELECT COUNT(DISTINCT race_date) FROM raw_files WHERE kind = 'result' AND race_date < ?",
            (race_date,),
        ).fetchone()[0]
        historical_races = conn.execute(
            "SELECT COUNT(*) FROM races WHERE race_date < ?",
            (race_date,),
        ).fetchone()[0]
        historical_results = conn.execute(
            """
            SELECT COUNT(DISTINCT r.race_id)
            FROM races r
            WHERE r.race_date < ?
              AND (SELECT COUNT(*) FROM race_results rr WHERE rr.race_id = r.race_id AND rr.rank IS NOT NULL) >= 3
            """,
            (race_date,),
        ).fetchone()[0]
        today_row = conn.execute(
            """
            SELECT
              COUNT(DISTINCT r.race_id) AS races,
              SUM(CASE WHEN entry_counts.entries = 6 THEN 1 ELSE 0 END) AS racelists,
              SUM(CASE WHEN odds_counts.odds_rows > 0 THEN 1 ELSE 0 END) AS odds_races,
              SUM(CASE WHEN result_counts.result_rows >= 3 THEN 1 ELSE 0 END) AS finals
            FROM races r
            LEFT JOIN (
              SELECT race_id, COUNT(*) AS entries FROM entries GROUP BY race_id
            ) entry_counts ON entry_counts.race_id = r.race_id
            LEFT JOIN (
              SELECT race_id, COUNT(*) AS odds_rows FROM odds_snapshots GROUP BY race_id
            ) odds_counts ON odds_counts.race_id = r.race_id
            LEFT JOIN (
              SELECT race_id, COUNT(*) AS result_rows FROM race_results WHERE rank IS NOT NULL GROUP BY race_id
            ) result_counts ON result_counts.race_id = r.race_id
            WHERE r.race_date = ?
            """,
            (race_date,),
        ).fetchone()
    today_counts = {
        "races": int(today_row["races"] or 0),
        "racelists": int(today_row["racelists"] or 0),
        "odds_races": int(today_row["odds_races"] or 0),
        "finals": int(today_row["finals"] or 0),
    }
    return {
        "date": race_date,
        "historical": {
            "target_days": HISTORICAL_TARGET_DAYS,
            "program_days": int(program_days or 0),
            "result_days": int(result_days or 0),
            "program_remaining_days": max(0, HISTORICAL_TARGET_DAYS - int(program_days or 0)),
            "result_remaining_days": max(0, HISTORICAL_TARGET_DAYS - int(result_days or 0)),
            "races": int(historical_races or 0),
            "result_races": int(historical_results or 0),
        },
        "today": {
            "target_races": TODAY_TARGET_RACES,
            **today_counts,
            "race_remaining": max(0, TODAY_TARGET_RACES - today_counts["races"]),
            "racelist_remaining": max(0, TODAY_TARGET_RACES - today_counts["racelists"]),
            "odds_remaining": max(0, TODAY_TARGET_RACES - today_counts["odds_races"]),
            "final_remaining": max(0, TODAY_TARGET_RACES - today_counts["finals"]),
        },
    }


def build_html() -> str:
    html = BASE_HTML
    html = html.replace(
        'const [s, vc, g, day, acc, bt] = await Promise.all([',
        'const [s, vc, g, day, acc, bt, prog] = await Promise.all([',
    )
    html = html.replace(
        'getJson("/api/backtest")\n  ]);',
        'getJson("/api/backtest"),\n    getJson(`/api/progress?date=${encodeURIComponent(d)}`)\n  ]);',
    )
    html = html.replace(
        '$("dataStatus").textContent = `取得状態: レース ${s.races} / 出走 ${s.entries} / 結果 ${s.results} / オッズ ${s.odds_snapshots} / 予測 ${s.predictions} / 更新 ${day.now_jst.replace("T"," ").slice(0,19)}`;',
        '$("dataStatus").textContent = footerStatus(s, prog, day.now_jst);',
    )
    html = html.replace(
        'function statusClass(v){ return v==="監視中" ? "live" : v==="終了" ? "done" : v==="未取得" ? "" : "wait"; }',
        'function statusClass(v){ return v==="監視中" ? "live" : v==="終了" ? "done" : v==="未取得" ? "" : "wait"; }\n'
        'function footerStatus(s, prog, nowIso){ const h=prog.historical||{}, t=prog.today||{}; return `過去分: 番組LZH 残${h.program_remaining_days ?? "-"}日 / 結果LZH 残${h.result_remaining_days ?? "-"}日 / parsed ${h.races ?? "-"}R / 結果 ${h.result_races ?? "-"}R | 本日分: レース 残${t.race_remaining ?? "-"} / 出走 残${t.racelist_remaining ?? "-"} / odds 残${t.odds_remaining ?? "-"} / 結果 残${t.final_remaining ?? "-"} / 予測 ${s.predictions} | 更新 ${nowIso.replace("T"," ").slice(0,19)}`; }',
    )
    return html


HTML = build_html()


if __name__ == "__main__":
    raise SystemExit(main())
