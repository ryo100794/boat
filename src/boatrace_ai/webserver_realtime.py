from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .db import connect, init_db
from .webserver_all import (
    HTML,
    backtest,
    odds,
    predictions,
    races,
    send_html,
    send_json,
    summary,
    venues,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="BOAT RACE AI realtime dashboard.")
    parser.add_argument("--db", default="data/boatrace.sqlite")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10001)
    parser.add_argument("--backtest", default="data/models/backtest.json")
    args = parser.parse_args(argv)
    init_db(args.db)
    handler = make_handler(Path(args.db), Path(args.backtest) if args.backtest else None)
    print(f"Serving BOAT RACE AI Monitor on http://{args.host}:{args.port}", flush=True)
    ThreadingHTTPServer((args.host, args.port), handler).serve_forever()
    return 0


def make_handler(db_path: Path, backtest_path: Path | None):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            try:
                if parsed.path == "/":
                    send_html(self, html_with_accuracy())
                elif parsed.path == "/api/summary":
                    send_json(self, summary(db_path))
                elif parsed.path == "/api/venues":
                    send_json(self, venues(db_path, query))
                elif parsed.path == "/api/races":
                    send_json(self, races(db_path, query))
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


def html_with_accuracy() -> str:
    html = HTML.replace(
        '<div id="backtest" class="empty"></div>',
        '<div id="accuracy" class="empty"></div><div id="backtest" class="empty"></div>',
    )
    html = html.replace(
        "await loadBacktest();",
        "await loadAccuracy();\n  await loadBacktest();",
    )
    html = html.replace(
        "async function loadBacktest() {",
        """
async function loadAccuracy() {
  const raceDate = $("raceDate").value || today;
  const data = await getJson(`/api/accuracy?date=${encodeURIComponent(raceDate)}`);
  $("accuracy").innerHTML =
    `本日: ${data.evaluated || 0} races / winner ${pct(data.winner_top1_accuracy)} / 3T top5 ${pct(data.trifecta_top5_hit_rate)}`;
}
async function loadBacktest() {""",
    )
    return html


def accuracy(db_path: Path, query: dict[str, list[str]]) -> dict[str, Any]:
    race_date = query.get("date", [date.today().isoformat()])[0]
    with connect(db_path) as conn:
        race_rows = conn.execute(
            """
            SELECT r.race_id
            FROM races r
            WHERE r.race_date = ?
              AND (SELECT COUNT(*) FROM race_results rr WHERE rr.race_id = r.race_id AND rr.rank IS NOT NULL) >= 3
              AND EXISTS (SELECT 1 FROM predictions p WHERE p.race_id = r.race_id)
            """,
            (race_date,),
        ).fetchall()
        evaluated = 0
        winner_hits = 0
        trifecta_top1_hits = 0
        trifecta_top5_hits = 0
        for race in race_rows:
            rid = race["race_id"]
            actual_rows = conn.execute(
                """
                SELECT lane, rank
                FROM race_results
                WHERE race_id = ? AND rank IS NOT NULL
                ORDER BY rank
                LIMIT 3
                """,
                (rid,),
            ).fetchall()
            if len(actual_rows) < 3:
                continue
            actual_combo = "-".join(str(row["lane"]) for row in actual_rows)
            actual_winner = actual_combo.split("-")[0]
            latest = conn.execute(
                """
                SELECT generated_at
                FROM predictions
                WHERE race_id = ?
                ORDER BY generated_at DESC
                LIMIT 1
                """,
                (rid,),
            ).fetchone()
            if not latest:
                continue
            pred_rows = conn.execute(
                """
                SELECT combination
                FROM predictions
                WHERE race_id = ? AND generated_at = ?
                ORDER BY COALESCE(expected_value, probability) DESC, probability DESC
                LIMIT 5
                """,
                (rid, latest["generated_at"]),
            ).fetchall()
            if not pred_rows:
                continue
            top = pred_rows[0]["combination"]
            top5 = [row["combination"] for row in pred_rows]
            evaluated += 1
            winner_hits += 1 if top.split("-")[0] == actual_winner else 0
            trifecta_top1_hits += 1 if top == actual_combo else 0
            trifecta_top5_hits += 1 if actual_combo in top5 else 0
    return {
        "date": race_date,
        "evaluated": evaluated,
        "winner_top1_accuracy": winner_hits / evaluated if evaluated else None,
        "trifecta_top1_hit_rate": trifecta_top1_hits / evaluated if evaluated else None,
        "trifecta_top5_hit_rate": trifecta_top5_hits / evaluated if evaluated else None,
    }


if __name__ == "__main__":
    raise SystemExit(main())
