from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from . import webserver_operational25 as base
from .db import connect, init_db
from .time_semantics import time_fields_from_stored_start
from .webserver_all import backtest, odds, send_html, send_json, summary
from .webserver_model_rank import accuracy_model_rank, predictions_model_rank
from .webserver_operational2 import now_jst


HTML = base.HTML


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="BOAT RACE AI corrected timing dashboard with corrected prediction details.")
    parser.add_argument("--db", default="data/boatrace.sqlite")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10001)
    parser.add_argument("--backtest", default="data/models/backtest_no_odds_v8.json")
    args = parser.parse_args(argv)
    init_db(args.db)
    handler = make_handler(Path(args.db), Path(args.backtest) if args.backtest else None)
    print(f"Serving BOAT RACE AI Corrected Timing Detail Monitor on http://{args.host}:{args.port}", flush=True)
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
                    send_json(self, base.venue_cards_start_time(db_path, query))
                elif parsed.path == "/api/day":
                    send_json(self, base.day_overview_start_time(db_path, query))
                elif parsed.path == "/api/guide":
                    send_json(self, base.purchase_guide_start_time(db_path, query))
                elif parsed.path == "/api/live-wipe":
                    send_json(self, base.live_wipe_start_time(db_path, query))
                elif parsed.path == "/api/progress":
                    send_json(self, base.progress_active(db_path, query))
                elif parsed.path == "/api/predictions":
                    send_json(self, predictions_start_time_model_rank(db_path, query))
                elif parsed.path == "/api/odds":
                    send_json(self, odds(db_path, query))
                elif parsed.path == "/api/backtest":
                    send_json(self, backtest(backtest_path))
                elif parsed.path == "/api/accuracy":
                    send_json(self, accuracy_model_rank(db_path, query))
                else:
                    self.send_error(404)
            except Exception as exc:
                send_json(self, {"error": str(exc)}, status=500)

        def log_message(self, fmt: str, *args: Any) -> None:
            return

    return Handler


def predictions_start_time_model_rank(db_path: Path, query: dict[str, list[str]]) -> dict[str, Any]:
    payload = predictions_model_rank(db_path, query)
    race = payload.get("race")
    if race:
        now = now_jst()
        race_id = str(race.get("race_id") or "")
        with connect(db_path) as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS result_rows
                FROM race_results
                WHERE race_id = ? AND rank IS NOT NULL
                """,
                (race_id,),
            ).fetchone()
        race.update(
            time_fields_from_stored_start(
                race.get("deadline_at"),
                now=now,
                before_minutes=5,
                result_rows=int(row["result_rows"] or 0) if row else 0,
            )
        )
    payload["time_basis"] = "stored_deadline_at_is_race_start"
    return payload


if __name__ == "__main__":
    raise SystemExit(main())

