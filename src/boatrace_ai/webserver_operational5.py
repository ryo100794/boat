from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .db import connect, init_db
from .webserver_all import backtest, odds, predictions, send_html, send_json, summary
from .webserver_operational2 import day_overview, purchase_guide, race_payload, venue_cards, now_jst
from .webserver_operational4 import HTML, result_summary
from .webserver_realtime import accuracy


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="BOAT RACE AI dense operational dashboard.")
    parser.add_argument("--db", default="data/boatrace.sqlite")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10001)
    parser.add_argument("--backtest", default="data/models/backtest.json")
    args = parser.parse_args(argv)
    init_db(args.db)
    handler = make_handler(Path(args.db), Path(args.backtest) if args.backtest else None)
    print(f"Serving BOAT RACE AI Dense Monitor on http://{args.host}:{args.port}", flush=True)
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
                    send_json(self, day_overview(db_path, query))
                elif parsed.path == "/api/guide":
                    send_json(self, purchase_guide_with_finished(db_path, query))
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


def purchase_guide_with_finished(db_path: Path, query: dict[str, list[str]]) -> dict[str, Any]:
    payload = purchase_guide(db_path, query)
    race_date = query.get("date", [payload["date"]])[0]
    before_minutes = int(query.get("before_minutes", ["10"])[0])
    finished_limit = int(query.get("finished_limit", ["4"])[0])
    now = now_jst()
    with connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT r.race_id, r.race_date, r.jcd, r.venue_name, r.rno, r.title,
                   r.status, r.deadline_at,
                   (SELECT COUNT(*) FROM entries e WHERE e.race_id = r.race_id) AS entries,
                   (SELECT COUNT(*) FROM odds_snapshots os WHERE os.race_id = r.race_id) AS odds_snapshots,
                   (SELECT MAX(captured_at) FROM odds_snapshots os WHERE os.race_id = r.race_id) AS latest_odds_at,
                   (SELECT COUNT(*) FROM race_results rr WHERE rr.race_id = r.race_id AND rr.rank IS NOT NULL) AS result_rows,
                   (SELECT MAX(generated_at) FROM predictions p WHERE p.race_id = r.race_id) AS latest_prediction,
                   (SELECT MAX(updated_at) FROM race_results rr WHERE rr.race_id = r.race_id) AS result_updated_at
            FROM races r
            WHERE r.race_date = ?
              AND (SELECT COUNT(*) FROM entries e WHERE e.race_id = r.race_id) = 6
              AND EXISTS (SELECT 1 FROM predictions p WHERE p.race_id = r.race_id)
              AND (SELECT COUNT(*) FROM race_results rr WHERE rr.race_id = r.race_id AND rr.rank IS NOT NULL) >= 3
            ORDER BY
              CASE WHEN r.deadline_at IS NULL THEN 1 ELSE 0 END,
              r.deadline_at DESC,
              result_updated_at DESC,
              r.rno DESC,
              r.jcd DESC
            LIMIT ?
            """,
            (race_date, finished_limit),
        ).fetchall()
        finished = []
        for row in rows:
            item = race_payload(conn, row, now=now, before_minutes=before_minutes)
            item.update(result_summary(conn, row["race_id"]))
            top = item.get("top_prediction") or {}
            result_combination = item.get("result_combination")
            item["top_hit"] = bool(result_combination and top.get("combination") == result_combination)
            item["top5_hit"] = bool(
                result_combination
                and any(pred.get("combination") == result_combination for pred in item.get("top5", []))
            )
            finished.append(item)
    payload["finished"] = finished
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
