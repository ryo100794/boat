from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .db import init_db
from .webserver_all import backtest, odds, predictions, send_html, send_json, summary
from .webserver_operational11 import day_overview_t5
from .webserver_operational12 import progress, purchase_guide_with_finished, venue_cards
from .webserver_operational16 import HTML as BASE_HTML
from .webserver_realtime import accuracy


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="BOAT RACE AI single-line timeline dashboard.")
    parser.add_argument("--db", default="data/boatrace.sqlite")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10001)
    parser.add_argument("--backtest", default="data/models/backtest.json")
    args = parser.parse_args(argv)
    init_db(args.db)
    handler = make_handler(Path(args.db), Path(args.backtest) if args.backtest else None)
    print(f"Serving BOAT RACE AI Single-Line Timeline Monitor on http://{args.host}:{args.port}", flush=True)
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


def build_html() -> str:
    html = BASE_HTML
    html = html.replace(
        ".timeline-scroll th { top:0; } .timeline-scroll tr { height:38px; }",
        ".timeline-scroll th { top:0; } .timeline-scroll tr { height:20px; } .timeline-scroll th,.timeline-scroll td { padding:1px 3px; line-height:1.05; vertical-align:middle; }",
    )
    html = html.replace(
        ".timeline-frame h2 { margin:0; padding:4px 6px; font-size:12px; letter-spacing:0; display:flex; justify-content:space-between; gap:6px; border-bottom:1px solid var(--line); }",
        ".timeline-frame h2 { margin:0; padding:3px 5px; font-size:12px; letter-spacing:0; display:flex; justify-content:space-between; gap:6px; border-bottom:1px solid var(--line); }",
    )
    html = html.replace(
        ".timeline-scroll { max-height:215px;",
        ".timeline-scroll { max-height:190px;",
    )
    return html


HTML = build_html()


if __name__ == "__main__":
    raise SystemExit(main())
