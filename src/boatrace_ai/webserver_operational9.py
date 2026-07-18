from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .db import init_db
from .webserver_all import backtest, odds, predictions, send_html, send_json, summary
from .webserver_operational2 import day_overview, venue_cards
from .webserver_operational5 import purchase_guide_with_finished
from .webserver_operational8 import HTML as BASE_HTML
from .webserver_realtime import accuracy


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="BOAT RACE AI fixed four-column venue dashboard.")
    parser.add_argument("--db", default="data/boatrace.sqlite")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10001)
    parser.add_argument("--backtest", default="data/models/backtest.json")
    args = parser.parse_args(argv)
    init_db(args.db)
    handler = make_handler(Path(args.db), Path(args.backtest) if args.backtest else None)
    print(f"Serving BOAT RACE AI Fixed Four-Column Monitor on http://{args.host}:{args.port}", flush=True)
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


def build_html() -> str:
    html = BASE_HTML
    html = html.replace(
        ".venue { background:#fff; border:1px solid var(--line); border-radius:5px; padding:4px 5px; cursor:pointer; min-height:58px; }",
        ".venue { background:#fff; border:1px solid var(--line); border-radius:5px; padding:3px 4px; cursor:pointer; min-height:52px; }",
    )
    html = html.replace(
        ".venue b { display:flex; align-items:center; justify-content:space-between; gap:4px; font-size:11px; line-height:1.1; white-space:nowrap; overflow:hidden; }",
        ".venue b { display:flex; align-items:center; justify-content:space-between; gap:3px; font-size:10px; line-height:1.05; white-space:nowrap; overflow:hidden; }",
    )
    html = html.replace(
        ".venue .next strong { font-size:10px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; } .venue .next span { color:var(--muted); font-size:10px; white-space:nowrap; }",
        ".venue .next strong { font-size:9px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; } .venue .next span { color:var(--muted); font-size:9px; white-space:nowrap; }",
    )
    html = html.replace(
        ".metric b { display:inline; font-size:10px; line-height:1; } .metric span { display:inline; color:var(--muted); font-size:9px; line-height:1; margin-left:1px; }",
        ".metric b { display:inline; font-size:9px; line-height:1; } .metric span { display:inline; color:var(--muted); font-size:8px; line-height:1; margin-left:1px; }",
    )
    html = html.replace(
        "@media (max-width:720px) { .stats { grid-template-columns:repeat(2,minmax(120px,1fr)); } .venue-grid { grid-template-columns:repeat(2,minmax(0,1fr)); }",
        "@media (max-width:720px) { .stats { grid-template-columns:repeat(2,minmax(120px,1fr)); } .venue-grid { grid-template-columns:repeat(4,minmax(0,1fr)); }",
    )
    return html


HTML = build_html()


if __name__ == "__main__":
    raise SystemExit(main())
