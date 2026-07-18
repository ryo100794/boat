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
from .webserver_operational7 import HTML as BASE_HTML
from .webserver_realtime import accuracy


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="BOAT RACE AI four-column venue dashboard.")
    parser.add_argument("--db", default="data/boatrace.sqlite")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10001)
    parser.add_argument("--backtest", default="data/models/backtest.json")
    args = parser.parse_args(argv)
    init_db(args.db)
    handler = make_handler(Path(args.db), Path(args.backtest) if args.backtest else None)
    print(f"Serving BOAT RACE AI Four-Column Monitor on http://{args.host}:{args.port}", flush=True)
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
        "main { display:grid; grid-template-columns:460px 1fr; min-height:calc(100vh - 49px); padding-bottom:22px; }",
        "main { display:grid; grid-template-columns:600px 1fr; min-height:calc(100vh - 49px); padding-bottom:22px; }",
    )
    html = html.replace(
        ".venue-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:4px; margin-top:0; }",
        ".venue-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:4px; margin-top:0; }",
    )
    html = html.replace(
        "@media (max-width:1180px) { main,.grid2 { grid-template-columns:1fr; }",
        "@media (max-width:1320px) { main,.grid2 { grid-template-columns:1fr; }",
    )
    html = html.replace(
        'function statusClass(v){ return v==="監視中" ? "live" : v==="終了" ? "done" : v==="未取得" ? "" : "wait"; }',
        'function statusClass(v){ return v==="監視中" ? "live" : v==="終了" ? "done" : v==="未取得" ? "" : "wait"; }\n'
        'function statusTitle(v){ return v==="監視中" ? "出走表とオッズを取得済みで、ライブ更新対象です。" : v==="出走表" ? "出走表は取得済み、オッズは未取得です。" : v==="取得中" ? "当日レース情報を取得中です。" : v==="終了" ? "全レースの結果が入っています。" : "当日データはまだ未取得です。"; }',
    )
    html = html.replace(
        '<b><span>${v.code} ${v.name}</span><span class="badge ${statusClass(v.status)}">${v.status}</span></b>',
        '<b><span>${v.code} ${v.name}</span><span class="badge ${statusClass(v.status)}" title="${statusTitle(v.status)}">${v.status}</span></b>',
    )
    return html


HTML = build_html()


if __name__ == "__main__":
    raise SystemExit(main())
