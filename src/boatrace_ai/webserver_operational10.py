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
from .webserver_operational9 import HTML as BASE_HTML
from .webserver_realtime import accuracy


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="BOAT RACE AI dense compact dashboard.")
    parser.add_argument("--db", default="data/boatrace.sqlite")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10001)
    parser.add_argument("--backtest", default="data/models/backtest.json")
    args = parser.parse_args(argv)
    init_db(args.db)
    handler = make_handler(Path(args.db), Path(args.backtest) if args.backtest else None)
    print(f"Serving BOAT RACE AI Dense Compact Monitor on http://{args.host}:{args.port}", flush=True)
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
    replacements = {
        "body { margin:0; color:var(--ink); background:#fff; font-size:12px; }":
            "body { margin:0; color:var(--ink); background:#fff; font-size:11px; }",
        "header { display:flex; align-items:center; justify-content:space-between; gap:14px; padding:9px 12px; border-bottom:1px solid var(--line); background:#fff; position:sticky; top:0; z-index:6; }":
            "header { display:flex; align-items:center; justify-content:space-between; gap:8px; padding:5px 8px; border-bottom:1px solid var(--line); background:#fff; position:sticky; top:0; z-index:6; }",
        "h1 { margin:0; font-size:17px; letter-spacing:0; } main { display:grid; grid-template-columns:600px 1fr; min-height:calc(100vh - 49px); padding-bottom:22px; }":
            "h1 { margin:0; font-size:14px; letter-spacing:0; } main { display:grid; grid-template-columns:600px 1fr; min-height:calc(100vh - 35px); padding-bottom:18px; }",
        "aside { background:var(--band); border-right:1px solid var(--line); padding:8px; overflow:auto; } section { padding:10px 12px; min-width:0; }":
            "aside { background:var(--band); border-right:1px solid var(--line); padding:5px; overflow:auto; } section { padding:6px 8px; min-width:0; }",
        "input, select, button { height:30px; border:1px solid var(--line); border-radius:6px; padding:0 8px; background:#fff; color:var(--ink); font:inherit; }":
            "input, select, button { height:24px; border:1px solid var(--line); border-radius:4px; padding:0 6px; background:#fff; color:var(--ink); font:inherit; }",
        "button { background:var(--accent); border-color:var(--accent); color:#fff; cursor:pointer; } .toolbar { display:flex; gap:8px; align-items:center; flex-wrap:wrap; }":
            "button { background:var(--accent); border-color:var(--accent); color:#fff; cursor:pointer; } .toolbar { display:flex; gap:5px; align-items:center; flex-wrap:wrap; }",
        ".venue-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:4px; margin-top:0; }":
            ".venue-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:3px; margin-top:0; }",
        ".venue { background:#fff; border:1px solid var(--line); border-radius:5px; padding:3px 4px; cursor:pointer; min-height:52px; }":
            ".venue { background:#fff; border:1px solid var(--line); border-radius:4px; padding:2px 3px; cursor:pointer; min-height:46px; }",
        ".venue .next { display:grid; grid-template-columns:1fr auto; gap:3px; align-items:center; margin:3px 0; padding:2px 3px; background:#f8fafb; border:1px solid #edf1f2; border-radius:3px; }":
            ".venue .next { display:grid; grid-template-columns:1fr auto; gap:2px; align-items:center; margin:2px 0; padding:1px 2px; background:#f8fafb; border:1px solid #edf1f2; border-radius:3px; }",
        ".metrics { display:grid; grid-template-columns:repeat(3,1fr); gap:1px; margin-bottom:2px; background:var(--line); border:1px solid var(--line); }":
            ".metrics { display:grid; grid-template-columns:repeat(3,1fr); gap:1px; margin-bottom:1px; background:var(--line); border:1px solid var(--line); }",
        ".metric { background:#fff; padding:2px 3px; text-align:right; min-width:0; }":
            ".metric { background:#fff; padding:1px 2px; text-align:right; min-width:0; }",
        ".timeline-frame { border:1px solid var(--line); border-top:2px solid var(--accent); background:#fff; margin-bottom:12px; }":
            ".timeline-frame { border:1px solid var(--line); border-top:2px solid var(--accent); background:#fff; margin-bottom:7px; }",
        ".timeline-frame h2 { margin:0; padding:7px 8px; font-size:14px; letter-spacing:0; display:flex; justify-content:space-between; gap:8px; border-bottom:1px solid var(--line); }":
            ".timeline-frame h2 { margin:0; padding:4px 6px; font-size:12px; letter-spacing:0; display:flex; justify-content:space-between; gap:6px; border-bottom:1px solid var(--line); }",
        ".timeline-scroll { max-height:235px; overflow-y:auto; scrollbar-gutter:stable; }":
            ".timeline-scroll { max-height:205px; overflow-y:auto; scrollbar-gutter:stable; }",
        ".timeline-scroll th { top:0; } .timeline-scroll tr { height:50px; }":
            ".timeline-scroll th { top:0; } .timeline-scroll tr { height:38px; }",
        ".grid2 { display:grid; grid-template-columns:minmax(520px,1.05fr) minmax(430px,.95fr); gap:12px; margin-top:12px; }":
            ".grid2 { display:grid; grid-template-columns:minmax(520px,1.05fr) minmax(430px,.95fr); gap:8px; margin-top:7px; }",
        ".panel { border-top:2px solid var(--accent); padding-top:8px; min-width:0; } .panel h2 { margin:0 0 7px; font-size:14px; letter-spacing:0; display:flex; justify-content:space-between; gap:8px; }":
            ".panel { border-top:2px solid var(--accent); padding-top:5px; min-width:0; } .panel h2 { margin:0 0 4px; font-size:12px; letter-spacing:0; display:flex; justify-content:space-between; gap:6px; }",
        "th,td { border-bottom:1px solid var(--line); padding:5px; text-align:right; vertical-align:top; overflow-wrap:anywhere; }":
            "th,td { border-bottom:1px solid var(--line); padding:3px 4px; text-align:right; vertical-align:top; overflow-wrap:anywhere; }",
        "th { color:var(--muted); font-weight:700; background:#fafbfb; position:sticky; top:49px; z-index:2; }":
            "th { color:var(--muted); font-weight:700; background:#fafbfb; position:sticky; top:35px; z-index:2; }",
        ".entries { display:grid; grid-template-columns:repeat(6,minmax(72px,1fr)); gap:1px; background:var(--line); border:1px solid var(--line); margin:7px 0 10px; }":
            ".entries { display:grid; grid-template-columns:repeat(6,minmax(60px,1fr)); gap:1px; background:var(--line); border:1px solid var(--line); margin:4px 0 6px; }",
        ".entry { background:#fff; min-height:68px; padding:6px; } .lane { display:inline-grid; place-items:center; width:22px; height:22px; border:1px solid var(--line); font-weight:700; margin-bottom:3px; }":
            ".entry { background:#fff; min-height:50px; padding:4px; } .lane { display:inline-grid; place-items:center; width:18px; height:18px; border:1px solid var(--line); font-weight:700; margin-bottom:2px; }",
        "canvas { width:100%; height:140px; border:1px solid var(--line); background:#fff; }":
            "canvas { width:100%; height:105px; border:1px solid var(--line); background:#fff; }",
        ".ops-status { position:fixed; left:0; right:0; bottom:0; z-index:8; padding:4px 10px; border-top:1px solid var(--line); background:#fff; color:var(--muted); font-size:10px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }":
            ".ops-status { position:fixed; left:0; right:0; bottom:0; z-index:8; padding:2px 8px; border-top:1px solid var(--line); background:#fff; color:var(--muted); font-size:9px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }",
    }
    for old, new in replacements.items():
        html = html.replace(old, new)
    return html


HTML = build_html()


if __name__ == "__main__":
    raise SystemExit(main())
