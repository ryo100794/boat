from __future__ import annotations

import argparse
from http.server import ThreadingHTTPServer
from pathlib import Path

from . import webserver_operational26 as base
from .db import init_db


SMALL_WIPE_CSS = """
    .live-wipe { width:360px; min-width:260px; max-width:min(360px,calc(100vw - 16px)); right:8px; bottom:18px; }
    .live-wipe-video { min-height:0; height:auto; aspect-ratio:16/9; }
    .live-wipe-head { padding:3px 5px; }
    .live-wipe-title { font-size:10px; }
    .live-wipe-actions a,.live-wipe-actions button { height:16px; padding:0 4px; font-size:9px; line-height:14px; }
    .live-wipe-meta { left:4px; right:4px; bottom:4px; gap:2px; }
    .live-wipe-meta span { padding:1px 3px; font-size:8.5px; }
    .live-wipe.zoom .live-wipe-video iframe { width:112%; height:112%; transform:translate(-5.35%,-5.35%); }
    @media (max-width:720px) { .live-wipe { width:320px; min-width:240px; max-width:calc(100vw - 10px); } .live-wipe-video { min-height:0; max-height:none; } }
"""


HTML = base.HTML.replace("</style>", SMALL_WIPE_CSS + "\n  </style>")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="BOAT RACE AI corrected timing dashboard with compact live wipe.")
    parser.add_argument("--db", default="data/boatrace.sqlite")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10001)
    parser.add_argument("--backtest", default="data/models/backtest_no_odds_v8.json")
    args = parser.parse_args(argv)
    init_db(args.db)
    base.HTML = HTML
    handler = base.make_handler(Path(args.db), Path(args.backtest) if args.backtest else None)
    print(f"Serving BOAT RACE AI Compact Live Wipe Monitor on http://{args.host}:{args.port}", flush=True)
    ThreadingHTTPServer((args.host, args.port), handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

