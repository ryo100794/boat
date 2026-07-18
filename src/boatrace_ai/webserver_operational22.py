from __future__ import annotations

import argparse
from http.server import ThreadingHTTPServer
from pathlib import Path

from . import webserver_operational21 as base
from .db import init_db


LIVE_WIPE_CSS = """
    .live-wipe { position:fixed; right:10px; bottom:24px; width:min(620px,calc(100vw - 20px)); min-width:min(420px,calc(100vw - 20px)); resize:both; overflow:auto; background:#11191b; color:#f6fbfc; border:1px solid #2d4247; box-shadow:0 10px 28px rgba(0,0,0,.28); z-index:30; font-size:11px; }
    .live-wipe.hidden { display:none; }
    .live-wipe-head { display:flex; align-items:center; justify-content:space-between; gap:6px; padding:4px 6px; background:#172126; border-bottom:1px solid #2d4247; }
    .live-wipe-title { font-weight:800; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .live-wipe-actions { display:flex; gap:4px; align-items:center; }
    .live-wipe-actions a,.live-wipe-actions button { height:18px; padding:0 5px; border:1px solid #48636a; border-radius:3px; background:#23363a; color:#fff; font-size:10px; text-decoration:none; line-height:16px; cursor:pointer; }
    .live-wipe-video { position:relative; aspect-ratio:16/9; background:#050808; min-height:230px; }
    .live-wipe-video iframe { position:absolute; inset:0; width:100%; height:100%; border:0; background:#050808; }
    .live-wipe-meta { display:grid; grid-template-columns:1fr 1fr; gap:1px; background:#263b40; border-top:1px solid #2d4247; }
    .live-wipe-meta span { min-width:0; padding:3px 5px; background:#11191b; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    @media (max-width:1320px) { .live-wipe { width:min(520px,calc(100vw - 20px)); } .live-wipe-video { min-height:190px; } }
    @media (max-width:720px) { .live-wipe { width:calc(100vw - 12px); min-width:0; right:6px; bottom:20px; } .live-wipe-video { min-height:180px; max-height:260px; } }
"""


HTML = base.HTML.replace(base.LIVE_WIPE_CSS, LIVE_WIPE_CSS)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="BOAT RACE AI dashboard with larger live wipe.")
    parser.add_argument("--db", default="data/boatrace.sqlite")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10001)
    parser.add_argument("--backtest", default="data/models/backtest_no_odds_v7.json")
    args = parser.parse_args(argv)
    init_db(args.db)
    base.HTML = HTML
    handler = base.make_handler(Path(args.db), Path(args.backtest) if args.backtest else None)
    print(f"Serving BOAT RACE AI Larger Live Wipe Monitor on http://{args.host}:{args.port}", flush=True)
    ThreadingHTTPServer((args.host, args.port), handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
