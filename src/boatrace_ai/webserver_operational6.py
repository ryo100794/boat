from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .db import init_db
from .webserver_all import backtest, odds, predictions, send_html, send_json, summary
from .webserver_operational2 import day_overview, venue_cards
from .webserver_operational5 import HTML as BASE_HTML, purchase_guide_with_finished
from .webserver_realtime import accuracy


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="BOAT RACE AI compact venue dashboard.")
    parser.add_argument("--db", default="data/boatrace.sqlite")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10001)
    parser.add_argument("--backtest", default="data/models/backtest.json")
    args = parser.parse_args(argv)
    init_db(args.db)
    handler = make_handler(Path(args.db), Path(args.backtest) if args.backtest else None)
    print(f"Serving BOAT RACE AI Compact Monitor on http://{args.host}:{args.port}", flush=True)
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
        "main { display:grid; grid-template-columns:430px 1fr; min-height:calc(100vh - 49px); padding-bottom:22px; }",
        "main { display:grid; grid-template-columns:460px 1fr; min-height:calc(100vh - 49px); padding-bottom:22px; }",
    )
    html = html.replace(
        ".venue-grid { display:grid; grid-template-columns:repeat(2,minmax(190px,1fr)); gap:5px; margin-top:8px; }",
        ".venue-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:4px; margin-top:0; }",
    )
    html = html.replace(
        ".venue { background:#fff; border:1px solid var(--line); border-radius:6px; padding:6px; cursor:pointer; min-height:112px; }",
        ".venue { background:#fff; border:1px solid var(--line); border-radius:5px; padding:4px 5px; cursor:pointer; min-height:58px; }",
    )
    html = html.replace(
        ".venue b { display:flex; align-items:center; justify-content:space-between; gap:6px; font-size:13px; line-height:1.15; }",
        ".venue b { display:flex; align-items:center; justify-content:space-between; gap:4px; font-size:11px; line-height:1.1; white-space:nowrap; overflow:hidden; }",
    )
    html = html.replace(
        ".venue .next { display:grid; grid-template-columns:1fr auto; gap:4px; align-items:center; margin:5px 0; padding:4px 5px; background:#f8fafb; border:1px solid #edf1f2; border-radius:4px; }",
        ".venue .next { display:grid; grid-template-columns:1fr auto; gap:3px; align-items:center; margin:3px 0; padding:2px 3px; background:#f8fafb; border:1px solid #edf1f2; border-radius:3px; }",
    )
    html = html.replace(
        ".venue .next strong { font-size:12px; } .venue .next span { color:var(--muted); font-size:11px; white-space:nowrap; }",
        ".venue .next strong { font-size:10px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; } .venue .next span { color:var(--muted); font-size:10px; white-space:nowrap; }",
    )
    html = html.replace(
        ".metrics { display:grid; grid-template-columns:repeat(4,1fr); gap:1px; margin-bottom:4px; background:var(--line); border:1px solid var(--line); }",
        ".metrics { display:grid; grid-template-columns:repeat(3,1fr); gap:1px; margin-bottom:2px; background:var(--line); border:1px solid var(--line); }",
    )
    html = html.replace(
        ".metric { background:#fff; padding:3px 4px; text-align:right; min-width:0; }",
        ".metric { background:#fff; padding:2px 3px; text-align:right; min-width:0; }",
    )
    html = html.replace(
        ".metric b { display:block; font-size:13px; line-height:1.05; } .metric span { display:block; color:var(--muted); font-size:10px; line-height:1.1; }",
        ".metric b { display:inline; font-size:10px; line-height:1; } .metric span { display:inline; color:var(--muted); font-size:9px; line-height:1; margin-left:1px; }",
    )
    html = html.replace(
        ".subline { display:grid; grid-template-columns:48px 1fr; gap:4px; color:var(--muted); font-size:11px; line-height:1.25; white-space:nowrap; overflow:hidden; }",
        ".subline { display:grid; grid-template-columns:28px 1fr; gap:2px; color:var(--muted); font-size:9px; line-height:1.1; white-space:nowrap; overflow:hidden; }",
    )
    html = html.replace(
        '.badge { display:inline-block; border-radius:999px; padding:1px 6px; color:#fff; background:var(--muted); font-size:11px; font-weight:700; white-space:nowrap; }',
        '.badge { display:inline-block; border-radius:999px; padding:1px 4px; color:#fff; background:var(--muted); font-size:9px; font-weight:700; white-space:nowrap; }',
    )
    html = html.replace(
        ".timeline-scroll { max-height:245px; overflow-y:auto; scrollbar-gutter:stable; }",
        ".timeline-scroll { max-height:235px; overflow-y:auto; scrollbar-gutter:stable; }",
    )
    html = html.replace(
        '@media (max-width:720px) { .stats { grid-template-columns:repeat(2,minmax(120px,1fr)); } .venue-grid { grid-template-columns:1fr; }',
        '@media (max-width:720px) { .stats { grid-template-columns:repeat(2,minmax(120px,1fr)); } .venue-grid { grid-template-columns:repeat(2,minmax(0,1fr)); }',
    )
    return replace_render_venues(html)


def replace_render_venues(html: str) -> str:
    start = html.index("function renderVenues(")
    end = html.index("\nfunction renderGuide", start)
    return html[:start] + RENDER_VENUES + html[end:]


RENDER_VENUES = """function renderVenues(items){
  $("venueFilter").innerHTML = `<option value="">全場</option>` + items.map(v => `<option value="${v.code}">${v.name}</option>`).join("");
  $("venueFilter").value = state.jcd;
  $("venueGrid").innerHTML = items.map(v => `<div class="venue ${v.code === state.jcd ? "active" : ""}" data-jcd="${v.code}">
    <b><span>${v.code} ${v.name}</span><span class="badge ${statusClass(v.status)}">${v.status}</span></b>
    <div class="next"><strong>${v.next_rno ? `${v.next_rno}R ${hm(v.next_deadline_at)}` : "-"}</strong><span>${minLabel(v.minutes_to_next_deadline)}</span></div>
    <div class="metrics">
      <div class="metric"><b>${v.racelists}</b><span>出</span></div>
      <div class="metric"><b>${v.odds_snapshots}</b><span>od</span></div>
      <div class="metric"><b>${v.finals}</b><span>確</span></div>
    </div>
    <div class="subline"><span>od</span><strong>${age(v.latest_odds_at)}</strong></div>
  </div>`).join("");
  document.querySelectorAll(".venue").forEach(el => el.onclick = () => { state.jcd = el.dataset.jcd; state.raceId = null; loadAll(); });
}
"""


HTML = build_html()


if __name__ == "__main__":
    raise SystemExit(main())
