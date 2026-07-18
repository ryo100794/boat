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
from .webserver_operational17 import HTML as BASE_HTML
from .webserver_realtime import accuracy


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="BOAT RACE AI compact merged venue-card dashboard.")
    parser.add_argument("--db", default="data/boatrace.sqlite")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10001)
    parser.add_argument("--backtest", default="data/models/backtest.json")
    args = parser.parse_args(argv)
    init_db(args.db)
    handler = make_handler(Path(args.db), Path(args.backtest) if args.backtest else None)
    print(f"Serving BOAT RACE AI Merged Venue Card Monitor on http://{args.host}:{args.port}", flush=True)
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
        ".venue { background:#fff; border:1px solid var(--line); border-radius:4px; padding:2px 3px; cursor:pointer; min-height:46px; }",
        ".venue { background:#fff; border:1px solid var(--line); border-radius:4px; padding:2px 3px; cursor:pointer; min-height:31px; }",
    )
    html = html.replace(
        ".venue .next { display:grid; grid-template-columns:1fr auto; gap:2px; align-items:center; margin:2px 0; padding:1px 2px; background:#f8fafb; border:1px solid #edf1f2; border-radius:3px; }",
        ".venue .next { display:grid; grid-template-columns:minmax(34px,1fr) auto auto; gap:2px; align-items:center; margin:1px 0 0; padding:1px 2px; background:#f8fafb; border:1px solid #edf1f2; border-radius:3px; }",
    )
    html = html.replace(
        ".venue .next span { color:var(--muted); font-size:9px; white-space:nowrap; }",
        ".venue .next span { color:var(--muted); font-size:9px; white-space:nowrap; } .venue .next .od { color:var(--ink); font-weight:600; overflow:hidden; text-overflow:ellipsis; }",
    )
    return replace_render_venues(html)


def replace_render_venues(html: str) -> str:
    start = html.index("function renderVenues(")
    end = html.index("\nfunction renderActionTable", start)
    return html[:start] + RENDER_VENUES + html[end:]


RENDER_VENUES = """function renderVenues(items){
  $("venueFilter").innerHTML = `<option value="">全場</option>` + items.map(v => `<option value="${v.code}">${v.name}</option>`).join("");
  $("venueFilter").value = state.jcd;
  $("venueGrid").innerHTML = items.map(v => `<div class="venue ${v.code === state.jcd ? "active" : ""} ${venueTone(v)}" data-jcd="${v.code}">
    <b><span>${v.code} ${v.name}</span><span class="badge ${statusClass(v.status)}" title="${statusTitle(v.status)}">${v.status}</span></b>
    <div class="next"><strong>${v.next_rno ? `${v.next_rno}R ${hm(v.next_deadline_at)}` : "-"}</strong><span>${minLabel(v.minutes_to_next_deadline)}</span><span class="od">od ${hm(v.latest_odds_at)}</span></div>
  </div>`).join("");
  document.querySelectorAll(".venue").forEach(el => el.onclick = () => { state.jcd = el.dataset.jcd; state.raceId = null; loadAll(); });
}
"""


HTML = build_html()


if __name__ == "__main__":
    raise SystemExit(main())
