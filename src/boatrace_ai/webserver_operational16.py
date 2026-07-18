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
from .webserver_operational15 import HTML as BASE_HTML
from .webserver_realtime import accuracy


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="BOAT RACE AI chronological timeline dashboard.")
    parser.add_argument("--db", default="data/boatrace.sqlite")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10001)
    parser.add_argument("--backtest", default="data/models/backtest.json")
    args = parser.parse_args(argv)
    init_db(args.db)
    handler = make_handler(Path(args.db), Path(args.backtest) if args.backtest else None)
    print(f"Serving BOAT RACE AI Chronological Monitor on http://{args.host}:{args.port}", flush=True)
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
    return replace_action_table(BASE_HTML)


def replace_action_table(html: str) -> str:
    start = html.index("function renderActionTable(")
    end = html.index("\nasync function selectRace", start)
    return html[:start] + ACTION_TABLE + html[end:]


ACTION_TABLE = """function renderActionTable(rows, candidates, finished){
  let candidateMap = new Map((candidates || []).map(r => [r.race_id, r]));
  const upcoming = futureRows(rows);
  candidateMap = fallbackCandidates(upcoming, candidateMap);
  const picks = upcoming.filter(r => candidateMap.has(r.race_id));
  const others = upcoming.filter(r => !candidateMap.has(r.race_id));
  const action = [
    ...finished.map(r => ({ kind:"確定", item:r, source:r, isPick:false, isFinal:true })),
    ...picks.map(r => ({ kind:"候補", item:candidateMap.get(r.race_id), source:r, isPick:true, isFinal:false })),
    ...others.map(r => ({ kind:"予定", item:r, source:r, isPick:false, isFinal:false })),
  ].sort((a,b) => {
    const av = actionTime(a);
    const bv = actionTime(b);
    if(av !== bv) return av - bv;
    return String(a.source.jcd).localeCompare(String(b.source.jcd)) || Number(a.source.rno || 0) - Number(b.source.rno || 0);
  });
  $("timelineInfo").textContent = `確定 ${finished.length}R / 候補 ${picks.length}R / 今後 ${upcoming.length}R / 時刻昇順`;
  $("actionRows").innerHTML = action.map((row,idx) => {
    const r = row.source;
    const item = row.item || r;
    const p = item.top_prediction || {};
    const shortRace = `${r.venue_name}${r.rno}R`;
    if(row.isFinal){
      const mark = item.top_hit ? "的中" : (item.top5_hit ? "上位5" : "外れ");
      const result = `${item.result_combination || "-"} ${mark}`;
      const payout = item.trifecta_payout_yen == null ? num(p.expected_value) : `${Number(item.trifecta_payout_yen).toLocaleString("ja-JP")}円`;
      return `<tr class="final" data-race="${r.race_id}"><td><span class="badge 確定">確定</span></td><td title="${r.venue_name} ${r.rno}R ${r.title || ""}"><b>${shortRace}</b></td><td class="mono">${hm(r.race_time_at || r.deadline_at)}</td><td class="mono" title="${result}">${result}</td><td class="mono">${p.combination || "-"}</td><td>${pct(p.probability)}</td><td>${num(p.odds)}</td><td class="mono">${payout}</td></tr>`;
    }
    const status = r.time_status === "T-10超過" ? "T-5超過" : r.time_status;
    const verdict = `<span class="badge ${cls(status)}">${status}</span>${row.isPick ? ` <span class="badge 候補">候補</span>` : ""}`;
    const tone = rowTone(r, row.isPick, idx);
    return `<tr class="${tone}" data-race="${r.race_id}"><td><span class="badge ${row.isPick ? "候補" : ""}">${row.kind}</span></td><td title="${r.venue_name} ${r.rno}R ${r.title || ""}"><b>${shortRace}</b></td><td class="mono">${hm(r.deadline_at)} <span class="muted">${minLabel(r.minutes_to_deadline)}</span></td><td>${verdict}</td><td class="mono">${p.combination || "-"}</td><td>${pct(p.probability)}</td><td>${num(p.odds)}</td><td>${num(p.expected_value)}</td></tr>`;
  }).join("") || `<tr><td colspan="8" class="empty">表示対象のレース情報はありません。</td></tr>`;
  document.querySelectorAll("#actionRows tr[data-race]").forEach(row => row.onclick = () => selectRace(row.dataset.race));
}
function actionTime(row){
  const r = row.source || {};
  const item = row.item || r;
  const value = row.isFinal ? (item.race_time_at || item.deadline_at || item.latest_odds_at) : (r.deadline_at || item.deadline_at);
  const parsed = value ? new Date(value).getTime() : Number.MAX_SAFE_INTEGER;
  return Number.isFinite(parsed) ? parsed : Number.MAX_SAFE_INTEGER;
}
"""


HTML = build_html()


if __name__ == "__main__":
    raise SystemExit(main())
