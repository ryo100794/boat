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
from .webserver_operational14 import HTML as BASE_HTML
from .webserver_realtime import accuracy


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="BOAT RACE AI compact header dashboard.")
    parser.add_argument("--db", default="data/boatrace.sqlite")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10001)
    parser.add_argument("--backtest", default="data/models/backtest.json")
    args = parser.parse_args(argv)
    init_db(args.db)
    handler = make_handler(Path(args.db), Path(args.backtest) if args.backtest else None)
    print(f"Serving BOAT RACE AI Compact Header Monitor on http://{args.host}:{args.port}", flush=True)
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
        "header { display:flex; align-items:center; justify-content:space-between; gap:8px; padding:5px 8px; border-bottom:1px solid var(--line); background:#fff; position:sticky; top:0; z-index:6; }",
        "header { display:grid; grid-template-columns:auto 1fr; align-items:center; gap:6px; padding:3px 6px; border-bottom:1px solid var(--line); background:#fff; position:sticky; top:0; z-index:6; overflow:hidden; }",
    )
    html = html.replace(
        "h1 { margin:0; font-size:14px; letter-spacing:0; } main { display:grid; grid-template-columns:600px 1fr; min-height:calc(100vh - 35px); padding-bottom:18px; }",
        "h1 { margin:0; font-size:12px; letter-spacing:0; white-space:nowrap; } main { display:grid; grid-template-columns:600px 1fr; min-height:calc(100vh - 29px); padding-bottom:18px; }",
    )
    html = html.replace(
        "input, select, button { height:24px; border:1px solid var(--line); border-radius:4px; padding:0 6px; background:#fff; color:var(--ink); font:inherit; }",
        "input, select, button { height:22px; border:1px solid var(--line); border-radius:4px; padding:0 5px; background:#fff; color:var(--ink); font:inherit; }",
    )
    html = html.replace(
        "button { background:var(--accent); border-color:var(--accent); color:#fff; cursor:pointer; } .toolbar { display:flex; gap:5px; align-items:center; flex-wrap:wrap; }",
        "button { background:var(--accent); border-color:var(--accent); color:#fff; cursor:pointer; } .toolbar { display:flex; gap:4px; align-items:center; justify-content:flex-end; flex-wrap:nowrap; min-width:0; overflow:hidden; } #clock { white-space:nowrap; overflow:hidden; text-overflow:ellipsis; } #raceDate { width:118px; } #venueFilter { width:74px; }",
    )
    html = html.replace(
        "<th>区分</th><th>場/R</th><th>締切</th><th>判定/結果</th><th>odds取得</th><th>予測</th><th>確率</th><th>オッズ</th><th>EV/払戻</th>",
        "<th>区分</th><th>場/R</th><th>締切</th><th>判定/結果</th><th>予測</th><th>確率</th><th>オッズ</th><th>EV/払戻</th>",
    )
    html = html.replace(
        ".timeline-scroll { max-height:215px; overflow-y:auto; scrollbar-gutter:stable; } .timeline-scroll table { table-layout:fixed; } .timeline-scroll th,.timeline-scroll td { white-space:nowrap; overflow:hidden; text-overflow:ellipsis; } .timeline-scroll th:nth-child(1),.timeline-scroll td:nth-child(1){width:42px;} .timeline-scroll th:nth-child(2),.timeline-scroll td:nth-child(2){width:86px;} .timeline-scroll th:nth-child(3),.timeline-scroll td:nth-child(3){width:58px;} .timeline-scroll th:nth-child(4),.timeline-scroll td:nth-child(4){width:82px;} .timeline-scroll th:nth-child(5),.timeline-scroll td:nth-child(5){width:72px;} .timeline-scroll th:nth-child(6),.timeline-scroll td:nth-child(6){width:58px;} .timeline-scroll th:nth-child(7),.timeline-scroll td:nth-child(7){width:54px;} .timeline-scroll th:nth-child(8),.timeline-scroll td:nth-child(8){width:52px;} .timeline-scroll th:nth-child(9),.timeline-scroll td:nth-child(9){width:68px;}",
        ".timeline-scroll { max-height:215px; overflow-y:auto; scrollbar-gutter:stable; } .timeline-scroll table { table-layout:fixed; } .timeline-scroll th,.timeline-scroll td { white-space:nowrap; overflow:hidden; text-overflow:ellipsis; } .timeline-scroll th:nth-child(1),.timeline-scroll td:nth-child(1){width:34px;} .timeline-scroll th:nth-child(2),.timeline-scroll td:nth-child(2){width:70px;} .timeline-scroll th:nth-child(3),.timeline-scroll td:nth-child(3){width:62px;} .timeline-scroll th:nth-child(4),.timeline-scroll td:nth-child(4){width:82px;} .timeline-scroll th:nth-child(5),.timeline-scroll td:nth-child(5){width:56px;} .timeline-scroll th:nth-child(6),.timeline-scroll td:nth-child(6){width:50px;} .timeline-scroll th:nth-child(7),.timeline-scroll td:nth-child(7){width:50px;} .timeline-scroll th:nth-child(8),.timeline-scroll td:nth-child(8){width:64px;}",
    )
    html = html.replace(
        "@media (max-width:720px) { .stats { grid-template-columns:repeat(2,minmax(120px,1fr)); } .venue-grid { grid-template-columns:repeat(4,minmax(0,1fr)); } .entries { grid-template-columns:repeat(3,minmax(72px,1fr)); } header { align-items:flex-start; flex-direction:column; } }",
        "@media (max-width:720px) { .stats { grid-template-columns:repeat(2,minmax(120px,1fr)); } .venue-grid { grid-template-columns:repeat(4,minmax(0,1fr)); } .entries { grid-template-columns:repeat(3,minmax(72px,1fr)); } header { grid-template-columns:auto 1fr; align-items:center; } }",
    )
    return replace_action_table(html)


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
  ];
  $("timelineInfo").textContent = `確定 ${finished.length}R / 候補 ${picks.length}R / 今後 ${upcoming.length}R`;
  $("actionRows").innerHTML = action.map((row,idx) => {
    const r = row.source;
    const item = row.item || r;
    const p = item.top_prediction || {};
    const shortRace = `${r.venue_name}${r.rno}R`;
    if(row.isFinal){
      const mark = item.top_hit ? "的中" : (item.top5_hit ? "上位5" : "外れ");
      const result = `${item.result_combination || "-"} ${mark}`;
      const payout = item.trifecta_payout_yen == null ? num(p.expected_value) : `${Number(item.trifecta_payout_yen).toLocaleString("ja-JP")}円`;
      return `<tr class="final" data-race="${r.race_id}"><td><span class="badge 確定">確定</span></td><td title="${r.venue_name} ${r.rno}R ${r.title || ""}"><b>${shortRace}</b></td><td class="mono">${hm(r.deadline_at)}</td><td class="mono" title="${result}">${result}</td><td class="mono">${p.combination || "-"}</td><td>${pct(p.probability)}</td><td>${num(p.odds)}</td><td class="mono">${payout}</td></tr>`;
    }
    const status = r.time_status === "T-10超過" ? "T-5超過" : r.time_status;
    const verdict = `<span class="badge ${cls(status)}">${status}</span>${row.isPick ? ` <span class="badge 候補">候補</span>` : ""}`;
    const tone = rowTone(r, row.isPick, idx);
    return `<tr class="${tone}" data-race="${r.race_id}"><td><span class="badge ${row.isPick ? "候補" : ""}">${row.kind}</span></td><td title="${r.venue_name} ${r.rno}R ${r.title || ""}"><b>${shortRace}</b></td><td class="mono">${hm(r.deadline_at)} <span class="muted">${minLabel(r.minutes_to_deadline)}</span></td><td>${verdict}</td><td class="mono">${p.combination || "-"}</td><td>${pct(p.probability)}</td><td>${num(p.odds)}</td><td>${num(p.expected_value)}</td></tr>`;
  }).join("") || `<tr><td colspan="8" class="empty">表示対象のレース情報はありません。</td></tr>`;
  document.querySelectorAll("#actionRows tr[data-race]").forEach(row => row.onclick = () => selectRace(row.dataset.race));
}
"""


HTML = build_html()


if __name__ == "__main__":
    raise SystemExit(main())
