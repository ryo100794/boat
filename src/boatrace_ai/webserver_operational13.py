from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .db import init_db
from .webserver_all import backtest, odds, predictions, send_html, send_json, summary
from .webserver_operational11 import day_overview_t5
from .webserver_operational12 import HTML as BASE_HTML, progress, purchase_guide_with_finished, venue_cards
from .webserver_realtime import accuracy


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="BOAT RACE AI unified race table dashboard.")
    parser.add_argument("--db", default="data/boatrace.sqlite")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10001)
    parser.add_argument("--backtest", default="data/models/backtest.json")
    args = parser.parse_args(argv)
    init_db(args.db)
    handler = make_handler(Path(args.db), Path(args.backtest) if args.backtest else None)
    print(f"Serving BOAT RACE AI Unified Table Monitor on http://{args.host}:{args.port}", flush=True)
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
        "<span>進行タイムライン・購入候補</span>",
        "<span>確定4件・購入候補・今後レース</span>",
    )
    html = html.replace(
        "<th>場/R</th><th>締切</th><th>判定</th><th>odds取得</th><th>確率</th><th>オッズ</th><th>EV</th>",
        "<th>区分</th><th>場/R</th><th>締切</th><th>判定/結果</th><th>odds取得</th><th>予測</th><th>確率</th><th>オッズ</th><th>EV/払戻</th>",
    )
    html = html.replace(
        "tr.pick { background:#f2fbfa; } tr.late { color:var(--muted); } tr.nowline { background:#fff9e9; }",
        "tr.pick { background:#f2fbfa; } tr.final { background:#eef5ff; } tr.late { color:var(--muted); } tr.nowline { background:#fff9e9; }",
    )
    html = html.replace(
        ".grid2 { display:grid; grid-template-columns:minmax(520px,1.05fr) minmax(430px,.95fr); gap:8px; margin-top:7px; }",
        ".grid2 { display:grid; grid-template-columns:1fr; gap:8px; margin-top:7px; }",
    )
    html = html.replace(
        '<div class="panel"><h2><span>直近終了4件</span><span class="muted">結果 / 判定</span></h2><table><thead><tr><th>レース</th><th>締切</th><th>予測</th><th>結果</th><th>判定</th><th>払戻</th><th>人気</th></tr></thead><tbody id="finishedGuide"></tbody></table></div>\n        ',
        "",
    )
    return replace_table_functions(html)


def replace_table_functions(html: str) -> str:
    start = html.index("function renderActionTable(")
    end = html.index("\nasync function selectRace", start)
    return html[:start] + ACTION_TABLE + html[end:]


ACTION_TABLE = """function renderActionTable(rows, candidates, finished){
  const candidateMap = new Map((candidates || []).map(r => [r.race_id, r]));
  const upcoming = futureRows(rows);
  const picks = upcoming.filter(r => candidateMap.has(r.race_id));
  const others = upcoming.filter(r => !candidateMap.has(r.race_id));
  const action = [
    ...finished.map(r => ({ kind:"確定", item:r, source:r, isPick:false, isFinal:true })),
    ...picks.map(r => ({ kind:"候補", item:candidateMap.get(r.race_id), source:r, isPick:true, isFinal:false })),
    ...others.map(r => ({ kind:"予定", item:r, source:r, isPick:false, isFinal:false })),
  ];
  $("timelineInfo").textContent = `確定 ${finished.length}R / T-5候補 ${picks.length}R / 今後 ${upcoming.length}R`;
  $("actionRows").innerHTML = action.map((row,idx) => {
    const r = row.source;
    const item = row.item || r;
    const p = item.top_prediction || {};
    if(row.isFinal){
      const mark = item.top_hit ? "的中" : (item.top5_hit ? "上位5" : "外れ");
      const result = `${item.result_combination || "-"} ${mark}`;
      const payout = item.trifecta_payout_yen == null ? num(p.expected_value) : `${Number(item.trifecta_payout_yen).toLocaleString("ja-JP")}円`;
      return `<tr class="final" data-race="${r.race_id}"><td><span class="badge 確定">確定</span></td><td><b>${r.venue_name} ${r.rno}R</b><br><span class="muted">${r.title || ""}</span></td><td class="mono">${hm(r.deadline_at)}</td><td class="mono">${result}</td><td class="mono">${age(item.latest_odds_at)}</td><td class="mono">${p.combination || "-"}</td><td>${pct(p.probability)}</td><td>${num(p.odds)}</td><td class="mono">${payout}</td></tr>`;
    }
    const status = r.time_status === "T-10超過" ? "T-5超過" : r.time_status;
    const verdict = `<span class="badge ${cls(status)}">${status}</span>${row.isPick ? ` <span class="badge 候補">候補</span>` : ""}`;
    const tone = rowTone(r, row.isPick, idx);
    return `<tr class="${tone}" data-race="${r.race_id}"><td><span class="badge ${row.isPick ? "候補" : ""}">${row.kind}</span></td><td><b>${r.venue_name} ${r.rno}R</b><br><span class="muted">${r.title || ""}</span></td><td class="mono">${hm(r.deadline_at)}<br><span class="muted">${minLabel(r.minutes_to_deadline)}</span></td><td>${verdict}</td><td class="mono">${age(item.latest_odds_at || r.latest_odds_at)}</td><td class="mono">${p.combination || "-"}</td><td>${pct(p.probability)}</td><td>${num(p.odds)}</td><td>${num(p.expected_value)}</td></tr>`;
  }).join("") || `<tr><td colspan="9" class="empty">表示対象のレース情報はありません。</td></tr>`;
  document.querySelectorAll("#actionRows tr[data-race]").forEach(row => row.onclick = () => selectRace(row.dataset.race));
}
"""


HTML = build_html()


if __name__ == "__main__":
    raise SystemExit(main())
