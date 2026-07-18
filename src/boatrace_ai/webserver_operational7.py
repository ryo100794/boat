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
from .webserver_operational6 import HTML as BASE_HTML
from .webserver_realtime import accuracy


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="BOAT RACE AI merged timeline dashboard.")
    parser.add_argument("--db", default="data/boatrace.sqlite")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10001)
    parser.add_argument("--backtest", default="data/models/backtest.json")
    args = parser.parse_args(argv)
    init_db(args.db)
    handler = make_handler(Path(args.db), Path(args.backtest) if args.backtest else None)
    print(f"Serving BOAT RACE AI Merged Monitor on http://{args.host}:{args.port}", flush=True)
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
        '<h2><span>現時点以降タイムライン</span><span id="timelineInfo" class="muted"></span></h2>\n'
        '        <div class="timeline-scroll"><table><thead><tr><th>場/R</th><th>締切</th><th>T-10</th><th>状態</th><th>odds取得</th><th>上位</th><th>EV</th></tr></thead><tbody id="timelineRows"></tbody></table></div>',
        '<h2><span>進行タイムライン・購入候補</span><span id="timelineInfo" class="muted"></span></h2>\n'
        '        <div class="timeline-scroll"><table><thead><tr><th>場/R</th><th>締切</th><th>状態</th><th>odds取得</th><th>購入</th><th>確率</th><th>オッズ</th><th>EV</th></tr></thead><tbody id="actionRows"></tbody></table></div>',
    )
    html = html.replace(
        '<div class="panel"><h2><span>次の購入候補</span><span class="muted">締切10分前まで / odds取得時刻表示</span></h2><table><thead><tr><th>レース</th><th>締切</th><th>T-10</th><th>odds取得</th><th>候補</th><th>確率</th><th>オッズ</th><th>EV</th></tr></thead><tbody id="guide"></tbody></table><h2 class="subtable-title"><span>直近終了4件</span><span class="muted">結果 / 判定</span></h2><table><thead><tr><th>レース</th><th>締切</th><th>予測</th><th>結果</th><th>判定</th><th>払戻</th><th>人気</th></tr></thead><tbody id="finishedGuide"></tbody></table></div>',
        '<div class="panel"><h2><span>直近終了4件</span><span class="muted">結果 / 判定</span></h2><table><thead><tr><th>レース</th><th>締切</th><th>予測</th><th>結果</th><th>判定</th><th>払戻</th><th>人気</th></tr></thead><tbody id="finishedGuide"></tbody></table></div>',
    )
    html = html.replace(
        "renderVenues(vc.venues); renderGuide(g.candidates, g.finished || []); renderTimeline(day.races);",
        "renderVenues(vc.venues); renderActionTable(day.races, g.candidates || [], g.finished || []);",
    )
    return replace_action_functions(html)


def replace_action_functions(html: str) -> str:
    start = html.index("function renderGuide(")
    end = html.index("\nasync function selectRace", start)
    return html[:start] + ACTION_FUNCTIONS + html[end:]


ACTION_FUNCTIONS = """function renderActionTable(rows, candidates, finished){
  const candidateMap = new Map((candidates || []).map(r => [r.race_id, r]));
  const upcoming = futureRows(rows);
  $("timelineInfo").textContent = `${state.jcd ? $("venueFilter").selectedOptions[0]?.textContent || "" : "全場"} / 現在以降 ${upcoming.length}R / 候補 ${candidateMap.size}R / 先頭4件表示`;
  $("actionRows").innerHTML = upcoming.map((r,idx) => {
    const candidate = candidateMap.get(r.race_id);
    const item = candidate || r;
    const p = item.top_prediction || {};
    const isPick = Boolean(candidate);
    const rowClass = isPick ? "pick" : (idx < 4 ? "nowline" : "");
    return `<tr class="${rowClass}" data-race="${r.race_id}"><td><b>${r.venue_name} ${r.rno}R</b><br><span class="muted">${r.title || ""}</span></td><td class="mono">${hm(r.deadline_at)}<br><span class="muted">${minLabel(r.minutes_to_deadline)}</span></td><td><span class="badge ${cls(r.time_status)}">${r.time_status}</span></td><td class="mono">${age(item.latest_odds_at || r.latest_odds_at)}</td><td>${isPick ? `<span class="badge 候補">候補</span>` : "-"}</td><td>${pct(p.probability)}</td><td>${num(p.odds)}</td><td>${num(p.expected_value)}</td></tr>`;
  }).join("") || `<tr><td colspan="8" class="empty">現時点以降のレース情報はありません。</td></tr>`;
  renderFinished(finished);
  document.querySelectorAll("#actionRows tr[data-race], #finishedGuide tr[data-race]").forEach(row => row.onclick = () => selectRace(row.dataset.race));
}
function renderFinished(finished){
  $("finishedGuide").innerHTML = finished.map(r => {
    const p = r.top_prediction || {};
    const mark = r.top_hit ? "的中" : (r.top5_hit ? "上位5" : "外れ");
    const clsName = r.top_hit || r.top5_hit ? "hit" : "miss";
    return `<tr data-race="${r.race_id}"><td><b>${r.venue_name} ${r.rno}R</b><br><span class="muted">${r.title || ""}</span></td><td class="mono">${hm(r.deadline_at)}</td><td class="mono">${p.combination || "-"}<br><span class="muted">EV ${num(p.expected_value)}</span></td><td class="mono">${r.result_combination || "-"}</td><td class="${clsName}">${mark}</td><td class="mono">${r.trifecta_payout_yen == null ? "-" : Number(r.trifecta_payout_yen).toLocaleString("ja-JP")}</td><td>${r.trifecta_popularity == null ? "-" : `${r.trifecta_popularity}`}</td></tr>`;
  }).join("") || `<tr><td colspan="7" class="empty">終了済み候補はまだありません。</td></tr>`;
}
"""


HTML = build_html()


if __name__ == "__main__":
    raise SystemExit(main())
