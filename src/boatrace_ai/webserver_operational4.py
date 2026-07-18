from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .db import connect, init_db
from .webserver_all import backtest, odds, predictions, send_html, send_json, summary
from .webserver_operational2 import (
    day_overview,
    purchase_guide,
    race_payload,
    venue_cards,
    now_jst,
)
from .webserver_operational3 import HTML as BASE_HTML
from .webserver_realtime import accuracy


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="BOAT RACE AI dense operational dashboard.")
    parser.add_argument("--db", default="data/boatrace.sqlite")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10001)
    parser.add_argument("--backtest", default="data/models/backtest.json")
    args = parser.parse_args(argv)
    init_db(args.db)
    handler = make_handler(Path(args.db), Path(args.backtest) if args.backtest else None)
    print(f"Serving BOAT RACE AI Dense Monitor on http://{args.host}:{args.port}", flush=True)
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


def purchase_guide_with_finished(db_path: Path, query: dict[str, list[str]]) -> dict[str, Any]:
    payload = purchase_guide(db_path, query)
    race_date = query.get("date", [payload["date"]])[0]
    before_minutes = int(query.get("before_minutes", ["10"])[0])
    finished_limit = int(query.get("finished_limit", ["4"])[0])
    now = now_jst()
    with connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT r.race_id, r.race_date, r.jcd, r.venue_name, r.rno, r.title,
                   r.status, r.deadline_at,
                   (SELECT COUNT(*) FROM entries e WHERE e.race_id = r.race_id) AS entries,
                   (SELECT COUNT(*) FROM odds_snapshots os WHERE os.race_id = r.race_id) AS odds_snapshots,
                   (SELECT MAX(captured_at) FROM odds_snapshots os WHERE os.race_id = r.race_id) AS latest_odds_at,
                   (SELECT COUNT(*) FROM race_results rr WHERE rr.race_id = r.race_id AND rr.rank IS NOT NULL) AS result_rows,
                   (SELECT MAX(generated_at) FROM predictions p WHERE p.race_id = r.race_id) AS latest_prediction
            FROM races r
            WHERE r.race_date = ?
              AND (SELECT COUNT(*) FROM entries e WHERE e.race_id = r.race_id) = 6
              AND EXISTS (SELECT 1 FROM predictions p WHERE p.race_id = r.race_id)
              AND (SELECT COUNT(*) FROM race_results rr WHERE rr.race_id = r.race_id AND rr.rank IS NOT NULL) >= 3
            ORDER BY r.deadline_at IS NULL, r.deadline_at DESC, r.jcd, r.rno
            LIMIT ?
            """,
            (race_date, finished_limit),
        ).fetchall()
        finished = []
        for row in rows:
            item = race_payload(conn, row, now=now, before_minutes=before_minutes)
            item.update(result_summary(conn, row["race_id"]))
            top = item.get("top_prediction") or {}
            result_combination = item.get("result_combination")
            item["top_hit"] = bool(result_combination and top.get("combination") == result_combination)
            item["top5_hit"] = bool(
                result_combination
                and any(pred.get("combination") == result_combination for pred in item.get("top5", []))
            )
            finished.append(item)
    payload["finished"] = finished
    return payload


def result_summary(conn, race_id: str) -> dict[str, Any]:
    ranks = conn.execute(
        """
        SELECT rank, lane
        FROM race_results
        WHERE race_id = ? AND rank BETWEEN 1 AND 3
        ORDER BY rank
        """,
        (race_id,),
    ).fetchall()
    combination = "-".join(str(row["lane"]) for row in ranks) if len(ranks) == 3 else None
    payout = None
    popularity = None
    if combination:
        payout_row = conn.execute(
            """
            SELECT payout_yen, popularity
            FROM payouts
            WHERE race_id = ? AND bet_type = '3連単' AND combination = ?
            LIMIT 1
            """,
            (race_id, combination),
        ).fetchone()
        if payout_row:
            payout = payout_row["payout_yen"]
            popularity = payout_row["popularity"]
    return {
        "result_combination": combination,
        "trifecta_payout_yen": payout,
        "trifecta_popularity": popularity,
    }


def build_html() -> str:
    html = BASE_HTML
    html = html.replace(
        "main { display:grid; grid-template-columns:430px 1fr; min-height:calc(100vh - 49px); }",
        "main { display:grid; grid-template-columns:430px 1fr; min-height:calc(100vh - 49px); padding-bottom:22px; }",
    )
    html = html.replace(
        "canvas { width:100%; height:140px; border:1px solid var(--line); background:#fff; } .empty { color:var(--muted); padding:10px 0; }",
        "canvas { width:100%; height:140px; border:1px solid var(--line); background:#fff; } .empty { color:var(--muted); padding:10px 0; } .ops-status { position:fixed; left:0; right:0; bottom:0; z-index:8; padding:4px 10px; border-top:1px solid var(--line); background:#fff; color:var(--muted); font-size:10px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; } .subtable-title { margin-top:10px; color:var(--muted); font-size:12px; } .hit { color:var(--ok); font-weight:700; } .miss { color:var(--bad); font-weight:700; }",
    )
    html = html.replace(
        '<aside><div id="summary" class="stats"></div><div id="venueGrid" class="venue-grid"></div></aside>',
        '<aside><div id="venueGrid" class="venue-grid"></div></aside>',
    )
    html = html.replace(
        '<div class="panel"><h2><span>次の購入候補</span><span class="muted">締切10分前まで / odds取得時刻表示</span></h2><table><thead><tr><th>レース</th><th>締切</th><th>T-10</th><th>odds取得</th><th>候補</th><th>確率</th><th>オッズ</th><th>EV</th></tr></thead><tbody id="guide"></tbody></table></div>',
        '<div class="panel"><h2><span>次の購入候補</span><span class="muted">締切10分前まで / odds取得時刻表示</span></h2><table><thead><tr><th>レース</th><th>締切</th><th>T-10</th><th>odds取得</th><th>候補</th><th>確率</th><th>オッズ</th><th>EV</th></tr></thead><tbody id="guide"></tbody></table><h2 class="subtable-title"><span>直近終了4件</span><span class="muted">結果 / 判定</span></h2><table><thead><tr><th>レース</th><th>締切</th><th>予測</th><th>結果</th><th>判定</th><th>払戻</th><th>人気</th></tr></thead><tbody id="finishedGuide"></tbody></table></div>',
    )
    html = html.replace(
        "</main>\n<script>",
        '</main>\n  <footer id="dataStatus" class="ops-status">取得状態を読み込み中</footer>\n<script>',
    )
    html = html.replace(
        '$("summary").innerHTML = stat("レース",s.races)+stat("出走",s.entries)+stat("結果",s.results)+stat("オッズ",s.odds_snapshots)+stat("予測済み",s.predictions);',
        '$("dataStatus").textContent = `取得状態: レース ${s.races} / 出走 ${s.entries} / 結果 ${s.results} / オッズ ${s.odds_snapshots} / 予測 ${s.predictions} / 更新 ${day.now_jst.replace("T"," ").slice(0,19)}`;',
    )
    html = html.replace(
        "renderVenues(vc.venues); renderGuide(g.candidates); renderTimeline(day.races);",
        "renderVenues(vc.venues); renderGuide(g.candidates, g.finished || []); renderTimeline(day.races);",
    )
    return replace_function(html, "renderGuide", GUIDE_FUNCTION)


def replace_function(html: str, name: str, replacement: str) -> str:
    start = html.index(f"function {name}(")
    end = html.index("\nfunction renderTimeline", start)
    return html[:start] + replacement + html[end:]


GUIDE_FUNCTION = """function renderGuide(rows, finished){
  $("guide").innerHTML = rows.map(r => { const p = r.top_prediction || {}; return `<tr class="pick" data-race="${r.race_id}"><td><b>${r.venue_name} ${r.rno}R</b><br><span class="muted">${r.title || ""}</span></td><td class="mono">${hm(r.deadline_at)}<br><span class="muted">${minLabel(r.minutes_to_deadline)}</span></td><td class="mono">${hm(r.buy_until_at)}<br><span class="muted">${minLabel(r.minutes_to_buy_until)}</span></td><td class="mono">${age(r.latest_odds_at)}</td><td class="mono">${p.combination || "-"}</td><td>${pct(p.probability)}</td><td>${num(p.odds)}</td><td>${num(p.expected_value)}</td></tr>`; }).join("") || `<tr><td colspan="8" class="empty">T-10分前までに判断できる候補はありません。</td></tr>`;
  $("finishedGuide").innerHTML = finished.map(r => { const p = r.top_prediction || {}; const mark = r.top_hit ? "的中" : (r.top5_hit ? "上位5" : "外れ"); const clsName = r.top_hit || r.top5_hit ? "hit" : "miss"; return `<tr data-race="${r.race_id}"><td><b>${r.venue_name} ${r.rno}R</b><br><span class="muted">${r.title || ""}</span></td><td class="mono">${hm(r.deadline_at)}</td><td class="mono">${p.combination || "-"}<br><span class="muted">EV ${num(p.expected_value)}</span></td><td class="mono">${r.result_combination || "-"}</td><td class="${clsName}">${mark}</td><td class="mono">${r.trifecta_payout_yen == null ? "-" : Number(r.trifecta_payout_yen).toLocaleString("ja-JP")}</td><td>${r.trifecta_popularity == null ? "-" : `${r.trifecta_popularity}`}</td></tr>`; }).join("") || `<tr><td colspan="7" class="empty">終了済み候補はまだありません。</td></tr>`;
  document.querySelectorAll("#guide tr[data-race], #finishedGuide tr[data-race]").forEach(row => row.onclick = () => selectRace(row.dataset.race));
}
"""


HTML = build_html()


if __name__ == "__main__":
    raise SystemExit(main())
