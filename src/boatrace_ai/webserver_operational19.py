from __future__ import annotations

import argparse
from datetime import timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .db import connect, init_db
from .webserver_all import backtest, odds, predictions, send_html, send_json, summary
from .webserver_operational2 import parse_jst, purchase_guide, race_payload
from .webserver_operational4 import result_summary
from .webserver_operational11 import day_overview_t5
from .webserver_operational12 import progress, venue_cards
from .webserver_operational18 import HTML as BASE_HTML
from .webserver_operational2 import now_jst
from .webserver_realtime import accuracy


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="BOAT RACE AI recent closed-race timeline dashboard.")
    parser.add_argument("--db", default="data/boatrace.sqlite")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10001)
    parser.add_argument("--backtest", default="data/models/backtest.json")
    args = parser.parse_args(argv)
    init_db(args.db)
    handler = make_handler(Path(args.db), Path(args.backtest) if args.backtest else None)
    print(f"Serving BOAT RACE AI Recent Closed Monitor on http://{args.host}:{args.port}", flush=True)
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
                    send_json(self, purchase_guide_with_recent_closed(db_path, query))
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


def purchase_guide_with_recent_closed(db_path: Path, query: dict[str, list[str]]) -> dict[str, Any]:
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
                   (SELECT MAX(generated_at) FROM predictions p WHERE p.race_id = r.race_id) AS latest_prediction,
                   (SELECT MAX(updated_at) FROM race_results rr WHERE rr.race_id = r.race_id) AS result_updated_at
            FROM races r
            WHERE r.race_date = ?
              AND r.deadline_at IS NOT NULL
              AND (SELECT COUNT(*) FROM entries e WHERE e.race_id = r.race_id) = 6
            ORDER BY r.deadline_at DESC, result_updated_at DESC, r.rno DESC, r.jcd DESC
            LIMIT 96
            """,
            (race_date,),
        ).fetchall()
        finished = []
        for row in rows:
            deadline = parse_jst(row["deadline_at"])
            if not deadline:
                continue
            race_time = deadline + timedelta(minutes=5)
            if race_time > now:
                continue
            item = race_payload(conn, row, now=now, before_minutes=before_minutes)
            result_ready = int(item.get("result_rows") or 0) >= 3
            if result_ready:
                item.update(result_summary(conn, row["race_id"]))
                top = item.get("top_prediction") or {}
                result_combination = item.get("result_combination")
                item["top_hit"] = bool(result_combination and top.get("combination") == result_combination)
                item["top5_hit"] = bool(
                    result_combination
                    and any(pred.get("combination") == result_combination for pred in item.get("top5", []))
                )
            else:
                item["time_status"] = "結果待"
                item["result_combination"] = None
                item["trifecta_payout_yen"] = None
                item["trifecta_popularity"] = None
                item["top_hit"] = False
                item["top5_hit"] = False
            finished.append(item)
            if len(finished) >= finished_limit:
                break
    payload["finished"] = finished
    return payload


def build_html() -> str:
    html = BASE_HTML
    html = html.replace(
        ".live,.候補 { background:var(--accent); }",
        ".live,.候補 { background:var(--accent); } .結果待 { background:var(--warn); }",
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
    ...finished.map(r => ({ kind:(Number(r.result_rows || 0) >= 3 ? "確定" : "結果待"), item:r, source:r, isPick:false, isFinal:true })),
    ...picks.map(r => ({ kind:"候補", item:candidateMap.get(r.race_id), source:r, isPick:true, isFinal:false })),
    ...others.map(r => ({ kind:"予定", item:r, source:r, isPick:false, isFinal:false })),
  ].sort((a,b) => {
    const av = actionTime(a);
    const bv = actionTime(b);
    if(av !== bv) return av - bv;
    return String(a.source.jcd).localeCompare(String(b.source.jcd)) || Number(a.source.rno || 0) - Number(b.source.rno || 0);
  });
  $("timelineInfo").textContent = `直近終了 ${finished.length}R / 候補 ${picks.length}R / 今後 ${upcoming.length}R / 時刻昇順`;
  $("actionRows").innerHTML = action.map((row,idx) => {
    const r = row.source;
    const item = row.item || r;
    const p = item.top_prediction || {};
    const shortRace = `${r.venue_name}${r.rno}R`;
    if(row.isFinal){
      const ready = Number(item.result_rows || 0) >= 3 && item.result_combination;
      const mark = ready ? (item.top_hit ? "的中" : (item.top5_hit ? "上位5" : "外れ")) : "結果待";
      const result = ready ? `${item.result_combination} ${mark}` : "結果待";
      const payout = ready ? (item.trifecta_payout_yen == null ? "-" : `${Number(item.trifecta_payout_yen).toLocaleString("ja-JP")}円`) : "-";
      const badge = ready ? "確定" : "結果待";
      return `<tr class="${ready ? "final" : "pending"}" data-race="${r.race_id}"><td><span class="badge ${badge}">${badge}</span></td><td title="${r.venue_name} ${r.rno}R ${r.title || ""}"><b>${shortRace}</b></td><td class="mono">${hm(r.race_time_at || r.deadline_at)}</td><td class="mono" title="${result}">${result}</td><td class="mono">${p.combination || "-"}</td><td>${pct(p.probability)}</td><td>${num(p.odds)}</td><td class="mono">${payout}</td></tr>`;
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
