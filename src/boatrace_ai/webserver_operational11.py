from __future__ import annotations

import argparse
import sqlite3
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .db import connect, init_db
from .webserver_all import backtest, odds, predictions, send_html, send_json, summary
from .webserver_operational2 import race_payload, venue_cards
from .webserver_operational5 import purchase_guide_with_finished
from .webserver_operational10 import HTML as BASE_HTML
from .webserver_operational2 import now_jst
from .webserver_realtime import accuracy


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="BOAT RACE AI T-5 dense dashboard.")
    parser.add_argument("--db", default="data/boatrace.sqlite")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10001)
    parser.add_argument("--backtest", default="data/models/backtest.json")
    args = parser.parse_args(argv)
    init_db(args.db)
    handler = make_handler(Path(args.db), Path(args.backtest) if args.backtest else None)
    print(f"Serving BOAT RACE AI T-5 Dense Monitor on http://{args.host}:{args.port}", flush=True)
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


def day_overview_t5(db_path: Path, query: dict[str, list[str]]) -> dict[str, Any]:
    race_date = query.get("date", [date.today().isoformat()])[0]
    jcd = query.get("jcd", [None])[0]
    now = now_jst()
    params: list[Any] = [race_date]
    filters = ["r.race_date = ?"]
    if jcd:
        filters.append("r.jcd = ?")
        params.append(jcd.zfill(2))
    with connect(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT r.race_id, r.race_date, r.jcd, r.venue_name, r.rno, r.title,
                   r.status, r.deadline_at,
                   (SELECT COUNT(*) FROM entries e WHERE e.race_id = r.race_id) AS entries,
                   (SELECT COUNT(*) FROM odds_snapshots os WHERE os.race_id = r.race_id) AS odds_snapshots,
                   (SELECT MAX(captured_at) FROM odds_snapshots os WHERE os.race_id = r.race_id) AS latest_odds_at,
                   (SELECT COUNT(*) FROM race_results rr WHERE rr.race_id = r.race_id AND rr.rank IS NOT NULL) AS result_rows,
                   (SELECT MAX(generated_at) FROM predictions p WHERE p.race_id = r.race_id) AS latest_prediction
            FROM races r
            WHERE {" AND ".join(filters)}
            ORDER BY r.deadline_at IS NULL, r.deadline_at, r.jcd, r.rno
            """,
            params,
        ).fetchall()
        races = [_race_payload_t5(conn, row, now=now) for row in rows]
    return {"date": race_date, "now_jst": now.isoformat(timespec="seconds"), "races": races}


def _race_payload_t5(conn: sqlite3.Connection, row: sqlite3.Row, *, now) -> dict[str, Any]:
    item = race_payload(conn, row, now=now, before_minutes=5)
    if item.get("time_status") == "T-10超過":
        item["time_status"] = "T-5超過"
    return item


def build_html() -> str:
    html = BASE_HTML
    html = html.replace("before_minutes=10&limit=16", "before_minutes=5&limit=16")
    html = html.replace(
        '<th>状態</th><th>odds取得</th><th>購入</th><th>確率</th><th>オッズ</th><th>EV</th>',
        '<th>判定</th><th>odds取得</th><th>確率</th><th>オッズ</th><th>EV</th>',
    )
    html = html.replace(
        ".live,.候補 { background:var(--accent); } .done,.確定 { background:var(--ok); } .wait,.T-10超過 { background:var(--warn); } .締切後 { background:var(--bad); }",
        ".live,.候補 { background:var(--accent); } .done,.確定 { background:var(--ok); } .wait,.T-10超過,.T-5超過 { background:var(--warn); } .締切後 { background:var(--bad); } .venue.s-live { background:#effafa; } .venue.s-wait { background:#fff8e8; } .venue.s-done { background:#eef8f2; } .venue.s-none { background:#fafbfb; } .venue.urgent { background:#fff1e6; border-color:#d99a5b; } .venue.due { background:#ffe4db; border-color:#c95c43; } tr.near { background:#fff7db; } tr.soon { background:#fff0e3; } tr.due { background:#ffe2d8; } tr.pick { background:#eaf8f6; } tr.pick.due { background:#ffdcd2; }",
    )
    html = html.replace(
        'function statusTitle(v){ return v==="監視中" ? "出走表とオッズを取得済みで、ライブ更新対象です。" : v==="出走表" ? "出走表は取得済み、オッズは未取得です。" : v==="取得中" ? "当日レース情報を取得中です。" : v==="終了" ? "全レースの結果が入っています。" : "当日データはまだ未取得です。"; }',
        'function statusTitle(v){ return v==="監視中" ? "出走表とオッズを取得済みで、ライブ更新対象です。" : v==="出走表" ? "出走表は取得済み、オッズは未取得です。" : v==="取得中" ? "当日レース情報を取得中です。" : v==="終了" ? "全レースの結果が入っています。" : "当日データはまだ未取得です。"; }\n'
        'function venueTone(v){ const m = v.minutes_to_next_deadline; if(m != null && m <= 5) return "due"; if(m != null && m <= 15) return "urgent"; if(v.status==="監視中") return "s-live"; if(v.status==="終了") return "s-done"; if(v.status==="未取得") return "s-none"; return "s-wait"; }\n'
        'function rowTone(r, isPick, idx){ const m = r.minutes_to_deadline; const parts = []; if(isPick) parts.push("pick"); if(m != null && m <= 5) parts.push("due"); else if(m != null && m <= 15) parts.push("soon"); else if(idx < 4) parts.push("near"); return parts.join(" "); }',
    )
    html = html.replace(
        '<div class="venue ${v.code === state.jcd ? "active" : ""}" data-jcd="${v.code}">',
        '<div class="venue ${v.code === state.jcd ? "active" : ""} ${venueTone(v)}" data-jcd="${v.code}">',
    )
    return replace_action_table(html)


def replace_action_table(html: str) -> str:
    start = html.index("function renderActionTable(")
    end = html.index("\nfunction renderFinished", start)
    return html[:start] + ACTION_TABLE + html[end:]


ACTION_TABLE = """function renderActionTable(rows, candidates, finished){
  const candidateMap = new Map((candidates || []).map(r => [r.race_id, r]));
  const upcoming = futureRows(rows);
  $("timelineInfo").textContent = `${state.jcd ? $("venueFilter").selectedOptions[0]?.textContent || "" : "全場"} / 現在以降 ${upcoming.length}R / T-5候補 ${candidateMap.size}R / 先頭4件表示`;
  $("actionRows").innerHTML = upcoming.map((r,idx) => {
    const candidate = candidateMap.get(r.race_id);
    const item = candidate || r;
    const p = item.top_prediction || {};
    const isPick = Boolean(candidate);
    const status = r.time_status === "T-10超過" ? "T-5超過" : r.time_status;
    const verdict = `<span class="badge ${cls(status)}">${status}</span>${isPick ? ` <span class="badge 候補">候補</span>` : ""}`;
    return `<tr class="${rowTone(r, isPick, idx)}" data-race="${r.race_id}"><td><b>${r.venue_name} ${r.rno}R</b><br><span class="muted">${r.title || ""}</span></td><td class="mono">${hm(r.deadline_at)}<br><span class="muted">${minLabel(r.minutes_to_deadline)}</span></td><td>${verdict}</td><td class="mono">${age(item.latest_odds_at || r.latest_odds_at)}</td><td>${pct(p.probability)}</td><td>${num(p.odds)}</td><td>${num(p.expected_value)}</td></tr>`;
  }).join("") || `<tr><td colspan="7" class="empty">現時点以降のレース情報はありません。</td></tr>`;
  renderFinished(finished);
  document.querySelectorAll("#actionRows tr[data-race], #finishedGuide tr[data-race]").forEach(row => row.onclick = () => selectRace(row.dataset.race));
}
"""


HTML = build_html()


if __name__ == "__main__":
    raise SystemExit(main())
