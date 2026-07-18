from __future__ import annotations

import argparse
import sqlite3
from datetime import date, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .db import connect, init_db
from .official import race_page_url, ymd
from .webserver_all import backtest, odds, predictions, send_html, send_json, summary
from .webserver_operational2 import iso, minutes_between, race_payload
from .webserver_operational4 import result_summary
from .webserver_operational20 import (
    HTML as BASE_HTML,
    PER_RACE_SQL,
    day_overview_t5_active,
    progress_active,
    purchase_guide_with_recent_closed,
    venue_cards_active,
)
from .webserver_operational2 import now_jst, parse_jst
from .webserver_realtime import accuracy


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="BOAT RACE AI dashboard with live wipe.")
    parser.add_argument("--db", default="data/boatrace.sqlite")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10001)
    parser.add_argument("--backtest", default="data/models/backtest_no_odds_v7.json")
    args = parser.parse_args(argv)
    init_db(args.db)
    handler = make_handler(Path(args.db), Path(args.backtest) if args.backtest else None)
    print(f"Serving BOAT RACE AI Live Wipe Monitor on http://{args.host}:{args.port}", flush=True)
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
                    send_json(self, venue_cards_active(db_path, query))
                elif parsed.path == "/api/day":
                    send_json(self, day_overview_t5_active(db_path, query))
                elif parsed.path == "/api/guide":
                    send_json(self, purchase_guide_with_recent_closed(db_path, query))
                elif parsed.path == "/api/live-wipe":
                    send_json(self, live_wipe(db_path, query))
                elif parsed.path == "/api/progress":
                    send_json(self, progress_active(db_path, query))
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


def live_wipe(db_path: Path, query: dict[str, list[str]]) -> dict[str, Any]:
    race_date = query.get("date", [date.today().isoformat()])[0]
    now = now_jst()
    with connect(db_path) as conn:
        rows = conn.execute(
            f"""
            WITH per_race AS ({PER_RACE_SQL})
            SELECT race_id, race_date, jcd, venue_name, rno, title, status, deadline_at,
                   entries, odds_snapshots, latest_odds_at, result_rows, latest_prediction
            FROM per_race
            WHERE is_active = 1
              AND deadline_at IS NOT NULL
              AND entries = 6
            ORDER BY deadline_at DESC, jcd DESC, rno DESC
            """,
            (race_date,),
        ).fetchall()
        for row in rows:
            deadline = parse_jst(row["deadline_at"])
            if not deadline:
                continue
            race_time = deadline + timedelta(minutes=5)
            if race_time > now:
                continue
            item = _live_race_payload(conn, row, now=now, deadline=deadline, race_time=race_time)
            return {"date": race_date, "now_jst": iso(now), "active": True, "race": item}
    return {"date": race_date, "now_jst": iso(now), "active": False, "race": None}


def _live_race_payload(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    *,
    now,
    deadline,
    race_time,
) -> dict[str, Any]:
    item = race_payload(conn, row, now=now, before_minutes=5)
    jcd = str(row["jcd"]).zfill(2)
    race_date = date.fromisoformat(str(row["race_date"]))
    item.update(
        {
            "race_time_at": iso(race_time),
            "minutes_since_deadline": int((now - deadline).total_seconds() // 60),
            "live_url": f"https://race.boatcast.jp/?jo={jcd}",
            "live_embed_url": f"https://race.boatcast.jp/?jo={jcd}",
            "official_url": race_page_url("racelist", race_date, jcd, int(row["rno"])),
            "official_result_url": (
                f"https://www.boatrace.jp/owpc/pc/race/raceresult"
                f"?rno={int(row['rno'])}&jcd={jcd}&hd={ymd(race_date)}"
            ),
        }
    )
    if int(item.get("result_rows") or 0) >= 3:
        item.update(result_summary(conn, row["race_id"]))
        top = item.get("top_prediction") or {}
        result_combination = item.get("result_combination")
        item["top_hit"] = bool(result_combination and top.get("combination") == result_combination)
        item["top5_hit"] = bool(
            result_combination
            and any(pred.get("combination") == result_combination for pred in item.get("top5", []))
        )
    return item


def build_html() -> str:
    html = BASE_HTML
    html = html.replace("</style>", LIVE_WIPE_CSS + "\n  </style>")
    html = html.replace("\n<script>", LIVE_WIPE_MARKUP + "\n<script>")
    html = html.replace(
        "const [s, vc, g, day, acc, bt, prog] = await Promise.all([",
        "const [s, vc, g, day, acc, bt, prog, live] = await Promise.all([",
    )
    html = html.replace(
        "getJson(`/api/progress?date=${encodeURIComponent(d)}`)\n  ]);",
        "getJson(`/api/progress?date=${encodeURIComponent(d)}`),\n"
        "    getJson(`/api/live-wipe?date=${encodeURIComponent(d)}`)\n"
        "  ]);",
    )
    html = html.replace(
        "renderVenues(vc.venues); renderActionTable(day.races, g.candidates || [], g.finished || []);",
        "renderVenues(vc.venues); renderActionTable(day.races, g.candidates || [], g.finished || []); renderLiveWipe(live);",
    )
    html = html.replace("loadAll(); setInterval(loadAll,30000);", LIVE_WIPE_JS + "\nloadAll(); setInterval(loadAll,30000);")
    return html


LIVE_WIPE_CSS = """
    .live-wipe { position:fixed; right:8px; bottom:22px; width:min(360px,calc(100vw - 16px)); background:#11191b; color:#f6fbfc; border:1px solid #2d4247; box-shadow:0 10px 28px rgba(0,0,0,.28); z-index:30; font-size:11px; }
    .live-wipe.hidden { display:none; }
    .live-wipe-head { display:flex; align-items:center; justify-content:space-between; gap:6px; padding:4px 6px; background:#172126; border-bottom:1px solid #2d4247; }
    .live-wipe-title { font-weight:800; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .live-wipe-actions { display:flex; gap:4px; align-items:center; }
    .live-wipe-actions a,.live-wipe-actions button { height:18px; padding:0 5px; border:1px solid #48636a; border-radius:3px; background:#23363a; color:#fff; font-size:10px; text-decoration:none; line-height:16px; cursor:pointer; }
    .live-wipe-video { position:relative; aspect-ratio:16/9; background:#050808; }
    .live-wipe-video iframe { position:absolute; inset:0; width:100%; height:100%; border:0; background:#050808; }
    .live-wipe-meta { display:grid; grid-template-columns:1fr 1fr; gap:1px; background:#263b40; border-top:1px solid #2d4247; }
    .live-wipe-meta span { min-width:0; padding:3px 5px; background:#11191b; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    @media (max-width:720px) { .live-wipe { width:min(320px,calc(100vw - 12px)); right:6px; bottom:20px; } .live-wipe-video { max-height:170px; } }
"""


LIVE_WIPE_MARKUP = """
  <div id="liveWipe" class="live-wipe hidden">
    <div class="live-wipe-head">
      <div id="liveWipeTitle" class="live-wipe-title">LIVE</div>
      <div class="live-wipe-actions">
        <button id="liveWipeSelect" type="button">選択</button>
        <a id="liveWipeOfficial" href="#" target="_blank" rel="noopener">出走</a>
        <a id="liveWipeOpen" href="#" target="_blank" rel="noopener">公式</a>
      </div>
    </div>
    <div class="live-wipe-video"><iframe id="liveWipeFrame" title="BOATCAST live" allow="autoplay; fullscreen; picture-in-picture"></iframe></div>
    <div id="liveWipeMeta" class="live-wipe-meta"></div>
  </div>
"""


LIVE_WIPE_JS = """function renderLiveWipe(payload){
  const box = $("liveWipe");
  if(!box) return;
  const r = payload && payload.active ? payload.race : null;
  if(!r){ box.classList.add("hidden"); return; }
  box.classList.remove("hidden");
  $("liveWipeTitle").textContent = `${r.jcd} ${r.venue_name} ${r.rno}R LIVE`;
  $("liveWipeMeta").innerHTML = [
    `締切 ${hm(r.deadline_at)} +${r.minutes_since_deadline ?? "-"}分`,
    `発走 ${hm(r.race_time_at)}`,
    `状態 ${r.time_status || "-"}`,
    `予測 ${(r.top_prediction && r.top_prediction.combination) || "-"}`,
    `確率 ${pct(r.top_prediction && r.top_prediction.probability)}`,
    wipeResultText(r)
  ].map(v => `<span title="${String(v).replaceAll('"',"&quot;")}">${v}</span>`).join("");
  const src = r.live_embed_url || r.live_url;
  const frame = $("liveWipeFrame");
  if(frame && src && frame.dataset.src !== src){ frame.src = src; frame.dataset.src = src; }
  $("liveWipeOpen").href = r.live_url || "#";
  $("liveWipeOfficial").href = r.official_url || "#";
  $("liveWipeSelect").onclick = () => selectRace(r.race_id);
}
function wipeResultText(r){
  if(!r.result_combination) return "結果 -";
  const hit = r.top_hit ? "的中" : (r.top5_hit ? "上位5" : "外れ");
  const payout = r.trifecta_payout_yen == null ? "" : ` ${Number(r.trifecta_payout_yen).toLocaleString("ja-JP")}円`;
  return `結果 ${r.result_combination} ${hit}${payout}`;
}"""


HTML = build_html()


if __name__ == "__main__":
    raise SystemExit(main())
