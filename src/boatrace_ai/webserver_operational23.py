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
from .webserver_all import backtest, odds, send_html, send_json, summary
from .webserver_model_rank import (
    accuracy_model_rank,
    buy_score,
    predictions_model_rank,
    race_payload_model_rank,
)
from .webserver_operational2 import iso, minutes_between
from .webserver_operational4 import result_summary
from .webserver_operational20 import BASE_HTML, PER_RACE_SQL, progress_active, venue_cards_active
from .webserver_operational2 import now_jst, parse_jst


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="BOAT RACE AI dashboard with model-ranked predictions.")
    parser.add_argument("--db", default="data/boatrace.sqlite")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10001)
    parser.add_argument("--backtest", default="data/models/backtest_no_odds_v7.json")
    args = parser.parse_args(argv)
    init_db(args.db)
    handler = make_handler(Path(args.db), Path(args.backtest) if args.backtest else None)
    print(f"Serving BOAT RACE AI Model Prediction Monitor on http://{args.host}:{args.port}", flush=True)
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
                    send_json(self, day_overview_model_rank(db_path, query))
                elif parsed.path == "/api/guide":
                    send_json(self, purchase_guide_model_rank(db_path, query))
                elif parsed.path == "/api/live-wipe":
                    send_json(self, live_wipe_model_rank(db_path, query))
                elif parsed.path == "/api/progress":
                    send_json(self, progress_active(db_path, query))
                elif parsed.path == "/api/predictions":
                    send_json(self, predictions_model_rank(db_path, query))
                elif parsed.path == "/api/odds":
                    send_json(self, odds(db_path, query))
                elif parsed.path == "/api/backtest":
                    send_json(self, backtest(backtest_path))
                elif parsed.path == "/api/accuracy":
                    send_json(self, accuracy_model_rank(db_path, query))
                else:
                    self.send_error(404)
            except Exception as exc:
                send_json(self, {"error": str(exc)}, status=500)

        def log_message(self, fmt: str, *args: Any) -> None:
            return

    return Handler


def day_overview_model_rank(db_path: Path, query: dict[str, list[str]]) -> dict[str, Any]:
    race_date = query.get("date", [date.today().isoformat()])[0]
    jcd = query.get("jcd", [None])[0]
    now = now_jst()
    params: list[Any] = [race_date]
    jcd_filter = ""
    if jcd:
        jcd_filter = "AND jcd = ?"
        params.append(jcd.zfill(2))
    with connect(db_path) as conn:
        rows = conn.execute(
            f"""
            WITH per_race AS ({PER_RACE_SQL})
            SELECT race_id, race_date, jcd, venue_name, rno, title, status, deadline_at,
                   entries, odds_snapshots, latest_odds_at, result_rows, latest_prediction
            FROM per_race
            WHERE is_active = 1 {jcd_filter}
            ORDER BY deadline_at IS NULL, deadline_at, jcd, rno
            """,
            params,
        ).fetchall()
        races = [_race_payload_t5_model_rank(conn, row, now=now) for row in rows]
    return {"date": race_date, "now_jst": iso(now), "races": races}


def purchase_guide_model_rank(db_path: Path, query: dict[str, list[str]]) -> dict[str, Any]:
    race_date = query.get("date", [date.today().isoformat()])[0]
    before_minutes = int(query.get("before_minutes", ["5"])[0])
    limit = int(query.get("limit", ["16"])[0])
    finished_limit = int(query.get("finished_limit", ["4"])[0])
    now = now_jst()
    with connect(db_path) as conn:
        future_rows = conn.execute(
            f"""
            WITH per_race AS ({PER_RACE_SQL})
            SELECT race_id, race_date, jcd, venue_name, rno, title, status, deadline_at,
                   entries, odds_snapshots, latest_odds_at, result_rows, latest_prediction
            FROM per_race
            WHERE is_active = 1
              AND entries = 6
              AND result_rows < 3
              AND latest_prediction IS NOT NULL
            ORDER BY deadline_at IS NULL, deadline_at, jcd, rno
            """,
            (race_date,),
        ).fetchall()
        candidates = []
        for row in future_rows:
            item = _race_payload_t5_model_rank(conn, row, now=now, before_minutes=before_minutes)
            deadline = parse_jst(row["deadline_at"])
            if not deadline:
                continue
            buy_until = deadline - timedelta(minutes=before_minutes)
            if now > buy_until:
                continue
            if item.get("top_prediction"):
                candidates.append(item)
        candidates.sort(key=lambda item: (item["buy_until_at"], -buy_score(item)))
        if candidates:
            first_cutoff = candidates[0]["buy_until_at"]
            same_cutoff = [item for item in candidates if item["buy_until_at"] == first_cutoff]
            later = [item for item in candidates if item["buy_until_at"] != first_cutoff]
            candidates = sorted(same_cutoff, key=buy_score, reverse=True) + later

        closed_rows = conn.execute(
            f"""
            WITH per_race AS ({PER_RACE_SQL})
            SELECT race_id, race_date, jcd, venue_name, rno, title, status, deadline_at,
                   entries, odds_snapshots, latest_odds_at, result_rows, latest_prediction
            FROM per_race
            WHERE is_active = 1
              AND deadline_at IS NOT NULL
              AND entries = 6
            ORDER BY deadline_at DESC, jcd DESC, rno DESC
            LIMIT 96
            """,
            (race_date,),
        ).fetchall()
        finished = []
        for row in closed_rows:
            deadline = parse_jst(row["deadline_at"])
            if not deadline or deadline + timedelta(minutes=5) > now:
                continue
            item = _race_payload_t5_model_rank(conn, row, now=now, before_minutes=before_minutes)
            if int(item.get("result_rows") or 0) >= 3:
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
    return {
        "date": race_date,
        "now_jst": iso(now),
        "before_minutes": before_minutes,
        "candidates": candidates[:limit],
        "finished": finished,
        "prediction_rank_basis": "model_probability",
    }


def live_wipe_model_rank(db_path: Path, query: dict[str, list[str]]) -> dict[str, Any]:
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
            item = _live_race_payload_model_rank(conn, row, now=now, deadline=deadline, race_time=race_time)
            return {"date": race_date, "now_jst": iso(now), "active": True, "race": item}
    return {"date": race_date, "now_jst": iso(now), "active": False, "race": None}


def _race_payload_t5_model_rank(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    *,
    now,
    before_minutes: int = 5,
) -> dict[str, Any]:
    item = race_payload_model_rank(conn, row, now=now, before_minutes=before_minutes)
    if item.get("time_status") == "T-10超過":
        item["time_status"] = "T-5超過"
    return item


def _live_race_payload_model_rank(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    *,
    now,
    deadline,
    race_time,
) -> dict[str, Any]:
    item = _race_payload_t5_model_rank(conn, row, now=now)
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
    html = html.replace("<th>予測</th>", "<th>モデル予測</th>")
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
    html = html.replace(
        'function guideScore(r){ const p=r.top_prediction||{}; return Number(p.expected_value ?? p.probability ?? 0); }',
        'function guideScore(r){ const p=r.buy_prediction||r.top_prediction||{}; return Number(p.expected_value ?? p.probability ?? 0); }',
    )
    html = html.replace(
        "${num(p.expected_value)}</td></tr>`;",
        "${num(((item.buy_prediction||p)||{}).expected_value)}</td></tr>`;",
    )
    html = html.replace("loadAll(); setInterval(loadAll,30000);", LIVE_WIPE_JS + "\nloadAll(); setInterval(loadAll,30000);")
    return html


LIVE_WIPE_CSS = """
    .live-wipe { position:fixed; right:8px; bottom:20px; width:min(760px,calc(100vw - 16px)); min-width:min(500px,calc(100vw - 16px)); resize:both; overflow:hidden; background:#050808; border:1px solid #263b40; box-shadow:0 12px 30px rgba(0,0,0,.32); z-index:30; }
    .live-wipe.hidden { display:none; }
    .live-wipe-video { position:relative; width:100%; aspect-ratio:16/9; min-height:360px; background:#050808; }
    .live-wipe-video iframe { position:absolute; inset:0; width:100%; height:100%; border:0; background:#050808; }
    .live-wipe-head { position:absolute; left:0; right:0; top:0; display:flex; align-items:center; justify-content:space-between; gap:8px; padding:5px 7px; color:#fff; background:linear-gradient(180deg,rgba(0,0,0,.72),rgba(0,0,0,.18)); z-index:2; pointer-events:none; }
    .live-wipe-title { font-size:12px; font-weight:800; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; text-shadow:0 1px 2px #000; }
    .live-wipe-actions { display:flex; gap:4px; align-items:center; pointer-events:auto; }
    .live-wipe-actions a,.live-wipe-actions button { height:18px; padding:0 5px; border:1px solid rgba(255,255,255,.35); border-radius:3px; background:rgba(16,28,31,.78); color:#fff; font-size:10px; text-decoration:none; line-height:16px; cursor:pointer; }
    .live-wipe-meta { position:absolute; left:6px; right:6px; bottom:6px; display:flex; gap:4px; flex-wrap:wrap; z-index:2; pointer-events:none; }
    .live-wipe-meta span { max-width:100%; padding:2px 5px; border-radius:3px; background:rgba(0,0,0,.7); color:#fff; font-size:10px; line-height:1.15; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; text-shadow:0 1px 2px #000; }
    @media (max-width:1320px) { .live-wipe { width:min(620px,calc(100vw - 16px)); min-width:min(420px,calc(100vw - 16px)); } .live-wipe-video { min-height:300px; } }
    @media (max-width:720px) { .live-wipe { width:calc(100vw - 10px); min-width:0; right:5px; bottom:18px; } .live-wipe-video { min-height:220px; } }
"""


LIVE_WIPE_MARKUP = """
  <div id="liveWipe" class="live-wipe hidden">
    <div class="live-wipe-video">
      <iframe id="liveWipeFrame" title="BOATCAST live" allow="autoplay; fullscreen; picture-in-picture"></iframe>
      <div class="live-wipe-head">
        <div id="liveWipeTitle" class="live-wipe-title">LIVE</div>
        <div class="live-wipe-actions">
          <button id="liveWipeSelect" type="button">選択</button>
          <a id="liveWipeOfficial" href="#" target="_blank" rel="noopener">出走</a>
          <a id="liveWipeOpen" href="#" target="_blank" rel="noopener">公式</a>
        </div>
      </div>
      <div id="liveWipeMeta" class="live-wipe-meta"></div>
    </div>
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
    `モデル ${(r.top_prediction && r.top_prediction.combination) || "-"}`,
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
