from __future__ import annotations

import argparse
import sqlite3
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .constants import VENUES
from .db import connect, init_db
from .official import race_page_url, ymd
from .time_semantics import estimated_deadline_from_start, stored_start_time, time_fields_from_stored_start
from .webserver_all import backtest, odds, send_html, send_json, summary
from .webserver_model_rank import (
    accuracy_model_rank,
    buy_score,
    predictions_model_rank,
    race_payload_model_rank,
)
from .webserver_operational2 import dict_row, iso, minutes_between, now_jst, parse_any_time
from .webserver_operational4 import result_summary
from .webserver_operational20 import HTML as BASE_HTML, PER_RACE_SQL, progress_active


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="BOAT RACE AI model dashboard with corrected deadline/start timing.")
    parser.add_argument("--db", default="data/boatrace.sqlite")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10001)
    parser.add_argument("--backtest", default="data/models/backtest_no_odds_v8.json")
    args = parser.parse_args(argv)
    init_db(args.db)
    handler = make_handler(Path(args.db), Path(args.backtest) if args.backtest else None)
    print(f"Serving BOAT RACE AI Corrected Timing Monitor on http://{args.host}:{args.port}", flush=True)
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
                    send_json(self, venue_cards_start_time(db_path, query))
                elif parsed.path == "/api/day":
                    send_json(self, day_overview_start_time(db_path, query))
                elif parsed.path == "/api/guide":
                    send_json(self, purchase_guide_start_time(db_path, query))
                elif parsed.path == "/api/live-wipe":
                    send_json(self, live_wipe_start_time(db_path, query))
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


def venue_cards_start_time(db_path: Path, query: dict[str, list[str]]) -> dict[str, Any]:
    race_date = query.get("date", [date.today().isoformat()])[0]
    now = now_jst()
    with connect(db_path) as conn:
        grouped = conn.execute(
            f"""
            WITH per_race AS ({PER_RACE_SQL})
            SELECT
              jcd,
              COUNT(*) AS raw_races,
              SUM(is_active) AS races,
              SUM(CASE WHEN is_active = 1 AND entries = 6 THEN 1 ELSE 0 END) AS racelists,
              COALESCE(SUM(CASE WHEN is_active = 1 THEN odds_snapshots ELSE 0 END), 0) AS odds_snapshots,
              SUM(CASE WHEN is_active = 1 AND result_rows >= 3 THEN 1 ELSE 0 END) AS finals,
              MAX(CASE WHEN is_active = 1 THEN latest_prediction ELSE NULL END) AS latest_prediction,
              MAX(CASE WHEN is_active = 1 THEN latest_odds_at ELSE NULL END) AS latest_odds_at
            FROM per_race
            GROUP BY jcd
            """,
            (race_date,),
        ).fetchall()
        upcoming = conn.execute(
            f"""
            WITH per_race AS ({PER_RACE_SQL})
            SELECT jcd, deadline_at, rno, result_rows
            FROM per_race
            WHERE is_active = 1 AND deadline_at IS NOT NULL
            ORDER BY deadline_at, jcd, rno
            """,
            (race_date,),
        ).fetchall()

    by_code = {row["jcd"]: dict_row(row) for row in grouped}
    next_by_code: dict[str, tuple[Any, Any, int]] = {}
    for row in upcoming:
        if int(row["result_rows"] or 0) >= 3:
            continue
        start_at = stored_start_time(row["deadline_at"])
        deadline_at = estimated_deadline_from_start(start_at)
        if deadline_at and deadline_at >= now and row["jcd"] not in next_by_code:
            next_by_code[row["jcd"]] = (deadline_at, start_at, int(row["rno"]))

    cards = []
    for venue in VENUES:
        stats = by_code.get(venue.code, {})
        active_races = int(stats.get("races") or 0)
        racelists = int(stats.get("racelists") or 0)
        odds_count = int(stats.get("odds_snapshots") or 0)
        finals = int(stats.get("finals") or 0)
        if active_races == 0:
            status = "開催なし"
        elif finals >= active_races:
            status = "終了"
        elif odds_count > 0:
            status = "監視中"
        elif racelists > 0:
            status = "出走表"
        else:
            status = "取得中"
        next_deadline, next_start, next_rno = next_by_code.get(venue.code, (None, None, None))
        latest_odds = parse_any_time(stats.get("latest_odds_at"))
        cards.append(
            {
                "code": venue.code,
                "name": venue.name,
                "status": status,
                "races": active_races,
                "raw_races": int(stats.get("raw_races") or 0),
                "racelists": racelists,
                "odds_snapshots": odds_count,
                "finals": finals,
                "latest_prediction": stats.get("latest_prediction"),
                "latest_odds_at": iso(latest_odds),
                "next_rno": next_rno,
                "next_deadline_at": iso(next_deadline),
                "next_race_time_at": iso(next_start),
                "minutes_to_next_deadline": minutes_between(now, next_deadline),
            }
        )
    return {"date": race_date, "now_jst": iso(now), "venues": cards}


def day_overview_start_time(db_path: Path, query: dict[str, list[str]]) -> dict[str, Any]:
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
        races = [_race_payload_start_time(conn, row, now=now) for row in rows]
    return {"date": race_date, "now_jst": iso(now), "races": races}


def purchase_guide_start_time(db_path: Path, query: dict[str, list[str]]) -> dict[str, Any]:
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
            item = _race_payload_start_time(conn, row, now=now, before_minutes=before_minutes)
            buy_until = stored_start_time(item.get("buy_until_at"))
            if not buy_until or now > buy_until:
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
            start_at = stored_start_time(row["deadline_at"])
            if not start_at or start_at > now:
                continue
            item = _race_payload_start_time(conn, row, now=now, before_minutes=before_minutes)
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
        "time_basis": "stored_deadline_at_is_race_start",
    }


def live_wipe_start_time(db_path: Path, query: dict[str, list[str]]) -> dict[str, Any]:
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
            start_at = stored_start_time(row["deadline_at"])
            if not start_at or start_at > now:
                continue
            item = _live_race_payload_start_time(conn, row, now=now)
            return {"date": race_date, "now_jst": iso(now), "active": True, "race": item}
    return {"date": race_date, "now_jst": iso(now), "active": False, "race": None}


def _race_payload_start_time(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    *,
    now,
    before_minutes: int = 5,
) -> dict[str, Any]:
    item = race_payload_model_rank(conn, row, now=now, before_minutes=before_minutes)
    item.update(
        time_fields_from_stored_start(
            row["deadline_at"],
            now=now,
            before_minutes=before_minutes,
            result_rows=int(row["result_rows"] or 0),
        )
    )
    return item


def _live_race_payload_start_time(conn: sqlite3.Connection, row: sqlite3.Row, *, now) -> dict[str, Any]:
    item = _race_payload_start_time(conn, row, now=now)
    jcd = str(row["jcd"]).zfill(2)
    race_date = date.fromisoformat(str(row["race_date"]))
    deadline_at = stored_start_time(item.get("deadline_at"))
    item.update(
        {
            "minutes_since_deadline": int((now - deadline_at).total_seconds() // 60) if deadline_at else None,
            "live_url": f"https://race.boatcast.jp/replay?jo={jcd}",
            "live_embed_url": f"https://race.boatcast.jp/replay?jo={jcd}",
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


LIVE_WIPE_CSS = """
    .live-wipe { position:fixed; right:8px; bottom:18px; width:min(540px,calc(100vw - 16px)); min-width:min(360px,calc(100vw - 16px)); resize:both; overflow:hidden; background:#050808; border:1px solid #263b40; box-shadow:0 10px 24px rgba(0,0,0,.30); z-index:30; }
    .live-wipe.hidden { display:none; }
    .live-wipe-video { position:relative; width:100%; aspect-ratio:16/9; min-height:288px; max-height:calc(100vh - 80px); overflow:hidden; background:#050808; }
    .live-wipe-video iframe { position:absolute; inset:0; width:100%; height:100%; border:0; background:#050808; transform-origin:center center; }
    .live-wipe.zoom .live-wipe-video iframe { width:122%; height:122%; transform:translate(-9%,-9%); }
    .live-wipe-head { position:absolute; left:0; right:0; top:0; display:flex; align-items:center; justify-content:space-between; gap:8px; padding:4px 6px; color:#fff; background:linear-gradient(180deg,rgba(0,0,0,.72),rgba(0,0,0,.08)); z-index:2; pointer-events:none; }
    .live-wipe-title { font-size:11px; font-weight:800; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; text-shadow:0 1px 2px #000; }
    .live-wipe-actions { display:flex; gap:4px; align-items:center; pointer-events:auto; }
    .live-wipe-actions a,.live-wipe-actions button { height:17px; padding:0 5px; border:1px solid rgba(255,255,255,.35); border-radius:3px; background:rgba(16,28,31,.78); color:#fff; font-size:10px; text-decoration:none; line-height:15px; cursor:pointer; }
    .live-wipe-meta { position:absolute; left:5px; right:5px; bottom:5px; display:flex; gap:3px; flex-wrap:wrap; z-index:2; pointer-events:none; }
    .live-wipe-meta span { max-width:100%; padding:2px 4px; border-radius:3px; background:rgba(0,0,0,.68); color:#fff; font-size:9px; line-height:1.12; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; text-shadow:0 1px 2px #000; }
    @media (max-width:1320px) { .live-wipe { width:min(480px,calc(100vw - 16px)); min-width:min(320px,calc(100vw - 16px)); } .live-wipe-video { min-height:252px; } }
    @media (max-width:720px) { .live-wipe { width:calc(100vw - 10px); min-width:0; right:5px; bottom:18px; } .live-wipe-video { min-height:205px; max-height:58vh; } }
"""


LIVE_WIPE_MARKUP = """
  <div id="liveWipe" class="live-wipe zoom hidden">
    <div class="live-wipe-video">
      <iframe id="liveWipeFrame" title="BOATCAST live" allow="autoplay; fullscreen; picture-in-picture"></iframe>
      <div class="live-wipe-head">
        <div id="liveWipeTitle" class="live-wipe-title">LIVE</div>
        <div class="live-wipe-actions">
          <button id="liveWipeZoom" type="button">全体</button>
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
    `出走 ${hm(r.race_time_at)}`,
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
  $("liveWipeZoom").onclick = () => { box.classList.toggle("zoom"); $("liveWipeZoom").textContent = box.classList.contains("zoom") ? "全体" : "拡大"; };
}
function wipeResultText(r){
  if(!r.result_combination) return "結果 -";
  const hit = r.top_hit ? "的中" : (r.top5_hit ? "上位5" : "外れ");
  const payout = r.trifecta_payout_yen == null ? "" : ` ${Number(r.trifecta_payout_yen).toLocaleString("ja-JP")}円`;
  return `結果 ${r.result_combination} ${hit}${payout}`;
}
function timePair(r){
  const d = hm(r && r.deadline_at);
  const s = hm(r && r.race_time_at);
  if(!r || d === "-" || s === "-" || d === s) return d;
  return `${d}/${s.slice(3)}`;
}"""


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
  $("timelineInfo").textContent = `直近終了 ${finished.length}R / 候補 ${picks.length}R / 今後 ${upcoming.length}R / 締切昇順`;
  $("actionRows").innerHTML = action.map((row,idx) => {
    const r = row.source;
    const item = row.item || r;
    const p = item.top_prediction || {};
    const ev = (item.buy_prediction || p || {}).expected_value;
    const shortRace = `${r.venue_name}${r.rno}R`;
    const when = timePair(r);
    if(row.isFinal){
      const ready = Number(item.result_rows || 0) >= 3 && item.result_combination;
      const mark = ready ? (item.top_hit ? "的中" : (item.top5_hit ? "上位5" : "外れ")) : "結果待";
      const result = ready ? `${item.result_combination} ${mark}` : "結果待";
      const payout = ready ? (item.trifecta_payout_yen == null ? "-" : `${Number(item.trifecta_payout_yen).toLocaleString("ja-JP")}円`) : "-";
      const badge = ready ? "確定" : "結果待";
      return `<tr class="${ready ? "final" : "pending"}" data-race="${r.race_id}"><td><span class="badge ${badge}">${badge}</span></td><td title="${r.venue_name} ${r.rno}R ${r.title || ""}"><b>${shortRace}</b></td><td class="mono" title="締切 ${hm(r.deadline_at)} / 出走 ${hm(r.race_time_at)}">${when}</td><td class="mono" title="${result}">${result}</td><td class="mono">${p.combination || "-"}</td><td>${pct(p.probability)}</td><td>${num(p.odds)}</td><td class="mono">${payout}</td></tr>`;
    }
    const status = r.time_status === "T-10超過" ? "T-5超過" : r.time_status;
    const verdict = `<span class="badge ${cls(status)}">${status}</span>${row.isPick ? ` <span class="badge 候補">候補</span>` : ""}`;
    const tone = rowTone(r, row.isPick, idx);
    return `<tr class="${tone}" data-race="${r.race_id}"><td><span class="badge ${row.isPick ? "候補" : ""}">${row.kind}</span></td><td title="${r.venue_name} ${r.rno}R ${r.title || ""}"><b>${shortRace}</b></td><td class="mono" title="締切 ${hm(r.deadline_at)} / 出走 ${hm(r.race_time_at)}">${when} <span class="muted">${minLabel(r.minutes_to_deadline)}</span></td><td>${verdict}</td><td class="mono">${p.combination || "-"}</td><td>${pct(p.probability)}</td><td>${num(p.odds)}</td><td>${num(ev)}</td></tr>`;
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


def build_html() -> str:
    html = BASE_HTML
    html = html.replace("</style>", LIVE_WIPE_CSS + "\n  </style>")
    html = html.replace("\n<script>", LIVE_WIPE_MARKUP + "\n<script>")
    html = html.replace("<th>締切</th>", "<th>締切/出</th>")
    html = html.replace("<th>予測</th>", "<th>モデル予測</th>")
    html = html.replace(
        ".timeline-scroll th:nth-child(3),.timeline-scroll td:nth-child(3){width:62px;}",
        ".timeline-scroll th:nth-child(3),.timeline-scroll td:nth-child(3){width:76px;}",
    )
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
    html = replace_action_table(html)
    html = html.replace("loadAll(); setInterval(loadAll,30000);", LIVE_WIPE_JS + "\nloadAll(); setInterval(loadAll,30000);")
    return html


def replace_action_table(html: str) -> str:
    start = html.index("function renderActionTable(")
    end = html.index("\nasync function selectRace", start)
    return html[:start] + ACTION_TABLE + html[end:]


HTML = build_html()


if __name__ == "__main__":
    raise SystemExit(main())

