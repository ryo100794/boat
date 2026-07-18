from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import date, datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .constants import VENUES
from .db import connect, init_db
from .webserver_all import backtest, odds, predictions, send_html, send_json, summary
from .webserver_realtime import accuracy


JST = timezone(timedelta(hours=9))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="BOAT RACE AI operational dashboard.")
    parser.add_argument("--db", default="data/boatrace.sqlite")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10001)
    parser.add_argument("--backtest", default="data/models/backtest.json")
    args = parser.parse_args(argv)
    init_db(args.db)
    handler = make_handler(Path(args.db), Path(args.backtest) if args.backtest else None)
    print(f"Serving BOAT RACE AI Monitor on http://{args.host}:{args.port}", flush=True)
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
                    send_json(self, purchase_guide(db_path, query))
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


def venue_cards(db_path: Path, query: dict[str, list[str]]) -> dict[str, Any]:
    race_date = query.get("date", [date.today().isoformat()])[0]
    now = now_jst()
    with connect(db_path) as conn:
        grouped = conn.execute(
            """
            SELECT
              r.jcd,
              COUNT(DISTINCT r.race_id) AS races,
              SUM(CASE WHEN entry_counts.entries = 6 THEN 1 ELSE 0 END) AS racelists,
              COALESCE(SUM(odds_counts.snapshots), 0) AS odds_snapshots,
              SUM(CASE WHEN result_counts.results >= 3 THEN 1 ELSE 0 END) AS finals,
              MAX(pred_counts.generated_at) AS latest_prediction,
              MAX(odds_counts.latest_odds_at) AS latest_odds_at
            FROM races r
            LEFT JOIN (
              SELECT race_id, COUNT(*) AS entries FROM entries GROUP BY race_id
            ) entry_counts ON entry_counts.race_id = r.race_id
            LEFT JOIN (
              SELECT race_id, COUNT(*) AS snapshots, MAX(captured_at) AS latest_odds_at
              FROM odds_snapshots GROUP BY race_id
            ) odds_counts ON odds_counts.race_id = r.race_id
            LEFT JOIN (
              SELECT race_id, COUNT(*) AS results FROM race_results GROUP BY race_id
            ) result_counts ON result_counts.race_id = r.race_id
            LEFT JOIN (
              SELECT race_id, MAX(generated_at) AS generated_at FROM predictions GROUP BY race_id
            ) pred_counts ON pred_counts.race_id = r.race_id
            WHERE r.race_date = ?
            GROUP BY r.jcd
            """,
            (race_date,),
        ).fetchall()
        upcoming = conn.execute(
            """
            SELECT r.jcd, r.deadline_at, r.rno,
                   (SELECT COUNT(*) FROM race_results rr WHERE rr.race_id = r.race_id AND rr.rank IS NOT NULL) AS result_rows
            FROM races r
            WHERE r.race_date = ? AND r.deadline_at IS NOT NULL
            ORDER BY r.deadline_at, r.jcd, r.rno
            """,
            (race_date,),
        ).fetchall()
    by_code = {row["jcd"]: dict_row(row) for row in grouped}
    next_by_code: dict[str, tuple[datetime, int]] = {}
    for row in upcoming:
        if int(row["result_rows"] or 0) >= 3:
            continue
        deadline = parse_jst(row["deadline_at"])
        if deadline and deadline >= now and row["jcd"] not in next_by_code:
            next_by_code[row["jcd"]] = (deadline, int(row["rno"]))

    cards = []
    for venue in VENUES:
        stats = by_code.get(venue.code, {})
        races = int(stats.get("races") or 0)
        racelists = int(stats.get("racelists") or 0)
        odds_count = int(stats.get("odds_snapshots") or 0)
        finals = int(stats.get("finals") or 0)
        if races == 0:
            status = "未取得"
        elif finals >= races:
            status = "終了"
        elif odds_count > 0:
            status = "監視中"
        elif racelists > 0:
            status = "出走表"
        else:
            status = "取得中"
        next_deadline, next_rno = next_by_code.get(venue.code, (None, None))
        latest_odds = parse_any_time(stats.get("latest_odds_at"))
        cards.append(
            {
                "code": venue.code,
                "name": venue.name,
                "status": status,
                "races": races,
                "racelists": racelists,
                "odds_snapshots": odds_count,
                "finals": finals,
                "latest_prediction": stats.get("latest_prediction"),
                "latest_odds_at": iso(latest_odds),
                "next_rno": next_rno,
                "next_deadline_at": iso(next_deadline),
                "minutes_to_next_deadline": minutes_between(now, next_deadline),
            }
        )
    return {"date": race_date, "now_jst": iso(now), "venues": cards}


def day_overview(db_path: Path, query: dict[str, list[str]]) -> dict[str, Any]:
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
        races = [race_payload(conn, row, now=now) for row in rows]
    return {"date": race_date, "now_jst": iso(now), "races": races}


def purchase_guide(db_path: Path, query: dict[str, list[str]]) -> dict[str, Any]:
    race_date = query.get("date", [date.today().isoformat()])[0]
    before_minutes = int(query.get("before_minutes", ["10"])[0])
    limit = int(query.get("limit", ["16"])[0])
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
              AND (SELECT COUNT(*) FROM race_results rr WHERE rr.race_id = r.race_id AND rr.rank IS NOT NULL) < 3
            ORDER BY r.deadline_at IS NULL, r.deadline_at, r.jcd, r.rno
            """,
            (race_date,),
        ).fetchall()
        candidates = []
        for row in rows:
            item = race_payload(conn, row, now=now, before_minutes=before_minutes)
            deadline = parse_jst(row["deadline_at"])
            if not deadline:
                continue
            buy_until = deadline - timedelta(minutes=before_minutes)
            if now > buy_until:
                continue
            if item.get("top_prediction"):
                candidates.append(item)
    candidates.sort(key=lambda item: (item["buy_until_at"], -guide_score(item)))
    if candidates:
        first_cutoff = candidates[0]["buy_until_at"]
        same_cutoff = [item for item in candidates if item["buy_until_at"] == first_cutoff]
        later = [item for item in candidates if item["buy_until_at"] != first_cutoff]
        candidates = sorted(same_cutoff, key=guide_score, reverse=True) + later
    return {
        "date": race_date,
        "now_jst": iso(now),
        "before_minutes": before_minutes,
        "candidates": candidates[:limit],
    }


def race_payload(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    *,
    now: datetime,
    before_minutes: int = 10,
) -> dict[str, Any]:
    deadline = parse_jst(row["deadline_at"])
    buy_until = deadline - timedelta(minutes=before_minutes) if deadline else None
    race_time = deadline + timedelta(minutes=5) if deadline else None
    latest_odds = parse_any_time(row["latest_odds_at"])
    if int(row["result_rows"] or 0) >= 3:
        time_status = "確定"
    elif not deadline:
        time_status = "時刻未取得"
    elif now > deadline:
        time_status = "締切後"
    elif buy_until and now > buy_until:
        time_status = "T-10超過"
    else:
        time_status = "候補"
    top_prediction, top5 = latest_prediction_rows(conn, row["race_id"])
    return {
        "race_id": row["race_id"],
        "race_date": row["race_date"],
        "jcd": row["jcd"],
        "venue_name": row["venue_name"],
        "rno": row["rno"],
        "title": row["title"],
        "status": row["status"],
        "deadline_at": iso(deadline),
        "race_time_at": iso(race_time),
        "buy_until_at": iso(buy_until),
        "minutes_to_deadline": minutes_between(now, deadline),
        "minutes_to_buy_until": minutes_between(now, buy_until),
        "time_status": time_status,
        "entries": row["entries"],
        "odds_snapshots": row["odds_snapshots"],
        "latest_odds_at": iso(latest_odds),
        "result_rows": row["result_rows"],
        "latest_prediction": row["latest_prediction"],
        "top_prediction": top_prediction,
        "top5": top5,
    }


def latest_prediction_rows(
    conn: sqlite3.Connection,
    race_id: str,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    latest = conn.execute(
        "SELECT generated_at FROM predictions WHERE race_id = ? ORDER BY generated_at DESC LIMIT 1",
        (race_id,),
    ).fetchone()
    if not latest:
        return None, []
    rows = conn.execute(
        """
        SELECT combination, probability, odds, expected_value, generated_at
        FROM predictions
        WHERE race_id = ? AND generated_at = ?
        ORDER BY COALESCE(expected_value, probability) DESC, probability DESC
        LIMIT 5
        """,
        (race_id, latest["generated_at"]),
    ).fetchall()
    mapped = [dict_row(row) for row in rows]
    return (mapped[0] if mapped else None), mapped


def dict_row(row: sqlite3.Row) -> dict[str, Any]:
    return {key: row[key] for key in row.keys()}


def guide_score(item: dict[str, Any]) -> float:
    top = item.get("top_prediction") or {}
    if top.get("expected_value") is not None:
        return float(top["expected_value"])
    return float(top.get("probability") or 0.0)


def now_jst() -> datetime:
    return datetime.now(timezone.utc).astimezone(JST)


def parse_jst(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=JST)
    return parsed.astimezone(JST)


def parse_any_time(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc).astimezone(JST)
    return parsed.astimezone(JST)


def minutes_between(start: datetime, end: datetime | None) -> int | None:
    if not end:
        return None
    return int((end - start).total_seconds() // 60)


def iso(value: datetime | None) -> str | None:
    return value.isoformat(timespec="seconds") if value else None


HTML = """<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BOAT RACE AI Ops</title>
  <style>
    :root { color-scheme: light; --ink:#172126; --muted:#637279; --line:#d8e0e3; --band:#f3f6f7; --accent:#006d77; --accent2:#8f2d56; --ok:#247a4b; --warn:#a76300; --bad:#a33a3a; font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    * { box-sizing: border-box; } body { margin: 0; color: var(--ink); background: #fff; font-size: 13px; }
    header { display:flex; align-items:center; justify-content:space-between; gap:14px; padding:10px 14px; border-bottom:1px solid var(--line); background:#fff; position:sticky; top:0; z-index:5; }
    h1 { margin:0; font-size:17px; letter-spacing:0; } main { display:grid; grid-template-columns: 330px 1fr; min-height: calc(100vh - 51px); }
    aside { background:var(--band); border-right:1px solid var(--line); padding:10px; overflow:auto; } section { padding:12px 14px; min-width:0; }
    input, select, button { height:30px; border:1px solid var(--line); border-radius:6px; padding:0 8px; background:#fff; color:var(--ink); font:inherit; }
    button { background:var(--accent); border-color:var(--accent); color:#fff; cursor:pointer; } .toolbar { display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
    .stats { display:grid; grid-template-columns: repeat(5, minmax(76px, 1fr)); gap:1px; background:var(--line); border:1px solid var(--line); }
    .stat { background:#fff; padding:8px; min-width:0; } .stat b { display:block; font-size:18px; line-height:1.1; } .stat span { color:var(--muted); font-size:11px; }
    .venue-grid { display:grid; grid-template-columns: repeat(2, minmax(140px, 1fr)); gap:5px; margin-top:8px; }
    .venue { background:#fff; border:1px solid var(--line); border-radius:6px; padding:7px; cursor:pointer; min-height:88px; }
    .venue.active { border-color:var(--accent); box-shadow: inset 3px 0 0 var(--accent); } .venue b { display:flex; justify-content:space-between; gap:6px; font-size:13px; }
    .venue small { display:block; color:var(--muted); line-height:1.35; margin-top:4px; }
    .badge { display:inline-block; border-radius:999px; padding:1px 6px; color:#fff; background:var(--muted); font-size:11px; font-weight:700; white-space:nowrap; }
    .live,.候補 { background:var(--accent); } .done,.確定 { background:var(--ok); } .wait,.T-10超過 { background:var(--warn); } .締切後 { background:var(--bad); }
    .grid2 { display:grid; grid-template-columns:minmax(520px,1.1fr) minmax(420px,.9fr); gap:12px; margin-top:12px; }
    .panel { border-top:2px solid var(--accent); padding-top:8px; min-width:0; } .panel h2 { margin:0 0 7px; font-size:14px; letter-spacing:0; display:flex; justify-content:space-between; gap:8px; }
    table { width:100%; border-collapse:collapse; table-layout:fixed; } th,td { border-bottom:1px solid var(--line); padding:5px; text-align:right; vertical-align:top; overflow-wrap:anywhere; }
    th { color:var(--muted); font-weight:700; background:#fafbfb; position:sticky; top:51px; z-index:2; } th:first-child,td:first-child { text-align:left; }
    tr.pick { background:#f2fbfa; } tr.late { color:var(--muted); } .mono { font-variant-numeric:tabular-nums; font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; } .muted { color:var(--muted); }
    .entries { display:grid; grid-template-columns:repeat(6,minmax(72px,1fr)); gap:1px; background:var(--line); border:1px solid var(--line); margin:7px 0 10px; }
    .entry { background:#fff; min-height:68px; padding:6px; } .lane { display:inline-grid; place-items:center; width:22px; height:22px; border:1px solid var(--line); font-weight:700; margin-bottom:3px; }
    canvas { width:100%; height:140px; border:1px solid var(--line); background:#fff; } .empty { color:var(--muted); padding:10px 0; }
    @media (max-width:1050px) { main,.grid2 { grid-template-columns:1fr; } aside { border-right:0; border-bottom:1px solid var(--line); max-height:45vh; } .stats { grid-template-columns:repeat(2,minmax(120px,1fr)); } .entries { grid-template-columns:repeat(3,minmax(72px,1fr)); } }
  </style>
</head>
<body>
  <header><h1>BOAT RACE AI Ops</h1><div class="toolbar"><input id="raceDate" type="date"><select id="venueFilter"><option value="">全場</option></select><button id="reload">更新</button><span id="clock" class="muted mono"></span></div></header>
  <main>
    <aside><div id="summary" class="stats"></div><div id="venueGrid" class="venue-grid"></div></aside>
    <section>
      <div class="panel"><h2><span>次の購入候補</span><span class="muted">T-10分で候補から除外 / odds取得時刻表示</span></h2><table><thead><tr><th>レース</th><th>締切</th><th>T-10</th><th>odds取得</th><th>候補</th><th>確率</th><th>オッズ</th><th>EV</th></tr></thead><tbody id="guide"></tbody></table></div>
      <div class="grid2">
        <div class="panel"><h2><span>当日タイムライン</span><span id="dayCount" class="muted"></span></h2><table><thead><tr><th>場/R</th><th>締切</th><th>発走目安</th><th>状態</th><th>odds取得</th><th>上位</th><th>EV</th></tr></thead><tbody id="dayRows"></tbody></table></div>
        <div class="panel"><h2><span id="raceTitle">レース詳細</span><span id="accuracy" class="muted"></span></h2><div id="entries" class="entries"></div><table><thead><tr><th>3連単</th><th>確率</th><th>オッズ</th><th>EV</th></tr></thead><tbody id="predictions"></tbody></table><h2 style="margin-top:12px;"><span>オッズ推移</span><select id="combo"></select></h2><canvas id="oddsChart" width="720" height="200"></canvas><div id="backtest" class="empty"></div></div>
      </div>
    </section>
  </main>
<script>
const state = { raceId:null, jcd:"", combo:"1-2-3" };
const $ = id => document.getElementById(id);
const today = new Date().toISOString().slice(0,10);
$("raceDate").value = today;
$("reload").onclick = loadAll;
$("venueFilter").onchange = () => { state.jcd = $("venueFilter").value; state.raceId = null; loadAll(); };
$("combo").onchange = () => { state.combo = $("combo").value; loadOdds(); };
async function getJson(url){ const res = await fetch(url,{cache:"no-store"}); if(!res.ok) throw new Error(await res.text()); return await res.json(); }
function stat(label,value){ return `<div class="stat"><b>${value ?? "-"}</b><span>${label}</span></div>`; }
function pct(v){ return v == null ? "-" : `${(Number(v)*100).toFixed(2)}%`; }
function num(v){ return v == null ? "-" : Number(v).toFixed(3); }
function hm(v){ if(!v) return "-"; return new Date(v).toLocaleTimeString("ja-JP",{hour:"2-digit",minute:"2-digit",timeZone:"Asia/Tokyo"}); }
function age(v){ if(!v) return "-"; const m = Math.floor((Date.now()-new Date(v).getTime())/60000); return `${hm(v)} (${m}分前)`; }
function minLabel(v){ return v == null ? "-" : `${v}分`; }
function cls(v){ return String(v || "").replaceAll(" ",""); }
async function loadAll(){
  const d = $("raceDate").value || today;
  const [s, vc, g, day, acc, bt] = await Promise.all([
    getJson("/api/summary"),
    getJson(`/api/venues?date=${encodeURIComponent(d)}`),
    getJson(`/api/guide?date=${encodeURIComponent(d)}&before_minutes=10&limit=16`),
    getJson(`/api/day?date=${encodeURIComponent(d)}${state.jcd ? `&jcd=${state.jcd}` : ""}`),
    getJson(`/api/accuracy?date=${encodeURIComponent(d)}`),
    getJson("/api/backtest")
  ]);
  $("clock").textContent = `JST ${day.now_jst.replace("T"," ").slice(0,16)}`;
  $("summary").innerHTML = stat("レース",s.races)+stat("出走",s.entries)+stat("結果",s.results)+stat("オッズ",s.odds_snapshots)+stat("予測済み",s.predictions);
  renderVenues(vc.venues); renderGuide(g.candidates); renderDay(day.races);
  $("accuracy").textContent = `本日 ${acc.evaluated || 0}R / 1着 ${pct(acc.winner_top1_accuracy)} / 3T5 ${pct(acc.trifecta_top5_hit_rate)}`;
  $("backtest").textContent = bt.available ? `BT ${bt.evaluated_races}R / 1着 ${pct(bt.winner_top1_accuracy)} / 3T5 ${pct(bt.trifecta_top5_hit_rate)}` : "バックテスト結果はまだありません。";
  if(!state.raceId && day.races.length){ const next = g.candidates[0] || day.races.find(r => r.top_prediction) || day.races[0]; await selectRace(next.race_id); }
  else if(state.raceId) await selectRace(state.raceId);
}
function renderVenues(items){
  $("venueFilter").innerHTML = `<option value="">全場</option>` + items.map(v => `<option value="${v.code}">${v.name}</option>`).join("");
  $("venueFilter").value = state.jcd;
  $("venueGrid").innerHTML = items.map(v => `<div class="venue ${v.code === state.jcd ? "active" : ""}" data-jcd="${v.code}">
    <b>${v.name}<span class="badge ${v.status==="監視中"?"live":v.status==="終了"?"done":v.status==="未取得"?"":"wait"}">${v.status}</span></b>
    <small>${v.races}R / 出走 ${v.racelists} / 結果 ${v.finals}<br>次 ${v.next_rno ? `${v.next_rno}R ${hm(v.next_deadline_at)} (${minLabel(v.minutes_to_next_deadline)})` : "-"}<br>odds ${v.odds_snapshots} / 最新 ${age(v.latest_odds_at)}</small>
  </div>`).join("");
  document.querySelectorAll(".venue").forEach(el => el.onclick = () => { state.jcd = el.dataset.jcd; state.raceId = null; loadAll(); });
}
function renderGuide(rows){
  $("guide").innerHTML = rows.map(r => { const p = r.top_prediction || {}; return `<tr class="pick" data-race="${r.race_id}"><td><b>${r.venue_name} ${r.rno}R</b><br><span class="muted">${r.title || ""}</span></td><td class="mono">${hm(r.deadline_at)}<br><span class="muted">${minLabel(r.minutes_to_deadline)}</span></td><td class="mono">${hm(r.buy_until_at)}<br><span class="muted">${minLabel(r.minutes_to_buy_until)}</span></td><td class="mono">${age(r.latest_odds_at)}</td><td class="mono">${p.combination || "-"}</td><td>${pct(p.probability)}</td><td>${num(p.odds)}</td><td>${num(p.expected_value)}</td></tr>`; }).join("") || `<tr><td colspan="8" class="empty">T-10分前までに判断できる候補はありません。</td></tr>`;
  document.querySelectorAll("#guide tr[data-race]").forEach(row => row.onclick = () => selectRace(row.dataset.race));
}
function renderDay(rows){
  $("dayCount").textContent = `${rows.length}R`;
  $("dayRows").innerHTML = rows.map(r => { const p = r.top_prediction || {}; const late = ["締切後","確定","T-10超過"].includes(r.time_status); return `<tr class="${late ? "late" : ""}" data-race="${r.race_id}"><td><b>${r.venue_name} ${r.rno}R</b><br><span class="muted">${r.title || ""}</span></td><td class="mono">${hm(r.deadline_at)}<br><span class="muted">${minLabel(r.minutes_to_deadline)}</span></td><td class="mono">${hm(r.race_time_at)}</td><td><span class="badge ${cls(r.time_status)}">${r.time_status}</span></td><td class="mono">${age(r.latest_odds_at)}</td><td class="mono">${p.combination || "-"}</td><td>${num(p.expected_value)}</td></tr>`; }).join("") || `<tr><td colspan="7" class="empty">レース情報はまだありません。</td></tr>`;
  document.querySelectorAll("#dayRows tr[data-race]").forEach(row => row.onclick = () => selectRace(row.dataset.race));
}
async function selectRace(raceId){
  state.raceId = raceId;
  const data = await getJson(`/api/predictions?race_id=${encodeURIComponent(raceId)}`);
  const race = data.race || {};
  $("raceTitle").textContent = `${race.venue_name || ""} ${race.rno || ""}R ${race.title || ""}`;
  $("entries").innerHTML = data.entries.map(e => `<div class="entry"><span class="lane">${e.lane}</span><br>${e.racer_name || ""}<br><small>${e.racer_class || ""} M${e.motor_no || "-"} B${e.boat_no || "-"}</small></div>`).join("");
  $("predictions").innerHTML = data.predictions.map(p => `<tr><td class="mono">${p.combination}</td><td>${pct(p.probability)}</td><td>${num(p.odds)}</td><td>${num(p.expected_value)}</td></tr>`).join("") || `<tr><td colspan="4" class="empty">予測はまだありません。</td></tr>`;
  $("combo").innerHTML = data.predictions.slice(0,20).map(p => `<option>${p.combination}</option>`).join("") || `<option>1-2-3</option>`;
  state.combo = $("combo").value; await loadOdds();
}
async function loadOdds(){ if(!state.raceId) return; const data = await getJson(`/api/odds?race_id=${encodeURIComponent(state.raceId)}&combination=${encodeURIComponent(state.combo || "1-2-3")}`); drawTrend(data.trend || []); }
function drawTrend(rows){ const c=$("oddsChart"),ctx=c.getContext("2d"); ctx.clearRect(0,0,c.width,c.height); ctx.strokeStyle="#d8e0e3"; ctx.beginPath(); ctx.moveTo(34,16); ctx.lineTo(34,178); ctx.lineTo(700,178); ctx.stroke(); const vals=rows.map(r=>Number(r.odds)).filter(Number.isFinite); if(vals.length<2){ ctx.fillStyle="#637279"; ctx.fillText("オッズ推移の点が不足しています。",46,98); return; } const min=Math.min(...vals),max=Math.max(...vals),span=Math.max(.01,max-min); ctx.strokeStyle="#8f2d56"; ctx.lineWidth=2; ctx.beginPath(); vals.forEach((v,i)=>{ const x=42+(640*i/Math.max(1,vals.length-1)); const y=170-((v-min)/span)*140; if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y); }); ctx.stroke(); ctx.fillStyle="#172126"; ctx.fillText(`min ${num(min)} / max ${num(max)}`,46,26); }
loadAll(); setInterval(loadAll,30000);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
