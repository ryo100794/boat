from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .db import init_db
from .webserver_all import backtest, odds, predictions, send_html, send_json, summary
from .webserver_operational2 import day_overview, purchase_guide, venue_cards
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


HTML = """<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BOAT RACE AI Ops</title>
  <style>
    :root { color-scheme: light; --ink:#172126; --muted:#637279; --line:#d8e0e3; --band:#f3f6f7; --accent:#006d77; --accent2:#8f2d56; --ok:#247a4b; --warn:#a76300; --bad:#a33a3a; font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    * { box-sizing: border-box; } body { margin:0; color:var(--ink); background:#fff; font-size:12px; }
    header { display:flex; align-items:center; justify-content:space-between; gap:14px; padding:9px 12px; border-bottom:1px solid var(--line); background:#fff; position:sticky; top:0; z-index:6; }
    h1 { margin:0; font-size:17px; letter-spacing:0; } main { display:grid; grid-template-columns:430px 1fr; min-height:calc(100vh - 49px); }
    aside { background:var(--band); border-right:1px solid var(--line); padding:8px; overflow:auto; } section { padding:10px 12px; min-width:0; }
    input, select, button { height:30px; border:1px solid var(--line); border-radius:6px; padding:0 8px; background:#fff; color:var(--ink); font:inherit; }
    button { background:var(--accent); border-color:var(--accent); color:#fff; cursor:pointer; } .toolbar { display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
    .stats { display:grid; grid-template-columns:repeat(5,minmax(68px,1fr)); gap:1px; background:var(--line); border:1px solid var(--line); }
    .stat { background:#fff; padding:6px; min-width:0; } .stat b { display:block; font-size:17px; line-height:1.1; } .stat span { color:var(--muted); font-size:10px; }
    .venue-grid { display:grid; grid-template-columns:repeat(2,minmax(190px,1fr)); gap:5px; margin-top:8px; }
    .venue { background:#fff; border:1px solid var(--line); border-radius:6px; padding:6px; cursor:pointer; min-height:112px; }
    .venue.active { border-color:var(--accent); box-shadow:inset 3px 0 0 var(--accent); }
    .venue b { display:flex; align-items:center; justify-content:space-between; gap:6px; font-size:13px; line-height:1.15; }
    .venue .next { display:grid; grid-template-columns:1fr auto; gap:4px; align-items:center; margin:5px 0; padding:4px 5px; background:#f8fafb; border:1px solid #edf1f2; border-radius:4px; }
    .venue .next strong { font-size:12px; } .venue .next span { color:var(--muted); font-size:11px; white-space:nowrap; }
    .metrics { display:grid; grid-template-columns:repeat(4,1fr); gap:1px; margin-bottom:4px; background:var(--line); border:1px solid var(--line); }
    .metric { background:#fff; padding:3px 4px; text-align:right; min-width:0; }
    .metric b { display:block; font-size:13px; line-height:1.05; } .metric span { display:block; color:var(--muted); font-size:10px; line-height:1.1; }
    .subline { display:grid; grid-template-columns:48px 1fr; gap:4px; color:var(--muted); font-size:11px; line-height:1.25; white-space:nowrap; overflow:hidden; }
    .subline strong { color:var(--ink); font-weight:600; overflow:hidden; text-overflow:ellipsis; }
    .badge { display:inline-block; border-radius:999px; padding:1px 6px; color:#fff; background:var(--muted); font-size:11px; font-weight:700; white-space:nowrap; }
    .live,.候補 { background:var(--accent); } .done,.確定 { background:var(--ok); } .wait,.T-10超過 { background:var(--warn); } .締切後 { background:var(--bad); }
    .timeline-frame { border:1px solid var(--line); border-top:2px solid var(--accent); background:#fff; margin-bottom:12px; }
    .timeline-frame h2 { margin:0; padding:7px 8px; font-size:14px; letter-spacing:0; display:flex; justify-content:space-between; gap:8px; border-bottom:1px solid var(--line); }
    .timeline-scroll { max-height:245px; overflow-y:auto; scrollbar-gutter:stable; }
    .timeline-scroll th { top:0; } .timeline-scroll tr { height:50px; }
    .grid2 { display:grid; grid-template-columns:minmax(520px,1.05fr) minmax(430px,.95fr); gap:12px; margin-top:12px; }
    .panel { border-top:2px solid var(--accent); padding-top:8px; min-width:0; } .panel h2 { margin:0 0 7px; font-size:14px; letter-spacing:0; display:flex; justify-content:space-between; gap:8px; }
    table { width:100%; border-collapse:collapse; table-layout:fixed; } th,td { border-bottom:1px solid var(--line); padding:5px; text-align:right; vertical-align:top; overflow-wrap:anywhere; }
    th { color:var(--muted); font-weight:700; background:#fafbfb; position:sticky; top:49px; z-index:2; } th:first-child,td:first-child { text-align:left; }
    tr.pick { background:#f2fbfa; } tr.late { color:var(--muted); } tr.nowline { background:#fff9e9; }
    .mono { font-variant-numeric:tabular-nums; font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; } .muted { color:var(--muted); }
    .entries { display:grid; grid-template-columns:repeat(6,minmax(72px,1fr)); gap:1px; background:var(--line); border:1px solid var(--line); margin:7px 0 10px; }
    .entry { background:#fff; min-height:68px; padding:6px; } .lane { display:inline-grid; place-items:center; width:22px; height:22px; border:1px solid var(--line); font-weight:700; margin-bottom:3px; }
    canvas { width:100%; height:140px; border:1px solid var(--line); background:#fff; } .empty { color:var(--muted); padding:10px 0; }
    @media (max-width:1180px) { main,.grid2 { grid-template-columns:1fr; } aside { border-right:0; border-bottom:1px solid var(--line); max-height:42vh; } }
    @media (max-width:720px) { .stats { grid-template-columns:repeat(2,minmax(120px,1fr)); } .venue-grid { grid-template-columns:1fr; } .entries { grid-template-columns:repeat(3,minmax(72px,1fr)); } header { align-items:flex-start; flex-direction:column; } }
  </style>
</head>
<body>
  <header><h1>BOAT RACE AI Ops</h1><div class="toolbar"><input id="raceDate" type="date"><select id="venueFilter"><option value="">全場</option></select><button id="reload">更新</button><span id="clock" class="muted mono"></span></div></header>
  <main>
    <aside><div id="summary" class="stats"></div><div id="venueGrid" class="venue-grid"></div></aside>
    <section>
      <div class="timeline-frame">
        <h2><span>現時点以降タイムライン</span><span id="timelineInfo" class="muted"></span></h2>
        <div class="timeline-scroll"><table><thead><tr><th>場/R</th><th>締切</th><th>T-10</th><th>状態</th><th>odds取得</th><th>上位</th><th>EV</th></tr></thead><tbody id="timelineRows"></tbody></table></div>
      </div>
      <div class="grid2">
        <div class="panel"><h2><span>次の購入候補</span><span class="muted">締切10分前まで / odds取得時刻表示</span></h2><table><thead><tr><th>レース</th><th>締切</th><th>T-10</th><th>odds取得</th><th>候補</th><th>確率</th><th>オッズ</th><th>EV</th></tr></thead><tbody id="guide"></tbody></table></div>
        <div class="panel"><h2><span id="raceTitle">レース詳細</span><span id="accuracy" class="muted"></span></h2><div id="entries" class="entries"></div><table><thead><tr><th>3連単</th><th>確率</th><th>オッズ</th><th>EV</th></tr></thead><tbody id="predictions"></tbody></table><h2 style="margin-top:12px;"><span>オッズ推移</span><select id="combo"></select></h2><canvas id="oddsChart" width="720" height="200"></canvas><div id="backtest" class="empty"></div></div>
      </div>
    </section>
  </main>
<script>
const state = { raceId:null, jcd:"", combo:"1-2-3", nowIso:null };
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
function statusClass(v){ return v==="監視中" ? "live" : v==="終了" ? "done" : v==="未取得" ? "" : "wait"; }
function futureRows(rows){
  const nowMs = state.nowIso ? new Date(state.nowIso).getTime() : Date.now();
  return rows.filter(r => {
    if(r.deadline_at) return new Date(r.deadline_at).getTime() >= nowMs;
    return !["確定","締切後"].includes(r.time_status || "");
  }).sort((a,b) => {
    const av = a.deadline_at ? new Date(a.deadline_at).getTime() : Number.MAX_SAFE_INTEGER;
    const bv = b.deadline_at ? new Date(b.deadline_at).getTime() : Number.MAX_SAFE_INTEGER;
    return av - bv || String(a.jcd).localeCompare(String(b.jcd)) || Number(a.rno || 0) - Number(b.rno || 0);
  });
}
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
  state.nowIso = day.now_jst;
  $("clock").textContent = `JST ${day.now_jst.replace("T"," ").slice(0,16)}`;
  $("summary").innerHTML = stat("レース",s.races)+stat("出走",s.entries)+stat("結果",s.results)+stat("オッズ",s.odds_snapshots)+stat("予測済み",s.predictions);
  renderVenues(vc.venues); renderGuide(g.candidates); renderTimeline(day.races);
  $("accuracy").textContent = `本日 ${acc.evaluated || 0}R / 1着 ${pct(acc.winner_top1_accuracy)} / 3T5 ${pct(acc.trifecta_top5_hit_rate)}`;
  $("backtest").textContent = bt.available ? `BT ${bt.evaluated_races}R / 1着 ${pct(bt.winner_top1_accuracy)} / 3T5 ${pct(bt.trifecta_top5_hit_rate)}` : "バックテスト結果はまだありません。";
  if(!state.raceId && day.races.length){ const next = g.candidates[0] || futureRows(day.races).find(r => r.top_prediction) || day.races.find(r => r.top_prediction) || day.races[0]; await selectRace(next.race_id); }
  else if(state.raceId) await selectRace(state.raceId);
}
function renderVenues(items){
  $("venueFilter").innerHTML = `<option value="">全場</option>` + items.map(v => `<option value="${v.code}">${v.name}</option>`).join("");
  $("venueFilter").value = state.jcd;
  $("venueGrid").innerHTML = items.map(v => `<div class="venue ${v.code === state.jcd ? "active" : ""}" data-jcd="${v.code}">
    <b><span>${v.code} ${v.name}</span><span class="badge ${statusClass(v.status)}">${v.status}</span></b>
    <div class="next"><strong>${v.next_rno ? `次 ${v.next_rno}R ${hm(v.next_deadline_at)}` : "次 -"} </strong><span>${minLabel(v.minutes_to_next_deadline)}</span></div>
    <div class="metrics">
      <div class="metric"><b>${v.races}</b><span>R</span></div>
      <div class="metric"><b>${v.racelists}</b><span>出走</span></div>
      <div class="metric"><b>${v.odds_snapshots}</b><span>odds</span></div>
      <div class="metric"><b>${v.finals}</b><span>結果</span></div>
    </div>
    <div class="subline"><span>odds</span><strong>${age(v.latest_odds_at)}</strong></div>
    <div class="subline"><span>予測</span><strong>${hm(v.latest_prediction)}</strong></div>
  </div>`).join("");
  document.querySelectorAll(".venue").forEach(el => el.onclick = () => { state.jcd = el.dataset.jcd; state.raceId = null; loadAll(); });
}
function renderGuide(rows){
  $("guide").innerHTML = rows.map(r => { const p = r.top_prediction || {}; return `<tr class="pick" data-race="${r.race_id}"><td><b>${r.venue_name} ${r.rno}R</b><br><span class="muted">${r.title || ""}</span></td><td class="mono">${hm(r.deadline_at)}<br><span class="muted">${minLabel(r.minutes_to_deadline)}</span></td><td class="mono">${hm(r.buy_until_at)}<br><span class="muted">${minLabel(r.minutes_to_buy_until)}</span></td><td class="mono">${age(r.latest_odds_at)}</td><td class="mono">${p.combination || "-"}</td><td>${pct(p.probability)}</td><td>${num(p.odds)}</td><td>${num(p.expected_value)}</td></tr>`; }).join("") || `<tr><td colspan="8" class="empty">T-10分前までに判断できる候補はありません。</td></tr>`;
  document.querySelectorAll("#guide tr[data-race]").forEach(row => row.onclick = () => selectRace(row.dataset.race));
}
function renderTimeline(rows){
  const upcoming = futureRows(rows);
  $("timelineInfo").textContent = `${state.jcd ? $("venueFilter").selectedOptions[0]?.textContent || "" : "全場"} / 現在以降 ${upcoming.length}R / 先頭4件表示`;
  $("timelineRows").innerHTML = upcoming.map((r,idx) => { const p = r.top_prediction || {}; return `<tr class="${idx < 4 ? "nowline" : ""}" data-race="${r.race_id}"><td><b>${r.venue_name} ${r.rno}R</b><br><span class="muted">${r.title || ""}</span></td><td class="mono">${hm(r.deadline_at)}<br><span class="muted">${minLabel(r.minutes_to_deadline)}</span></td><td class="mono">${hm(r.buy_until_at)}<br><span class="muted">${minLabel(r.minutes_to_buy_until)}</span></td><td><span class="badge ${cls(r.time_status)}">${r.time_status}</span></td><td class="mono">${age(r.latest_odds_at)}</td><td class="mono">${p.combination || "-"}</td><td>${num(p.expected_value)}</td></tr>`; }).join("") || `<tr><td colspan="7" class="empty">現時点以降のレース情報はありません。</td></tr>`;
  document.querySelectorAll("#timelineRows tr[data-race]").forEach(row => row.onclick = () => selectRace(row.dataset.race));
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
