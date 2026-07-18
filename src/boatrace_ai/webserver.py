from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .db import connect, init_db


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="BOAT RACE AI realtime dashboard.")
    parser.add_argument("--db", default="data/boatrace.sqlite")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10001)
    parser.add_argument("--backtest", default="data/models/backtest.json")
    args = parser.parse_args(argv)
    init_db(args.db)
    print(f"Serving BOAT RACE AI Monitor on http://{args.host}:{args.port}", flush=True)
    handler = make_handler(Path(args.db), Path(args.backtest) if args.backtest else None)
    ThreadingHTTPServer((args.host, args.port), handler).serve_forever()
    return 0


def make_handler(db_path: Path, backtest_path: Path | None):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            try:
                if parsed.path == "/":
                    send_html(self, DASHBOARD_HTML)
                elif parsed.path == "/api/summary":
                    send_json(self, summary(db_path))
                elif parsed.path == "/api/races":
                    send_json(self, races(db_path, query))
                elif parsed.path == "/api/predictions":
                    send_json(self, predictions(db_path, query))
                elif parsed.path == "/api/odds":
                    send_json(self, odds(db_path, query))
                elif parsed.path == "/api/backtest":
                    send_json(self, backtest(backtest_path))
                else:
                    self.send_error(404)
            except Exception as exc:
                send_json(self, {"error": str(exc)}, status=500)

        def log_message(self, fmt: str, *args: Any) -> None:
            return

    return Handler


def summary(db_path: Path) -> dict[str, Any]:
    with connect(db_path) as conn:
        return {
            "races": scalar(conn, "SELECT COUNT(*) FROM races"),
            "entries": scalar(conn, "SELECT COUNT(*) FROM entries"),
            "results": scalar(conn, "SELECT COUNT(DISTINCT race_id) FROM race_results"),
            "odds_snapshots": scalar(conn, "SELECT COUNT(*) FROM odds_snapshots"),
            "predictions": scalar(conn, "SELECT COUNT(DISTINCT race_id) FROM predictions"),
            "latest_prediction": scalar(conn, "SELECT MAX(generated_at) FROM predictions"),
        }


def races(db_path: Path, query: dict[str, list[str]]) -> dict[str, Any]:
    race_date = query.get("date", [date.today().isoformat()])[0]
    with connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT
              r.race_id, r.race_date, r.jcd, r.venue_name, r.rno,
              r.title, r.status, r.deadline_at,
              (SELECT COUNT(*) FROM entries e WHERE e.race_id = r.race_id) AS entries,
              (SELECT COUNT(*) FROM odds_snapshots os WHERE os.race_id = r.race_id) AS odds_snapshots,
              (SELECT MAX(generated_at) FROM predictions p WHERE p.race_id = r.race_id) AS latest_prediction
            FROM races r
            WHERE r.race_date = ?
            ORDER BY r.jcd, r.rno
            """,
            (race_date,),
        ).fetchall()
    return {"date": race_date, "races": [rowdict(row) for row in rows]}


def predictions(db_path: Path, query: dict[str, list[str]]) -> dict[str, Any]:
    race_id = required(query, "race_id")
    with connect(db_path) as conn:
        race = conn.execute("SELECT * FROM races WHERE race_id = ?", (race_id,)).fetchone()
        entries = conn.execute(
            """
            SELECT lane, racer_no, racer_name, racer_class, motor_no, boat_no
            FROM entries
            WHERE race_id = ?
            ORDER BY lane
            """,
            (race_id,),
        ).fetchall()
        latest = conn.execute(
            """
            SELECT generated_at
            FROM predictions
            WHERE race_id = ?
            ORDER BY generated_at DESC
            LIMIT 1
            """,
            (race_id,),
        ).fetchone()
        pred_rows = []
        if latest:
            pred_rows = conn.execute(
                """
                SELECT combination, probability, odds, expected_value, generated_at
                FROM predictions
                WHERE race_id = ? AND generated_at = ?
                ORDER BY COALESCE(expected_value, probability) DESC, probability DESC
                LIMIT 60
                """,
                (race_id, latest["generated_at"]),
            ).fetchall()
    return {
        "race": rowdict(race) if race else None,
        "entries": [rowdict(row) for row in entries],
        "predictions": [rowdict(row) for row in pred_rows],
    }


def odds(db_path: Path, query: dict[str, list[str]]) -> dict[str, Any]:
    race_id = required(query, "race_id")
    combo = query.get("combination", ["1-2-3"])[0]
    with connect(db_path) as conn:
        trend = conn.execute(
            """
            SELECT os.captured_at, os.source_update_time, ot.odds
            FROM odds_snapshots os
            JOIN odds_trifecta ot ON ot.snapshot_id = os.snapshot_id
            WHERE os.race_id = ? AND ot.combination = ?
            ORDER BY os.captured_at
            """,
            (race_id, combo),
        ).fetchall()
        latest_count = scalar(
            conn,
            """
            SELECT COUNT(*)
            FROM odds_trifecta
            WHERE snapshot_id = (
              SELECT snapshot_id
              FROM odds_snapshots
              WHERE race_id = ?
              ORDER BY captured_at DESC, snapshot_id DESC
              LIMIT 1
            )
            """,
            (race_id,),
        )
    return {
        "race_id": race_id,
        "combination": combo,
        "trend": [rowdict(row) for row in trend],
        "latest_count": latest_count,
    }


def backtest(path: Path | None) -> dict[str, Any]:
    if not path or not path.exists():
        return {"available": False}
    return {"available": True, **json.loads(path.read_text(encoding="utf-8"))}


def scalar(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> Any:
    row = conn.execute(sql, params).fetchone()
    return row[0] if row else None


def required(query: dict[str, list[str]], key: str) -> str:
    values = query.get(key)
    if not values or not values[0]:
        raise ValueError(f"missing query parameter: {key}")
    return values[0]


def rowdict(row: sqlite3.Row) -> dict[str, Any]:
    return {key: row[key] for key in row.keys()}


def send_html(handler: BaseHTTPRequestHandler, body: str, status: int = 200) -> None:
    payload = body.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "text/html; charset=utf-8")
    handler.send_header("Content-Length", str(len(payload)))
    handler.end_headers()
    handler.wfile.write(payload)


def send_json(handler: BaseHTTPRequestHandler, value: Any, status: int = 200) -> None:
    payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Content-Length", str(len(payload)))
    handler.end_headers()
    handler.wfile.write(payload)


DASHBOARD_HTML = """<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BOAT RACE AI Monitor</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #172126;
      --muted: #667780;
      --line: #d7e0e4;
      --band: #f4f7f8;
      --accent: #006d77;
      --accent-2: #8f2d56;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    * { box-sizing: border-box; }
    body { margin: 0; color: var(--ink); background: #fff; }
    header {
      display: flex; align-items: center; justify-content: space-between;
      gap: 16px; padding: 14px 20px; border-bottom: 1px solid var(--line);
      background: #fff; position: sticky; top: 0; z-index: 3;
    }
    h1 { font-size: 18px; margin: 0; font-weight: 700; letter-spacing: 0; }
    main { display: grid; grid-template-columns: 320px 1fr; min-height: calc(100vh - 58px); }
    aside { border-right: 1px solid var(--line); background: var(--band); padding: 14px; overflow: auto; }
    section { padding: 18px 22px; min-width: 0; }
    .toolbar { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
    input, select, button {
      height: 34px; border: 1px solid var(--line); background: #fff; color: var(--ink);
      padding: 0 10px; border-radius: 6px; font: inherit;
    }
    button { cursor: pointer; color: #fff; background: var(--accent); border-color: var(--accent); }
    .metrics { display: grid; grid-template-columns: repeat(5, minmax(86px, 1fr)); gap: 1px; background: var(--line); border: 1px solid var(--line); }
    .metric { background: #fff; padding: 12px; min-width: 0; }
    .metric strong { display: block; font-size: 22px; line-height: 1.1; }
    .metric span { color: var(--muted); font-size: 12px; }
    .race-list { display: grid; gap: 6px; margin-top: 12px; }
    .race-button {
      width: 100%; text-align: left; background: #fff; color: var(--ink);
      border: 1px solid var(--line); border-radius: 6px; padding: 9px 10px; height: auto;
    }
    .race-button.active { border-color: var(--accent); box-shadow: inset 3px 0 0 var(--accent); }
    .race-button b { font-size: 14px; }
    .race-button small { display: block; color: var(--muted); margin-top: 3px; }
    .split { display: grid; grid-template-columns: minmax(360px, 1.2fr) minmax(320px, .8fr); gap: 18px; margin-top: 18px; }
    .panel { border-top: 2px solid var(--accent); padding-top: 10px; min-width: 0; }
    .panel h2 { font-size: 15px; margin: 0 0 10px; letter-spacing: 0; }
    table { width: 100%; border-collapse: collapse; table-layout: fixed; }
    th, td { border-bottom: 1px solid var(--line); padding: 8px 6px; font-size: 13px; text-align: right; overflow-wrap: anywhere; }
    th:first-child, td:first-child { text-align: left; }
    th { color: var(--muted); font-weight: 600; background: #fbfcfc; }
    .entries { display: grid; grid-template-columns: repeat(6, minmax(70px, 1fr)); gap: 1px; background: var(--line); border: 1px solid var(--line); margin-top: 10px; }
    .entry { background: #fff; padding: 8px; min-height: 74px; }
    .lane { display: inline-grid; place-items: center; width: 24px; height: 24px; border: 1px solid var(--line); font-weight: 700; margin-bottom: 4px; }
    canvas { width: 100%; height: 180px; border: 1px solid var(--line); background: #fff; }
    .empty { color: var(--muted); padding: 18px 0; }
    @media (max-width: 900px) {
      main { grid-template-columns: 1fr; }
      aside { border-right: 0; border-bottom: 1px solid var(--line); max-height: 42vh; }
      .metrics { grid-template-columns: repeat(2, minmax(120px, 1fr)); }
      .split { grid-template-columns: 1fr; }
      .entries { grid-template-columns: repeat(3, minmax(70px, 1fr)); }
    }
  </style>
</head>
<body>
  <header>
    <h1>BOAT RACE AI Monitor</h1>
    <div class="toolbar">
      <input id="raceDate" type="date">
      <button id="reload">更新</button>
    </div>
  </header>
  <main>
    <aside>
      <div id="summary" class="metrics"></div>
      <div id="raceList" class="race-list"></div>
    </aside>
    <section>
      <div id="raceTitle" class="empty">レースを選択してください。</div>
      <div id="entries" class="entries"></div>
      <div class="split">
        <div class="panel">
          <h2>最新予測</h2>
          <table>
            <thead><tr><th>3連単</th><th>確率</th><th>オッズ</th><th>期待値</th></tr></thead>
            <tbody id="predictions"></tbody>
          </table>
        </div>
        <div class="panel">
          <h2>オッズ推移</h2>
          <select id="combo"></select>
          <canvas id="oddsChart" width="720" height="240"></canvas>
          <div id="backtest" class="empty"></div>
        </div>
      </div>
    </section>
  </main>
<script>
const state = { raceId: null, combo: null };
const $ = (id) => document.getElementById(id);
const today = new Date().toISOString().slice(0, 10);
$("raceDate").value = today;
$("reload").onclick = loadAll;
$("combo").onchange = () => { state.combo = $("combo").value; loadOdds(); };

async function getJson(url) {
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) throw new Error(await res.text());
  return await res.json();
}
function metric(label, value) {
  return `<div class="metric"><strong>${value ?? "-"}</strong><span>${label}</span></div>`;
}
async function loadAll() {
  const summary = await getJson("/api/summary");
  $("summary").innerHTML =
    metric("レース", summary.races) +
    metric("出走", summary.entries) +
    metric("結果", summary.results) +
    metric("オッズ", summary.odds_snapshots) +
    metric("予測済み", summary.predictions);
  await loadRaces();
  await loadBacktest();
}
async function loadRaces() {
  const raceDate = $("raceDate").value || today;
  const data = await getJson(`/api/races?date=${encodeURIComponent(raceDate)}`);
  $("raceList").innerHTML = data.races.map(r => `
    <button class="race-button ${r.race_id === state.raceId ? "active" : ""}" data-race="${r.race_id}">
      <b>${r.venue_name} ${r.rno}R</b>
      <small>${r.entries}艇 / odds ${r.odds_snapshots} / ${r.status || ""}</small>
    </button>
  `).join("") || `<div class="empty">対象日のレースはまだありません。</div>`;
  document.querySelectorAll(".race-button").forEach(btn => {
    btn.onclick = () => selectRace(btn.dataset.race);
  });
}
async function selectRace(raceId) {
  state.raceId = raceId;
  await loadRaces();
  const data = await getJson(`/api/predictions?race_id=${encodeURIComponent(raceId)}`);
  const race = data.race || {};
  $("raceTitle").textContent = `${race.venue_name || ""} ${race.rno || ""}R ${race.title || ""}`;
  $("entries").innerHTML = data.entries.map(e => `
    <div class="entry"><span class="lane">${e.lane}</span><br>${e.racer_name || ""}<br><small>${e.racer_class || ""} M${e.motor_no || "-"} B${e.boat_no || "-"}</small></div>
  `).join("");
  $("predictions").innerHTML = data.predictions.map(p => `
    <tr data-combo="${p.combination}"><td>${p.combination}</td><td>${pct(p.probability)}</td><td>${num(p.odds)}</td><td>${num(p.expected_value)}</td></tr>
  `).join("") || `<tr><td colspan="4">予測はまだありません。</td></tr>`;
  $("combo").innerHTML = data.predictions.slice(0, 20).map(p => `<option>${p.combination}</option>`).join("") || `<option>1-2-3</option>`;
  state.combo = $("combo").value;
  await loadOdds();
}
async function loadOdds() {
  if (!state.raceId) return;
  const combo = state.combo || "1-2-3";
  const data = await getJson(`/api/odds?race_id=${encodeURIComponent(state.raceId)}&combination=${encodeURIComponent(combo)}`);
  drawTrend(data.trend || []);
}
async function loadBacktest() {
  const data = await getJson("/api/backtest");
  if (!data.available) {
    $("backtest").textContent = "バックテスト結果はまだありません。";
    return;
  }
  $("backtest").innerHTML =
    `BT: ${data.evaluated_races || 0} races / winner ${pct(data.winner_top1_accuracy)} / 3T top5 ${pct(data.trifecta_top5_hit_rate)}`;
}
function drawTrend(rows) {
  const canvas = $("oddsChart");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = "#d7e0e4";
  ctx.beginPath();
  ctx.moveTo(40, 18); ctx.lineTo(40, 210); ctx.lineTo(700, 210); ctx.stroke();
  const values = rows.map(r => Number(r.odds)).filter(v => Number.isFinite(v));
  if (values.length < 2) {
    ctx.fillStyle = "#667780";
    ctx.fillText("オッズ推移の点が不足しています。", 52, 118);
    return;
  }
  const min = Math.min(...values), max = Math.max(...values);
  const span = Math.max(0.01, max - min);
  ctx.strokeStyle = "#8f2d56";
  ctx.lineWidth = 2;
  ctx.beginPath();
  values.forEach((v, i) => {
    const x = 48 + (640 * i / Math.max(1, values.length - 1));
    const y = 202 - ((v - min) / span) * 168;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();
  ctx.fillStyle = "#172126";
  ctx.fillText(`min ${num(min)} / max ${num(max)}`, 52, 28);
}
function pct(value) {
  return value == null ? "-" : `${(Number(value) * 100).toFixed(2)}%`;
}
function num(value) {
  return value == null ? "-" : Number(value).toFixed(3);
}
loadAll();
setInterval(loadAll, 60000);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
