from __future__ import annotations

import argparse
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

from . import webserver_operational26 as api_base
from .db import connect, init_db
from .webserver_operational27 import HTML as PREV_HTML


ORIGINAL_PREDICTIONS = api_base.predictions_start_time_model_rank


ENTRY_AND_WIPE_CSS = """
    .entries { grid-template-columns:repeat(6,minmax(76px,1fr)); gap:2px; background:transparent; border:0; }
    .entry { min-height:42px; padding:3px 4px; border:1px solid rgba(0,0,0,.18); color:#111; overflow:hidden; }
    .entry-main { display:grid; grid-template-columns:20px minmax(0,1fr); gap:4px; align-items:center; min-width:0; }
    .entry .lane { margin:0; width:20px; height:20px; border:1px solid currentColor; background:rgba(255,255,255,.22); font-weight:800; }
    .entry .racer-name { min-width:0; font-weight:800; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    .entry-meta { display:flex; gap:4px; justify-content:space-between; margin-top:2px; font-size:9px; opacity:.86; white-space:nowrap; overflow:hidden; }
    .entry.lane1 { background:#fff; color:#111; }
    .entry.lane2 { background:#202529; color:#fff; }
    .entry.lane3 { background:#c83232; color:#fff; }
    .entry.lane4 { background:#1769c2; color:#fff; }
    .entry.lane5 { background:#f5d84a; color:#111; }
    .entry.lane6 { background:#24824d; color:#fff; }
    .live-wipe.zoom .live-wipe-video iframe { width:170%; height:170%; transform:translate(-20.6%,-20.6%); }
    @media (max-width:720px) { .entries { grid-template-columns:repeat(3,minmax(82px,1fr)); } }
"""


SELECT_RACE_JS = """function escapeHtml(v){
  return String(v ?? "").replace(/[&<>"']/g, ch => ({ "&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;" }[ch]));
}
function racerDisplayName(e){
  const raw = String(e.racer_name || "").trim();
  const no = String(e.racer_no || "").trim();
  if(raw && raw !== no && !/^\\d+$/.test(raw)) return raw;
  return "選手名未取得";
}
function renderEntryCard(e){
  const lane = Number(e.lane || 0);
  const name = racerDisplayName(e);
  const no = e.racer_no ? `#${e.racer_no}` : "#-";
  const meta = `${e.racer_class || "-"} M${e.motor_no || "-"} B${e.boat_no || "-"}`;
  return `<div class="entry lane${lane}" title="${escapeHtml(`${lane}号艇 ${name} ${no} ${meta}`)}">
    <div class="entry-main"><span class="lane">${lane || "-"}</span><span class="racer-name">${escapeHtml(name)}</span></div>
    <div class="entry-meta"><span>${escapeHtml(no)}</span><span>${escapeHtml(meta)}</span></div>
  </div>`;
}
async function selectRace(raceId){
  state.raceId = raceId;
  const data = await getJson(`/api/predictions?race_id=${encodeURIComponent(raceId)}`);
  const race = data.race || {};
  $("raceTitle").textContent = `${race.venue_name || ""} ${race.rno || ""}R ${race.title || ""}`;
  $("entries").innerHTML = data.entries.map(renderEntryCard).join("");
  $("predictions").innerHTML = data.predictions.map(p => `<tr><td class="mono">${p.combination}</td><td>${pct(p.probability)}</td><td>${num(p.odds)}</td><td>${num(p.expected_value)}</td></tr>`).join("") || `<tr><td colspan="4" class="empty">予測はまだありません。</td></tr>`;
  $("combo").innerHTML = data.predictions.slice(0,20).map(p => `<option>${p.combination}</option>`).join("") || `<option>1-2-3</option>`;
  state.combo = $("combo").value; await loadOdds();
}
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="BOAT RACE AI dashboard with lane-colored entry cards and tighter live video crop.")
    parser.add_argument("--db", default="data/boatrace.sqlite")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10001)
    parser.add_argument("--backtest", default="data/models/backtest_no_odds_v8.json")
    args = parser.parse_args(argv)
    init_db(args.db)
    api_base.HTML = HTML
    api_base.predictions_start_time_model_rank = predictions_with_names
    handler = api_base.make_handler(Path(args.db), Path(args.backtest) if args.backtest else None)
    print(f"Serving BOAT RACE AI Entry Card Monitor on http://{args.host}:{args.port}", flush=True)
    ThreadingHTTPServer((args.host, args.port), handler).serve_forever()
    return 0


def predictions_with_names(db_path: Path, query: dict[str, list[str]]) -> dict[str, Any]:
    payload = ORIGINAL_PREDICTIONS(db_path, query)
    entries = payload.get("entries") or []
    missing = [entry for entry in entries if needs_name_fix(entry)]
    if missing:
        with connect(db_path) as conn:
            for entry in missing:
                lookup = conn.execute(
                    """
                    SELECT racer_name, COUNT(*) AS c
                    FROM entries
                    WHERE racer_no = ?
                      AND racer_name IS NOT NULL
                      AND TRIM(racer_name) != ''
                      AND racer_name NOT GLOB '[0-9]*'
                    GROUP BY racer_name
                    ORDER BY c DESC
                    LIMIT 1
                    """,
                    (entry.get("racer_no"),),
                ).fetchone()
                if lookup:
                    entry["racer_name"] = lookup["racer_name"]
                    entry["racer_name_source"] = "history_lookup"
                else:
                    entry["racer_name_source"] = "missing"
    return payload


def needs_name_fix(entry: dict[str, Any]) -> bool:
    name = str(entry.get("racer_name") or "").strip()
    no = str(entry.get("racer_no") or "").strip()
    return not name or name == no or name.isdigit()


def build_html() -> str:
    html = PREV_HTML.replace("</style>", ENTRY_AND_WIPE_CSS + "\n  </style>")
    start = html.index("async function selectRace(")
    end = html.index("\nasync function loadOdds", start)
    return html[:start] + SELECT_RACE_JS + html[end:]


HTML = build_html()


if __name__ == "__main__":
    raise SystemExit(main())

