from __future__ import annotations

import argparse
from http.server import ThreadingHTTPServer
from pathlib import Path

from . import webserver_operational23 as base
from .db import init_db
from .webserver_operational20 import HTML as BASE_HTML


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
    html = html.replace("loadAll(); setInterval(loadAll,30000);", LIVE_WIPE_JS + "\nloadAll(); setInterval(loadAll,30000);")
    return html


HTML = build_html()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="BOAT RACE AI larger live model-prediction dashboard.")
    parser.add_argument("--db", default="data/boatrace.sqlite")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10001)
    parser.add_argument("--backtest", default="data/models/backtest_no_odds_v7.json")
    args = parser.parse_args(argv)
    init_db(args.db)
    base.HTML = HTML
    handler = base.make_handler(Path(args.db), Path(args.backtest) if args.backtest else None)
    print(f"Serving BOAT RACE AI Larger Live Model Monitor on http://{args.host}:{args.port}", flush=True)
    ThreadingHTTPServer((args.host, args.port), handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
