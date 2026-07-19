#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_HOST = "root@213.173.105.92"
DEFAULT_PORT = "28659"
DEFAULT_IDENTITY = str(Path.home() / ".ssh" / "id_ed25519")
DEFAULT_WORKDIR = "/workspace/boat-milestone-e07badb"
DEFAULT_OUTPUT = Path("data/remote_eval_status.json")

JOBS: list[dict[str, Any]] = [
    {"pid": 171805, "name": "adaptive_real_odds_97cc181", "milestone": "M6", "kind": "real_odds", "output": "data/models/adaptive_real_odds_97cc181.json", "log": "logs/adaptive_real_odds_97cc181.log"},
    {"pid": 172555, "name": "m6_norm_ev105_top80_19d5c35", "milestone": "M6", "kind": "bankroll_norm", "output": "data/models/m6_norm_ev105_top80_19d5c35.json", "log": "logs/m6_norm_ev105_top80_19d5c35.log"},
    {"pid": 172556, "name": "m6_norm_ev110_top40_19d5c35", "milestone": "M6", "kind": "bankroll_norm", "output": "data/models/m6_norm_ev110_top40_19d5c35.json", "log": "logs/m6_norm_ev110_top40_19d5c35.log"},
    {"pid": 172557, "name": "m6_norm_ev120_top30_19d5c35", "milestone": "M6", "kind": "bankroll_norm", "output": "data/models/m6_norm_ev120_top30_19d5c35.json", "log": "logs/m6_norm_ev120_top30_19d5c35.log"},
    {"pid": 172558, "name": "m6_norm_ev150_top20_19d5c35", "milestone": "M6", "kind": "bankroll_norm", "output": "data/models/m6_norm_ev150_top20_19d5c35.json", "log": "logs/m6_norm_ev150_top20_19d5c35.log"},
    {"pid": 172559, "name": "m6_norm_ev200_top10_19d5c35", "milestone": "M6", "kind": "bankroll_norm", "output": "data/models/m6_norm_ev200_top10_19d5c35.json", "log": "logs/m6_norm_ev200_top10_19d5c35.log"},
    {"pid": 172873, "name": "m6_norm_sanity_fold_19d5c35", "milestone": "M6", "kind": "bankroll_sanity", "output": "data/models/m6_norm_sanity_fold_19d5c35.json", "log": "logs/m6_norm_sanity_fold_19d5c35.log"},
    {"pid": 171811, "name": "feature_ablation_97cc181", "milestone": "M4", "kind": "feature_ablation", "output": "data/models/feature_ablation_97cc181.json", "log": "logs/feature_ablation_97cc181.log"},
    {"pid": 173485, "name": "feature_correlation_advanced_e07badb", "milestone": "M4-2", "kind": "feature_correlation", "output": "data/models/feature_diagnostics_stream_advanced_e07badb.json", "log": "logs/feature_correlation_advanced_e07badb.log", "superseded_by": "feature_correlation_advanced_retry"},
    {"pid": 174501, "name": "feature_correlation_advanced_retry", "milestone": "M4-2", "kind": "feature_correlation", "output": "data/models/feature_result_correlation_v8_stream_advanced_retry.json", "log": "logs/feature_correlation_advanced_retry.log"},
    {"pid": 175652, "name": "m6_best_roi_attribution_b53debe", "milestone": "M4-2/M6", "kind": "bankroll_roi_attribution", "output": "data/models/m6_best_roi_attribution_b53debe.json", "log": "logs/m6_best_roi_attribution_b53debe.log", "superseded_by": "m6_best_roi_attribution_retry3"},
    {"pid": 178254, "name": "m6_best_roi_attribution_retry3", "milestone": "M4-2/M6", "kind": "bankroll_roi_attribution", "output": "data/models/m6_best_roi_attribution_retry3.json", "log": "logs/m6_best_roi_attribution_retry3.log"},
    {"pid": 184638, "name": "calibrated_linear_shadow_2fold", "milestone": "M4-1", "kind": "calibrated_linear", "output": "data/models/calibrated_linear_shadow_2fold.json", "log": "logs/calibrated_linear_shadow_2fold.log"},
    {"pid": 184700, "name": "calibrated_mlp_shadow_2fold", "milestone": "M4-1", "kind": "calibrated_mlp", "output": "data/models/calibrated_mlp_shadow_2fold.json", "log": "logs/calibrated_mlp_shadow_2fold.log"},
    {"pid": 192206, "name": "bankroll_no_odds_v8_normalized_kelly_5fold", "milestone": "M6", "kind": "bankroll_operational_same_policy", "output": "data/models/bankroll_no_odds_v8_normalized_kelly_5fold.json", "log": "logs/bankroll_no_odds_v8_normalized_kelly_5fold.log"},
    {"pid": 196980, "name": "listwise_feature_teacher_search_v1", "milestone": "M4-3", "kind": "feature_teacher_search", "output": "data/models/listwise_feature_teacher_search_v1.json", "log": "logs/listwise_feature_teacher_newton_loop.log"},
    {"pid": 196979, "name": "listwise_newton_cg_v1", "milestone": "M4-3/M6", "kind": "newton_listwise_bankroll", "output": "data/models/listwise_newton_cg_v1.json", "log": "logs/listwise_feature_teacher_newton_loop.log", "wait_for": "data/models/listwise_feature_teacher_search_v1.json"},
    {"pid": 199131, "name": "listwise_temporal_stability_v1", "milestone": "M4-3/M6", "kind": "listwise_temporal_stability", "output": "data/models/listwise_temporal_stability_v1.json", "log": "logs/listwise_temporal_stability_v1.log"},
]

REMOTE_CODE = r'''
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

WORKDIR = Path(__WORKDIR_JSON__)
JOBS = __JOBS_JSON__
HOST = __HOST_JSON__
METRIC_KEYS = (
    "roi", "profit_yen", "stake_yen", "return_yen", "evaluated_races", "selected_races",
    "tickets", "hit_tickets", "ticket_hit_rate", "race_hit_rate", "max_drawdown_yen",
    "skipped_no_real_odds", "real_odds_races", "entry_log_loss", "entry_brier",
    "winner_top1_accuracy", "trifecta_top5_hit_rate", "ranking_log_loss",
    "examples", "races", "positive_labels", "global_win_rate", "race_days",
    "candidate_tickets", "winning_days", "losing_days", "budget_utilization",
    "promotion_eligible", "holdout_races", "profitable_folds",
)

def iso_mtime(path):
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).replace(microsecond=0).isoformat()

def ps_row(pid):
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "pid=", "-o", "stat=", "-o", "etime=", "-o", "pcpu=", "-o", "pmem=", "-o", "cmd="],
        text=True,
        capture_output=True,
    )
    line = result.stdout.strip()
    if result.returncode != 0 or not line:
        return None
    parts = line.split(None, 5)
    if len(parts) < 6:
        return {"pid": pid, "raw": line}
    return {"pid": int(parts[0]), "stat": parts[1], "elapsed": parts[2], "pcpu": parts[3], "pmem": parts[4], "cmd": parts[5]}

def tail_text(path, lines=12):
    if not path.exists():
        return []
    try:
        return path.read_text(errors="replace").splitlines()[-lines:]
    except Exception as exc:
        return [f"log_read_error: {exc}"]

def completed_fold_count(path):
    if not path.exists():
        return 0
    completed = 0
    try:
        with path.open(errors="replace") as handle:
            for line in handle:
                try:
                    item = json.loads(line)
                except Exception:
                    continue
                if isinstance(item, dict):
                    completed = max(completed, int(item.get("fold") or 0))
    except Exception:
        return completed
    return completed


def result_summary(path):
    if not path.exists():
        return None
    row = {"file": str(path.relative_to(WORKDIR)), "size_bytes": path.stat().st_size, "modified_at": iso_mtime(path)}
    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        row["error"] = str(exc)
        return row
    row["metrics"] = {key: data.get(key) for key in METRIC_KEYS if key in data}
    holdout = data.get("holdout_after_newton") or data.get("holdout") or {}
    if isinstance(holdout, dict):
        for key in METRIC_KEYS:
            if key in holdout:
                row["metrics"][key] = holdout.get(key)
        bankroll = holdout.get("bankroll") or data.get("bankroll") or {}
        if isinstance(bankroll, dict):
            for key in METRIC_KEYS:
                if key in bankroll:
                    row["metrics"][key] = bankroll.get(key)
    if "feature_family_summary" in data or "suspect_features" in data:
        row["feature_family_summary"] = (data.get("feature_family_summary") or [])[:16]
        row["suspect_features"] = (data.get("suspect_features") or [])[:24]
        row["coefficient_alignment"] = (data.get("coefficient_alignment") or [])[:16]
        row["action_items"] = (data.get("action_items") or data.get("diagnosis") or [])[:10]
        row["roi_link"] = data.get("roi_link") or {}
    if "ticket_roi_attribution" in data:
        attribution = data.get("ticket_roi_attribution") or {}
        row["ticket_roi_attribution"] = {
            "method": attribution.get("method"),
            "diagnosis": attribution.get("diagnosis"),
            "minimum_evidence": attribution.get("minimum_evidence") or {},
            "top_signals": (attribution.get("top_signals") or [])[:16],
            "fold_stability": attribution.get("fold_stability") or {},
        }
    if "drop_results" in data:
        base = data.get("base") or {}
        row["base_metrics"] = {key: base.get(key) for key in METRIC_KEYS if key in base}
        drops = []
        for item in data.get("drop_results") or []:
            drops.append({"dropped": item.get("dropped"), "metrics": {key: item.get(key) for key in METRIC_KEYS if key in item}})
        row["drop_results"] = drops
    if "folds" in data:
        row["folds"] = len(data.get("folds") or [])
    if "daily" in data:
        row["daily_rows"] = len(data.get("daily") or [])
    return row

jobs = []
for job in JOBS:
    proc = ps_row(job["pid"])
    output_path = WORKDIR / job["output"]
    log_path = WORKDIR / job["log"]
    result = result_summary(output_path)
    log_tail = tail_text(log_path)
    log_joined = "\n".join(log_tail)
    error_seen = "Traceback" in log_joined or "ValueError" in log_joined or "Error" in log_joined
    waiting_for = job.get("wait_for")
    dependency_pending = bool(waiting_for and not (WORKDIR / waiting_for).exists())
    if job.get("superseded_by") and not proc:
        status = "差替済み"
    elif proc:
        status = "待機中" if dependency_pending or (log_tail and "waiting_for_pid" in log_tail[-1]) else "実行中"
    elif result and not result.get("error"):
        status = "完了"
    elif error_seen or (result and result.get("error")):
        status = "失敗"
    else:
        status = "未生成"
    item = dict(job)
    item.update({
        "status": status,
        "running": bool(proc),
        "process": proc,
        "result": result,
        "log_tail": log_tail,
        "completed_folds": completed_fold_count(log_path),
    })
    jobs.append(item)

print(json.dumps({
    "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    "host": HOST,
    "workdir": str(WORKDIR),
    "jobs": jobs,
}, ensure_ascii=False))
'''


def build_remote_code(host: str, workdir: str) -> str:
    return (
        REMOTE_CODE
        .replace("__HOST_JSON__", json.dumps(host))
        .replace("__WORKDIR_JSON__", json.dumps(workdir))
        .replace("__JOBS_JSON__", json.dumps(JOBS))
    )


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def poll_once(args: argparse.Namespace) -> dict[str, Any]:
    ssh = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
        "-i",
        args.identity,
        "-p",
        args.port,
        args.host,
        "python3",
        "-",
    ]
    result = subprocess.run(
        ssh,
        input=build_remote_code(args.host, args.workdir),
        text=True,
        capture_output=True,
        timeout=args.timeout,
    )
    if result.returncode != 0:
        return {
            "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "host": args.host,
            "workdir": args.workdir,
            "status": "取得失敗",
            "error": result.stderr.strip() or result.stdout.strip() or f"ssh exit {result.returncode}",
            "jobs": JOBS,
        }
    payload = json.loads(result.stdout)
    payload["status"] = "取得済み"
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Poll remote boat race model evaluation jobs into a local status JSON.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--identity", default=DEFAULT_IDENTITY)
    parser.add_argument("--workdir", default=DEFAULT_WORKDIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--interval", type=float, default=120.0)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--loop", action="store_true")
    args = parser.parse_args()

    while True:
        payload = poll_once(args)
        write_json_atomic(args.output, payload)
        print(json.dumps({"generated_at": payload.get("generated_at"), "status": payload.get("status"), "jobs": [job.get("status") for job in payload.get("jobs", [])]}, ensure_ascii=False), flush=True)
        if not args.loop:
            return 0 if payload.get("status") == "取得済み" else 1
        time.sleep(max(10.0, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
