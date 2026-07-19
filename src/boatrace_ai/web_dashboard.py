from __future__ import annotations

import argparse
import json
import re
import sqlite3
import time
from datetime import date, datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .constants import RACES_PER_DAY, VENUES
from .db import connect, init_db
from .official import race_page_url, ymd


JST = timezone(timedelta(hours=9))
START_TO_DEADLINE_MINUTES = 5
HISTORICAL_TARGET_DAYS = 3650
TODAY_TARGET_RACES = len(VENUES) * len(RACES_PER_DAY)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_STATUS_PATH = PROJECT_ROOT / "docs" / "PROJECT_STATUS.md"
REMOTE_EVAL_STATUS_NAME = "remote_eval_status.json"
TEMPLATE_DIR = Path(__file__).with_name("templates")


def _load_template(name: str) -> str:
    return (TEMPLATE_DIR / name).read_text(encoding="utf-8")


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


def stored_start_time(value: str | None) -> datetime | None:
    return parse_jst(value)


def estimated_deadline_from_start(start: datetime | None) -> datetime | None:
    if start is None:
        return None
    return start - timedelta(minutes=START_TO_DEADLINE_MINUTES)


def time_fields_from_stored_start(
    stored_value: str | None,
    *,
    now: datetime,
    before_minutes: int = 5,
    result_rows: int = 0,
) -> dict[str, object]:
    start_at = stored_start_time(stored_value)
    deadline_at = estimated_deadline_from_start(start_at)
    buy_until_at = deadline_at - timedelta(minutes=before_minutes) if deadline_at else None
    if result_rows >= 3:
        status = "確定"
    elif not start_at:
        status = "時刻未取得"
    elif deadline_at and now >= deadline_at:
        status = "締切後"
    elif buy_until_at and now > buy_until_at:
        status = "T-5超過"
    else:
        status = "候補"
    return {
        "stored_schedule_at": iso(start_at),
        "deadline_at": iso(deadline_at),
        "race_time_at": iso(start_at),
        "buy_until_at": iso(buy_until_at),
        "minutes_to_deadline": minutes_between(now, deadline_at),
        "minutes_to_race_time": minutes_between(now, start_at),
        "minutes_to_buy_until": minutes_between(now, buy_until_at),
        "time_status": status,
        "time_basis": "stored_deadline_at_is_race_start",
    }


def dict_row(row: sqlite3.Row) -> dict[str, Any]:
    return {key: row[key] for key in row.keys()}


def required(query: dict[str, list[str]], key: str) -> str:
    values = query.get(key)
    if not values or not values[0]:
        raise ValueError(f"missing query parameter: {key}")
    return values[0]


def send_html(handler: BaseHTTPRequestHandler, body: str, status: int = 200) -> None:
    payload = body.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "text/html; charset=utf-8")
    handler.send_header("Content-Length", str(len(payload)))
    handler.end_headers()
    try:
        handler.wfile.write(payload)
    except BrokenPipeError:
        return


def send_json(handler: BaseHTTPRequestHandler, value: Any, status: int = 200) -> None:
    payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Content-Length", str(len(payload)))
    handler.end_headers()
    try:
        handler.wfile.write(payload)
    except BrokenPipeError:
        return



def query_race_date(db_path: Path, query: dict[str, list[str]]) -> str:
    values = query.get("date")
    if values and values[0]:
        return values[0]
    return default_race_date(db_path)


def default_race_date(db_path: Path) -> str:
    now = time.monotonic()
    today = now_jst().date().isoformat()
    cached = _DEFAULT_DATE_CACHE.get(db_path)
    if cached and cached[1] == today and now - cached[0] < 300.0:
        return cached[1]
    selected = today
    try:
        with connect(db_path) as conn:
            row = conn.execute(
                """
                SELECT MAX(race_date)
                FROM races
                WHERE race_date <= ? AND deadline_at IS NOT NULL
                """,
                (today,),
            ).fetchone()
            if row and row[0]:
                selected = str(row[0])
    except Exception:
        selected = today
    _DEFAULT_DATE_CACHE[db_path] = (now, selected)
    return selected


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
    return {"race_id": race_id, "combination": combo, "trend": [dict_row(row) for row in trend]}


def predictions_model_rank(db_path: Path, query: dict[str, list[str]]) -> dict[str, Any]:
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
                ORDER BY probability DESC, COALESCE(expected_value, 0) DESC, combination
                LIMIT 120
                """,
                (race_id, latest["generated_at"]),
            ).fetchall()
    return {
        "race": dict_row(race) if race else None,
        "entries": [dict_row(row) for row in entries],
        "predictions": [dict_row(row) for row in pred_rows],
        "prediction_rank_basis": "model_probability",
    }


def latest_prediction_rows_by_probability(
    conn: sqlite3.Connection,
    race_id: str,
    *,
    limit: int,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    return _latest_prediction_rows(
        conn,
        race_id,
        limit=limit,
        order_sql="probability DESC, COALESCE(expected_value, 0) DESC, combination",
    )


def _latest_prediction_rows(
    conn: sqlite3.Connection,
    race_id: str,
    *,
    limit: int,
    order_sql: str,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    latest = conn.execute(
        "SELECT generated_at FROM predictions WHERE race_id = ? ORDER BY generated_at DESC LIMIT 1",
        (race_id,),
    ).fetchone()
    if not latest:
        return None, []
    rows = conn.execute(
        f"""
        SELECT combination, probability, odds, expected_value, generated_at
        FROM predictions
        WHERE race_id = ? AND generated_at = ?
        ORDER BY {order_sql}
        LIMIT ?
        """,
        (race_id, latest["generated_at"], limit),
    ).fetchall()
    mapped = [dict_row(row) for row in rows]
    return (mapped[0] if mapped else None), mapped


def accuracy_model_rank(db_path: Path, query: dict[str, list[str]]) -> dict[str, Any]:
    race_date = query_race_date(db_path, query)
    with connect(db_path) as conn:
        race_rows = conn.execute(
            """
            SELECT r.race_id
            FROM races r
            WHERE r.race_date = ?
              AND (SELECT COUNT(*) FROM race_results rr WHERE rr.race_id = r.race_id AND rr.rank IS NOT NULL) >= 3
              AND EXISTS (SELECT 1 FROM predictions p WHERE p.race_id = r.race_id)
            """,
            (race_date,),
        ).fetchall()
        evaluated = 0
        winner_hits = 0
        trifecta_top1_hits = 0
        trifecta_top5_hits = 0
        for race in race_rows:
            rid = race["race_id"]
            actual_rows = conn.execute(
                """
                SELECT lane, rank
                FROM race_results
                WHERE race_id = ? AND rank IS NOT NULL
                ORDER BY rank
                LIMIT 3
                """,
                (rid,),
            ).fetchall()
            if len(actual_rows) < 3:
                continue
            actual_combo = "-".join(str(row["lane"]) for row in actual_rows)
            actual_winner = actual_combo.split("-")[0]
            _, pred_rows = latest_prediction_rows_by_probability(conn, rid, limit=5)
            if not pred_rows:
                continue
            top = pred_rows[0]["combination"]
            top5 = [row["combination"] for row in pred_rows]
            evaluated += 1
            winner_hits += 1 if top.split("-")[0] == actual_winner else 0
            trifecta_top1_hits += 1 if top == actual_combo else 0
            trifecta_top5_hits += 1 if actual_combo in top5 else 0
    return {
        "date": race_date,
        "evaluated": evaluated,
        "winner_top1_accuracy": winner_hits / evaluated if evaluated else None,
        "trifecta_top1_hit_rate": trifecta_top1_hits / evaluated if evaluated else None,
        "trifecta_top5_hit_rate": trifecta_top5_hits / evaluated if evaluated else None,
        "prediction_rank_basis": "model_probability",
    }


def buy_score(item: dict[str, Any]) -> float:
    top = item.get("buy_prediction") or item.get("top_prediction") or {}
    if top.get("expected_value") is not None:
        return float(top["expected_value"])
    return float(top.get("probability") or 0.0)


def result_summary(conn: sqlite3.Connection, race_id: str) -> dict[str, Any]:
    ranks = conn.execute(
        """
        SELECT rank, lane
        FROM race_results
        WHERE race_id = ? AND rank BETWEEN 1 AND 3
        ORDER BY rank
        """,
        (race_id,),
    ).fetchall()
    combination = "-".join(str(row["lane"]) for row in ranks) if len(ranks) == 3 else None
    payout = None
    popularity = None
    if combination:
        payout_row = conn.execute(
            """
            SELECT payout_yen, popularity
            FROM payouts
            WHERE race_id = ? AND bet_type = '3連単' AND combination = ?
            LIMIT 1
            """,
            (race_id, combination),
        ).fetchone()
        if payout_row:
            payout = payout_row["payout_yen"]
            popularity = payout_row["popularity"]
    return {
        "result_combination": combination,
        "trifecta_payout_yen": payout,
        "trifecta_popularity": popularity,
    }


def base_progress(db_path: Path, query: dict[str, list[str]]) -> dict[str, Any]:
    race_date = query_race_date(db_path, query)
    with connect(db_path) as conn:
        program_days = conn.execute(
            "SELECT COUNT(DISTINCT race_date) FROM raw_files WHERE kind = 'program' AND race_date < ?",
            (race_date,),
        ).fetchone()[0]
        result_days = conn.execute(
            "SELECT COUNT(DISTINCT race_date) FROM raw_files WHERE kind = 'result' AND race_date < ?",
            (race_date,),
        ).fetchone()[0]
        historical_races = conn.execute(
            "SELECT COUNT(*) FROM races WHERE race_date < ?",
            (race_date,),
        ).fetchone()[0]
        historical_results = conn.execute(
            "SELECT COUNT(*) FROM races WHERE race_date < ? AND status = 'final'",
            (race_date,),
        ).fetchone()[0]
        day_rows = [dict_row(row) for row in _day_metric_rows(conn, race_date, include_predictions=False)]
    today_counts = {
        "races": len(day_rows),
        "racelists": sum(1 for row in day_rows if int(row.get("entries") or 0) == 6),
        "odds_races": sum(1 for row in day_rows if int(row.get("odds_snapshots") or 0) > 0),
        "finals": sum(1 for row in day_rows if int(row.get("result_rows") or 0) >= 3),
    }
    return {
        "date": race_date,
        "historical": {
            "target_days": HISTORICAL_TARGET_DAYS,
            "program_days": int(program_days or 0),
            "result_days": int(result_days or 0),
            "program_remaining_days": max(0, HISTORICAL_TARGET_DAYS - int(program_days or 0)),
            "result_remaining_days": max(0, HISTORICAL_TARGET_DAYS - int(result_days or 0)),
            "races": int(historical_races or 0),
            "result_races": int(historical_results or 0),
        },
        "today": {
            "target_races": TODAY_TARGET_RACES,
            **today_counts,
            "race_remaining": max(0, TODAY_TARGET_RACES - today_counts["races"]),
            "racelist_remaining": max(0, TODAY_TARGET_RACES - today_counts["racelists"]),
            "odds_remaining": max(0, TODAY_TARGET_RACES - today_counts["odds_races"]),
            "final_remaining": max(0, TODAY_TARGET_RACES - today_counts["finals"]),
        },
    }

_CACHE_TTL_SECONDS = 15.0
_SUMMARY_CACHE: dict[Path, tuple[float, dict[str, Any]]] = {}
_PROGRESS_CACHE: dict[tuple[Path, str], tuple[float, dict[str, Any]]] = {}
_ACCURACY_CACHE: dict[tuple[Path, str], tuple[float, dict[str, Any]]] = {}
_BACKTEST_CACHE: dict[Path, tuple[float, int, dict[str, Any]]] = {}
_MODEL_REPORT_CACHE: dict[Path, tuple[float, dict[str, Any]]] = {}
_ROADMAP_CACHE: dict[Path, tuple[float, dict[str, Any]]] = {}
_DEFAULT_DATE_CACHE: dict[Path, tuple[float, str]] = {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="BOAT RACE AI dashboard with staged loading.")
    parser.add_argument("--db", default="data/boatrace.sqlite")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10001)
    parser.add_argument("--backtest", default="data/models/backtest_no_odds_v8.json")
    args = parser.parse_args(argv)

    init_db(args.db)
    _ensure_dashboard_indexes(Path(args.db))
    handler = make_handler(Path(args.db), Path(args.backtest) if args.backtest else None)
    print(f"Serving BOAT RACE AI Dashboard on http://{args.host}:{args.port}", flush=True)
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
                    send_json(self, summary_cached(db_path))
                elif parsed.path == "/api/venues":
                    send_json(self, venue_cards_fast(db_path, query))
                elif parsed.path == "/api/day":
                    send_json(self, day_overview_fast(db_path, query))
                elif parsed.path == "/api/guide":
                    send_json(self, purchase_guide_fast(db_path, query))
                elif parsed.path == "/api/live-wipe":
                    send_json(self, live_wipe_fast(db_path, query))
                elif parsed.path == "/api/progress":
                    send_json(self, progress_active_fast(db_path, query))
                elif parsed.path == "/api/predictions":
                    send_json(self, predictions_with_names(db_path, query))
                elif parsed.path == "/api/odds":
                    send_json(self, odds(db_path, query))
                elif parsed.path == "/api/backtest":
                    send_json(self, backtest_cached(backtest_path))
                elif parsed.path == "/api/accuracy":
                    send_json(self, accuracy_cached(db_path, query))
                elif parsed.path == "/reports/models":
                    send_html(self, MODEL_REPORT_HTML)
                elif parsed.path == "/api/reports/model-performance":
                    send_json(self, model_performance_report(db_path, query))
                elif parsed.path == "/reports/roadmap":
                    send_html(self, ROADMAP_REPORT_HTML)
                elif parsed.path == "/api/reports/roadmap-status":
                    send_json(self, roadmap_status(db_path, query))
                elif parsed.path == "/api/archive/overview":
                    send_json(self, archive_overview(db_path, query))
                elif parsed.path == "/api/archive/today":
                    send_json(self, archive_today(db_path, query))
                elif parsed.path == "/api/archive/history":
                    send_json(self, archive_history(db_path, query))
                elif parsed.path == "/api/archive/stats":
                    send_json(self, archive_stats(db_path, query))
                else:
                    self.send_error(404)
            except Exception as exc:
                send_json(self, {"error": str(exc)}, status=500)

        def log_message(self, fmt: str, *args: Any) -> None:
            return

    return Handler


def _ensure_dashboard_indexes(db_path: Path) -> None:
    with connect(db_path) as conn:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_races_date_jcd_deadline ON races(race_date, jcd, deadline_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_races_date_deadline ON races(race_date, deadline_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_race_generated_prob ON predictions(race_id, generated_at, probability)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_race_generated_ev ON predictions(race_id, generated_at, expected_value)")
        conn.commit()


def summary_cached(db_path: Path) -> dict[str, Any]:
    now = time.monotonic()
    cached = _SUMMARY_CACHE.get(db_path)
    if cached and now - cached[0] < 300.0:
        return cached[1]
    with connect(db_path) as conn:
        payload = {
            "races": None,
            "entries": None,
            "results": None,
            "odds_snapshots": _scalar(conn, "SELECT COUNT(*) FROM odds_snapshots"),
            "predictions": _scalar(conn, "SELECT COUNT(DISTINCT race_id) FROM predictions"),
            "latest_prediction": _scalar(conn, "SELECT MAX(generated_at) FROM predictions"),
            "summary_scope": "lightweight",
        }
    _SUMMARY_CACHE[db_path] = (now, payload)
    return payload


def backtest_cached(path: Path | None) -> dict[str, Any]:
    if not path or not path.exists():
        return {"available": False}
    stat = path.stat()
    cached = _BACKTEST_CACHE.get(path)
    if cached and cached[1] == stat.st_mtime_ns:
        return cached[2]
    payload = {"available": True, **json.loads(path.read_text(encoding="utf-8"))}
    _BACKTEST_CACHE[path] = (time.monotonic(), stat.st_mtime_ns, payload)
    return payload


def accuracy_cached(db_path: Path, query: dict[str, list[str]]) -> dict[str, Any]:
    race_date = query_race_date(db_path, query)
    key = (db_path, race_date)
    now = time.monotonic()
    cached = _ACCURACY_CACHE.get(key)
    if cached and now - cached[0] < 60.0:
        return cached[1]
    payload = accuracy_model_rank(db_path, query)
    _ACCURACY_CACHE[key] = (now, payload)
    return payload




def model_performance_report(db_path: Path, query: dict[str, list[str]]) -> dict[str, Any]:
    model_dir = Path(query.get("model_dir", [str(db_path.parent / "models")])[0])
    now = time.monotonic()
    cached = _MODEL_REPORT_CACHE.get(model_dir)
    if cached and now - cached[0] < 60.0:
        return cached[1]

    backtests: list[dict[str, Any]] = []
    fold_metrics: list[dict[str, Any]] = []
    bankroll: list[dict[str, Any]] = []
    bankroll_daily: dict[str, list[dict[str, Any]]] = {}
    sweeps: list[dict[str, Any]] = []
    feature_diagnostics: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for path in sorted(model_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append({"file": path.name, "error": str(exc)})
            continue
        label = _report_label(path, data)
        if _is_feature_correlation_result(data):
            feature_diagnostics.append(_feature_correlation_summary(path, label, data))
        if _is_bankroll_result(data):
            bankroll.append(_bankroll_summary(path, label, data))
            daily_rows = _daily_report_rows(data.get("daily") or [])
            if daily_rows:
                bankroll_daily[label] = daily_rows
        elif _is_backtest_result(data):
            backtests.append(_backtest_summary(path, label, data))
            for fold in data.get("folds") or []:
                fold_metrics.append(_fold_report_row(label, fold))
        if isinstance(data.get("results"), list):
            for row in data["results"]:
                if isinstance(row, dict):
                    sweeps.append(_sweep_report_row(path, row))
                    for fold in row.get("folds") or []:
                        fold_metrics.append(_fold_report_row(str(row.get("variant") or path.stem), fold))

    remote_evaluations = _read_remote_eval_status(db_path.parent / REMOTE_EVAL_STATUS_NAME)
    feature_diagnostics.extend(_remote_feature_correlation_summaries(remote_evaluations))
    bankroll.extend(_remote_bankroll_report_summaries(remote_evaluations))
    backtests.sort(key=lambda item: (item.get("generated_at") or "", item["name"]))
    bankroll.sort(key=lambda item: (item.get("generated_at") or "", item["name"]))
    sweeps.sort(key=lambda item: (item.get("entry_log_loss") is None, item.get("entry_log_loss") or 999, item["name"]))
    feature_diagnostics.sort(key=lambda item: (item.get("generated_at") or "", item["file"]))
    payload = {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "model_dir": str(model_dir),
        "backtests": backtests,
        "fold_metrics": fold_metrics,
        "bankroll": bankroll,
        "bankroll_daily": bankroll_daily,
        "sweeps": sweeps,
        "feature_diagnostics": feature_diagnostics,
        "evaluation_jobs": _remote_evaluation_job_summaries(remote_evaluations),
        "remote_generated_at": remote_evaluations.get("generated_at"),
        "errors": errors,
    }
    _MODEL_REPORT_CACHE[model_dir] = (now, payload)
    return payload



def _remote_evaluation_job_summaries(remote_evaluations: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    jobs = remote_evaluations.get("jobs") if isinstance(remote_evaluations, dict) else []
    for job in jobs or []:
        result = job.get("result") or {}
        metrics = {**(result.get("base_metrics") or {}), **(result.get("metrics") or {})}
        command = str((job.get("process") or {}).get("cmd") or "")
        expected_match = re.search(r"(?:^|\s)--folds\s+(\d+)(?:\s|$)", command)
        expected_folds = int(expected_match.group(1)) if expected_match else None
        completed_folds = 0
        for line in job.get("log_tail") or []:
            try:
                parsed_line = json.loads(line)
                if isinstance(parsed_line, dict):
                    completed_folds = max(completed_folds, int(parsed_line.get("fold") or 0))
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
        if job.get("status") == "完了" and expected_folds:
            completed_folds = expected_folds
        rows.append({
            "name": job.get("name"),
            "milestone": job.get("milestone"),
            "kind": job.get("kind"),
            "status": job.get("status"),
            "running": bool(job.get("running")),
            "elapsed": (job.get("process") or {}).get("elapsed"),
            "completed_folds": completed_folds or None,
            "expected_folds": expected_folds,
            "roi": _float_or_none(metrics.get("roi")),
            "profit_yen": metrics.get("profit_yen"),
            "evaluated_races": metrics.get("evaluated_races"),
            "entry_log_loss": _float_or_none(metrics.get("entry_log_loss")),
            "winner_top1_accuracy": _float_or_none(metrics.get("winner_top1_accuracy")),
            "trifecta_top5_hit_rate": _float_or_none(metrics.get("trifecta_top5_hit_rate")),
            "real_odds_races": metrics.get("real_odds_races"),
            "skipped_no_real_odds": metrics.get("skipped_no_real_odds"),
            "error": (job.get("log_tail") or [])[-1] if job.get("status") == "失敗" else None,
        })
    return rows

def _is_backtest_result(data: dict[str, Any]) -> bool:
    return "entry_log_loss" in data or "winner_top1_accuracy" in data or "trifecta_top5_hit_rate" in data


def _is_bankroll_result(data: dict[str, Any]) -> bool:
    return "roi" in data and ("stake_yen" in data or "return_yen" in data or "daily" in data)


def _is_feature_correlation_result(data: dict[str, Any]) -> bool:
    return "top_numeric_abs_correlation" in data or "feature_family_summary" in data or "suspect_features" in data


def _feature_correlation_summary(path: Path, label: str, data: dict[str, Any]) -> dict[str, Any]:
    roi_link = data.get("roi_link") or {}
    families = data.get("feature_family_summary") or []
    suspects = data.get("suspect_features") or []
    return {
        "name": label,
        "file": path.name,
        "generated_at": data.get("generated_at"),
        "feature_set": data.get("feature_set"),
        "examples": data.get("examples"),
        "races": data.get("races"),
        "global_win_rate": _float_or_none(data.get("global_win_rate")),
        "roi_status": roi_link.get("status"),
        "roi": _float_or_none(roi_link.get("roi")),
        "profit_yen": roi_link.get("profit_yen"),
        "suspect_count": len(suspects),
        "family_summary": families[:16],
        "suspect_features": suspects[:24],
        "coefficient_alignment": (data.get("coefficient_alignment") or [])[:16],
        "action_items": (data.get("action_items") or data.get("diagnosis") or [])[:10],
    }


def _remote_feature_correlation_summaries(remote_evaluations: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for job in (remote_evaluations.get("jobs") if isinstance(remote_evaluations, dict) else []) or []:
        if job.get("kind") != "feature_correlation":
            continue
        result = job.get("result") or {}
        families = result.get("feature_family_summary") or []
        suspects = result.get("suspect_features") or []
        if not families and not suspects:
            continue
        metrics = result.get("metrics") or {}
        roi_link = result.get("roi_link") or {}
        rows.append({
            "name": job.get("name") or "remote_feature_correlation",
            "file": result.get("file") or job.get("output"),
            "generated_at": result.get("modified_at") or remote_evaluations.get("generated_at"),
            "feature_set": None,
            "examples": metrics.get("examples"),
            "races": metrics.get("races"),
            "global_win_rate": _float_or_none(metrics.get("global_win_rate")),
            "roi_status": roi_link.get("status"),
            "roi": _float_or_none(roi_link.get("roi")),
            "profit_yen": roi_link.get("profit_yen"),
            "suspect_count": len(suspects),
            "family_summary": families[:16],
            "suspect_features": suspects[:24],
            "coefficient_alignment": (result.get("coefficient_alignment") or [])[:16],
            "action_items": (result.get("action_items") or [])[:10],
        })
    return rows


def _report_label(path: Path, data: dict[str, Any]) -> str:
    model = str(data.get("model") or "").strip()
    if model:
        return model
    feature_set = str(data.get("feature_set") or "").strip()
    if feature_set:
        return feature_set
    return path.stem


def _backtest_summary(path: Path, label: str, data: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": label,
        "file": path.name,
        "generated_at": data.get("generated_at"),
        "feature_set": data.get("feature_set"),
        "include_odds": data.get("include_odds"),
        "evaluated_races": data.get("evaluated_races"),
        "entry_log_loss": _float_or_none(data.get("entry_log_loss")),
        "entry_brier": _float_or_none(data.get("entry_brier")),
        "winner_top1_accuracy": _float_or_none(data.get("winner_top1_accuracy")),
        "trifecta_top1_hit_rate": _float_or_none(data.get("trifecta_top1_hit_rate")),
        "trifecta_top5_hit_rate": _float_or_none(data.get("trifecta_top5_hit_rate")),
    }


def _bankroll_summary(path: Path, label: str, data: dict[str, Any]) -> dict[str, Any]:
    policy = data.get("policy") or {}
    return {
        "name": label,
        "file": path.name,
        "generated_at": data.get("generated_at"),
        "feature_set": data.get("feature_set") or policy.get("feature_set"),
        "model": data.get("model") or policy.get("model"),
        "daily_budget_yen": policy.get("daily_budget_yen"),
        "stake_model": policy.get("stake_model"),
        "evaluated_races": data.get("evaluated_races"),
        "race_days": data.get("race_days"),
        "selected_races": data.get("selected_races"),
        "tickets": data.get("tickets"),
        "candidate_tickets": data.get("candidate_tickets"),
        "stake_yen": data.get("stake_yen"),
        "return_yen": data.get("return_yen"),
        "profit_yen": data.get("profit_yen"),
        "roi": _float_or_none(data.get("roi")),
        "ticket_hit_rate": _float_or_none(data.get("ticket_hit_rate")),
        "race_hit_rate": _float_or_none(data.get("race_hit_rate")),
        "winning_days": data.get("winning_days"),
        "losing_days": data.get("losing_days"),
        "budget_utilization": _float_or_none(data.get("budget_utilization")),
        "avg_stake_yen_per_ticket": _float_or_none(data.get("avg_stake_yen_per_ticket")),
        "avg_tickets_per_selected_race": _float_or_none(data.get("avg_tickets_per_selected_race")),
        "max_drawdown_yen": data.get("max_drawdown_yen"),
        "ticket_roi_attribution": _compact_ticket_roi_attribution(data.get("ticket_roi_attribution")),
    }


def _compact_ticket_roi_attribution(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    return {
        "method": value.get("method"),
        "diagnosis": value.get("diagnosis"),
        "minimum_evidence": value.get("minimum_evidence") or {},
        "top_signals": (value.get("top_signals") or [])[:16],
        "fold_stability": value.get("fold_stability") or {},
    }


def _remote_bankroll_report_summaries(remote_evaluations: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for job in (remote_evaluations.get("jobs") if isinstance(remote_evaluations, dict) else []) or []:
        if not str(job.get("kind") or "").startswith("bankroll"):
            continue
        result = job.get("result") or {}
        metrics = result.get("metrics") or {}
        if metrics.get("roi") is None:
            continue
        attribution = result.get("ticket_roi_attribution")
        rows.append(
            {
                "name": job.get("name") or result.get("file") or "remote_bankroll",
                "file": result.get("file") or job.get("output"),
                "generated_at": result.get("modified_at") or remote_evaluations.get("generated_at"),
                "feature_set": None,
                "model": "remote",
                "daily_budget_yen": None,
                "stake_model": None,
                "evaluated_races": metrics.get("evaluated_races"),
                "race_days": metrics.get("race_days"),
                "selected_races": metrics.get("selected_races"),
                "tickets": metrics.get("tickets"),
                "candidate_tickets": metrics.get("candidate_tickets"),
                "stake_yen": metrics.get("stake_yen"),
                "return_yen": metrics.get("return_yen"),
                "profit_yen": metrics.get("profit_yen"),
                "roi": _float_or_none(metrics.get("roi")),
                "ticket_hit_rate": _float_or_none(metrics.get("ticket_hit_rate")),
                "race_hit_rate": _float_or_none(metrics.get("race_hit_rate")),
                "winning_days": metrics.get("winning_days"),
                "losing_days": metrics.get("losing_days"),
                "budget_utilization": _float_or_none(metrics.get("budget_utilization")),
                "avg_stake_yen_per_ticket": None,
                "avg_tickets_per_selected_race": None,
                "max_drawdown_yen": metrics.get("max_drawdown_yen"),
                "ticket_roi_attribution": attribution,
                "remote": True,
            }
        )
    return rows


def _daily_report_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        tickets = int(row.get("tickets") or 0)
        out.append(
            {
                "date": row.get("race_date"),
                "evaluated_races": row.get("evaluated_races"),
                "tickets": tickets,
                "races_bet": row.get("races_bet"),
                "stake_yen": row.get("stake_yen"),
                "return_yen": row.get("return_yen"),
                "profit_yen": row.get("profit_yen"),
                "cumulative_profit_yen": row.get("cumulative_profit_yen"),
                "roi": _float_or_none(row.get("roi")),
                "budget_used_fraction": _float_or_none(row.get("budget_used_fraction")),
                "ticket_hit_rate": (float(row.get("hit_tickets") or 0) / tickets) if tickets else None,
            }
        )
    return out


def _sweep_report_row(path: Path, row: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": str(row.get("variant") or path.stem),
        "file": path.name,
        "evaluated_races": row.get("evaluated_races"),
        "entry_log_loss": _float_or_none(row.get("entry_log_loss")),
        "entry_brier": _float_or_none(row.get("entry_brier")),
        "winner_top1_accuracy": _float_or_none(row.get("winner_top1_accuracy")),
        "trifecta_top1_hit_rate": _float_or_none(row.get("trifecta_top1_hit_rate")),
        "trifecta_top5_hit_rate": _float_or_none(row.get("trifecta_top5_hit_rate")),
    }


def _fold_report_row(model: str, fold: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": model,
        "fold": fold.get("fold"),
        "train_races": fold.get("train_races"),
        "test_races": fold.get("test_races"),
        "entry_log_loss": _float_or_none(fold.get("entry_log_loss")),
        "entry_brier": _float_or_none(fold.get("entry_brier")),
    }


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def venue_cards_fast(db_path: Path, query: dict[str, list[str]]) -> dict[str, Any]:
    race_date = query_race_date(db_path, query)
    now = now_jst()
    with connect(db_path) as conn:
        rows = _day_metric_rows(conn, race_date)

    by_code: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        item = dict_row(row)
        by_code.setdefault(str(item["jcd"]).zfill(2), []).append(item)

    cards = []
    for venue in VENUES:
        venue_rows = by_code.get(venue.code, [])
        active_rows = [row for row in venue_rows if _is_active_row(row)]
        active_races = len(active_rows)
        racelists = sum(1 for row in active_rows if int(row.get("entries") or 0) == 6)
        odds_count = sum(int(row.get("odds_snapshots") or 0) for row in active_rows)
        finals = sum(1 for row in active_rows if int(row.get("result_rows") or 0) >= 3)
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

        next_deadline = None
        next_start = None
        next_rno = None
        for row in sorted(active_rows, key=lambda item: (item.get("deadline_at") is None, item.get("deadline_at") or "", item.get("rno") or 0)):
            if int(row.get("result_rows") or 0) >= 3:
                continue
            start_at = stored_start_time(row.get("deadline_at"))
            deadline_at = estimated_deadline_from_start(start_at)
            if deadline_at and deadline_at >= now:
                next_deadline = deadline_at
                next_start = start_at
                next_rno = int(row.get("rno") or 0)
                break

        latest_odds_values = [parse_any_time(str(row.get("latest_odds_at") or "")) for row in active_rows if row.get("latest_odds_at")]
        latest_odds = max((value for value in latest_odds_values if value), default=None)
        latest_prediction = max((str(row.get("latest_prediction")) for row in active_rows if row.get("latest_prediction")), default=None)
        cards.append(
            {
                "code": venue.code,
                "name": venue.name,
                "status": status,
                "races": active_races,
                "raw_races": len(venue_rows),
                "racelists": racelists,
                "odds_snapshots": odds_count,
                "finals": finals,
                "latest_prediction": latest_prediction,
                "latest_odds_at": iso(latest_odds),
                "next_rno": next_rno,
                "next_deadline_at": iso(next_deadline),
                "next_race_time_at": iso(next_start),
                "minutes_to_next_deadline": minutes_between(now, next_deadline),
            }
        )
    return {"date": race_date, "now_jst": iso(now), "venues": cards}


def day_overview_fast(db_path: Path, query: dict[str, list[str]]) -> dict[str, Any]:
    race_date = query_race_date(db_path, query)
    jcd = query.get("jcd", [None])[0]
    lite = (query.get("lite", ["0"])[0] or "0").lower() in {"1", "true", "yes"}
    now = now_jst()
    with connect(db_path) as conn:
        rows = _day_metric_rows(conn, race_date, jcd=jcd, include_predictions=not lite)
    races = [_race_payload_from_row(row, now=now, before_minutes=5) for row in rows if _is_active_row(row)]
    return {"date": race_date, "now_jst": iso(now), "races": races}


def purchase_guide_fast(db_path: Path, query: dict[str, list[str]]) -> dict[str, Any]:
    race_date = query_race_date(db_path, query)
    before_minutes = int(query.get("before_minutes", ["5"])[0])
    limit = int(query.get("limit", ["16"])[0])
    finished_limit = int(query.get("finished_limit", ["4"])[0])
    now = now_jst()

    with connect(db_path) as conn:
        rows = [row for row in _day_metric_rows(conn, race_date, include_predictions=True) if _is_active_row(row)]
        candidates = []
        for row in rows:
            item = _race_payload_from_row(row, now=now, before_minutes=before_minutes)
            if int(item.get("entries") or 0) != 6:
                continue
            if int(item.get("result_rows") or 0) >= 3:
                continue
            buy_until = stored_start_time(item.get("buy_until_at"))
            if not buy_until or now > buy_until:
                continue
            if item.get("top_prediction"):
                candidates.append(item)

        candidates.sort(key=lambda item: (item["buy_until_at"] or "", -buy_score(item)))
        if candidates:
            first_cutoff = candidates[0]["buy_until_at"]
            same_cutoff = [item for item in candidates if item["buy_until_at"] == first_cutoff]
            later = [item for item in candidates if item["buy_until_at"] != first_cutoff]
            candidates = sorted(same_cutoff, key=buy_score, reverse=True) + later

        closed = []
        for row in sorted(rows, key=lambda item: item["deadline_at"] or "", reverse=True):
            start_at = stored_start_time(row["deadline_at"])
            if not start_at or start_at > now:
                continue
            item = _race_payload_from_row(row, now=now, before_minutes=before_minutes)
            if int(item.get("entries") or 0) != 6:
                continue
            if int(item.get("result_rows") or 0) >= 3:
                item.update(result_summary(conn, row["race_id"]))
                _attach_prediction_hits(conn, item)
            else:
                item.update(
                    {
                        "time_status": "結果待",
                        "result_combination": None,
                        "trifecta_payout_yen": None,
                        "trifecta_popularity": None,
                        "top_hit": False,
                        "top5_hit": False,
                    }
                )
            closed.append(item)
            if len(closed) >= finished_limit:
                break

    return {
        "date": race_date,
        "now_jst": iso(now),
        "before_minutes": before_minutes,
        "candidates": candidates[:limit],
        "finished": closed,
        "prediction_rank_basis": "model_probability",
        "time_basis": "stored_deadline_at_is_race_start",
    }


def _latest_live_window_row(rows: list[Any], *, now: datetime, window_minutes: int = 5) -> Any | None:
    eligible = []
    for row in rows:
        start_at = stored_start_time(row["deadline_at"])
        if not start_at:
            continue
        elapsed = (now - start_at).total_seconds()
        if 0 <= elapsed < window_minutes * 60:
            eligible.append((start_at, row))
    return max(eligible, key=lambda item: item[0], default=(None, None))[1]


def live_wipe_fast(db_path: Path, query: dict[str, list[str]]) -> dict[str, Any]:
    race_date = query_race_date(db_path, query)
    now = now_jst()
    with connect(db_path) as conn:
        rows = [row for row in _day_metric_rows(conn, race_date, include_predictions=True) if _is_active_row(row)]
        row = _latest_live_window_row(rows, now=now, window_minutes=5)
        if row is not None:
            start_at = stored_start_time(row["deadline_at"])
            item = _race_payload_from_row(row, now=now, before_minutes=5)
            item.update(
                {
                    "minutes_since_start": int((now - start_at).total_seconds() // 60) if start_at else None,
                    "live_window_seconds": 300,
                    "live_url": f"https://race.boatcast.jp/replay?jo={str(row['jcd']).zfill(2)}",
                    "live_embed_url": f"https://race.boatcast.jp/replay?jo={str(row['jcd']).zfill(2)}",
                    "official_url": race_page_url("racelist", date.fromisoformat(str(row["race_date"])), str(row["jcd"]).zfill(2), int(row["rno"])),
                    "official_result_url": (
                        f"https://www.boatrace.jp/owpc/pc/race/raceresult"
                        f"?rno={int(row['rno'])}&jcd={str(row['jcd']).zfill(2)}&hd={ymd(date.fromisoformat(str(row['race_date'])))}"
                    ),
                }
            )
            if int(item.get("result_rows") or 0) >= 3:
                item.update(result_summary(conn, row["race_id"]))
                _attach_prediction_hits(conn, item)
            return {"date": race_date, "now_jst": iso(now), "active": True, "race": item}
    return {"date": race_date, "now_jst": iso(now), "active": False, "race": None}

def progress_active_fast(db_path: Path, query: dict[str, list[str]]) -> dict[str, Any]:
    race_date = query_race_date(db_path, query)
    cache_key = (db_path, race_date)
    now_mono = time.monotonic()
    cached = _PROGRESS_CACHE.get(cache_key)
    if cached and now_mono - cached[0] < 300.0:
        return cached[1]
    payload = base_progress(db_path, query)
    with connect(db_path) as conn:
        rows = [dict_row(row) for row in _day_metric_rows(conn, race_date, include_predictions=False)]
    active = [row for row in rows if _is_active_row(row)]
    active_counts = {
        "races": len(active),
        "racelists": sum(1 for row in active if int(row.get("entries") or 0) == 6),
        "odds_races": sum(1 for row in active if int(row.get("odds_snapshots") or 0) > 0),
        "finals": sum(1 for row in active if int(row.get("result_rows") or 0) >= 3),
    }
    payload["today"].update(
        {
            "target_races": active_counts["races"],
            **active_counts,
            "race_remaining": 0,
            "racelist_remaining": max(0, active_counts["races"] - active_counts["racelists"]),
            "odds_remaining": max(0, active_counts["races"] - active_counts["odds_races"]),
            "final_remaining": max(0, active_counts["races"] - active_counts["finals"]),
        }
    )
    _PROGRESS_CACHE[cache_key] = (now_mono, payload)
    return payload


def predictions_with_names(db_path: Path, query: dict[str, list[str]]) -> dict[str, Any]:
    payload = predictions_model_rank(db_path, query)
    race = payload.get("race")
    if race:
        now = now_jst()
        with connect(db_path) as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS result_rows
                FROM race_results
                WHERE race_id = ? AND rank IS NOT NULL
                """,
                (race.get("race_id"),),
            ).fetchone()
            result_rows = int(row["result_rows"] or 0) if row else 0
            race.update(time_fields_from_stored_start(race.get("deadline_at"), now=now, before_minutes=5, result_rows=result_rows))
            _fill_missing_racer_names(conn, payload.get("entries") or [])
    payload["time_basis"] = "stored_deadline_at_is_race_start"
    payload["prediction_rank_basis"] = "model_probability"
    return payload


def _day_metric_rows(
    conn: sqlite3.Connection,
    race_date: str,
    *,
    jcd: str | None = None,
    include_predictions: bool = False,
) -> list[sqlite3.Row]:
    params: list[Any] = [race_date]
    jcd_sql = ""
    if jcd:
        jcd_sql = "AND r.jcd = ?"
        params.append(jcd.zfill(2))

    prediction_ctes = ""
    prediction_select = "NULL AS top_combination, NULL AS top_probability, NULL AS top_odds, NULL AS top_expected_value, NULL AS top_generated_at, NULL AS buy_combination, NULL AS buy_probability, NULL AS buy_odds, NULL AS buy_expected_value, NULL AS buy_generated_at"
    prediction_join = ""
    if include_predictions:
        prediction_ctes = """
        , latest_pred AS MATERIALIZED (
          SELECT p.race_id, MAX(p.generated_at) AS generated_at
          FROM predictions p
          JOIN races r ON r.race_id = p.race_id
          WHERE r.race_date = ?
          GROUP BY p.race_id
        ),
        top_rank AS MATERIALIZED (
          SELECT
            p.race_id, p.combination, p.probability, p.odds, p.expected_value, p.generated_at,
            ROW_NUMBER() OVER (
              PARTITION BY p.race_id
              ORDER BY p.probability DESC, COALESCE(p.expected_value, 0) DESC, p.combination
            ) AS rn
          FROM predictions p
          JOIN latest_pred lp ON lp.race_id = p.race_id AND lp.generated_at = p.generated_at
        ),
        buy_rank AS MATERIALIZED (
          SELECT
            p.race_id, p.combination, p.probability, p.odds, p.expected_value, p.generated_at,
            ROW_NUMBER() OVER (
              PARTITION BY p.race_id
              ORDER BY p.expected_value IS NOT NULL DESC, p.expected_value DESC, p.probability DESC, p.combination
            ) AS rn
          FROM predictions p
          JOIN latest_pred lp ON lp.race_id = p.race_id AND lp.generated_at = p.generated_at
        )
        """
        params.append(race_date)
        prediction_select = """
          tr.combination AS top_combination,
          tr.probability AS top_probability,
          tr.odds AS top_odds,
          tr.expected_value AS top_expected_value,
          tr.generated_at AS top_generated_at,
          br.combination AS buy_combination,
          br.probability AS buy_probability,
          br.odds AS buy_odds,
          br.expected_value AS buy_expected_value,
          br.generated_at AS buy_generated_at
        """
        prediction_join = """
        LEFT JOIN top_rank tr ON tr.race_id = a.race_id AND tr.rn = 1
        LEFT JOIN buy_rank br ON br.race_id = a.race_id AND br.rn = 1
        """

    return conn.execute(
        f"""
        WITH base AS MATERIALIZED (
          SELECT
            r.race_id, r.race_date, r.jcd, r.venue_name, r.rno, r.title,
            r.status, r.deadline_at,
            (SELECT COUNT(*) FROM entries e WHERE e.race_id = r.race_id) AS entries,
            (SELECT COUNT(*) FROM odds_snapshots os WHERE os.race_id = r.race_id) AS odds_snapshots,
            (SELECT MAX(captured_at) FROM odds_snapshots os WHERE os.race_id = r.race_id) AS latest_odds_at,
            (SELECT COUNT(*) FROM race_results rr WHERE rr.race_id = r.race_id AND rr.rank IS NOT NULL) AS result_rows,
            (SELECT MAX(generated_at) FROM predictions p WHERE p.race_id = r.race_id) AS latest_prediction
          FROM races r
          WHERE r.race_date = ? {jcd_sql}
        ),
        active AS MATERIALIZED (
          SELECT *,
            CASE
              WHEN deadline_at IS NOT NULL
                OR entries = 6
                OR odds_snapshots > 0
                OR result_rows >= 3
                OR latest_prediction IS NOT NULL
              THEN 1 ELSE 0
            END AS is_active
          FROM base
        )
        {prediction_ctes}
        SELECT
          a.*,
          {prediction_select}
        FROM active a
        {prediction_join}
        ORDER BY a.deadline_at IS NULL, a.deadline_at, a.jcd, a.rno
        """,
        tuple(params),
    ).fetchall()


def _race_payload_from_row(row: sqlite3.Row, *, now, before_minutes: int) -> dict[str, Any]:
    result_rows = int(row["result_rows"] or 0)
    item = {
        "race_id": row["race_id"],
        "race_date": row["race_date"],
        "jcd": row["jcd"],
        "venue_name": row["venue_name"],
        "rno": row["rno"],
        "title": row["title"],
        "status": row["status"],
        "entries": int(row["entries"] or 0),
        "odds_snapshots": int(row["odds_snapshots"] or 0),
        "latest_odds_at": iso(parse_any_time(row["latest_odds_at"])),
        "result_rows": result_rows,
        "latest_prediction": row["latest_prediction"],
        "top_prediction": _prediction_from_row(row, "top"),
        "buy_prediction": _prediction_from_row(row, "buy"),
        "top5": [],
        "buy_top5": [],
        "prediction_rank_basis": "model_probability",
    }
    if item["top_prediction"]:
        item["top5"] = [item["top_prediction"]]
    if item["buy_prediction"]:
        item["buy_top5"] = [item["buy_prediction"]]
    item.update(time_fields_from_stored_start(row["deadline_at"], now=now, before_minutes=before_minutes, result_rows=result_rows))
    return item


def _prediction_from_row(row: sqlite3.Row, prefix: str) -> dict[str, Any] | None:
    combination = row[f"{prefix}_combination"]
    if not combination:
        return None
    return {
        "combination": combination,
        "probability": row[f"{prefix}_probability"],
        "odds": row[f"{prefix}_odds"],
        "expected_value": row[f"{prefix}_expected_value"],
        "generated_at": row[f"{prefix}_generated_at"],
    }


def _attach_prediction_hits(conn: sqlite3.Connection, item: dict[str, Any]) -> None:
    result_combination = item.get("result_combination")
    rows = conn.execute(
        """
        WITH latest AS (
          SELECT generated_at
          FROM predictions
          WHERE race_id = ?
          ORDER BY generated_at DESC
          LIMIT 1
        )
        SELECT combination, probability, odds, expected_value, generated_at
        FROM predictions
        WHERE race_id = ? AND generated_at = (SELECT generated_at FROM latest)
        ORDER BY probability DESC, COALESCE(expected_value, 0) DESC, combination
        LIMIT 5
        """,
        (item["race_id"], item["race_id"]),
    ).fetchall()
    top5 = [dict_row(row) for row in rows]
    item["top5"] = top5
    if top5:
        item["top_prediction"] = top5[0]
    item["top_hit"] = bool(result_combination and top5 and top5[0].get("combination") == result_combination)
    item["top5_hit"] = bool(result_combination and any(pred.get("combination") == result_combination for pred in top5))


def _fill_missing_racer_names(conn: sqlite3.Connection, entries: list[dict[str, Any]]) -> None:
    for entry in entries:
        name = str(entry.get("racer_name") or "").strip()
        no = str(entry.get("racer_no") or "").strip()
        if name and name != no and not name.isdigit():
            continue
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


def _is_active_row(row: sqlite3.Row | dict[str, Any]) -> bool:
    getter = row.get if isinstance(row, dict) else (lambda key, default=None: row[key])
    return bool(getter("is_active", 0))


def _scalar(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> Any:
    row = conn.execute(sql, params).fetchone()
    return row[0] if row else None

# Archive API is kept in this module so the dashboard has no numbered webserver dependency chain.
_DEFAULT_DAYS = 90
_DEFAULT_HISTORY_DAYS = 90
_EQUIPMENT_DAYS = 90
_DEFAULT_LIMIT = 120
_MAX_LIMIT = 500
_MAX_DAYS = 3650
_SCOPES = {"lane", "venue", "rno", "class", "motor", "boat"}


def archive_overview(db_path: Path, query: dict[str, list[str]]) -> dict[str, Any]:
    race_date = _archive_one(query, "date", default_race_date(db_path))
    with connect(db_path) as conn:
        totals = _archive_row(
            conn,
            """
            SELECT
              COUNT(*) AS races,
              MIN(race_date) AS first_date,
              MAX(race_date) AS last_date,
              (SELECT COUNT(*) FROM entries) AS entries,
              (SELECT COUNT(DISTINCT race_id) FROM race_results WHERE rank IS NOT NULL) AS result_races,
              (SELECT COUNT(*) FROM odds_snapshots) AS odds_snapshots,
              (SELECT COUNT(DISTINCT race_id) FROM predictions) AS prediction_races,
              (SELECT COUNT(DISTINCT race_id) FROM beforeinfo) AS beforeinfo_races
            FROM races
            """,
        )
        today = {
            "races": _archive_scalar(conn, "SELECT COUNT(*) FROM races WHERE race_date = ?", (race_date,)),
            "entry_races": _archive_scalar(
                conn,
                "SELECT COUNT(DISTINCT e.race_id) FROM entries e JOIN races r ON r.race_id = e.race_id WHERE r.race_date = ?",
                (race_date,),
            ),
            "result_races": _archive_scalar(
                conn,
                "SELECT COUNT(DISTINCT rr.race_id) FROM race_results rr JOIN races r ON r.race_id = rr.race_id WHERE r.race_date = ? AND rr.rank IS NOT NULL",
                (race_date,),
            ),
            "odds_races": _archive_scalar(
                conn,
                "SELECT COUNT(DISTINCT os.race_id) FROM odds_snapshots os JOIN races r ON r.race_id = os.race_id WHERE r.race_date = ?",
                (race_date,),
            ),
            "prediction_races": _archive_scalar(
                conn,
                "SELECT COUNT(DISTINCT p.race_id) FROM predictions p JOIN races r ON r.race_id = p.race_id WHERE r.race_date = ?",
                (race_date,),
            ),
        }
        years = _archive_rows(
            conn,
            """
            SELECT
              substr(race_date, 1, 4) AS year,
              COUNT(*) AS races,
              NULL AS entry_races,
              NULL AS result_races,
              NULL AS prediction_races
            FROM races
            GROUP BY year
            ORDER BY year DESC
            LIMIT 14
            """,
        )
        venues = _archive_rows(
            conn,
            """
            SELECT
              jcd,
              MAX(venue_name) AS venue_name,
              COUNT(*) AS races,
              NULL AS entry_races,
              NULL AS result_races,
              NULL AS odds_races
            FROM races
            GROUP BY jcd
            ORDER BY jcd
            """,
        )
    return {
        "date": race_date,
        "generated_at": _archive_now(),
        "totals": totals,
        "today": today,
        "years": years,
        "venues": venues,
    }


def archive_today(db_path: Path, query: dict[str, list[str]]) -> dict[str, Any]:
    race_date = _archive_one(query, "date", default_race_date(db_path))
    race_id = _archive_one(query, "race_id")
    jcd = _archive_one(query, "jcd")
    with connect(db_path) as conn:
        if race_id:
            return {
                "date": race_date,
                "generated_at": _archive_now(),
                "mode": "race",
                **_race_archive(conn, race_id),
            }
        params: list[Any] = [race_date]
        jcd_sql = ""
        if jcd:
            jcd_sql = "AND r.jcd = ?"
            params.append(jcd.zfill(2))
        races = _archive_rows(
            conn,
            f"""
            SELECT
              r.race_id, r.race_date, r.jcd, r.venue_name, r.rno, r.title, r.race_type,
              r.distance_m, r.deadline_at, r.status, r.updated_at,
              (SELECT COUNT(*) FROM entries e WHERE e.race_id = r.race_id) AS entries,
              (SELECT COUNT(*) FROM odds_snapshots os WHERE os.race_id = r.race_id) AS odds_snapshots,
              (SELECT MAX(captured_at) FROM odds_snapshots os WHERE os.race_id = r.race_id) AS latest_odds_at,
              (SELECT COUNT(*) FROM beforeinfo b WHERE b.race_id = r.race_id) AS beforeinfo_rows,
              (SELECT COUNT(*) FROM race_results rr WHERE rr.race_id = r.race_id AND rr.rank IS NOT NULL) AS result_rows,
              (SELECT MAX(generated_at) FROM predictions p WHERE p.race_id = r.race_id) AS latest_prediction
            FROM races r
            WHERE r.race_date = ? {jcd_sql}
            ORDER BY r.deadline_at IS NULL, r.deadline_at, r.jcd, r.rno
            """,
            tuple(params),
        )
    return {
        "date": race_date,
        "jcd": jcd,
        "generated_at": _archive_now(),
        "mode": "day",
        "races": races,
    }


def archive_history(db_path: Path, query: dict[str, list[str]]) -> dict[str, Any]:
    kind = (_archive_one(query, "kind", "racer") or "racer").lower()
    days = _bounded_int(_archive_one(query, "days", str(_DEFAULT_HISTORY_DAYS)), _DEFAULT_HISTORY_DAYS, 1, _MAX_DAYS)
    with connect(db_path) as conn:
        cutoff_date, latest_date = _recent_cutoff(conn, days)
        if kind == "race":
            race_id = _archive_required(query, "race_id")
            return {"kind": kind, "generated_at": _archive_now(), **_race_archive(conn, race_id)}
        if kind == "racer":
            payload = _history_racer(conn, _archive_required(query, "racer_no"), cutoff_date)
        elif kind == "venue":
            payload = _history_venue(conn, _archive_required(query, "jcd"), cutoff_date)
        elif kind == "motor":
            payload = _history_equipment(conn, "motor", _archive_required(query, "motor_no"), _archive_one(query, "jcd"), cutoff_date)
        elif kind == "boat":
            payload = _history_equipment(conn, "boat", _archive_required(query, "boat_no"), _archive_one(query, "jcd"), cutoff_date)
        elif kind == "lane":
            payload = _history_lane_fast(db_path, conn, _archive_required(query, "lane"), _archive_one(query, "jcd"), _archive_one(query, "rno"), cutoff_date)
        elif kind == "combo":
            payload = _history_combo(conn, _archive_required(query, "combination"), cutoff_date)
        else:
            raise ValueError(f"unsupported history kind: {kind}")
    payload["period_days"] = days
    payload["cutoff_date"] = cutoff_date
    payload["latest_date"] = latest_date
    payload.setdefault("summary", {})["period_days"] = days
    payload.setdefault("summary", {})["cutoff_date"] = cutoff_date
    return payload


def archive_stats(db_path: Path, query: dict[str, list[str]]) -> dict[str, Any]:
    scope = (_archive_one(query, "scope", "lane") or "lane").lower()
    if scope not in _SCOPES:
        scope = "lane"

    default_days = _EQUIPMENT_DAYS if scope in {"motor", "boat"} else _DEFAULT_DAYS
    days = _bounded_int(_archive_one(query, "days", str(default_days)), default_days, 1, _MAX_DAYS)
    limit = _bounded_int(_archive_one(query, "limit", str(_DEFAULT_LIMIT)), _DEFAULT_LIMIT, 1, _MAX_LIMIT)
    min_starts = _bounded_int(_archive_one(query, "min_starts", str(_default_min_starts(scope))), _default_min_starts(scope), 1, 1000)

    with connect(db_path) as conn:
        cutoff_date, latest_date = _recent_cutoff(conn, days)
        rows = _stat_rows_fast(conn, scope, cutoff_date, limit, min_starts)

    return {
        "scope": scope,
        "generated_at": _archive_now(),
        "period_days": days,
        "cutoff_date": cutoff_date,
        "latest_date": latest_date,
        "rows": rows,
    }


def _race_archive(conn: sqlite3.Connection, race_id: str) -> dict[str, Any]:
    race = _archive_row(
        conn,
        """
        SELECT
          r.*,
          (SELECT COUNT(*) FROM entries e WHERE e.race_id = r.race_id) AS entries,
          (SELECT COUNT(*) FROM odds_snapshots os WHERE os.race_id = r.race_id) AS odds_snapshots,
          (SELECT MIN(captured_at) FROM odds_snapshots os WHERE os.race_id = r.race_id) AS first_odds_at,
          (SELECT MAX(captured_at) FROM odds_snapshots os WHERE os.race_id = r.race_id) AS latest_odds_at,
          (SELECT COUNT(*) FROM beforeinfo b WHERE b.race_id = r.race_id) AS beforeinfo_rows,
          (SELECT COUNT(*) FROM race_results rr WHERE rr.race_id = r.race_id AND rr.rank IS NOT NULL) AS result_rows,
          (SELECT MAX(generated_at) FROM predictions p WHERE p.race_id = r.race_id) AS latest_prediction
        FROM races r
        WHERE r.race_id = ?
        """,
        (race_id,),
    )
    entries = _archive_rows(
        conn,
        """
        WITH latest_before AS (
          SELECT MAX(captured_at) AS captured_at FROM beforeinfo WHERE race_id = ?
        )
        SELECT
          e.lane, e.racer_no, e.racer_name, e.racer_class, e.branch, e.origin,
          e.age, e.weight_kg, e.f_count, e.l_count, e.avg_st,
          e.national_win_rate, e.national_2_rate, e.national_3_rate,
          e.local_win_rate, e.local_2_rate, e.local_3_rate,
          e.motor_no, e.motor_2_rate, e.motor_3_rate,
          e.boat_no, e.boat_2_rate, e.boat_3_rate,
          rr.rank, rr.course AS result_course, rr.start_timing AS result_start_timing,
          b.captured_at AS beforeinfo_at, b.exhibition_time, b.course AS exhibition_course,
          b.start_timing AS exhibition_start_timing, b.weather, b.wind_direction, b.wind_speed_m,
          b.air_temp_c, b.water_temp_c, b.wave_cm
        FROM entries e
        LEFT JOIN race_results rr ON rr.race_id = e.race_id AND rr.lane = e.lane
        LEFT JOIN latest_before lb
        LEFT JOIN beforeinfo b ON b.race_id = e.race_id AND b.lane = e.lane AND b.captured_at = lb.captured_at
        WHERE e.race_id = ?
        ORDER BY e.lane
        """,
        (race_id, race_id),
    )
    predictions = _archive_latest_prediction_rows(conn, race_id, limit=30)
    payouts = _archive_rows(
        conn,
        """
        SELECT bet_type, combination, payout_yen, popularity
        FROM payouts
        WHERE race_id = ?
        ORDER BY bet_type, popularity IS NULL, popularity
        """,
        (race_id,),
    )
    return {"race": race, "entries": entries, "predictions": predictions, "payouts": payouts}


def _history_racer(conn: sqlite3.Connection, racer_no: str, cutoff_date: str) -> dict[str, Any]:
    summary = _add_rates(
        _archive_row(
            conn,
            """
            WITH recent AS MATERIALIZED (
              SELECT race_id, race_date, jcd, venue_name, rno, title
              FROM races
              WHERE race_date >= ?
            )
            SELECT
              e.racer_no,
              MAX(e.racer_name) AS racer_name,
              MAX(e.racer_class) AS latest_class,
              MAX(e.branch) AS branch,
              MAX(e.origin) AS origin,
              COUNT(*) AS starts,
              SUM(CASE WHEN rr.rank IS NOT NULL THEN 1 ELSE 0 END) AS result_rows,
              SUM(CASE WHEN rr.rank = 1 THEN 1 ELSE 0 END) AS wins,
              SUM(CASE WHEN rr.rank <= 3 THEN 1 ELSE 0 END) AS top3,
              AVG(CASE WHEN rr.rank IS NOT NULL THEN rr.rank END) AS avg_rank,
              AVG(rr.start_timing) AS avg_start,
              AVG(e.national_win_rate) AS avg_national_win_rate,
              AVG(e.local_win_rate) AS avg_local_win_rate
            FROM entries e
            JOIN recent r ON r.race_id = e.race_id
            LEFT JOIN race_results rr ON rr.race_id = e.race_id AND rr.lane = e.lane
            WHERE e.racer_no = ?
            """,
            (cutoff_date, racer_no),
        )
    )
    summary.setdefault("racer_no", racer_no)
    rows = _archive_rows(
        conn,
        """
        WITH recent AS MATERIALIZED (
          SELECT race_id, race_date, jcd, venue_name, rno, title
          FROM races
          WHERE race_date >= ?
        )
        SELECT
          r.race_id, r.race_date, r.jcd, r.venue_name, r.rno, r.title,
          e.lane, e.racer_no, e.racer_name, e.racer_class,
          e.motor_no, e.boat_no, rr.rank, rr.course, rr.start_timing,
          p.combination AS result_combination, p.payout_yen
        FROM entries e
        JOIN recent r ON r.race_id = e.race_id
        LEFT JOIN race_results rr ON rr.race_id = e.race_id AND rr.lane = e.lane
        LEFT JOIN payouts p ON p.race_id = e.race_id AND p.bet_type = '3連単'
        WHERE e.racer_no = ?
        ORDER BY r.race_date DESC, r.jcd DESC, r.rno DESC
        LIMIT 80
        """,
        (cutoff_date, racer_no),
    )
    return {"kind": "racer", "generated_at": _archive_now(), "summary": summary, "rows": rows}


def _history_venue(conn: sqlite3.Connection, jcd: str, cutoff_date: str) -> dict[str, Any]:
    jcd = jcd.zfill(2)
    summary = _archive_row(
        conn,
        """
        WITH recent AS MATERIALIZED (
          SELECT race_id, race_date, jcd, venue_name, rno, title, race_type, distance_m
          FROM races
          WHERE race_date >= ? AND jcd = ?
        )
        SELECT
          jcd,
          MAX(venue_name) AS venue_name,
          COUNT(*) AS races,
          SUM(CASE WHEN (SELECT COUNT(*) FROM race_results rr WHERE rr.race_id = recent.race_id AND rr.rank IS NOT NULL) >= 3 THEN 1 ELSE 0 END) AS result_races,
          AVG(distance_m) AS avg_distance_m
        FROM recent
        """,
        (cutoff_date, jcd),
    )
    summary = dict(summary or {})
    summary["jcd"] = summary.get("jcd") or jcd
    summary["venue_name"] = summary.get("venue_name") or _archive_venue_name(jcd)
    facets = _archive_rows(
        conn,
        """
        WITH recent AS MATERIALIZED (
          SELECT race_id FROM races WHERE race_date >= ? AND jcd = ?
        )
        SELECT
          rr.lane,
          COUNT(*) AS starts,
          SUM(CASE WHEN rr.rank = 1 THEN 1 ELSE 0 END) AS wins,
          SUM(CASE WHEN rr.rank <= 3 THEN 1 ELSE 0 END) AS top3,
          AVG(rr.start_timing) AS avg_start
        FROM recent r
        JOIN race_results rr ON rr.race_id = r.race_id AND rr.rank IS NOT NULL
        GROUP BY rr.lane
        ORDER BY rr.lane
        """,
        (cutoff_date, jcd),
    )
    rows = _recent_races(conn, "r.race_date >= ? AND r.jcd = ?", (cutoff_date, jcd))
    return {"kind": "venue", "generated_at": _archive_now(), "summary": summary, "facets": facets, "rows": rows}


def _history_equipment(conn: sqlite3.Connection, kind: str, number: str, jcd: str | None, cutoff_date: str) -> dict[str, Any]:
    column = "motor_no" if kind == "motor" else "boat_no"
    rate2 = "motor_2_rate" if kind == "motor" else "boat_2_rate"
    rate3 = "motor_3_rate" if kind == "motor" else "boat_3_rate"
    params: list[Any] = [cutoff_date, number]
    filters = [f"e.{column} = ?"]
    if jcd:
        filters.append("r.jcd = ?")
        params.append(jcd.zfill(2))
    where = " AND ".join(filters)
    summary = _add_rates(
        _archive_row(
            conn,
            f"""
            WITH recent AS MATERIALIZED (
              SELECT race_id, race_date, jcd, venue_name, rno
              FROM races
              WHERE race_date >= ?
            )
            SELECT
              MAX(r.jcd) AS jcd,
              MAX(r.venue_name) AS venue_name,
              e.{column} AS number,
              COUNT(*) AS starts,
              SUM(CASE WHEN rr.rank IS NOT NULL THEN 1 ELSE 0 END) AS result_rows,
              SUM(CASE WHEN rr.rank = 1 THEN 1 ELSE 0 END) AS wins,
              SUM(CASE WHEN rr.rank <= 3 THEN 1 ELSE 0 END) AS top3,
              AVG(rr.rank) AS avg_rank,
              AVG(rr.start_timing) AS avg_start,
              AVG(e.{rate2}) AS avg_2_rate,
              AVG(e.{rate3}) AS avg_3_rate
            FROM recent r
            JOIN entries e ON e.race_id = r.race_id
            LEFT JOIN race_results rr ON rr.race_id = e.race_id AND rr.lane = e.lane
            WHERE {where}
            """,
            tuple(params),
        )
    )
    summary.setdefault("number", number)
    rows = _archive_rows(
        conn,
        f"""
        WITH recent AS MATERIALIZED (
          SELECT race_id, race_date, jcd, venue_name, rno
          FROM races
          WHERE race_date >= ?
        )
        SELECT
          r.race_id, r.race_date, r.jcd, r.venue_name, r.rno,
          e.lane, e.racer_no, e.racer_name, e.racer_class,
          e.motor_no, e.boat_no, rr.rank, rr.course, rr.start_timing
        FROM recent r
        JOIN entries e ON e.race_id = r.race_id
        LEFT JOIN race_results rr ON rr.race_id = e.race_id AND rr.lane = e.lane
        WHERE {where}
        ORDER BY r.race_date DESC, r.jcd DESC, r.rno DESC
        LIMIT 80
        """,
        tuple(params),
    )
    return {"kind": kind, "generated_at": _archive_now(), "summary": summary, "rows": rows}


def _history_lane_fast(
    db_path: Path,
    conn: sqlite3.Connection,
    lane: str,
    jcd: str | None,
    rno: str | None,
    cutoff_date: str,
) -> dict[str, Any]:
    days = _days_from_cutoff(conn, cutoff_date)
    summary = _lane_summary(db_path, lane, days)
    rows = _lane_rows_fast(conn, lane, cutoff_date, jcd, rno)
    return {"kind": "lane", "generated_at": _archive_now(), "summary": summary, "rows": rows}


def _history_combo(conn: sqlite3.Connection, combination: str, cutoff_date: str) -> dict[str, Any]:
    summary = _archive_row(
        conn,
        """
        WITH recent AS MATERIALIZED (
          SELECT race_id FROM races WHERE race_date >= ?
        )
        SELECT
          ? AS combination,
          COUNT(*) AS hits,
          AVG(p.payout_yen) AS avg_payout_yen,
          MIN(p.payout_yen) AS min_payout_yen,
          MAX(p.payout_yen) AS max_payout_yen,
          AVG(p.popularity) AS avg_popularity
        FROM recent r
        JOIN payouts p ON p.race_id = r.race_id
        WHERE p.bet_type = '3連単' AND p.combination = ?
        """,
        (cutoff_date, combination, combination),
    )
    rows = _archive_rows(
        conn,
        """
        WITH recent AS MATERIALIZED (
          SELECT race_id, race_date, jcd, venue_name, rno, title
          FROM races
          WHERE race_date >= ?
        )
        SELECT
          r.race_id, r.race_date, r.jcd, r.venue_name, r.rno, r.title,
          p.combination, p.payout_yen, p.popularity
        FROM recent r
        JOIN payouts p ON p.race_id = r.race_id
        WHERE p.bet_type = '3連単' AND p.combination = ?
        ORDER BY r.race_date DESC, r.jcd DESC, r.rno DESC
        LIMIT 80
        """,
        (cutoff_date, combination),
    )
    return {"kind": "combo", "generated_at": _archive_now(), "summary": dict(summary or {}), "rows": rows}


def _lane_summary(db_path: Path, lane: str, days: int) -> dict[str, Any]:
    payload = archive_stats(db_path, {"scope": ["lane"], "days": [str(days)], "min_starts": ["1"]})
    lane_text = str(lane)
    for row in payload.get("rows", []):
        if str(row.get("key")) == lane_text:
            return {
                "lane": int(lane),
                "starts": row.get("starts"),
                "result_rows": row.get("starts"),
                "wins": row.get("wins"),
                "top3": row.get("top3"),
                "win_rate": row.get("win_rate"),
                "top3_rate": row.get("top3_rate"),
                "avg_rank": row.get("avg_rank"),
                "avg_start": row.get("avg_start"),
                "avg_national_win_rate": row.get("avg_national_win_rate"),
                "avg_local_win_rate": row.get("avg_local_win_rate"),
                "avg_motor_2_rate": row.get("avg_motor_2_rate"),
                "avg_boat_2_rate": row.get("avg_boat_2_rate"),
            }
    return {"lane": int(lane), "starts": 0, "result_rows": 0, "wins": 0, "top3": 0}


def _lane_rows_fast(conn: sqlite3.Connection, lane: str, cutoff_date: str, jcd: str | None, rno: str | None) -> list[dict[str, Any]]:
    params: list[Any] = [int(lane), cutoff_date]
    filters = ["rr.lane = ?", "r.race_date >= ?", "rr.rank IS NOT NULL"]
    if jcd:
        filters.append("r.jcd = ?")
        params.append(jcd.zfill(2))
    if rno:
        filters.append("r.rno = ?")
        params.append(int(rno))
    return _archive_rows(
        conn,
        f"""
        SELECT
          r.race_id, r.race_date, r.jcd, r.venue_name, r.rno,
          rr.lane, e.racer_no, e.racer_name, e.racer_class,
          e.motor_no, e.boat_no, rr.rank, rr.course, rr.start_timing
        FROM race_results rr
        JOIN races r ON r.race_id = rr.race_id
        LEFT JOIN entries e ON e.race_id = rr.race_id AND e.lane = rr.lane
        WHERE {" AND ".join(filters)}
        ORDER BY r.race_date DESC, r.jcd DESC, r.rno DESC
        LIMIT 80
        """,
        tuple(params),
    )


def _recent_races(conn: sqlite3.Connection, where_sql: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
    return _archive_rows(
        conn,
        f"""
        SELECT
          r.race_id, r.race_date, r.jcd, r.venue_name, r.rno, r.title,
          r.race_type, r.distance_m,
          (SELECT COUNT(*) FROM entries e WHERE e.race_id = r.race_id) AS entries,
          (SELECT COUNT(*) FROM race_results rr WHERE rr.race_id = r.race_id AND rr.rank IS NOT NULL) AS result_rows,
          (SELECT combination FROM payouts p WHERE p.race_id = r.race_id AND p.bet_type = '3連単' LIMIT 1) AS result_combination,
          (SELECT payout_yen FROM payouts p WHERE p.race_id = r.race_id AND p.bet_type = '3連単' LIMIT 1) AS payout_yen
        FROM races r
        WHERE {where_sql}
        ORDER BY r.race_date DESC, r.jcd DESC, r.rno DESC
        LIMIT 80
        """,
        params,
    )


def _archive_latest_prediction_rows(conn: sqlite3.Connection, race_id: str, *, limit: int) -> list[dict[str, Any]]:
    latest = conn.execute(
        "SELECT generated_at FROM predictions WHERE race_id = ? ORDER BY generated_at DESC LIMIT 1",
        (race_id,),
    ).fetchone()
    if not latest:
        return []
    return _archive_rows(
        conn,
        """
        SELECT combination, probability, odds, expected_value, generated_at
        FROM predictions
        WHERE race_id = ? AND generated_at = ?
        ORDER BY probability DESC, COALESCE(expected_value, 0) DESC, combination
        LIMIT ?
        """,
        (race_id, latest["generated_at"], limit),
    )


def _stat_rows_fast(conn: sqlite3.Connection, scope: str, cutoff_date: str, limit: int, min_starts: int) -> list[dict[str, Any]]:
    select_key, group_sql, where_sql, order_sql = _scope_sql(scope)
    return _archive_rows(
        conn,
        f"""
        WITH recent_races AS MATERIALIZED (
          SELECT race_id, jcd, venue_name, rno
          FROM races
          WHERE race_date >= ?
        )
        SELECT
          {select_key},
          COUNT(*) AS starts,
          SUM(CASE WHEN rr.rank = 1 THEN 1 ELSE 0 END) AS wins,
          SUM(CASE WHEN rr.rank <= 3 THEN 1 ELSE 0 END) AS top3,
          SUM(CASE WHEN rr.rank = 1 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS win_rate,
          SUM(CASE WHEN rr.rank <= 3 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS top3_rate,
          AVG(rr.rank) AS avg_rank,
          AVG(rr.start_timing) AS avg_start,
          AVG(e.national_win_rate) AS avg_national_win_rate,
          AVG(e.local_win_rate) AS avg_local_win_rate,
          AVG(e.motor_2_rate) AS avg_motor_2_rate,
          AVG(e.boat_2_rate) AS avg_boat_2_rate
        FROM recent_races r
        JOIN race_results rr ON rr.race_id = r.race_id AND rr.rank IS NOT NULL
        LEFT JOIN entries e ON e.race_id = rr.race_id AND e.lane = rr.lane
        WHERE {where_sql}
        GROUP BY {group_sql}
        HAVING COUNT(*) >= ?
        ORDER BY {order_sql}
        LIMIT ?
        """,
        (cutoff_date, min_starts, limit),
    )


def _scope_sql(scope: str) -> tuple[str, str, str, str]:
    if scope == "venue":
        return (
            "r.jcd AS key, MAX(r.venue_name) AS label",
            "r.jcd",
            "1 = 1",
            "win_rate DESC, starts DESC, r.jcd",
        )
    if scope == "rno":
        return (
            "r.rno AS key, printf('%02dR', r.rno) AS label",
            "r.rno",
            "1 = 1",
            "win_rate DESC, starts DESC, r.rno",
        )
    if scope == "class":
        class_expr = "COALESCE(NULLIF(e.racer_class, ''), '-')"
        return (
            f"{class_expr} AS key, {class_expr} AS label",
            class_expr,
            "1 = 1",
            "win_rate DESC, starts DESC, label",
        )
    if scope == "motor":
        return (
            "printf('%s-M%s', r.jcd, e.motor_no) AS key, printf('%s M%s', MAX(r.venue_name), e.motor_no) AS label",
            "r.jcd, e.motor_no",
            "e.motor_no IS NOT NULL",
            "starts DESC, win_rate DESC, label",
        )
    if scope == "boat":
        return (
            "printf('%s-B%s', r.jcd, e.boat_no) AS key, printf('%s B%s', MAX(r.venue_name), e.boat_no) AS label",
            "r.jcd, e.boat_no",
            "e.boat_no IS NOT NULL",
            "starts DESC, win_rate DESC, label",
        )
    return (
        "rr.lane AS key, printf('%d号艇', rr.lane) AS label",
        "rr.lane",
        "1 = 1",
        "rr.lane",
    )


def _recent_cutoff(conn: sqlite3.Connection, days: int) -> tuple[str, str | None]:
    row = conn.execute("SELECT MAX(race_date) FROM races").fetchone()
    latest = row[0] if row else None
    try:
        latest_date = date.fromisoformat(str(latest)) if latest else now_jst().date()
    except ValueError:
        latest_date = now_jst().date()
    cutoff = latest_date - timedelta(days=days - 1)
    return cutoff.isoformat(), latest


def _days_from_cutoff(conn: sqlite3.Connection, cutoff_date: str) -> int:
    cutoff, latest = _recent_cutoff(conn, _DEFAULT_HISTORY_DAYS)
    if cutoff == cutoff_date:
        return _DEFAULT_HISTORY_DAYS
    latest_value = latest or now_jst().date().isoformat()
    try:
        return (date.fromisoformat(latest_value) - date.fromisoformat(cutoff_date)).days + 1
    except ValueError:
        return _DEFAULT_HISTORY_DAYS


def _default_min_starts(scope: str) -> int:
    return 5 if scope in {"motor", "boat"} else 20


def _bounded_int(raw: str | None, default: int, min_value: int, max_value: int) -> int:
    try:
        value = int(raw or default)
    except (TypeError, ValueError):
        value = default
    return min(max(value, min_value), max_value)


def _add_rates(summary: dict[str, Any] | None) -> dict[str, Any]:
    out = dict(summary or {})
    results = float(out.get("result_rows") or 0)
    if results:
        out["win_rate"] = float(out.get("wins") or 0) / results
        out["top3_rate"] = float(out.get("top3") or 0) / results
    return out


def _archive_row(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
    row = conn.execute(sql, params).fetchone()
    return _archive_rowdict(row) if row else None


def _archive_rows(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    return [_archive_rowdict(row) for row in conn.execute(sql, params).fetchall()]


def _archive_scalar(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> int:
    row = conn.execute(sql, params).fetchone()
    return int(row[0] or 0) if row else 0


def _archive_rowdict(row: sqlite3.Row) -> dict[str, Any]:
    return {key: row[key] for key in row.keys()}


def _archive_one(query: dict[str, list[str]], key: str, default: str | None = None) -> str | None:
    values = query.get(key)
    if not values or not values[0]:
        return default
    return values[0]


def _archive_required(query: dict[str, list[str]], key: str) -> str:
    value = _archive_one(query, key)
    if value is None:
        raise ValueError(f"missing query parameter: {key}")
    return value


def _archive_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _archive_venue_name(jcd: str) -> str:
    return next((venue.name for venue in VENUES if venue.code == jcd), jcd)


def roadmap_status(db_path: Path, query: dict[str, list[str]]) -> dict[str, Any]:
    race_date = query_race_date(db_path, query)
    now = time.monotonic()
    cached = _ROADMAP_CACHE.get(db_path)
    if cached and now - cached[0] < 30.0:
        return cached[1]

    progress: dict[str, Any]
    try:
        progress = progress_active_fast(db_path, {"date": [race_date]})
    except Exception as exc:
        progress = {"error": str(exc)}

    summary: dict[str, Any]
    try:
        summary = summary_cached(db_path)
    except Exception as exc:
        summary = {"error": str(exc)}

    remote_evaluations = _read_remote_eval_status(db_path.parent / REMOTE_EVAL_STATUS_NAME)

    payload = {
        "generated_at": now_jst().isoformat(timespec="seconds"),
        "date": race_date,
        "record_markdown": _read_project_status_markdown(),
        "milestones": _roadmap_milestones(),
        "improvements": _roadmap_improvements(),
        "agents": _roadmap_agents(),
        "progress": progress,
        "summary": summary,
        "processes": _process_snapshots(),
        "remote_evaluations": remote_evaluations,
        "quality_gates": _quality_gates(db_path.parent / "models", remote_evaluations),
        "model_artifacts": _latest_model_artifacts(db_path.parent / "models"),
        "v_file_inventory": _v_file_inventory(PROJECT_ROOT / "src" / "boatrace_ai"),
    }
    _ROADMAP_CACHE[db_path] = (now, payload)
    return payload


def _read_project_status_markdown() -> str:
    try:
        return PROJECT_STATUS_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        return "# BOAT RACE AI 懸案・進捗\n\n記録ファイルがまだ作成されていません。"


def _read_remote_eval_status(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"status": "未取得", "jobs": [], "note": "scripts/update_remote_eval_status.py --loop で更新"}
    except Exception as exc:
        return {"status": "読込失敗", "error": str(exc), "jobs": []}
    if not isinstance(payload, dict):
        return {"status": "形式不正", "jobs": []}
    payload.setdefault("jobs", [])
    return payload


def _quality_gates(model_dir: Path, remote_evaluations: dict[str, Any]) -> list[dict[str, Any]]:
    bankrolls = _bankroll_gate_records(model_dir) + _remote_bankroll_gate_records(remote_evaluations)
    best = max(bankrolls, key=lambda row: row.get("roi") or -1.0, default=None)
    latest = max(bankrolls, key=lambda row: row.get("modified_at") or "", default=None)
    remote_jobs = remote_evaluations.get("jobs") if isinstance(remote_evaluations, dict) else []
    remote_counts: dict[str, int] = {}
    for job in remote_jobs or []:
        status = str((job or {}).get("status") or "不明")
        remote_counts[status] = remote_counts.get(status, 0) + 1
    remote_text = " / ".join(f"{key}:{value}" for key, value in sorted(remote_counts.items())) or "未取得"

    best_roi = _float_or_none(best.get("roi") if best else None)
    best_profit = _float_or_none(best.get("profit_yen") if best else None)
    latest_roi = _float_or_none(latest.get("roi") if latest else None)
    latest_profit = _float_or_none(latest.get("profit_yen") if latest else None)
    latest_drawdown = latest.get("max_drawdown_yen") if latest else None
    attribution_rows = [row for row in bankrolls if row.get("roi_attribution_gate")]
    attribution_best = max(attribution_rows, key=lambda row: int(row.get("stable_signals") or 0), default=None)
    attribution_candidate = bool(
        attribution_best
        and attribution_best.get("roi_attribution_gate") == "candidate"
        and int(attribution_best.get("stable_signals") or 0) > 0
    )

    roi_ok = best_roi is not None and best_roi >= 1.0
    profit_ok = best_profit is not None and best_profit > 0
    return [
        {
            "target": "M6 ROI",
            "status": "達成候補" if roi_ok else "未達",
            "evidence": _gate_bankroll_text(best),
            "next": "ROI 1.0以上の候補を正規化KellyスイープPID 172555-172559で確認する" if not roi_ok else "損益/ドローダウン/購入日数も確認する",
        },
        {
            "target": "M6 損益",
            "status": "達成候補" if profit_ok else "未達",
            "evidence": _gate_bankroll_text(best),
            "next": "損益プラスまで購入条件と特徴量を再調整する" if not profit_ok else "ドローダウンと購入分散を確認する",
        },
        {
            "target": "M6 最新適応型",
            "status": "未達" if latest_roi is not None and latest_roi < 1.0 else "確認中",
            "evidence": _gate_bankroll_text(latest),
            "next": "正規化KellyスイープPID 172555-172559の結果で置き換える",
        },
        {
            "target": "M6 ドローダウン",
            "status": "ROI/損益待ち" if not (roi_ok and profit_ok) else "要判定",
            "evidence": f"latest maxDD={latest_drawdown if latest_drawdown is not None else '-'}",
            "next": "ROI/損益ゲート達成後に日次DD許容を判定する",
        },
        {
            "target": "M4-2 ROI帰属再現性",
            "status": "達成候補" if attribution_candidate else ("未達" if attribution_rows else "未評価"),
            "evidence": (
                f"{attribution_best.get('file')} / stable={int(attribution_best.get('stable_signals') or 0)} / gate={attribution_best.get('roi_attribution_gate')}"
                if attribution_best else "ROI帰属つき時間fold成果物なし"
            ),
            "next": "同じ方向が後続foldでも再現し、資金運用ROI/損益も改善するか確認する" if attribution_candidate else "最良Kelly条件でROI帰属つき時間foldバックテストを実行する",
        },
        {
            "target": "Remote Eval",
            "status": "実行中" if remote_counts.get("実行中") else ("完了" if remote_counts.get("完了") else str(remote_evaluations.get("status") or "未取得")),
            "evidence": remote_text,
            "next": "監視JSONから結果JSON生成と失敗ログを回収する",
        },
    ]


def _remote_bankroll_gate_records(remote_evaluations: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for job in (remote_evaluations.get("jobs") if isinstance(remote_evaluations, dict) else []) or []:
        if not str((job or {}).get("kind") or "").startswith("bankroll"):
            continue
        result = (job or {}).get("result") or {}
        metrics = result.get("metrics") or {}
        attribution = result.get("ticket_roi_attribution") or {}
        stability = attribution.get("fold_stability") or {}
        if metrics.get("roi") is None:
            continue
        rows.append(
            {
                "file": result.get("file") or job.get("name"),
                "modified_at": result.get("modified_at") or remote_evaluations.get("generated_at") or "",
                "roi": _float_or_none(metrics.get("roi")),
                "profit_yen": metrics.get("profit_yen"),
                "stake_yen": metrics.get("stake_yen"),
                "evaluated_races": metrics.get("evaluated_races"),
                "max_drawdown_yen": metrics.get("max_drawdown_yen"),
                "roi_attribution_gate": stability.get("gate"),
                "stable_signals": stability.get("stable_signals"),
            }
        )
    return rows


def _bankroll_gate_records(model_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not model_dir.exists():
        return rows
    for path in sorted(model_dir.glob("*.json")):
        if path.stat().st_size > 8_000_000:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not _is_bankroll_result(data):
            continue
        rows.append(
            {
                "file": path.name,
                "modified_at": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
                "roi": _float_or_none(data.get("roi")),
                "profit_yen": data.get("profit_yen"),
                "stake_yen": data.get("stake_yen"),
                "evaluated_races": data.get("evaluated_races"),
                "max_drawdown_yen": data.get("max_drawdown_yen"),
                "roi_attribution_gate": ((data.get("ticket_roi_attribution") or {}).get("fold_stability") or {}).get("gate"),
                "stable_signals": ((data.get("ticket_roi_attribution") or {}).get("fold_stability") or {}).get("stable_signals"),
            }
        )
    return rows


def _gate_bankroll_text(row: dict[str, Any] | None) -> str:
    if not row:
        return "資金運用JSONなし"
    parts = [row.get("file") or "-"]
    if row.get("roi") is not None:
        parts.append(f"ROI={float(row['roi']):.3f}")
    if row.get("profit_yen") is not None:
        parts.append(f"損益={int(float(row['profit_yen'])):,}円")
    if row.get("evaluated_races") is not None:
        parts.append(f"R={int(row['evaluated_races']):,}")
    return " / ".join(parts)


def _roadmap_improvements() -> list[dict[str, Any]]:
    return [
        {
            "id": "M4-1",
            "milestone": "M4/M5",
            "status": "設計",
            "progress": 10,
            "item": "NN shadowモデル導入",
            "next": "選手/場/モーター/ボートのembedding、天候/展示/時系列実績、オッズ系列branchを持つNNを別系統で短縮学習し、資金運用バックチェックで既存主系と比較する。",
        },
        {
            "id": "M4-2",
            "milestone": "M4/M6",
            "status": "実行中/要改善",
            "progress": 45,
            "item": "相関監査とROI接続",
            "next": "単相関だけでは採否を決めない。相関監査retry PID 174501と正規化Kelly 5条件を回収し、PID 175652が最良条件の選択券ROI帰属を時間foldで自動再検証する。",
        },
        {
            "id": "M6-1",
            "milestone": "M6",
            "status": "実行中",
            "progress": 25,
            "item": "実オッズ履歴不足の扱い",
            "next": "実オッズ必須バックチェックはfold1で全skipを確認。リアルタイム蓄積が増えるまでは過去ログ中心評価を主判定にする。",
        },
        {
            "id": "M6-2",
            "milestone": "M6",
            "status": "再設計/再実行",
            "progress": 45,
            "item": "資金配分パラメータ探索",
            "next": "旧スイープは候補あり選択0件のため停止。正規化KellyスイープPID 172555-172559で結果回収待ち。",
        },
        {
            "id": "M6-3",
            "milestone": "M6",
            "status": "未達/動的判定",
            "progress": 35,
            "item": "完了ゲート",
            "next": "ROI 1.0以上、損益プラス、ドローダウン許容、購入日数/的中率劣化なしを完了ゲート表で動的判定する。",
        },
        {
            "id": "M6-4",
            "milestone": "M6",
            "status": "実装済み/評価待ち",
            "progress": 30,
            "item": "特徴量改善の反映",
            "next": "選手/場/モーター/ボート/選択条件のROI帰属を資金運用出力へ追加。M4 ablationと最良Kelly条件を回収後、除外/採用候補を同一foldで再評価する。",
        },
        {
            "id": "M6-5",
            "milestone": "M6",
            "status": "修正済み/監視中",
            "progress": 90,
            "item": "疎行列index互換",
            "next": "FeatureHasher出力int32化を検証済み。正規化Kelly再実行でも同エラーが再発しないか監視する。",
        },
        {
            "id": "M6-6",
            "milestone": "M6",
            "status": "修正済み/再評価中",
            "progress": 55,
            "item": "候補あり選択0件の解消",
            "next": "日次上位候補制限とnormalized_kelly配分を追加。sanity PID 172873と本評価PID 172555-172559でselected_tickets>0を確認する。",
        },
    ]


def _roadmap_agents() -> list[dict[str, str]]:
    return [
        {"name": "Fermat", "area": "モデル/特徴量", "status": "完了", "task": "特徴量チューニング、相関、資金運用バックチェックの棚卸し"},
        {"name": "Hume", "area": "データ収集/リカバリ", "status": "完了", "task": "本日分/過去分の取得進捗、結果待ち、リトライ処理の棚卸し"},
        {"name": "Plato", "area": "WebUI/コード整理", "status": "完了", "task": "残存v系ファイル、WebUI性能、専用ページ確認の棚卸し"},
        {"name": "Hubble", "area": "v系依存解析", "status": "完了", "task": "番号付きファイル安定名移行の依存表作成"},
        {"name": "Dirac", "area": "特殊結果パーサ", "status": "完了", "task": "F/返還/不成立ケースの完了判定と保存方針"},
        {"name": "Linnaeus", "area": "特殊結果実装", "status": "完了", "task": "結果取得済み/3連単評価不可の保存と再取得除外"},
        {"name": "Helmholtz", "area": "資金運用", "status": "完了", "task": "100円単位・最低100円の適応型資金運用へ更新"},
        {"name": "Ohm", "area": "実オッズ評価", "status": "完了", "task": "締切前実オッズ必須/欠損skipのバックチェック設計"},
        {"name": "Sartre", "area": "特徴量ablation", "status": "完了", "task": "特徴量グループ別ablationの最小改修点"},
        {"name": "Russell", "area": "資金運用実装", "status": "完了", "task": "--require-real-odds による実オッズ必須/skipモード"},
        {"name": "Euler", "area": "特徴量実装", "status": "完了", "task": "drop-feature-groups と ablation サブコマンド"},
        {"name": "Noether", "area": "NN shadowモデル", "status": "完了", "task": "設計をM4-1改善事項へ移管して回収"},
        {"name": "Curie", "area": "相関監査", "status": "完了", "task": "retry PID 174501のフル相関診断を回収。採否判定はM4-2へ移管"},
        {"name": "Fisher", "area": "ROI帰属", "status": "完了", "task": "ROI帰属実装を回収。PID 175652の評価待ちはM4-2/M6へ移管"},
        {"name": "Ptolemy", "area": "懸案UI監査", "status": "完了", "task": "M6改善事項/完了ゲート/API表示の抜け漏れ確認。リモートPID静的表示のリスクを回収"},
        {"name": "Mendel", "area": "M7棚卸し", "status": "完了", "task": "v系ファイルをmust-keep依存とsafe-to-clean候補へ分離"},
    ]


def _roadmap_milestones() -> list[dict[str, Any]]:
    return [
        {"id": "M0", "title": "当日ダッシュボード運用", "status": "進行中", "progress": 70, "next": "表示を当日固定にし、重いAPIを段階読み込み・キャッシュで抑える"},
        {"id": "M1", "title": "懸案・進捗ページ", "status": "進行中", "progress": 86, "next": "リモート評価監視JSONを10001へ反映し、ジョブ結果を継続回収する"},
        {"id": "M2", "title": "公式データ収集", "status": "進行中", "progress": 58, "next": "特殊結果適用後の常駐収集ループを監視し、残る取得失敗を再試行キュー化する"},
        {"id": "M3", "title": "過去10年バックフィル", "status": "進行中", "progress": 35, "next": "新しい日付から古い日付へ、欠損日を優先して再取得する"},
        {"id": "M4", "title": "過去ログ中心モデル", "status": "進行中", "progress": 74, "next": "retry相関診断とROI帰属バックテストを回収し、時間foldで再現する特徴だけを採用候補にする"},
        {"id": "M5", "title": "リアルタイム併用モデル", "status": "設計/並走", "progress": 25, "next": "リアルタイムオッズ系列が十分貯まるまでは shadow 評価に限定する"},
        {"id": "M6", "title": "資金運用モデル", "status": "要改善", "progress": 68, "next": "正規化Kelly結果と選択券ROI帰属を回収し、ROI/損益/fold再現性ゲート達成まで完了扱いしない"},
        {"id": "M7", "title": "v系ファイル整理", "status": "棚卸し完了/移行待ち", "progress": 38, "next": "Mendel棚卸しのsafe-to-clean候補から安定名移行済み範囲を削除する"},
    ]


def _process_snapshots() -> list[dict[str, Any]]:
    patterns = [
        ("web_dashboard", "Webサーバ"),
        ("predict_loop", "予測ループ"),
        ("adaptive_odds_loop", "リアルタイム収集"),
        ("live_slow", "ライブ収集"),
        ("backfill", "過去バックフィル"),
    ]
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    proc = Path("/proc")
    for child in proc.iterdir() if proc.exists() else []:
        if not child.name.isdigit():
            continue
        try:
            raw = (child / "cmdline").read_bytes().replace(b"\x00", b" ").decode("utf-8", "replace").strip()
        except (OSError, PermissionError):
            continue
        if not raw or "boatrace_ai" not in raw:
            continue
        for pattern, label in patterns:
            if pattern in raw:
                key = (child.name, label)
                if key in seen:
                    continue
                seen.add(key)
                rows.append({"pid": int(child.name), "kind": label, "pattern": pattern, "cmd": raw[:240]})
    return sorted(rows, key=lambda row: (row["kind"], row["pid"]))


def _latest_model_artifacts(model_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not model_dir.exists():
        return rows
    for path in sorted(model_dir.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)[:24]:
        if not path.is_file():
            continue
        item: dict[str, Any] = {
            "file": path.name,
            "size_bytes": path.stat().st_size,
            "modified_at": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).astimezone(JST).isoformat(timespec="seconds"),
            "kind": path.suffix.lstrip(".") or "file",
        }
        if path.suffix == ".json":
            item.update(_json_file_hint(path))
        rows.append(item)
    return rows


def _json_file_hint(path: Path) -> dict[str, Any]:
    if path.stat().st_size > 512_000:
        return {"hint": "large_json_skipped"}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"error": str(exc)}
    keys = ["generated_at", "feature_set", "model", "evaluated_races", "roi", "profit_yen", "entry_log_loss", "winner_top1_accuracy", "trifecta_top5_hit_rate"]
    return {key: data.get(key) for key in keys if key in data}


def _v_file_inventory(src_dir: Path) -> dict[str, Any]:
    files = sorted(path.name for path in src_dir.glob("*.py") if _looks_versioned(path.name))
    return {
        "count": len(files),
        "sample": files[:30],
        "note": "WebUIの webserver_operational 系は削除対象。モデル/収集のv系は稼働中依存を安定名へ移してから整理する。",
    }


def _looks_versioned(name: str) -> bool:
    stem = name[:-3] if name.endswith(".py") else name
    return any(part.startswith("v") and part[1:].isdigit() for part in stem.replace("-", "_").split("_")) or stem[-1:].isdigit()

ROADMAP_REPORT_HTML = _load_template("roadmap_report.html")

MODEL_REPORT_HTML = _load_template("model_report.html")

HTML = _load_template("dashboard.html")

if __name__ == "__main__":
    raise SystemExit(main())
