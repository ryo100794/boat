from __future__ import annotations

import argparse
import json
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
    cached = _DEFAULT_DATE_CACHE.get(db_path)
    if cached and now - cached[0] < 300.0:
        return cached[1]
    today = now_jst().date().isoformat()
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
    errors: list[dict[str, str]] = []

    for path in sorted(model_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append({"file": path.name, "error": str(exc)})
            continue
        label = _report_label(path, data)
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

    backtests.sort(key=lambda item: (item.get("generated_at") or "", item["name"]))
    bankroll.sort(key=lambda item: (item.get("generated_at") or "", item["name"]))
    sweeps.sort(key=lambda item: (item.get("entry_log_loss") is None, item.get("entry_log_loss") or 999, item["name"]))
    payload = {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "model_dir": str(model_dir),
        "backtests": backtests,
        "fold_metrics": fold_metrics,
        "bankroll": bankroll,
        "bankroll_daily": bankroll_daily,
        "sweeps": sweeps,
        "errors": errors,
    }
    _MODEL_REPORT_CACHE[model_dir] = (now, payload)
    return payload


def _is_backtest_result(data: dict[str, Any]) -> bool:
    return "entry_log_loss" in data or "winner_top1_accuracy" in data or "trifecta_top5_hit_rate" in data


def _is_bankroll_result(data: dict[str, Any]) -> bool:
    return "roi" in data and ("stake_yen" in data or "return_yen" in data or "daily" in data)


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
    }


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


def live_wipe_fast(db_path: Path, query: dict[str, list[str]]) -> dict[str, Any]:
    race_date = query_race_date(db_path, query)
    now = now_jst()
    with connect(db_path) as conn:
        rows = [row for row in _day_metric_rows(conn, race_date, include_predictions=True) if _is_active_row(row)]
        for row in sorted(rows, key=lambda item: item["deadline_at"] or "", reverse=True):
            start_at = stored_start_time(row["deadline_at"])
            if not start_at or start_at > now:
                continue
            item = _race_payload_from_row(row, now=now, before_minutes=5)
            item.update(
                {
                    "minutes_since_deadline": int((now - estimated_deadline_from_start(start_at)).total_seconds() // 60)
                    if estimated_deadline_from_start(start_at)
                    else None,
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
            "status": "待ち",
            "progress": 15,
            "item": "特徴量改善の反映",
            "next": "M4 ablation結果を回収し、資金運用モデルの入力特徴量と購入判断へ反映する。",
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
            "next": "日次上位候補制限とnormalized_kelly配分を追加。PID 172555-172559のfoldでselected_tickets>0を確認する。",
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
        {"name": "Remote-M6", "area": "資金運用評価", "status": "再実行中", "task": "PID 171805実オッズ / 172555-172559正規化Kelly資金配分"},
        {"name": "Remote-M4", "area": "特徴量評価", "status": "再実行中", "task": "固定版PID 171811 / drop-one-feature-group ablation"},
        {"name": "Ptolemy", "area": "懸案UI監査", "status": "完了", "task": "M6改善事項/完了ゲート/API表示の抜け漏れ確認。リモートPID静的表示のリスクを回収"},
        {"name": "Mendel", "area": "M7棚卸し", "status": "完了", "task": "v系ファイルをmust-keep依存とsafe-to-clean候補へ分離"},
    ]


def _roadmap_milestones() -> list[dict[str, Any]]:
    return [
        {"id": "M0", "title": "当日ダッシュボード運用", "status": "進行中", "progress": 70, "next": "表示を当日固定にし、重いAPIを段階読み込み・キャッシュで抑える"},
        {"id": "M1", "title": "懸案・進捗ページ", "status": "進行中", "progress": 86, "next": "リモート評価監視JSONを10001へ反映し、ジョブ結果を継続回収する"},
        {"id": "M2", "title": "公式データ収集", "status": "進行中", "progress": 58, "next": "特殊結果適用後の常駐収集ループを監視し、残る取得失敗を再試行キュー化する"},
        {"id": "M3", "title": "過去10年バックフィル", "status": "進行中", "progress": 35, "next": "新しい日付から古い日付へ、欠損日を優先して再取得する"},
        {"id": "M4", "title": "過去ログ中心モデル", "status": "進行中", "progress": 68, "next": "リモートablation結果を回収して効く特徴量へ寄せる"},
        {"id": "M5", "title": "リアルタイム併用モデル", "status": "設計/並走", "progress": 25, "next": "リアルタイムオッズ系列が十分貯まるまでは shadow 評価に限定する"},
        {"id": "M6", "title": "資金運用モデル", "status": "要改善", "progress": 66, "next": "改善事項M6-1..M6-6を追跡し、ROI/損益ゲート達成まで完了扱いしない"},
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

ROADMAP_REPORT_HTML = '<!doctype html>\n<html lang="ja">\n<head>\n  <meta charset="utf-8">\n  <meta name="viewport" content="width=device-width, initial-scale=1">\n  <title>懸案・進捗マイルストーン</title>\n  <style>\n    :root { --ink:#172126; --muted:#637279; --line:#d8e0e3; --band:#f3f6f7; --accent:#006d77; --warn:#a76300; --bad:#a33a3a; --ok:#247a4b; font-family:ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }\n    * { box-sizing:border-box; } body { margin:0; color:var(--ink); background:#fff; font-size:10px; }\n    header { position:sticky; top:0; z-index:4; display:flex; align-items:center; justify-content:space-between; gap:6px; padding:4px 6px; border-bottom:1px solid var(--line); background:#fff; }\n    h1 { margin:0; font-size:12px; white-space:nowrap; } a,button { height:19px; border:1px solid var(--line); border-radius:3px; background:#fff; color:var(--ink); padding:1px 5px; text-decoration:none; font:inherit; } button,.primary { background:var(--accent); color:#fff; border-color:var(--accent); }\n    main { padding:5px 6px 14px; } .kpis { display:grid; grid-template-columns:repeat(6,minmax(90px,1fr)); gap:1px; background:var(--line); border:1px solid var(--line); }\n    .kpi { background:#fff; padding:3px 4px; min-width:0; } .kpi b { display:block; font-size:12px; line-height:1.05; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; } .kpi span { color:var(--muted); font-size:9px; }\n    .grid { display:grid; grid-template-columns:1.25fr .75fr; gap:6px; margin-top:6px; align-items:start; } .panel { min-width:0; border-top:1px solid var(--accent); padding-top:3px; }\n    .panel h2 { margin:0 0 2px; font-size:11px; display:flex; justify-content:space-between; gap:5px; } table { width:100%; border-collapse:collapse; table-layout:fixed; }\n    th,td { border-bottom:1px solid var(--line); padding:1px 3px; line-height:1.12; text-align:left; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; } th { color:var(--muted); background:#fafbfb; }\n    .record { white-space:pre-wrap; border:1px solid var(--line); background:#fbfcfc; padding:5px; max-height:520px; overflow:auto; line-height:1.25; } .bar { height:6px; background:#edf1f2; border-radius:3px; overflow:hidden; } .bar i { display:block; height:100%; background:var(--accent); }\n    .tag { display:inline-block; border-radius:999px; color:#fff; background:var(--muted); padding:0 4px; font-size:9px; } .ok { background:var(--ok); } .warn { background:var(--warn); } .bad { background:var(--bad); } .mono { font-variant-numeric:tabular-nums; font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; } .muted { color:var(--muted); }\n    @media (max-width:920px){ .kpis{grid-template-columns:repeat(2,minmax(110px,1fr));}.grid{grid-template-columns:1fr;} }\n  </style>\n</head>\n<body>\n<header><h1>懸案・進捗マイルストーン</h1><div><button id="reload" type="button">更新</button><a href="/reports/models">モデル</a><a class="primary" href="/">ダッシュボード</a></div></header>\n<main>\n  <div id="kpis" class="kpis"></div>\n  <div class="grid">\n    <div class="panel"><h2><span>進行中の懸案</span><span id="meta" class="muted"></span></h2><div id="record" class="record"></div></div>\n    <div class="panel"><h2><span>常駐プロセス</span><span class="muted">boatrace_ai</span></h2><table><thead><tr><th>種別</th><th>PID</th><th>cmd</th></tr></thead><tbody id="processRows"></tbody></table></div>\n    <div class="panel"><h2><span>マイルストーン</span><span class="muted">更新対象</span></h2><table><thead><tr><th>ID</th><th>状態</th><th>進捗</th><th>次作業</th></tr></thead><tbody id="milestoneRows"></tbody></table></div>\n    <div class="panel"><h2><span>改善事項</span><span class="muted">要改善を完了扱いしない</span></h2><table><thead><tr><th>ID</th><th>対象</th><th>状態</th><th>進捗</th><th>改善内容/次作業</th></tr></thead><tbody id="improvementRows"></tbody></table></div>\n    <div class="panel"><h2><span>完了ゲート</span><span class="muted">動的判定</span></h2><table><thead><tr><th>対象</th><th>判定</th><th>根拠</th><th>次作業</th></tr></thead><tbody id="gateRows"></tbody></table></div>\n    <div class="panel"><h2><span>リモート評価</span><span id="remoteEvalMeta" class="muted"></span></h2><table><thead><tr><th>PID</th><th>対象</th><th>状態</th><th>経過</th><th>指標</th></tr></thead><tbody id="remoteEvalRows"></tbody></table></div>\n    <div class="panel"><h2><span>専門エージェント</span><span class="muted">回収待ち含む</span></h2><table><thead><tr><th>名前</th><th>領域</th><th>状態</th><th>担当</th></tr></thead><tbody id="agentRows"></tbody></table></div>\n    <div class="panel"><h2><span>モデル成果物</span><span class="muted">最新順</span></h2><table><thead><tr><th>file</th><th>種別</th><th>サイズ</th><th>指標</th></tr></thead><tbody id="artifactRows"></tbody></table></div>\n    <div class="panel"><h2><span>v系ファイル棚卸し</span><span id="vMeta" class="muted"></span></h2><table><thead><tr><th>sample</th></tr></thead><tbody id="vRows"></tbody></table></div>\n  </div>\n</main>\n<script>\nconst $ = id => document.getElementById(id);\nconst esc = v => String(v ?? "").replace(/[&<>"\']/g, ch => ({"&":"&amp;","<":"&lt;",">":"&gt;",\'"\':"&quot;","\'":"&#39;"}[ch]));\nconst fmt = v => v == null ? "-" : Number(v).toLocaleString("ja-JP");\nconst pct = v => v == null ? "-" : `${(Number(v)*100).toFixed(1)}%`;\nconst bytes = v => { v=Number(v||0); if(v>1048576) return `${(v/1048576).toFixed(1)}MB`; if(v>1024) return `${(v/1024).toFixed(1)}KB`; return `${v}B`; };\nfunction tag(status){ const s=String(status||""); const c=s.includes("完了")?"ok":s.includes("要")||s.includes("失敗")?"bad":s.includes("進行")||s.includes("調査")?"warn":""; return `<span class="tag ${c}">${esc(s)}</span>`; }\nasync function load(){ const res=await fetch(\'/api/reports/roadmap-status\',{cache:\'no-store\'}); const data=await res.json(); render(data); }\nfunction render(data){\n  const p=(data.progress||{}), t=(p.today||{}), h=(p.historical||{}), s=(data.summary||{}), v=(data.v_file_inventory||{});\n  $(\'meta\').textContent=`更新 ${String(data.generated_at||\'\').replace(\'T\',\' \').slice(0,19)}`;\n  $(\'kpis\').innerHTML=[[\'当日R\',`${fmt(t.races)}/${fmt(t.target_races)}`],[\'当日結果残\',fmt(t.final_remaining)],[\'過去結果日残\',fmt(h.result_remaining_days)],[\'予測R\',fmt(s.predictions)],[\'odds snap\',fmt(s.odds_snapshots)],[\'v系py\',fmt(v.count)]].map(([l,x])=>`<div class="kpi"><b title="${esc(x)}">${esc(x)}</b><span>${esc(l)}</span></div>`).join(\'\');\n  $(\'record\').textContent=data.record_markdown||\'\';\n  $(\'processRows\').innerHTML=(data.processes||[]).map(r=>`<tr><td>${esc(r.kind)}</td><td class="mono">${esc(r.pid)}</td><td title="${esc(r.cmd)}">${esc(r.cmd)}</td></tr>`).join(\'\') || \'<tr><td colspan="3" class="muted">稼働プロセスなし</td></tr>\';\n  $(\'milestoneRows\').innerHTML=(data.milestones||[]).map(r=>`<tr><td class="mono" title="${esc(r.title)}">${esc(r.id)} ${esc(r.title)}</td><td>${tag(r.status)}</td><td><div class="bar"><i style="width:${Number(r.progress||0)}%"></i></div></td><td title="${esc(r.next)}">${esc(r.next)}</td></tr>`).join(\'\');\n  $(\'improvementRows\').innerHTML=(data.improvements||[]).map(r=>`<tr><td class="mono">${esc(r.id)}</td><td>${esc(r.milestone)}</td><td>${tag(r.status)}</td><td><div class="bar"><i style="width:${Number(r.progress||0)}%"></i></div></td><td title="${esc((r.item||\'\')+\' / \'+(r.next||\'\'))}">${esc(r.item)} / ${esc(r.next)}</td></tr>`).join(\'\') || \'<tr><td colspan="5" class="muted">改善事項なし</td></tr>\';\n  $(\'gateRows\').innerHTML=(data.quality_gates||[]).map(r=>`<tr><td>${esc(r.target)}</td><td>${tag(r.status)}</td><td title=\"${esc(r.evidence)}\">${esc(r.evidence)}</td><td title=\"${esc(r.next)}\">${esc(r.next)}</td></tr>`).join(\'\') || \'<tr><td colspan=\"4\" class=\"muted\">ゲート情報なし</td></tr>\';\n  const re=data.remote_evaluations||{}; $(\'remoteEvalMeta\').textContent=[re.status||\'\', String(re.generated_at||\'\').replace(\'T\',\' \').slice(0,19)].filter(Boolean).join(\' / \');\n  $(\'remoteEvalRows\').innerHTML=(re.jobs||[]).map(r=>`<tr><td class=\"mono\">${esc(r.pid)}</td><td title=\"${esc(r.name)}\">${esc(r.milestone||\'\')} ${esc(r.kind||\'\')}</td><td>${tag(r.status)}</td><td class=\"mono\">${esc((r.process&&r.process.elapsed)||\'-\')}</td><td title=\"${esc(remoteMetricText(r))}\">${esc(remoteMetricText(r))}</td></tr>`).join(\'\') || \'<tr><td colspan=\"5\" class=\"muted\">リモート評価状態なし</td></tr>\';\n  $(\'agentRows\').innerHTML=(data.agents||[]).map(r=>`<tr><td>${esc(r.name)}</td><td>${esc(r.area)}</td><td>${tag(r.status)}</td><td title="${esc(r.task)}">${esc(r.task)}</td></tr>`).join(\'\');\n  $(\'artifactRows\').innerHTML=(data.model_artifacts||[]).map(r=>`<tr><td title="${esc(r.file)}">${esc(r.file)}</td><td>${esc(r.kind)}</td><td class="mono">${bytes(r.size_bytes)}</td><td title="${esc(metricText(r))}">${esc(metricText(r))}</td></tr>`).join(\'\') || \'<tr><td colspan="4" class="muted">成果物なし</td></tr>\';\n  $(\'vMeta\').textContent=`${fmt(v.count)}件 / ${v.note||\'\'}`;\n  $(\'vRows\').innerHTML=(v.sample||[]).map(x=>`<tr><td title="${esc(x)}">${esc(x)}</td></tr>`).join(\'\');\n}\nfunction metricText(r){ return [[\'ROI\',r.roi],[\'損益\',r.profit_yen],[\'R\',r.evaluated_races],[\'LL\',r.entry_log_loss],[\'1着\',r.winner_top1_accuracy],[\'3T5\',r.trifecta_top5_hit_rate]].filter(([,v])=>v!=null).map(([k,v])=>`${k}:${k===\'ROI\'||k===\'1着\'||k===\'3T5\'?Number(v).toFixed(3):fmt(v)}`).join(\' / \'); }\nfunction remoteMetricText(r){ const m=(r.result&&r.result.metrics)||{}; const b=(r.result&&r.result.base_metrics)||{}; const x={...b,...m}; return [[\'ROI\',x.roi],[\'損益\',x.profit_yen],[\'R\',x.evaluated_races],[\'実od\',x.real_odds_races],[\'skip\',x.skipped_no_real_odds],[\'LL\',x.entry_log_loss],[\'1着\',x.winner_top1_accuracy],[\'3T5\',x.trifecta_top5_hit_rate]].filter(([,v])=>v!=null).map(([k,v])=>`${k}:${k===\'ROI\'||k===\'LL\'||k===\'1着\'||k===\'3T5\'?Number(v).toFixed(3):fmt(v)}`).join(\' / \') || (r.running?\'計算中\':\'結果未生成\'); }\n$(\'reload\').onclick=load; load(); setInterval(load,30000);\n</script>\n</body>\n</html>'

MODEL_REPORT_HTML = '<!doctype html>\n<html lang="ja">\n<head>\n  <meta charset="utf-8">\n  <meta name="viewport" content="width=device-width, initial-scale=1">\n  <title>モデル性能レポート</title>\n  <style>\n    :root { --ink:#172126; --muted:#637279; --line:#d8e0e3; --band:#f3f6f7; --accent:#006d77; --accent2:#8f2d56; --ok:#247a4b; --bad:#a33a3a; font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }\n    * { box-sizing:border-box; } body { margin:0; color:var(--ink); background:#fff; font-size:10px; }\n    header { position:sticky; top:0; z-index:4; display:flex; align-items:center; justify-content:space-between; gap:6px; padding:4px 6px; border-bottom:1px solid var(--line); background:#fff; }\n    h1 { margin:0; font-size:12px; } a,button,select { height:19px; border:1px solid var(--line); border-radius:3px; background:#fff; color:var(--ink); padding:0 5px; text-decoration:none; font:inherit; } button,.primary { background:var(--accent); color:#fff; border-color:var(--accent); }\n    main { padding:5px 6px 14px; }\n    .kpis { display:grid; grid-template-columns:repeat(6,minmax(92px,1fr)); gap:1px; border:1px solid var(--line); background:var(--line); }\n    .kpi { background:#fff; padding:3px 4px; min-width:0; } .kpi b { display:block; font-size:12px; line-height:1.05; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; } .kpi span { color:var(--muted); font-size:9px; }\n    .grid { display:grid; grid-template-columns:repeat(2,minmax(300px,1fr)); gap:6px; margin-top:6px; }\n    .panel { min-width:0; border-top:1px solid var(--accent); padding-top:3px; } .panel h2 { margin:0 0 2px; font-size:11px; display:flex; justify-content:space-between; gap:5px; }\n    canvas { width:100%; height:155px; border:1px solid var(--line); background:#fff; }\n    table { width:100%; border-collapse:collapse; table-layout:fixed; margin-top:3px; } th,td { border-bottom:1px solid var(--line); padding:1px 3px; line-height:1.12; text-align:right; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; } th { color:var(--muted); background:#fafbfb; } th:first-child,td:first-child { text-align:left; }\n    .mono { font-variant-numeric:tabular-nums; font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; } .muted { color:var(--muted); }\n    .pos { color:var(--ok); font-weight:700; } .neg { color:var(--bad); font-weight:700; }\n    @media (max-width:920px){ .kpis{grid-template-columns:repeat(2,minmax(120px,1fr));}.grid{grid-template-columns:1fr;} }\n  </style>\n</head>\n<body>\n<header><h1>モデル性能レポート</h1><div><select id="dailyModel"></select><button id="reload" type="button">更新</button><a class="primary" href="/">ダッシュボード</a></div></header>\n<main>\n  <div id="kpis" class="kpis"></div>\n  <div class="grid">\n    <div class="panel"><h2><span>モデル別 的中率</span><span class="muted">1着 / 3連単Top5</span></h2><canvas id="hitChart" width="900" height="330"></canvas></div>\n    <div class="panel"><h2><span>fold別 LogLoss</span><span class="muted">低いほど良い</span></h2><canvas id="lossChart" width="900" height="330"></canvas></div>\n    <div class="panel"><h2><span>資金運用 ROI</span><span class="muted">払戻 ÷ 投資</span></h2><canvas id="roiChart" width="900" height="330"></canvas></div>\n    <div class="panel"><h2><span>累積損益</span><span id="profitTitle" class="muted"></span></h2><canvas id="profitChart" width="900" height="330"></canvas></div>\n    <div class="panel"><h2><span>日次ROI推移</span><span class="muted">選択モデル</span></h2><canvas id="dailyRoiChart" width="900" height="330"></canvas></div>\n    <div class="panel"><h2><span>日次購入密度</span><span class="muted">点数 / 使用率</span></h2><canvas id="ticketsChart" width="900" height="330"></canvas></div>\n  </div>\n  <div class="panel" style="margin-top:10px"><h2><span>総合表</span><span id="meta" class="muted"></span></h2><table><thead><tr><th>モデル</th><th>種別</th><th>評価R</th><th>LogLoss</th><th>1着</th><th>3T5</th><th>ROI</th><th>損益</th><th>投資</th></tr></thead><tbody id="summaryRows"></tbody></table></div>\n  <div class="panel" style="margin-top:10px"><h2><span>スイープ</span><span class="muted">候補比較</span></h2><table><thead><tr><th>variant</th><th>LogLoss</th><th>Brier</th><th>1着</th><th>3T1</th><th>3T5</th><th>評価R</th></tr></thead><tbody id="sweepRows"></tbody></table></div>\n</main>\n<script>\nconst $ = id => document.getElementById(id);\nconst fmt = v => v == null ? "-" : Number(v).toLocaleString("ja-JP");\nconst pct = v => v == null ? "-" : `${(Number(v)*100).toFixed(2)}%`;\nconst yen = v => v == null ? "-" : `${Number(v).toLocaleString("ja-JP")}円`;\nconst ratio = v => v == null ? "-" : Number(v).toFixed(3);\nfunction shortName(s){ s=String(s||""); return s.replace(/^win_model_/,\'\').replace(/^bankroll_backtest_/,\'\').replace(/_10000$/,\'\').replace(/pastlog_/,\'pl_\'); }\nasync function loadReport(){ const res=await fetch(\'/api/reports/model-performance\',{cache:\'no-store\'}); const data=await res.json(); render(data); }\nfunction render(data){\n  const bank=data.bankroll||[], tests=data.backtests||[], folds=data.fold_metrics||[], daily=data.bankroll_daily||{};\n  const bestRoi=[...bank].sort((a,b)=>(b.roi??-9)-(a.roi??-9))[0]; const adaptive=bank.find(x=>String(x.file||\'\').includes(\'adaptive\')) || bank[bank.length-1];\n  $(\'kpis\').innerHTML=[[\'評価モデル\', tests.length+bank.length],[\'最高ROI\', bestRoi?`${ratio(bestRoi.roi)} ${shortName(bestRoi.name)}`:\'-\'],[\'最新適応ROI\', adaptive?ratio(adaptive.roi):\'-\'],[\'適応損益\', adaptive?yen(adaptive.profit_yen):\'-\'],[\'評価R\', adaptive?fmt(adaptive.evaluated_races):(tests[tests.length-1]?fmt(tests[tests.length-1].evaluated_races):\'-\')],[\'日次系列\', Object.keys(daily).length]].map(([l,v])=>`<div class="kpi"><b title="${String(v)}">${v}</b><span>${l}</span></div>`).join(\'\');\n  const dailyNames=Object.keys(daily); $(\'dailyModel\').innerHTML=dailyNames.map(n=>`<option>${n}</option>`).join(\'\'); const defaultName=(adaptive&&daily[adaptive.name])?adaptive.name:dailyNames[dailyNames.length-1]; if(defaultName) $(\'dailyModel\').value=defaultName;\n  $(\'dailyModel\').onchange=()=>renderDaily(data); $(\'reload\').onclick=loadReport; $(\'meta\').textContent=`生成 ${String(data.generated_at||\'\').replace(\'T\',\' \').slice(0,19)} / ${data.model_dir||\'\'}`;\n  drawGroupedBars($(\'hitChart\'), tests.map(x=>({label:shortName(x.name), a:x.winner_top1_accuracy, b:x.trifecta_top5_hit_rate})), {a:\'1着\', b:\'3T5\', percent:true});\n  drawLines($(\'lossChart\'), groupFoldSeries(folds, \'entry_log_loss\'), {baseline:null});\n  drawBars($(\'roiChart\'), bank.map(x=>({label:shortName(x.name), value:x.roi})), {baseline:1});\n  $(\'summaryRows\').innerHTML=[...tests.map(x=>summaryRow(x,\'BT\')),...bank.map(x=>summaryRow(x,\'資金\'))].join(\'\');\n  $(\'sweepRows\').innerHTML=(data.sweeps||[]).map(x=>`<tr><td title="${x.name}">${shortName(x.name)}</td><td>${ratio(x.entry_log_loss)}</td><td>${ratio(x.entry_brier)}</td><td>${pct(x.winner_top1_accuracy)}</td><td>${pct(x.trifecta_top1_hit_rate)}</td><td>${pct(x.trifecta_top5_hit_rate)}</td><td>${fmt(x.evaluated_races)}</td></tr>`).join(\'\') || \'<tr><td colspan="7" class="muted">スイープ結果なし</td></tr>\';\n  renderDaily(data);\n}\nfunction summaryRow(x,type){ const cls=Number(x.profit_yen||0)>=0?\'pos\':\'neg\'; return `<tr><td title="${x.name}">${shortName(x.name)}</td><td>${type}</td><td>${fmt(x.evaluated_races)}</td><td>${ratio(x.entry_log_loss)}</td><td>${pct(x.winner_top1_accuracy)}</td><td>${pct(x.trifecta_top5_hit_rate)}</td><td>${ratio(x.roi)}</td><td class="${x.profit_yen==null?\'\':cls}">${x.profit_yen==null?\'-\':yen(x.profit_yen)}</td><td>${x.stake_yen==null?\'-\':yen(x.stake_yen)}</td></tr>`; }\nfunction renderDaily(data){ const name=$(\'dailyModel\').value; const rows=(data.bankroll_daily||{})[name]||[]; $(\'profitTitle\').textContent=shortName(name); drawLines($(\'profitChart\'), [{label:\'累積損益\', rows:rows.map(r=>({x:r.date, y:r.cumulative_profit_yen}))}], {yen:true, baseline:0}); drawLines($(\'dailyRoiChart\'), [{label:\'日次ROI\', rows:rows.map(r=>({x:r.date, y:r.roi}))}], {baseline:1}); drawGroupedBars($(\'ticketsChart\'), rows.slice(-90).map(r=>({label:String(r.date||\'\').slice(5), a:r.tickets, b:(r.budget_used_fraction||0)*100})), {a:\'点数\', b:\'使用率%\'}); }\nfunction groupFoldSeries(folds, key){ const by=new Map(); for(const f of folds){ if(f[key]==null) continue; const k=shortName(f.model); if(!by.has(k)) by.set(k, []); by.get(k).push({x:f.fold, y:f[key]}); } return [...by.entries()].slice(-8).map(([label, rows])=>({label, rows:rows.sort((a,b)=>Number(a.x)-Number(b.x))})); }\nconst AX={l:54,t:12,b:30,r:8};\nfunction fmtAxis(v,opt={}){ const n=Number(v); if(!Number.isFinite(n)) return "-"; if(opt.percent) return `${(n*100).toFixed(0)}%`; if(opt.yen) return Math.abs(n)>=1000000 ? `${(n/1000000).toFixed(1)}M` : Math.abs(n)>=1000 ? `${Math.round(n/1000)}k` : `${Math.round(n)}`; if(Math.abs(n)>=1000) return Math.round(n).toLocaleString("ja-JP"); if(Math.abs(n)>=10) return n.toFixed(0); if(Math.abs(n)>=1) return n.toFixed(2); return n.toFixed(3); }\nfunction drawAxes(ctx,w,h,min,max,baseline,opt={}){ ctx.clearRect(0,0,w,h); const l=AX.l,t=AX.t,b=h-AX.b,r=w-AX.r; ctx.font=\'10px sans-serif\'; ctx.textBaseline=\'middle\'; ctx.textAlign=\'right\'; ctx.strokeStyle=\'#edf1f2\'; ctx.fillStyle=\'#637279\'; ctx.lineWidth=1; for(let i=0;i<=4;i++){ const v=min+(max-min)*(i/4); const y=scaleY(v,min,max,h); ctx.beginPath(); ctx.moveTo(l-4,y); ctx.lineTo(r,y); ctx.stroke(); ctx.fillText(fmtAxis(v,opt),l-7,y); } ctx.strokeStyle=\'#d8e0e3\'; ctx.beginPath(); ctx.moveTo(l,t); ctx.lineTo(l,b); ctx.lineTo(r,b); ctx.stroke(); if(baseline!=null && baseline>=min && baseline<=max){ const y=scaleY(baseline,min,max,h); ctx.strokeStyle=\'#a33a3a\'; ctx.setLineDash([4,3]); ctx.beginPath(); ctx.moveTo(l,y); ctx.lineTo(r,y); ctx.stroke(); ctx.setLineDash([]); ctx.fillStyle=\'#a33a3a\'; ctx.fillText(fmtAxis(baseline,opt),l-7,y); } ctx.textAlign=\'left\'; ctx.textBaseline=\'alphabetic\'; }\nfunction scaleY(v,min,max,h){ const span=Math.max(1e-9,max-min); return h-AX.b-((Number(v)-min)/span)*(h-AX.b-AX.t); }\nfunction drawXLabel(ctx,x,y,label,rotate=false){ ctx.save(); ctx.translate(x,y); if(rotate) ctx.rotate(-0.65); ctx.fillStyle=\'#172126\'; ctx.font=\'10px sans-serif\'; ctx.textAlign=rotate?\'left\':\'center\'; ctx.fillText(String(label||\'\').slice(0,18),0,0); ctx.restore(); }\nfunction drawXTicks(ctx,w,h,labels){ const l=AX.l,r=w-AX.r,b=h-AX.b; if(!labels.length) return; const step=Math.max(1,Math.ceil(labels.length/6)); ctx.strokeStyle=\'#d8e0e3\'; ctx.fillStyle=\'#172126\'; labels.forEach((label,i)=>{ if(i!==0 && i!==labels.length-1 && i%step!==0) return; const x=l+(r-l)*(i/Math.max(1,labels.length-1)); ctx.beginPath(); ctx.moveTo(x,b); ctx.lineTo(x,b+4); ctx.stroke(); drawXLabel(ctx,x,b+14,label,labels.length>8); }); }\nfunction drawBars(c, rows, opt={}){ const ctx=c.getContext(\'2d\'), w=c.width, h=c.height; const vals=rows.map(r=>Number(r.value)).filter(Number.isFinite); const min=Math.min(0,opt.baseline??0,...vals), max=Math.max(opt.baseline??0,...vals); drawAxes(ctx,w,h,min,max,opt.baseline,opt); const l=AX.l,r=w-AX.r,b=h-AX.b; const bw=Math.max(8,(r-l)/Math.max(1,rows.length)); rows.forEach((row,i)=>{ const v=Number(row.value); if(!Number.isFinite(v)) return; const x=l+i*bw, y=scaleY(Math.max(v,0),min,max,h), y0=scaleY(Math.min(0,opt.baseline??0),min,max,h); ctx.fillStyle=v>=1?\'#247a4b\':\'#8f2d56\'; ctx.fillRect(x,Math.min(y,y0),Math.max(4,bw-4),Math.abs(y0-y)); if(rows.length<=18 || i%Math.ceil(rows.length/10||1)===0 || i===rows.length-1){ ctx.strokeStyle=\'#d8e0e3\'; ctx.beginPath(); ctx.moveTo(x+bw/2,b); ctx.lineTo(x+bw/2,b+4); ctx.stroke(); drawXLabel(ctx,x+2,h-8,shortName(row.label),true); } }); }\nfunction drawGroupedBars(c, rows, opt={}){ const ctx=c.getContext(\'2d\'), w=c.width, h=c.height; const vals=rows.flatMap(r=>[Number(r.a),Number(r.b)]).filter(Number.isFinite); const min=Math.min(0,...vals), max=Math.max(opt.percent?1:0,...vals); drawAxes(ctx,w,h,min,max,opt.baseline,opt); const l=AX.l,r=w-AX.r,b=h-AX.b; const bw=Math.max(10,(r-l)/Math.max(1,rows.length)); rows.forEach((row,i)=>{ const x=l+i*bw; [[\'a\',\'#006d77\'],[\'b\',\'#8f2d56\']].forEach(([k,color],j)=>{ const v=Number(row[k]); if(!Number.isFinite(v)) return; const y=scaleY(v,min,max,h), y0=scaleY(0,min,max,h); ctx.fillStyle=color; ctx.fillRect(x+j*(bw/2),Math.min(y,y0),Math.max(3,bw/2-3),Math.abs(y0-y)); }); if(rows.length<=18 || i%Math.ceil(rows.length/10||1)===0 || i===rows.length-1){ ctx.strokeStyle=\'#d8e0e3\'; ctx.beginPath(); ctx.moveTo(x+bw/2,b); ctx.lineTo(x+bw/2,b+4); ctx.stroke(); drawXLabel(ctx,x,b+16,String(row.label).slice(0,14),rows.length>10); } }); ctx.fillStyle=\'#006d77\'; ctx.fillText(opt.a||\'A\',AX.l,18); ctx.fillStyle=\'#8f2d56\'; ctx.fillText(opt.b||\'B\',AX.l+56,18); }\nfunction drawLines(c, series, opt={}){ const ctx=c.getContext(\'2d\'), w=c.width, h=c.height; const vals=series.flatMap(s=>s.rows.map(r=>Number(r.y))).filter(Number.isFinite); if(!vals.length){ ctx.clearRect(0,0,w,h); ctx.fillStyle=\'#637279\'; ctx.fillText(\'データなし\',AX.l,80); return; } let min=Math.min(...vals), max=Math.max(...vals); if(opt.baseline!=null){ min=Math.min(min,opt.baseline); max=Math.max(max,opt.baseline); } if(min===max){ min-=1; max+=1; } drawAxes(ctx,w,h,min,max,opt.baseline,opt); const l=AX.l,r=w-AX.r; const colors=[\'#006d77\',\'#8f2d56\',\'#247a4b\',\'#a76300\',\'#1769c2\',\'#c83232\',\'#24824d\',\'#202529\']; const ref=(series.find(s=>s.rows&&s.rows.length)||{rows:[]}).rows; drawXTicks(ctx,w,h,ref.map(row=>row.x)); series.forEach((s,si)=>{ const rows=s.rows.filter(row=>Number.isFinite(Number(row.y))); ctx.strokeStyle=colors[si%colors.length]; ctx.lineWidth=2; ctx.beginPath(); rows.forEach((row,i)=>{ const x=l+(r-l)*(i/Math.max(1,rows.length-1)); const y=scaleY(row.y,min,max,h); if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y); }); ctx.stroke(); ctx.fillStyle=colors[si%colors.length]; ctx.fillText(shortName(s.label).slice(0,28),AX.l+6,18+si*12); }); }\nloadReport();\n</script>\n</body>\n</html>'

HTML = '<!doctype html>\n<html lang="ja">\n<head>\n  <meta charset="utf-8">\n  <meta name="viewport" content="width=device-width, initial-scale=1">\n  <title>BOAT RACE AI Ops</title>\n  <style>\n    :root { color-scheme: light; --ink:#172126; --muted:#637279; --line:#d8e0e3; --band:#f3f6f7; --accent:#006d77; --accent2:#8f2d56; --ok:#247a4b; --warn:#a76300; --bad:#a33a3a; font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }\n    * { box-sizing: border-box; } body { margin:0; color:var(--ink); background:#fff; font-size:11px; }\n    header { display:grid; grid-template-columns:auto 1fr; align-items:center; gap:6px; padding:3px 6px; border-bottom:1px solid var(--line); background:#fff; position:sticky; top:0; z-index:6; overflow:hidden; }\n    h1 { margin:0; font-size:12px; letter-spacing:0; white-space:nowrap; } main { display:grid; grid-template-columns:600px 1fr; min-height:calc(100vh - 29px); padding-bottom:18px; }\n    aside { background:var(--band); border-right:1px solid var(--line); padding:5px; overflow:auto; } section { padding:6px 8px; min-width:0; }\n    input, select, button { height:22px; border:1px solid var(--line); border-radius:4px; padding:0 5px; background:#fff; color:var(--ink); font:inherit; }\n    button { background:var(--accent); border-color:var(--accent); color:#fff; cursor:pointer; } .toolbar { display:flex; gap:4px; align-items:center; justify-content:flex-end; flex-wrap:nowrap; min-width:0; overflow:hidden; } #clock { white-space:nowrap; overflow:hidden; text-overflow:ellipsis; } #raceDate { display:none; } #venueFilter { width:74px; }\n    .stats { display:grid; grid-template-columns:repeat(5,minmax(68px,1fr)); gap:1px; background:var(--line); border:1px solid var(--line); }\n    .stat { background:#fff; padding:6px; min-width:0; } .stat b { display:block; font-size:17px; line-height:1.1; } .stat span { color:var(--muted); font-size:10px; }\n    .venue-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:3px; margin-top:0; }\n    .venue { background:#fff; border:1px solid var(--line); border-radius:4px; padding:2px 3px; cursor:pointer; min-height:31px; }\n    .venue.active { border-color:var(--accent); box-shadow:inset 3px 0 0 var(--accent); }\n    .venue b { display:flex; align-items:center; justify-content:space-between; gap:3px; font-size:10px; line-height:1.05; white-space:nowrap; overflow:hidden; }\n    .venue .next { display:grid; grid-template-columns:minmax(34px,1fr) auto auto; gap:2px; align-items:center; margin:1px 0 0; padding:1px 2px; background:#f8fafb; border:1px solid #edf1f2; border-radius:3px; }\n    .venue .next strong { font-size:9px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; } .venue .next span { color:var(--muted); font-size:9px; white-space:nowrap; } .venue .next .od { color:var(--ink); font-weight:600; overflow:hidden; text-overflow:ellipsis; }\n    .metrics { display:grid; grid-template-columns:repeat(3,1fr); gap:1px; margin-bottom:1px; background:var(--line); border:1px solid var(--line); }\n    .metric { background:#fff; padding:1px 2px; text-align:right; min-width:0; }\n    .metric b { display:inline; font-size:9px; line-height:1; } .metric span { display:inline; color:var(--muted); font-size:8px; line-height:1; margin-left:1px; }\n    .subline { display:grid; grid-template-columns:28px 1fr; gap:2px; color:var(--muted); font-size:9px; line-height:1.1; white-space:nowrap; overflow:hidden; }\n    .subline strong { color:var(--ink); font-weight:600; overflow:hidden; text-overflow:ellipsis; }\n    .badge { display:inline-block; border-radius:999px; padding:1px 4px; color:#fff; background:var(--muted); font-size:9px; font-weight:700; white-space:nowrap; }\n    .live,.候補 { background:var(--accent); } .結果待 { background:var(--warn); } .done,.確定 { background:var(--ok); } .wait,.T-10超過,.T-5超過 { background:var(--warn); } .締切後 { background:var(--bad); } .venue.s-live { background:#effafa; } .venue.s-wait { background:#fff8e8; } .venue.s-done { background:#eef8f2; } .venue.s-none { background:#fafbfb; } .venue.urgent { background:#fff1e6; border-color:#d99a5b; } .venue.due { background:#ffe4db; border-color:#c95c43; } tr.near { background:#fff7db; } tr.soon { background:#fff0e3; } tr.due { background:#ffe2d8; } tr.pick { background:#eaf8f6; } tr.pick.due { background:#ffdcd2; }\n    .timeline-frame { border:1px solid var(--line); border-top:2px solid var(--accent); background:#fff; margin-bottom:7px; }\n    .timeline-frame h2 { margin:0; padding:3px 5px; font-size:12px; letter-spacing:0; display:flex; justify-content:space-between; gap:6px; border-bottom:1px solid var(--line); }\n    .timeline-scroll { max-height:190px; overflow-y:auto; scrollbar-gutter:stable; } .timeline-scroll table { table-layout:fixed; } .timeline-scroll th,.timeline-scroll td { white-space:nowrap; overflow:hidden; text-overflow:ellipsis; } .timeline-scroll th:nth-child(1),.timeline-scroll td:nth-child(1){width:34px;} .timeline-scroll th:nth-child(2),.timeline-scroll td:nth-child(2){width:70px;} .timeline-scroll th:nth-child(3),.timeline-scroll td:nth-child(3){width:76px;} .timeline-scroll th:nth-child(4),.timeline-scroll td:nth-child(4){width:82px;} .timeline-scroll th:nth-child(5),.timeline-scroll td:nth-child(5){width:56px;} .timeline-scroll th:nth-child(6),.timeline-scroll td:nth-child(6){width:50px;} .timeline-scroll th:nth-child(7),.timeline-scroll td:nth-child(7){width:50px;} .timeline-scroll th:nth-child(8),.timeline-scroll td:nth-child(8){width:64px;}\n    .timeline-scroll th { top:0; } .timeline-scroll tr { height:20px; } .timeline-scroll th,.timeline-scroll td { padding:1px 3px; line-height:1.05; vertical-align:middle; }\n    .grid2 { display:grid; grid-template-columns:1fr; gap:8px; margin-top:7px; }\n    .panel { border-top:2px solid var(--accent); padding-top:5px; min-width:0; } .panel h2 { margin:0 0 4px; font-size:12px; letter-spacing:0; display:flex; justify-content:space-between; gap:6px; }\n    table { width:100%; border-collapse:collapse; table-layout:fixed; } th,td { border-bottom:1px solid var(--line); padding:3px 4px; text-align:right; vertical-align:top; overflow-wrap:anywhere; }\n    th { color:var(--muted); font-weight:700; background:#fafbfb; position:sticky; top:35px; z-index:2; } th:first-child,td:first-child { text-align:left; }\n    tr.pick { background:#f2fbfa; } tr.final { background:#eef5ff; } tr.late { color:var(--muted); } tr.nowline { background:#fff9e9; }\n    .mono { font-variant-numeric:tabular-nums; font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; } .muted { color:var(--muted); }\n    .entries { display:grid; grid-template-columns:repeat(6,minmax(60px,1fr)); gap:1px; background:var(--line); border:1px solid var(--line); margin:4px 0 6px; }\n    .entry { background:#fff; min-height:50px; padding:4px; } .lane { display:inline-grid; place-items:center; width:18px; height:18px; border:1px solid var(--line); font-weight:700; margin-bottom:2px; }\n    canvas { width:100%; height:105px; border:1px solid var(--line); background:#fff; } .empty { color:var(--muted); padding:10px 0; }\n    .lane-bg1 { background:#fff !important; color:#111 !important; }\n    .lane-bg2 { background:#202529 !important; color:#fff !important; }\n    .lane-bg3 { background:#c83232 !important; color:#fff !important; }\n    .lane-bg4 { background:#1769c2 !important; color:#fff !important; }\n    .lane-bg5 { background:#f5d84a !important; color:#111 !important; }\n    .lane-bg6 { background:#24824d !important; color:#fff !important; }\n    .pred-matrix-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:2px; margin:2px 0 4px; }\n    .pred-matrix { min-width:0; border:1px solid var(--line); background:#fff; overflow:hidden; }\n    .pred-matrix-title { display:grid; grid-template-columns:auto minmax(0,1fr) auto; align-items:center; gap:3px; padding:2px 3px; border-bottom:1px solid rgba(0,0,0,.18); font-size:9px; line-height:1.05; font-weight:900; white-space:nowrap; overflow:hidden; }\n    .pred-title-main { display:flex; align-items:center; gap:3px; min-width:0; }\n    .pred-title-lane,.pred-chip { display:inline-grid; place-items:center; border:1px solid rgba(0,0,0,.28); font-weight:900; line-height:1; }\n    .pred-title-lane { width:14px; height:14px; font-size:9px; }\n    .pred-title-order { display:flex; align-items:center; justify-content:center; gap:1px; min-width:0; overflow:hidden; }\n    .pred-chip { width:12px; height:12px; font-size:7px; }\n    .pred-title-order em { margin-left:2px; font-style:normal; font-size:7px; opacity:.78; }\n    .pred-title-best { color:inherit; opacity:.76; font-weight:700; overflow:hidden; text-overflow:ellipsis; }\n    .pred-matrix table { width:100%; table-layout:fixed; border-collapse:collapse; margin:0; }\n    .pred-matrix th,.pred-matrix td { width:16.666%; height:22px; padding:0 1px; border:1px solid #dde5e7; text-align:center; vertical-align:middle; line-height:1; white-space:nowrap; overflow:hidden; }\n    .pred-matrix th { position:static; top:auto; z-index:auto; padding:0; font-size:7.5px; font-weight:900; }\n    .pred-corner,.pred-axis { letter-spacing:0; }\n    .pred-corner b,.pred-axis b { display:block; font-size:9px; line-height:1; }\n    .pred-corner small,.pred-axis small { display:block; margin-top:1px; font-size:6px; line-height:1; opacity:.78; }\n    .pred-matrix td { position:relative; cursor:pointer; background:#fff; color:var(--ink); }\n    .pred-matrix td:hover { outline:1px solid var(--accent2); outline-offset:-1px; }\n    .pred-matrix td.invalid { background:#f3f5f5; cursor:default; color:#bdc5c8; }\n    .pred-matrix td.model-top { box-shadow:inset 0 0 0 1px var(--accent2); }\n    .pred-matrix td.odds-r1 { background:#005f69; color:#fff; border-color:#004b52; }\n    .pred-matrix td.odds-r2 { background:#268b8a; color:#fff; border-color:#1d7474; }\n    .pred-matrix td.odds-r3 { background:#8fcac5; color:#102f32; border-color:#68b4af; }\n    .pred-matrix td.odds-r4 { background:#d9f0ed; color:#173538; border-color:#b8dfdb; }\n    .pred-odds { display:block; font-size:10px; font-weight:900; letter-spacing:0; }\n    .pred-prob { display:block; margin-top:1px; font-size:6.8px; font-weight:600; opacity:.72; letter-spacing:0; }\n    .ops-status { position:fixed; left:0; right:0; bottom:0; z-index:8; padding:2px 8px; border-top:1px solid var(--line); background:#fff; color:var(--muted); font-size:9px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; } .subtable-title { margin-top:10px; color:var(--muted); font-size:12px; } .hit { color:var(--ok); font-weight:700; } .miss { color:var(--bad); font-weight:700; }\n    @media (max-width:1320px) { main,.grid2 { grid-template-columns:1fr; } aside { border-right:0; border-bottom:1px solid var(--line); max-height:42vh; } }\n    @media (max-width:720px) { .stats { grid-template-columns:repeat(2,minmax(120px,1fr)); } .venue-grid { grid-template-columns:repeat(4,minmax(0,1fr)); } .entries { grid-template-columns:repeat(3,minmax(72px,1fr)); } header { grid-template-columns:auto 1fr; align-items:center; } }\n  \n    .live-wipe { position:fixed; right:8px; bottom:18px; width:min(540px,calc(100vw - 16px)); min-width:min(360px,calc(100vw - 16px)); resize:both; overflow:hidden; background:#050808; border:1px solid #263b40; box-shadow:0 10px 24px rgba(0,0,0,.30); z-index:30; }\n    .live-wipe.hidden { display:none; }\n    .live-wipe-video { position:relative; width:100%; aspect-ratio:16/9; min-height:288px; max-height:calc(100vh - 80px); overflow:hidden; background:#050808; }\n    .live-wipe-video iframe { position:absolute; inset:0; width:100%; height:100%; border:0; background:#050808; transform-origin:center center; }\n    .live-wipe.zoom .live-wipe-video iframe { width:122%; height:122%; transform:translate(-9%,-9%); }\n    .live-wipe-head { position:absolute; left:0; right:0; top:0; display:flex; align-items:center; justify-content:space-between; gap:8px; padding:4px 6px; color:#fff; background:linear-gradient(180deg,rgba(0,0,0,.72),rgba(0,0,0,.08)); z-index:2; pointer-events:none; }\n    .live-wipe-title { font-size:11px; font-weight:800; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; text-shadow:0 1px 2px #000; }\n    .live-wipe-actions { display:flex; gap:4px; align-items:center; pointer-events:auto; }\n    .live-wipe-actions a,.live-wipe-actions button { height:17px; padding:0 5px; border:1px solid rgba(255,255,255,.35); border-radius:3px; background:rgba(16,28,31,.78); color:#fff; font-size:10px; text-decoration:none; line-height:15px; cursor:pointer; }\n    .live-wipe-meta { position:absolute; left:5px; right:5px; bottom:5px; display:flex; gap:3px; flex-wrap:wrap; z-index:2; pointer-events:none; }\n    .live-wipe-meta span { max-width:100%; padding:2px 4px; border-radius:3px; background:rgba(0,0,0,.68); color:#fff; font-size:9px; line-height:1.12; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; text-shadow:0 1px 2px #000; }\n    @media (max-width:1320px) { .live-wipe { width:min(480px,calc(100vw - 16px)); min-width:min(320px,calc(100vw - 16px)); } .live-wipe-video { min-height:252px; } }\n    @media (max-width:720px) { .live-wipe { width:calc(100vw - 10px); min-width:0; right:5px; bottom:18px; } .live-wipe-video { min-height:205px; max-height:58vh; } }\n\n  \n    .live-wipe { width:360px; min-width:260px; max-width:min(360px,calc(100vw - 16px)); right:8px; bottom:18px; }\n    .live-wipe-video { min-height:0; height:auto; aspect-ratio:16/9; }\n    .live-wipe-head { padding:3px 5px; }\n    .live-wipe-title { font-size:10px; }\n    .live-wipe-actions a,.live-wipe-actions button { height:16px; padding:0 4px; font-size:9px; line-height:14px; }\n    .live-wipe-meta { left:4px; right:4px; bottom:4px; gap:2px; }\n    .live-wipe-meta span { padding:1px 3px; font-size:8.5px; }\n    .live-wipe.zoom .live-wipe-video iframe { width:112%; height:112%; transform:translate(-5.35%,-5.35%); }\n    @media (max-width:720px) { .live-wipe { width:320px; min-width:240px; max-width:calc(100vw - 10px); } .live-wipe-video { min-height:0; max-height:none; } }\n\n  \n    .entries { grid-template-columns:repeat(6,minmax(86px,1fr)); gap:2px; background:transparent; border:0; }\n    .entry { min-height:44px; padding:3px 4px; border:1px solid rgba(0,0,0,.18); color:#111; overflow:hidden; }\n    .entry-main { display:grid; grid-template-columns:20px minmax(0,1fr); gap:4px; align-items:start; min-width:0; }\n    .entry-info { min-width:0; display:grid; gap:1px; }\n    .entry .lane { margin:0; width:20px; height:20px; border:1px solid currentColor; background:rgba(255,255,255,.22); font-weight:800; }\n    .entry .racer-name { min-width:0; font-weight:900; line-height:1.08; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }\n    .entry-meta { display:flex; gap:3px; justify-content:flex-start; margin-top:0; font-size:8.5px; line-height:1.05; opacity:.88; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }\n    .entry-meta span { min-width:0; overflow:hidden; text-overflow:ellipsis; }\n    .entry-actions { display:flex; gap:2px; margin-top:3px; }\n    .entry-actions button { flex:1 1 0; min-width:0; }\n    .entry.lane1 { background:#fff; color:#111; }\n    .entry.lane2 { background:#202529; color:#fff; }\n    .entry.lane3 { background:#c83232; color:#fff; }\n    .entry.lane4 { background:#1769c2; color:#fff; }\n    .entry.lane5 { background:#f5d84a; color:#111; }\n    .entry.lane6 { background:#24824d; color:#fff; }\n    .live-wipe.zoom .live-wipe-video iframe { width:170%; height:170%; transform:translate(-20.6%,-20.6%); }\n    @media (max-width:720px) { .entries { grid-template-columns:repeat(3,minmax(82px,1fr)); } }\n\n  \n    .live-wipe-video iframe { width:100%; height:100%; transform:none; }\n    .live-wipe.zoom .live-wipe-video iframe { width:190%; height:190%; transform:translate(-23.7%,-23.7%); }\n\n  \n    .navbtn { height:22px; padding:0 6px; border-radius:4px; font-size:10px; background:#fff; color:var(--accent); }\n    .archive-panel { min-width:0; }\n    .archive-head { display:flex; align-items:center; justify-content:space-between; gap:6px; margin-bottom:4px; }\n    .archive-head h2 { margin:0; font-size:12px; }\n    .archive-tabs { display:flex; gap:3px; align-items:center; flex-wrap:wrap; }\n    .archive-tabs button,.entry-actions button,.race-title-actions button { height:18px; padding:0 5px; border-radius:3px; font-size:9px; background:#fff; color:var(--accent); border-color:#9bbdc1; }\n    .archive-meta { color:var(--muted); font-size:10px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }\n    .archive-content { max-height:430px; overflow:auto; border-top:1px solid var(--line); }\n    .archive-kv { display:grid; grid-template-columns:repeat(4,minmax(82px,1fr)); gap:1px; background:var(--line); border:1px solid var(--line); margin:4px 0; }\n    .archive-kv div { background:#fff; padding:3px 4px; min-width:0; }\n    .archive-kv b { display:block; font-size:12px; line-height:1.1; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }\n    .archive-kv span { color:var(--muted); font-size:9px; }\n    .archive-table { margin-top:4px; table-layout:auto; }\n    .archive-table th,.archive-table td { padding:2px 4px; font-size:10px; line-height:1.15; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }\n    .archive-table tr.clickable,.pred-link { cursor:pointer; }\n    .archive-table tr.clickable:hover,.pred-link:hover { background:#eef8f7; }\n    .entry-actions { display:flex; gap:2px; margin-top:2px; }\n    .entry-actions button { flex:1 1 auto; min-width:0; }\n    .race-title-line { display:flex; align-items:center; gap:5px; min-width:0; }\n    .race-title-name { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }\n    .race-title-actions { display:flex; gap:3px; }\n    .venue-archive { height:15px; min-width:18px; padding:0 3px; margin-left:3px; font-size:9px; border-radius:3px; background:#fff; color:var(--accent); border-color:#9bbdc1; }\n    @media (max-width:720px) { .archive-kv { grid-template-columns:repeat(2,minmax(86px,1fr)); } .archive-content { max-height:320px; } }\n\n  </style>\n</head>\n<body>\n  <header><h1>BOAT RACE AI Ops</h1><div class="toolbar"><input id="raceDate" type="hidden"><select id="venueFilter"><option value="">全場</option></select><button id="reload">更新</button><button id="navPast" class="navbtn" type="button">過去</button><button id="navToday" class="navbtn" type="button">当日</button><button id="navStats" class="navbtn" type="button">統計</button><a href="/reports/models" style="height:18px;padding:1px 6px;border:1px solid var(--accent);border-radius:3px;background:#fff;color:var(--accent);text-decoration:none;font-size:10px;line-height:15px;">モデル</a><a href="/reports/roadmap" style="height:18px;padding:1px 6px;border:1px solid var(--accent);border-radius:3px;background:#fff;color:var(--accent);text-decoration:none;font-size:10px;line-height:15px;">懸案</a><span id="clock" class="muted mono"></span></div></header>\n  <main>\n    <aside><div id="venueGrid" class="venue-grid"></div></aside>\n    <section>\n      <div class="timeline-frame">\n        <h2><span>確定4件・購入候補・今後レース</span><span id="timelineInfo" class="muted"></span></h2>\n        <div class="timeline-scroll"><table><thead><tr><th>区分</th><th>場/R</th><th>締切/出</th><th>判定/結果</th><th>モデル予測</th><th>確率</th><th>オッズ</th><th>EV/払戻</th></tr></thead><tbody id="actionRows"></tbody></table></div>\n      </div>\n      <div class="grid2">\n        <div class="panel"><h2><span id="raceTitle">レース詳細</span><span id="accuracy" class="muted"></span></h2><div id="entries" class="entries"></div><div id="predictions" class="pred-matrix-grid"></div><h2 style="margin-top:6px;"><span>オッズ推移</span><select id="combo"></select></h2><canvas id="oddsChart" width="720" height="200"></canvas><div id="backtest" class="empty"></div></div>\n\n        <div class="panel archive-panel">\n          <div class="archive-head">\n            <h2 id="archiveTitle">データ参照</h2>\n            <div class="archive-tabs">\n              <button id="archivePast" type="button">過去総合</button>\n              <button id="archiveToday" type="button">当日</button>\n              <button id="archiveStats" type="button">統計</button>\n            </div>\n          </div>\n          <div id="archiveMeta" class="archive-meta">選択したレース/選手/場/モーター/ボート/枠の蓄積データを表示</div>\n          <div id="archiveContent" class="archive-content"></div>\n        </div>\n\n      </div>\n    </section>\n  </main>\n  <footer id="dataStatus" class="ops-status">取得状態を読み込み中</footer>\n  <div id="liveWipe" class="live-wipe hidden">\n    <div class="live-wipe-video">\n      <iframe id="liveWipeFrame" title="BOATCAST live" allow="autoplay; fullscreen; picture-in-picture"></iframe>\n      <div class="live-wipe-head">\n        <div id="liveWipeTitle" class="live-wipe-title">LIVE</div>\n        <div class="live-wipe-actions">\n          <button id="liveWipeClose" type="button" title="このレースのワイプを閉じる">X</button>\n          <button id="liveWipeZoom" type="button">動画</button>\n          <button id="liveWipeSelect" type="button">選択</button>\n          <a id="liveWipeOfficial" href="#" target="_blank" rel="noopener">出走</a>\n          <a id="liveWipeOpen" href="#" target="_blank" rel="noopener">公式</a>\n        </div>\n      </div>\n      <div id="liveWipeMeta" class="live-wipe-meta"></div>\n    </div>\n  </div>\n\n<script>\nconst state = { raceId:null, jcd:"", combo:"1-2-3", nowIso:null };\nconst $ = id => document.getElementById(id);\nfunction jstDate(){\n  const parts = new Intl.DateTimeFormat("en-CA", { timeZone:"Asia/Tokyo", year:"numeric", month:"2-digit", day:"2-digit" }).formatToParts(new Date());\n  const m = Object.fromEntries(parts.map(p => [p.type, p.value]));\n  return `${m.year}-${m.month}-${m.day}`;\n}\nconst today = jstDate();\n$("raceDate").value = "";\n$("reload").onclick = loadAll;\n$("venueFilter").onchange = () => { state.jcd = $("venueFilter").value; state.raceId = null; loadAll(); };\n$("combo").onchange = () => { state.combo = $("combo").value; loadOdds(); };\nasync function getJson(url){ const res = await fetch(url,{cache:"no-store"}); if(!res.ok) throw new Error(await res.text()); return await res.json(); }\nfunction stat(label,value){ return `<div class="stat"><b>${value ?? "-"}</b><span>${label}</span></div>`; }\nfunction pct(v){ return v == null ? "-" : `${(Number(v)*100).toFixed(2)}%`; }\nfunction num(v){ return v == null ? "-" : Number(v).toFixed(3); }\nfunction hm(v){ if(!v) return "-"; return new Date(v).toLocaleTimeString("ja-JP",{hour:"2-digit",minute:"2-digit",timeZone:"Asia/Tokyo"}); }\nfunction age(v){ if(!v) return "-"; const m = Math.floor((Date.now()-new Date(v).getTime())/60000); return `${hm(v)} (${m}分前)`; }\nfunction minLabel(v){ return v == null ? "-" : `${v}分`; }\nfunction cls(v){ return String(v || "").replaceAll(" ",""); }\nfunction statusClass(v){ return v==="監視中" ? "live" : v==="終了" ? "done" : (v==="未取得"||v==="開催なし") ? "" : "wait"; }\nfunction footerStatus(s, prog, nowIso){ const h=prog.historical||{}, t=prog.today||{}; return `過去分: 番組LZH 残${h.program_remaining_days ?? "-"}日 / 結果LZH 残${h.result_remaining_days ?? "-"}日 / parsed ${h.races ?? "-"}R / 結果 ${h.result_races ?? "-"}R | 本日分: レース 残${t.race_remaining ?? "-"} / 出走 残${t.racelist_remaining ?? "-"} / odds 残${t.odds_remaining ?? "-"} / 結果 残${t.final_remaining ?? "-"} / 予測 ${s.predictions} | 更新 ${nowIso.replace("T"," ").slice(0,19)}`; }\nfunction statusTitle(v){ return v==="監視中" ? "出走表とオッズを取得済みで、ライブ更新対象です。" : v==="出走表" ? "出走表は取得済み、オッズは未取得です。" : v==="開催なし" ? "本日はこの場の開催が確認されていません。" : v==="取得中" ? "当日レース情報を取得中です。" : v==="終了" ? "全レースの結果が入っています。" : "当日データはまだ未取得です。"; }\nfunction venueTone(v){ const m = v.minutes_to_next_deadline; if(m != null && m <= 5) return "due"; if(m != null && m <= 15) return "urgent"; if(v.status==="監視中") return "s-live"; if(v.status==="終了") return "s-done"; if(v.status==="未取得" || v.status==="開催なし") return "s-none"; return "s-wait"; }\nfunction rowTone(r, isPick, idx){ const m = r.minutes_to_deadline; const parts = []; if(isPick) parts.push("pick"); if(m != null && m <= 5) parts.push("due"); else if(m != null && m <= 15) parts.push("soon"); else if(idx < 4) parts.push("near"); return parts.join(" "); }\nfunction guideScore(r){ const p=r.buy_prediction||r.top_prediction||{}; return Number(p.expected_value ?? p.probability ?? 0); }\nfunction fallbackCandidates(upcoming, candidateMap){ if(candidateMap.size) return candidateMap; const rows=upcoming.filter(r => r.top_prediction && r.entries === 6 && r.time_status !== "締切後" && r.time_status !== "確定" && (r.minutes_to_deadline == null || r.minutes_to_deadline >= 5)).sort((a,b)=>guideScore(b)-guideScore(a)).slice(0,8); return new Map(rows.map(r => [r.race_id, r])); }\nfunction futureRows(rows){\n  const nowMs = state.nowIso ? new Date(state.nowIso).getTime() : Date.now();\n  return rows.filter(r => {\n    if(r.deadline_at) return new Date(r.deadline_at).getTime() >= nowMs;\n    return !["確定","締切後"].includes(r.time_status || "");\n  }).sort((a,b) => {\n    const av = a.deadline_at ? new Date(a.deadline_at).getTime() : Number.MAX_SAFE_INTEGER;\n    const bv = b.deadline_at ? new Date(b.deadline_at).getTime() : Number.MAX_SAFE_INTEGER;\n    return av - bv || String(a.jcd).localeCompare(String(b.jcd)) || Number(a.rno || 0) - Number(b.rno || 0);\n  });\n}\nasync function loadAll(){\n  const d = $("raceDate").value || "";\n  const loadId = Date.now();\n  state.loadId = loadId;\n  $("clock").textContent = "JST 読込中";\n  $("dataStatus").textContent = "当日一覧を読み込み中";\n  $("accuracy").textContent = "集計中";\n  const dayUrl = `/api/day?date=${encodeURIComponent(d)}&lite=1${state.jcd ? `&jcd=${state.jcd}` : ""}`;\n  try {\n    const [vcRes, dayRes] = await Promise.allSettled([\n      getJson(`/api/venues?date=${encodeURIComponent(d)}`),\n      getJson(dayUrl)\n    ]);\n    if(loadId !== state.loadId) return;\n    const vc = settledValue(vcRes, { venues:[] }, "venues");\n    const day = settledValue(dayRes, { now_jst:new Date().toISOString(), races:[] }, "day");\n    state.nowIso = day.now_jst;\n    state.dayRows = day.races || [];\n    $("clock").textContent = `JST ${(day.now_jst || "").replace("T"," ").slice(0,16)}`;\n    renderVenues(vc.venues || []);\n    renderActionTable(state.dayRows, state.guideCandidates || [], state.finishedRows || []);\n    $("dataStatus").textContent = `当日一覧 ${state.dayRows.length}R / 詳細集計は後追い中 / 更新 ${(day.now_jst || "").replace("T"," ").slice(0,19)}`;\n    scheduleSelectedDetail(false);\n    loadSecondary(d, loadId);\n  } catch(err) {\n    console.error(err);\n    $("dataStatus").textContent = `初回表示エラー: ${err.message || err}`;\n  }\n}\nasync function loadSecondary(d, loadId){\n  const [guideRes, liveRes] = await Promise.allSettled([\n    getJson(`/api/guide?date=${encodeURIComponent(d)}&before_minutes=5&limit=16&finished_limit=4`),\n    getJson(`/api/live-wipe?date=${encodeURIComponent(d)}`)\n  ]);\n  if(loadId !== state.loadId) return;\n  const guide = settledValue(guideRes, { candidates:[], finished:[] }, "guide");\n  state.guideCandidates = guide.candidates || [];\n  state.finishedRows = guide.finished || [];\n  renderActionTable(state.dayRows || [], state.guideCandidates, state.finishedRows);\n  const live = settledValue(liveRes, null, "live-wipe");\n  if(live) renderLiveWipe(live);\n  scheduleSelectedDetail(true);\n\n  const [summaryRes, progressRes, accuracyRes, backtestRes] = await Promise.allSettled([\n    getJson("/api/summary"),\n    getJson(`/api/progress?date=${encodeURIComponent(d)}`),\n    getJson(`/api/accuracy?date=${encodeURIComponent(d)}`),\n    getJson("/api/backtest")\n  ]);\n  if(loadId !== state.loadId) return;\n  const s = settledValue(summaryRes, null, "summary");\n  const prog = settledValue(progressRes, null, "progress");\n  const acc = settledValue(accuracyRes, null, "accuracy");\n  const bt = settledValue(backtestRes, null, "backtest");\n  if(s && prog) $("dataStatus").textContent = footerStatus(s, prog, state.nowIso || new Date().toISOString());\n  if(acc) $("accuracy").textContent = `本日 ${acc.evaluated || 0}R / 1着 ${pct(acc.winner_top1_accuracy)} / 3T5 ${pct(acc.trifecta_top5_hit_rate)}`;\n  if(bt) $("backtest").textContent = bt.available ? `BT ${bt.evaluated_races}R / 1着 ${pct(bt.winner_top1_accuracy)} / 3T5 ${pct(bt.trifecta_top5_hit_rate)}` : "バックテスト結果はまだありません。";\n}\nfunction settledValue(result, fallback, label){\n  if(result.status === "fulfilled") return result.value;\n  console.warn(`${label} load failed`, result.reason);\n  return fallback;\n}\nfunction scheduleSelectedDetail(allowRefresh){\n  const rows = state.dayRows || [];\n  let raceId = state.raceId;\n  if(!raceId && rows.length){\n    const byGuide = (state.guideCandidates || [])[0];\n    const next = byGuide || futureRows(rows).find(r => r.top_prediction) || rows.find(r => r.top_prediction) || rows[0];\n    raceId = next && next.race_id;\n    state.raceId = raceId || null;\n  }\n  if(!raceId) return;\n  const now = Date.now();\n  if(!allowRefresh && state.detailRaceId === raceId) return;\n  if(allowRefresh && state.detailRaceId === raceId && state.lastDetailAt && now - state.lastDetailAt < 55000) return;\n  state.detailRaceId = raceId;\n  state.lastDetailAt = now;\n  setTimeout(() => selectRace(raceId).catch(err => {\n    console.error(err);\n    $("raceTitle").textContent = `レース詳細エラー: ${err.message || err}`;\n  }), 0);\n}\n\nfunction renderVenues(items){\n  $("venueFilter").innerHTML = `<option value="">全場</option>` + items.map(v => `<option value="${v.code}">${v.name}</option>`).join("");\n  $("venueFilter").value = state.jcd;\n  $("venueGrid").innerHTML = items.map(v => `<div class="venue ${v.code === state.jcd ? "active" : ""} ${venueTone(v)}" data-jcd="${v.code}">\n    <b><span>${v.code} ${v.name}</span><span><button class="venue-archive" type="button" data-jcd="${v.code}" title="場の履歴">履</button><span class="badge ${statusClass(v.status)}" title="${statusTitle(v.status)}">${v.status}</span></span></b>\n    <div class="next"><strong>${v.next_rno ? `${v.next_rno}R ${hm(v.next_deadline_at)}` : "-"}</strong><span>${minLabel(v.minutes_to_next_deadline)}</span><span class="od">od ${hm(v.latest_odds_at)}</span></div>\n  </div>`).join("");\n  document.querySelectorAll(".venue").forEach(el => el.onclick = () => { state.jcd = el.dataset.jcd; state.raceId = null; loadAll(); });\n  document.querySelectorAll(".venue-archive").forEach(btn => btn.onclick = ev => { ev.stopPropagation(); loadArchiveView("history", { kind:"venue", jcd:btn.dataset.jcd || "" }); });\n}\n\nfunction renderActionTable(rows, candidates, finished){\n  let candidateMap = new Map((candidates || []).map(r => [r.race_id, r]));\n  const upcoming = futureRows(rows);\n  candidateMap = fallbackCandidates(upcoming, candidateMap);\n  const picks = upcoming.filter(r => candidateMap.has(r.race_id));\n  const others = upcoming.filter(r => !candidateMap.has(r.race_id));\n  const action = [\n    ...finished.map(r => ({ kind:(Number(r.result_rows || 0) >= 3 ? "確定" : "結果待"), item:r, source:r, isPick:false, isFinal:true })),\n    ...picks.map(r => ({ kind:"候補", item:candidateMap.get(r.race_id), source:r, isPick:true, isFinal:false })),\n    ...others.map(r => ({ kind:"予定", item:r, source:r, isPick:false, isFinal:false })),\n  ].sort((a,b) => {\n    const av = actionTime(a);\n    const bv = actionTime(b);\n    if(av !== bv) return av - bv;\n    return String(a.source.jcd).localeCompare(String(b.source.jcd)) || Number(a.source.rno || 0) - Number(b.source.rno || 0);\n  });\n  $("timelineInfo").textContent = `直近終了 ${finished.length}R / 候補 ${picks.length}R / 今後 ${upcoming.length}R / 締切昇順`;\n  $("actionRows").innerHTML = action.map((row,idx) => {\n    const r = row.source;\n    const item = row.item || r;\n    const p = item.top_prediction || {};\n    const ev = (item.buy_prediction || p || {}).expected_value;\n    const shortRace = `${r.venue_name}${r.rno}R`;\n    const when = timePair(r);\n    if(row.isFinal){\n      const ready = Number(item.result_rows || 0) >= 3 && item.result_combination;\n      const mark = ready ? (item.top_hit ? "的中" : (item.top5_hit ? "上位5" : "外れ")) : "結果待";\n      const result = ready ? `${item.result_combination} ${mark}` : "結果待";\n      const payout = ready ? (item.trifecta_payout_yen == null ? "-" : `${Number(item.trifecta_payout_yen).toLocaleString("ja-JP")}円`) : "-";\n      const badge = ready ? "確定" : "結果待";\n      return `<tr class="${ready ? "final" : "pending"}" data-race="${r.race_id}"><td><span class="badge ${badge}">${badge}</span></td><td title="${r.venue_name} ${r.rno}R ${r.title || ""}"><b>${shortRace}</b></td><td class="mono" title="締切 ${hm(r.deadline_at)} / 出走 ${hm(r.race_time_at)}">${when}</td><td class="mono" title="${result}">${result}</td><td class="mono">${p.combination || "-"}</td><td>${pct(p.probability)}</td><td>${num(p.odds)}</td><td class="mono">${payout}</td></tr>`;\n    }\n    const status = r.time_status === "T-10超過" ? "T-5超過" : r.time_status;\n    const verdict = `<span class="badge ${cls(status)}">${status}</span>${row.isPick ? ` <span class="badge 候補">候補</span>` : ""}`;\n    const tone = rowTone(r, row.isPick, idx);\n    return `<tr class="${tone}" data-race="${r.race_id}"><td><span class="badge ${row.isPick ? "候補" : ""}">${row.kind}</span></td><td title="${r.venue_name} ${r.rno}R ${r.title || ""}"><b>${shortRace}</b></td><td class="mono" title="締切 ${hm(r.deadline_at)} / 出走 ${hm(r.race_time_at)}">${when} <span class="muted">${minLabel(r.minutes_to_deadline)}</span></td><td>${verdict}</td><td class="mono">${p.combination || "-"}</td><td>${pct(p.probability)}</td><td>${num(p.odds)}</td><td>${num(ev)}</td></tr>`;\n  }).join("") || `<tr><td colspan="8" class="empty">表示対象のレース情報はありません。</td></tr>`;\n  document.querySelectorAll("#actionRows tr[data-race]").forEach(row => row.onclick = () => selectRace(row.dataset.race));\n}\nfunction actionTime(row){\n  const r = row.source || {};\n  const item = row.item || r;\n  const value = row.isFinal ? (item.race_time_at || item.deadline_at || item.latest_odds_at) : (r.deadline_at || item.deadline_at);\n  const parsed = value ? new Date(value).getTime() : Number.MAX_SAFE_INTEGER;\n  return Number.isFinite(parsed) ? parsed : Number.MAX_SAFE_INTEGER;\n}\n\nfunction escapeHtml(v){\n  return String(v ?? "").replace(/[&<>"\']/g, ch => ({ "&":"&amp;", "<":"&lt;", ">":"&gt;", \'"\':"&quot;", "\'":"&#39;" }[ch]));\n}\nfunction racerDisplayName(e){\n  const raw = String(e.racer_name || "").trim();\n  const no = String(e.racer_no || "").trim();\n  if(raw && raw !== no && !/^\\d+$/.test(raw)) return raw;\n  return "選手名未取得";\n}\nfunction renderEntryCard(e){\n  const lane = Number(e.lane || 0);\n  const name = racerDisplayName(e);\n  const no = e.racer_no ? `#${e.racer_no}` : "#-";\n  const cls = e.racer_class || "-";\n  const branchOrigin = [e.branch, e.origin].filter(Boolean).join("/") || "-";\n  const machine = `M${e.motor_no || "-"} B${e.boat_no || "-"}`;\n  const meta = [no, cls, branchOrigin, machine];\n  return `<div class="entry lane${lane}" title="${escapeHtml(`${lane}号艇 ${name} ${meta.join(" ")}`)}">\n    <div class="entry-main"><span class="lane">${lane || "-"}</span><div class="entry-info"><span class="racer-name">${escapeHtml(name)}</span><div class="entry-meta">${meta.map(v => `<span>${escapeHtml(v)}</span>`).join("")}</div></div></div>\n    <div class="entry-actions">\n      <button type="button" data-archive-kind="racer" data-racer-no="${escapeHtml(e.racer_no || "")}">選手</button>\n      <button type="button" data-archive-kind="lane" data-lane="${lane || ""}">枠</button>\n      <button type="button" data-archive-kind="motor" data-motor-no="${escapeHtml(e.motor_no || "")}">M</button>\n      <button type="button" data-archive-kind="boat" data-boat-no="${escapeHtml(e.boat_no || "")}">B</button>\n    </div>\n  </div>`;\n}\nfunction tightPct(v){ if(v == null) return "-"; const n=Number(v)*100; if(!Number.isFinite(n)) return "-"; return n>=10 ? `${n.toFixed(0)}%` : n>=1 ? `${n.toFixed(1)}%` : `${n.toFixed(2)}%`; }\nfunction tightNum(v){ if(v == null) return "-"; const n=Number(v); if(!Number.isFinite(n)) return "-"; if(n>=100) return String(Math.round(n)); if(n>=10) return n.toFixed(1); return n.toFixed(2); }\nfunction oddsRankClass(rank){ const r=Number(rank || 0); if(!r) return ""; if(r<=6) return "odds-r1"; if(r<=18) return "odds-r2"; if(r<=36) return "odds-r3"; if(r<=60) return "odds-r4"; return ""; }\nfunction laneChips(lanes){ return lanes.map(v => `<span class="pred-chip lane-bg${v}">${v}</span>`).join(""); }\nfunction renderPredictionMatrix(predictions){\n  const root = $("predictions");\n  const preds = Array.isArray(predictions) ? predictions : [];\n  if(!preds.length){ root.innerHTML = `<div class="empty">予測はまだありません。</div>`; return; }\n  const byCombo = new Map();\n  const byFirst = new Map();\n  preds.forEach((p,idx) => {\n    const combo = String(p.combination || "");\n    const full = { ...p, modelRank:idx + 1 };\n    byCombo.set(combo, full);\n    const first = combo.split("-")[0];\n    if(first && !byFirst.has(first)) byFirst.set(first, full);\n  });\n  [...byCombo.values()].filter(p => Number.isFinite(Number(p.odds)) && Number(p.odds) > 0).sort((a,b) => Number(a.odds) - Number(b.odds)).forEach((p,idx) => { p.oddsRank = idx + 1; });\n  root.innerHTML = [1,2,3,4,5,6].map(first => {\n    const lanes = [1,2,3,4,5,6].filter(v => v !== first);\n    const top = byFirst.get(String(first));\n    const best = top ? `モデル${top.modelRank}位 ${tightPct(top.probability)}` : "-";\n    const head = `<thead><tr><th class="pred-corner lane-bg${first}" title="${first}号艇 1着固定"><b>${first}</b><small>1着</small></th>${lanes.map(third => `<th class="pred-axis lane-bg${third}" title="${third}号艇 3着候補"><b>${third}</b><small>3着</small></th>`).join("")}</tr></thead>`;\n    const rows = lanes.map(second => `<tr><th class="pred-axis lane-bg${second}" title="${second}号艇 2着候補"><b>${second}</b><small>2着</small></th>${lanes.map(third => predictionMatrixCell(first, second, third, byCombo)).join("")}</tr>`).join("");\n    return `<div class="pred-matrix"><div class="pred-matrix-title lane-bg${first}"><span class="pred-title-main"><b class="pred-title-lane lane-bg${first}">${first}</b><span>1着</span></span><span class="pred-title-order" title="列3着/行2着 ${lanes.join("-")}">${laneChips(lanes)}<em>列3/行2</em></span><span class="pred-title-best" title="${escapeHtml(best)}">${escapeHtml(best)}</span></div><table>${head}<tbody>${rows}</tbody></table></div>`;\n  }).join("");\n}\nfunction predictionMatrixCell(first, second, third, byCombo){\n  if(second===third) return `<td class="invalid" title="${first}-${second}-${third} 無効"></td>`;\n  const combo = `${first}-${second}-${third}`;\n  const p = byCombo.get(combo);\n  if(!p) return `<td class="invalid" title="${escapeHtml(combo)} 欠損"></td>`;\n  const cls = [oddsRankClass(p.oddsRank), p.modelRank <= 6 ? "model-top" : ""].filter(Boolean).join(" ");\n  const title = `${combo} オッズ ${num(p.odds)} 人気 ${p.oddsRank || "-"}位 モデル ${p.modelRank || "-"}位 確率 ${pct(p.probability)} EV ${num(p.expected_value)}`;\n  return `<td class="${cls}" data-combo="${escapeHtml(combo)}" title="${escapeHtml(title)}"><span class="pred-odds">${escapeHtml(tightNum(p.odds))}</span><span class="pred-prob">${escapeHtml(tightPct(p.probability))}</span></td>`;\n}\nfunction selectCombo(combo){\n  state.combo = combo || "1-2-3";\n  const comboSelect = $("combo");\n  if(comboSelect){\n    if(![...comboSelect.options].some(o => o.value === state.combo)) comboSelect.add(new Option(state.combo, state.combo), 0);\n    comboSelect.value = state.combo;\n  }\n  loadOdds();\n}\n\nasync function selectRace(raceId){\n  state.raceId = raceId;\n  const data = await getJson(`/api/predictions?race_id=${encodeURIComponent(raceId)}`);\n  const race = data.race || {};\n  state.currentRace = race;\n  $("raceTitle").innerHTML = `<span class="race-title-line"><span class="race-title-name">${escapeHtml(`${race.venue_name || ""} ${race.rno || ""}R ${race.title || ""}`)}</span><span class="race-title-actions"><button id="raceArchiveBtn" type="button">R</button><button id="venueArchiveBtn" type="button">場</button></span></span>`;\n  $("entries").innerHTML = data.entries.map(renderEntryCard).join("");\n  renderPredictionMatrix(data.predictions || []);\n  $("combo").innerHTML = data.predictions.slice(0,20).map(p => `<option>${p.combination}</option>`).join("") || `<option>1-2-3</option>`;\n  $("raceArchiveBtn").onclick = () => loadArchiveView("today", { race_id: raceId });\n  $("venueArchiveBtn").onclick = () => loadArchiveView("history", { kind:"venue", jcd: race.jcd || "" });\n  document.querySelectorAll("#entries button[data-archive-kind]").forEach(btn => btn.onclick = ev => {\n    ev.stopPropagation();\n    const kind = btn.dataset.archiveKind;\n    const payload = { kind, jcd: race.jcd || "", rno: race.rno || "" };\n    if(kind === "racer") payload.racer_no = btn.dataset.racerNo || "";\n    if(kind === "lane") payload.lane = btn.dataset.lane || "";\n    if(kind === "motor") payload.motor_no = btn.dataset.motorNo || "";\n    if(kind === "boat") payload.boat_no = btn.dataset.boatNo || "";\n    loadArchiveView("history", payload);\n  });\n  document.querySelectorAll("#predictions [data-combo]").forEach(cell => cell.onclick = () => selectCombo(cell.dataset.combo || "1-2-3"));\n  state.combo = $("combo").value;\n  await loadOdds();\n}\n\nasync function loadOdds(){ if(!state.raceId) return; const data = await getJson(`/api/odds?race_id=${encodeURIComponent(state.raceId)}&combination=${encodeURIComponent(state.combo || "1-2-3")}`); drawTrend(data.trend || []); }\nfunction drawTrend(rows){ const c=$("oddsChart"),ctx=c.getContext("2d"); ctx.clearRect(0,0,c.width,c.height); ctx.strokeStyle="#d8e0e3"; ctx.beginPath(); ctx.moveTo(34,16); ctx.lineTo(34,178); ctx.lineTo(700,178); ctx.stroke(); const vals=rows.map(r=>Number(r.odds)).filter(Number.isFinite); if(vals.length<2){ ctx.fillStyle="#637279"; ctx.fillText("オッズ推移の点が不足しています。",46,98); return; } const min=Math.min(...vals),max=Math.max(...vals),span=Math.max(.01,max-min); ctx.strokeStyle="#8f2d56"; ctx.lineWidth=2; ctx.beginPath(); vals.forEach((v,i)=>{ const x=42+(640*i/Math.max(1,vals.length-1)); const y=170-((v-min)/span)*140; if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y); }); ctx.stroke(); ctx.fillStyle="#172126"; ctx.fillText(`min ${num(min)} / max ${num(max)}`,46,26); }\nfunction renderLiveWipe(payload){\n  const box = $("liveWipe");\n  if(!box) return;\n  const r = payload && payload.active ? payload.race : null;\n  if(!r){ box.classList.add("hidden"); return; }\n  const closedId = localStorage.getItem("boatLiveWipeClosedRaceId");\n  if(closedId === String(r.race_id || "")){ box.classList.add("hidden"); return; }\n  box.classList.remove("hidden");\n  $("liveWipeTitle").textContent = `${r.jcd} ${r.venue_name} ${r.rno}R LIVE`;\n  $("liveWipeMeta").innerHTML = [\n    `締切 ${hm(r.deadline_at)} +${r.minutes_since_deadline ?? "-"}分`,\n    `出走 ${hm(r.race_time_at)}`,\n    `モデル ${(r.top_prediction && r.top_prediction.combination) || "-"}`,\n    `確率 ${pct(r.top_prediction && r.top_prediction.probability)}`,\n    wipeResultText(r)\n  ].map(v => `<span title="${String(v).replaceAll(\'"\',"&quot;")}">${v}</span>`).join("");\n  const src = r.live_embed_url || r.live_url;\n  const frame = $("liveWipeFrame");\n  if(frame && src && frame.dataset.src !== src){ frame.src = src; frame.dataset.src = src; }\n  $("liveWipeOpen").href = r.live_url || "#";\n  $("liveWipeOfficial").href = r.official_url || "#";\n  $("liveWipeSelect").onclick = () => selectRace(r.race_id);\n  $("liveWipeClose").onclick = () => {\n    localStorage.setItem("boatLiveWipeClosedRaceId", String(r.race_id || ""));\n    box.classList.add("hidden");\n  };\n  $("liveWipeZoom").onclick = () => {\n    box.classList.toggle("zoom");\n    $("liveWipeZoom").textContent = box.classList.contains("zoom") ? "同意用" : "動画";\n  };\n}\nfunction wipeResultText(r){\n  if(!r.result_combination) return "結果 -";\n  const hit = r.top_hit ? "的中" : (r.top5_hit ? "上位5" : "外れ");\n  const payout = r.trifecta_payout_yen == null ? "" : ` ${Number(r.trifecta_payout_yen).toLocaleString("ja-JP")}円`;\n  return `結果 ${r.result_combination} ${hit}${payout}`;\n}\nfunction timePair(r){\n  const d = hm(r && r.deadline_at);\n  const s = hm(r && r.race_time_at);\n  if(!r || d === "-" || s === "-" || d === s) return d;\n  return `${d}/${s.slice(3)}`;\n}\nfunction queryString(params){\n  return Object.entries(params || {}).filter(([,v]) => v !== undefined && v !== null && String(v) !== "").map(([k,v]) => `${encodeURIComponent(k)}=${encodeURIComponent(v)}`).join("&");\n}\nfunction fmtCell(v){\n  if(v === null || v === undefined || v === "") return "-";\n  if(typeof v === "number"){\n    if(Math.abs(v) < 1 && v !== 0) return v.toFixed(4);\n    if(Math.abs(v) >= 1000) return Math.round(v).toLocaleString("ja-JP");\n    return Number.isInteger(v) ? String(v) : v.toFixed(3);\n  }\n  return escapeHtml(String(v));\n}\nfunction pctCell(v){ return v == null ? "-" : `${(Number(v)*100).toFixed(2)}%`; }\nfunction yenCell(v){ return v == null ? "-" : `${Math.round(Number(v)).toLocaleString("ja-JP")}円`; }\nfunction archiveKvs(items){\n  return `<div class="archive-kv">${items.map(([label,value,mode]) => `<div><b>${mode==="pct" ? pctCell(value) : mode==="yen" ? yenCell(value) : fmtCell(value)}</b><span>${escapeHtml(label)}</span></div>`).join("")}</div>`;\n}\nfunction archiveTable(rows, columns, options={}){\n  if(!rows || !rows.length) return `<div class="empty">表示できるデータがありません。</div>`;\n  const body = rows.map(row => {\n    const cls = options.raceLink && row.race_id ? "clickable" : "";\n    const attr = options.raceLink && row.race_id ? ` data-race="${escapeHtml(row.race_id)}"` : "";\n    return `<tr class="${cls}"${attr}>${columns.map(c => `<td class="${c.mono ? "mono" : ""}" title="${escapeHtml(row[c.key] ?? "")}">${c.format ? c.format(row[c.key], row) : fmtCell(row[c.key])}</td>`).join("")}</tr>`;\n  }).join("");\n  const html = `<table class="archive-table"><thead><tr>${columns.map(c => `<th>${escapeHtml(c.label)}</th>`).join("")}</tr></thead><tbody>${body}</tbody></table>`;\n  setTimeout(() => document.querySelectorAll("#archiveContent tr[data-race]").forEach(row => row.onclick = () => selectRace(row.dataset.race)), 0);\n  return html;\n}\nasync function loadArchiveView(view, params={}, opts={}){\n  const d = $("raceDate").value || "";\n  const merged = { date:d, ...params };\n  const endpoint = view === "overview" ? "/api/archive/overview" : view === "today" ? "/api/archive/today" : view === "stats" ? "/api/archive/stats" : "/api/archive/history";\n  $("archiveTitle").textContent = archiveTitle(view, merged);\n  if(!opts.silent) $("archiveMeta").textContent = "読み込み中";\n  const data = await getJson(`${endpoint}?${queryString(merged)}`);\n  renderArchive(view, merged, data);\n}\nfunction archiveTitle(view, params){\n  if(view === "overview") return "過去データ総合";\n  if(view === "today") return params.race_id ? "当日レース詳細" : "当日データ";\n  if(view === "stats") return "統計データ";\n  const names = { racer:"選手履歴", venue:"場履歴", motor:"モーター履歴", boat:"ボート履歴", lane:"枠番履歴", combo:"組番履歴", race:"レース履歴" };\n  return names[params.kind] || "履歴データ";\n}\nfunction renderArchive(view, params, data){\n  $("archiveMeta").textContent = `更新 ${String(data.generated_at || "").replace("T"," ").slice(0,19)}`;\n  if(view === "overview") return renderArchiveOverview(data);\n  if(view === "today") return renderArchiveToday(data);\n  if(view === "stats") return renderArchiveStats(data);\n  return renderArchiveHistory(data);\n}\nfunction renderArchiveOverview(data){\n  const t = data.totals || {}, td = data.today || {};\n  $("archiveContent").innerHTML =\n    archiveKvs([["総レース",t.races],["期間",`${t.first_date || "-"} - ${t.last_date || "-"}`],["出走行",t.entries],["結果R",t.result_races],["odds",t.odds_snapshots],["予測R",t.prediction_races],["展示R",t.beforeinfo_races],["当日R",td.races]]) +\n    `<h3>年別</h3>` + archiveTable(data.years || [], [{key:"year",label:"年"},{key:"races",label:"R"},{key:"entry_races",label:"出走"},{key:"result_races",label:"結果"},{key:"prediction_races",label:"予測"}]) +\n    `<h3>場別</h3>` + archiveTable(data.venues || [], [{key:"jcd",label:"場"},{key:"venue_name",label:"名称"},{key:"races",label:"R"},{key:"entry_races",label:"出走"},{key:"result_races",label:"結果"},{key:"odds_races",label:"od"}]);\n}\nfunction renderArchiveToday(data){\n  if(data.mode === "race"){\n    const r = data.race || {};\n    $("archiveContent").innerHTML =\n      archiveKvs([["場/R",`${r.venue_name || "-"} ${r.rno || "-"}R`],["日付",r.race_date],["締切/出",timePair(r)],["出走",r.entries],["odds",r.odds_snapshots],["展示",r.beforeinfo_rows],["結果",r.result_rows],["予測",r.latest_prediction]]) +\n      `<h3>出走・展示・結果</h3>` + archiveTable(data.entries || [], [\n        {key:"lane",label:"枠",mono:true},{key:"racer_name",label:"選手"},{key:"racer_no",label:"登番",mono:true},{key:"racer_class",label:"級"},{key:"branch",label:"支部"},{key:"origin",label:"出身"},\n        {key:"national_win_rate",label:"全国"},{key:"local_win_rate",label:"当地"},{key:"motor_no",label:"M",mono:true},{key:"motor_2_rate",label:"M2"},{key:"boat_no",label:"B",mono:true},{key:"boat_2_rate",label:"B2"},\n        {key:"exhibition_time",label:"展示"},{key:"exhibition_course",label:"展進"},{key:"exhibition_start_timing",label:"展ST"},{key:"rank",label:"着"},{key:"result_start_timing",label:"ST"}\n      ]) +\n      `<h3>モデル予測</h3>` + archiveTable(data.predictions || [], [{key:"combination",label:"3連単",mono:true},{key:"probability",label:"確率",format:pctCell},{key:"odds",label:"od"},{key:"expected_value",label:"EV"},{key:"generated_at",label:"生成"}]) +\n      `<h3>払戻</h3>` + archiveTable(data.payouts || [], [{key:"bet_type",label:"式別"},{key:"combination",label:"組番",mono:true},{key:"payout_yen",label:"払戻",format:yenCell},{key:"popularity",label:"人気"}]);\n    return;\n  }\n  $("archiveContent").innerHTML = archiveTable(data.races || [], [\n    {key:"race_date",label:"日"},{key:"venue_name",label:"場"},{key:"rno",label:"R",mono:true},{key:"title",label:"タイトル"},{key:"race_type",label:"種別"},\n    {key:"deadline_at",label:"締切/出",format:hm},{key:"entries",label:"出走"},{key:"odds_snapshots",label:"od"},{key:"beforeinfo_rows",label:"展示"},{key:"result_rows",label:"結果"},{key:"latest_prediction",label:"予測"}\n  ], { raceLink:true });\n}\nfunction renderArchiveHistory(data){\n  const s = data.summary || {};\n  $("archiveContent").innerHTML =\n    archiveKvs([["対象",s.racer_name || s.venue_name || s.number || s.combination || s.lane || "-"],["出走/件数",s.starts || s.races || s.hits],["結果",s.result_rows || s.result_races],["1着率",s.win_rate,"pct"],["3着内",s.top3_rate,"pct"],["平均着",s.avg_rank],["平均ST",s.avg_start],["平均払戻",s.avg_payout_yen,"yen"]]) +\n    (data.facets ? `<h3>内訳</h3>` + archiveTable(data.facets, [{key:"lane",label:"枠"},{key:"starts",label:"出走"},{key:"wins",label:"1着"},{key:"top3",label:"3内"},{key:"avg_start",label:"ST"}]) : "") +\n    `<h3>履歴</h3>` + archiveTable(data.rows || [], [\n      {key:"race_date",label:"日"},{key:"venue_name",label:"場"},{key:"rno",label:"R",mono:true},{key:"lane",label:"枠",mono:true},{key:"racer_name",label:"選手"},{key:"racer_class",label:"級"},\n      {key:"motor_no",label:"M",mono:true},{key:"boat_no",label:"B",mono:true},{key:"rank",label:"着"},{key:"start_timing",label:"ST"},{key:"result_combination",label:"結果",mono:true},{key:"payout_yen",label:"払戻",format:yenCell}\n    ], { raceLink:true });\n}\nfunction renderArchiveStats(data){\n  $("archiveContent").innerHTML =\n    `<div class="archive-tabs"><button type="button" onclick="loadArchiveView(\'stats\',{scope:\'lane\'})">枠</button><button type="button" onclick="loadArchiveView(\'stats\',{scope:\'venue\'})">場</button><button type="button" onclick="loadArchiveView(\'stats\',{scope:\'rno\'})">R</button><button type="button" onclick="loadArchiveView(\'stats\',{scope:\'class\'})">級</button><button type="button" onclick="loadArchiveView(\'stats\',{scope:\'motor\'})">M</button><button type="button" onclick="loadArchiveView(\'stats\',{scope:\'boat\'})">B</button></div>` +\n    archiveTable(data.rows || [], [\n      {key:"label",label:"対象"},{key:"starts",label:"件数"},{key:"wins",label:"1着"},{key:"top3",label:"3内"},{key:"win_rate",label:"1着率",format:pctCell},{key:"top3_rate",label:"3内率",format:pctCell},\n      {key:"avg_rank",label:"平均着"},{key:"avg_start",label:"ST"},{key:"avg_national_win_rate",label:"全国"},{key:"avg_local_win_rate",label:"当地"},{key:"avg_motor_2_rate",label:"M2"},{key:"avg_boat_2_rate",label:"B2"}\n    ]);\n}\nfunction wireArchiveNav(){\n  $("navPast").onclick = () => loadArchiveView("overview");\n  $("navToday").onclick = () => loadArchiveView("today", { jcd: state.jcd || "" });\n  $("navStats").onclick = () => loadArchiveView("stats", { scope:"lane" });\n  $("archivePast").onclick = () => loadArchiveView("overview");\n  $("archiveToday").onclick = () => loadArchiveView("today", { jcd: state.jcd || "" });\n  $("archiveStats").onclick = () => loadArchiveView("stats", { scope:"lane" });\n}\nwireArchiveNav();\n\nloadAll(); setInterval(loadAll,30000);\n</script>\n</body>\n</html>\n'

if __name__ == "__main__":
    raise SystemExit(main())
