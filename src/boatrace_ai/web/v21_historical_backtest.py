from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
MODEL_ID = "v21_daily"


def build_projection(
    result: Mapping[str, Any], *, race_date: str, source_job_id: int
) -> dict[str, Any]:
    if "triple_head_v21" not in str(result.get("model") or ""):
        raise ValueError("source result is not V21")
    days = (result.get("chronological_bankroll") or {}).get("daily") or []
    day = next(
        (row for row in days if str(row.get("race_date")) == race_date),
        None,
    )
    if not isinstance(day, Mapping):
        raise ValueError(f"V21 result has no chronological day: {race_date}")
    coverage_days = (result.get("coverage_gate") or {}).get("days") or []
    coverage = next(
        (row for row in coverage_days if str(row.get("race_date")) == race_date),
        {},
    )
    fold = next(
        (
            row
            for row in result.get("folds") or []
            if str(row.get("evaluation_date")) == race_date
        ),
        {},
    )
    probability = fold.get("probability_metrics") or {}
    model_metrics = {
        "winner_log_loss": probability.get("calibrated_winner_log_loss"),
        "winner_top1_accuracy": probability.get("calibrated_winner_top1_accuracy"),
        "trifecta_log_loss": probability.get("calibrated_trifecta_log_loss"),
        "trifecta_top5_hit_rate": probability.get(
            "calibrated_trifecta_top5_hit_rate"
        ),
        "market_trifecta_log_loss": probability.get("market_trifecta_log_loss"),
        "market_trifecta_top5_hit_rate": probability.get(
            "market_trifecta_top5_hit_rate"
        ),
    }
    decisions: dict[str, Mapping[str, Any]] = {}
    series = []
    initial = int(day.get("initial_bankroll_yen") or 10_000)
    for event in day.get("ledger") or []:
        race_id = str(event.get("race_id") or "")
        if not race_id:
            continue
        if event.get("event") == "decision":
            decisions[race_id] = event
            continue
        if event.get("event") != "settlement":
            continue
        decision = decisions.get(race_id) or {}
        selections = decision.get("selections") or []
        stake = int(event.get("stake_yen") or 0)
        returned = int(event.get("return_yen") or 0)
        equity = int(event.get("cash_after_yen") or 0) + int(
            event.get("outstanding_stake_yen") or 0
        )
        series.append({
            "race_id": race_id,
            "stake_yen": stake,
            "return_yen": returned,
            "profit_yen": equity - initial,
            "race_profit_yen": returned - stake,
            "tickets": len(selections),
            "hit": returned > 0,
            "odds_basis": "締切5分前V21 walk-forward終値予測",
        })
    stake = int(day.get("stake_yen") or 0)
    stats = {
        "starting_bankroll_yen": initial,
        "current_bankroll_yen": int(day.get("closing_bankroll_yen") or initial),
        "profit_yen": int(day.get("profit_yen") or 0),
        "stake_yen": stake,
        "return_yen": int(day.get("return_yen") or 0),
        "roi": float(day.get("roi")) if day.get("roi") is not None else None,
        "evaluated_races": int(day.get("evaluated_races") or 0),
        "prediction_races": int(day.get("evaluated_races") or 0),
        "valid_odds_races": int(coverage.get("eligible_t5_races") or 0),
        "historical_odds_races": 0,
        "selected_races": int(day.get("races_bet") or 0),
        "tickets": int(day.get("tickets") or 0),
        "hit_tickets": int(day.get("hit_tickets") or 0),
        "ticket_hit_rate": (
            int(day.get("hit_tickets") or 0) / int(day.get("tickets") or 1)
            if int(day.get("tickets") or 0) else None
        ),
        "max_drawdown_yen": int(day.get("max_drawdown_yen") or 0),
        "fallback_prediction_races": 0,
        "rejected_odds_snapshots": max(
            0,
            int(coverage.get("complete_races") or 0)
            - int(coverage.get("eligible_t5_races") or 0),
        ),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "model_id": MODEL_ID,
        "race_date": race_date,
        "source_job_id": int(source_job_id),
        "source_model": result.get("model"),
        "generated_at": result.get("generated_at"),
        "validation_design": result.get("validation_design"),
        "minimum_day_coverage": (result.get("coverage_gate") or {}).get(
            "minimum_day_coverage"
        ),
        "coverage": dict(coverage),
        "stats": stats,
        "model_metrics": model_metrics,
        "series": series,
        "real_betting_enabled": False,
    }


def write_projection(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_dashboard_payload(
    path: Path,
    *,
    race_date: str,
    models: Sequence[Mapping[str, Any]],
    selected_model: Mapping[str, Any],
    schedule: Sequence[Mapping[str, Any]],
    now_jst: datetime,
) -> dict[str, Any] | None:
    try:
        projection = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    if (
        projection.get("schema_version") != SCHEMA_VERSION
        or projection.get("model_id") != MODEL_ID
        or projection.get("race_date") != race_date
        or projection.get("real_betting_enabled") is not False
    ):
        return None
    schedule_by_race = {str(row["race_id"]): row for row in schedule}
    series = []
    for raw in projection.get("series") or []:
        row = dict(raw)
        scheduled = schedule_by_race.get(str(row.get("race_id") or "")) or {}
        row.update({
            "venue": scheduled.get("venue"),
            "jcd": scheduled.get("jcd"),
            "rno": scheduled.get("rno"),
            "race_time_at": scheduled.get("race_time_at"),
        })
        series.append(row)
    coverage = projection.get("coverage") or {}
    return {
        "available": True,
        "date": race_date,
        "generated_at": now_jst.isoformat(timespec="seconds"),
        "through_race_time_at": series[-1].get("race_time_at") if series else None,
        "models": list(models),
        "selected_model": MODEL_ID,
        "selected_model_label": selected_model["label"],
        "policy": {
            "starting_bankroll_yen": int(
                (projection.get("stats") or {}).get("starting_bankroll_yen") or 10_000
            ),
            "bet_type": "3連単",
            "decision": "締切5分前のV21日付順walk-forward判断",
            "model": selected_model["label"],
            "profit_reinvestment": True,
            "real_betting_enabled": False,
        },
        "stats": projection.get("stats") or {},
        "model_metrics": projection.get("model_metrics") or {},
        "series": series,
        "schedule": list(schedule),
        "warnings": [
            f"V21過去日walk-forward backtest / job {int(projection['source_job_id'])}。",
            f"T-5有効 {int(coverage.get('eligible_t5_races') or 0)}/"
            f"{int(coverage.get('complete_races') or 0)}R。実投票は無効です。",
        ],
        "backtest_source_job_id": int(projection["source_job_id"]),
    }
