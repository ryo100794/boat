from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
import json
import os
from pathlib import Path
from typing import Any

from . import historical_model, operational_bankroll, standard_evaluation
from .db import connection


TASK_TYPE = "historical_coverage_safe"
DAILY_BUDGET_YEN = 10_000


def _parse_evaluation_date(value: str) -> date:
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "evaluation date must use YYYY-MM-DD"
        ) from exc
    if parsed.isoformat() != value:
        raise argparse.ArgumentTypeError("evaluation date must use YYYY-MM-DD")
    return parsed


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def evaluate_historical_candidate(
    conn: Any,
    *,
    output_path: Path,
    evaluation_date: date,
) -> dict[str, Any]:
    protocol = standard_evaluation.build_protocol(
        conn,
        as_of_date=evaluation_date + timedelta(days=1),
    )
    training_races = int(protocol["training_races"])
    if training_races < 1:
        raise ValueError("standard evaluation protocol has no training races")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_output_path = output_path.with_suffix(".joblib")
    prediction_output_path = output_path.with_name(
        f".{output_path.name}.prediction.json"
    )
    bankroll_output_path = output_path.with_name(
        f".{output_path.name}.bankroll.json"
    )

    previous_max_race_date = os.environ.get("BOATRACE_EVAL_MAX_RACE_DATE")
    os.environ["BOATRACE_EVAL_MAX_RACE_DATE"] = evaluation_date.isoformat()
    try:
        prediction = historical_model.backtest_model(
            conn,
            output_path=prediction_output_path,
            folds=1,
            min_train_races=training_races,
            model_output_path=model_output_path,
        )
        bankroll = operational_bankroll.operational_adaptive_bankroll(
            conn,
            output_path=bankroll_output_path,
            folds=1,
            min_train_races=training_races,
            model_input_path=model_output_path,
            daily_budget_yen=DAILY_BUDGET_YEN,
        )
    finally:
        if previous_max_race_date is None:
            os.environ.pop("BOATRACE_EVAL_MAX_RACE_DATE", None)
        else:
            os.environ["BOATRACE_EVAL_MAX_RACE_DATE"] = previous_max_race_date

    daily = list(bankroll.get("daily") or [])
    metrics = {
        "entry_log_loss": prediction.get("entry_log_loss"),
        "entry_brier": prediction.get("entry_brier"),
        "winner_top1_accuracy": prediction.get("winner_top1_accuracy"),
        "trifecta_top1_hit_rate": prediction.get("trifecta_top1_hit_rate"),
        "trifecta_top5_hit_rate": prediction.get("trifecta_top5_hit_rate"),
        "roi": bankroll.get("roi"),
        "profit_yen": bankroll.get("profit_yen"),
        "stake_yen": bankroll.get("stake_yen"),
        "return_yen": bankroll.get("return_yen"),
        "evaluation_days": bankroll.get("race_days", len(daily)),
    }
    result = {
        "status": "completed",
        "task_type": TASK_TYPE,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_date": evaluation_date.isoformat(),
        "model": bankroll.get("model") or "win_model_no_odds_v8",
        "model_path": str(model_output_path),
        "include_odds": False,
        "daily_budget_yen": DAILY_BUDGET_YEN,
        "promotion_eligible": False,
        "promotion_note": "requires comparison on the same frozen holdout",
        "protocol": protocol,
        "prediction": prediction,
        "bankroll": bankroll,
        "metrics": metrics,
        "daily": daily,
    }
    _write_json_atomic(output_path, result)
    prediction_output_path.unlink(missing_ok=True)
    bankroll_output_path.unlink(missing_ok=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a no-odds historical candidate on a frozen holdout"
    )
    parser.add_argument("--db", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--evaluation-date",
        type=_parse_evaluation_date,
        required=True,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    with connection(args.db) as conn:
        result = evaluate_historical_candidate(
            conn,
            output_path=args.output,
            evaluation_date=args.evaluation_date,
        )
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
