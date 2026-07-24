from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
import json
import os
from pathlib import Path
from typing import Any

from . import historical_model, operational_bankroll, standard_evaluation
from .db import connection


TASK_TYPE = "historical_research_logit"
MODEL_ID = "no_odds_v9_research_logit"


def _date(value: str) -> date:
    parsed = date.fromisoformat(value)
    if parsed.isoformat() != value:
        raise argparse.ArgumentTypeError("evaluation date must use YYYY-MM-DD")
    return parsed


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _comparison(
    *,
    protocol: dict[str, Any],
    prediction: dict[str, Any],
    bankroll: dict[str, Any],
    baseline_prediction: dict[str, Any] | None,
    baseline_bankroll: dict[str, Any] | None,
) -> dict[str, Any]:
    expected_hash = str(protocol["race_set_sha256"])
    expected_races = int(protocol["prediction_races"])
    ready = bool(baseline_prediction and baseline_bankroll)
    checks = {
        "comparison_ready": ready,
        "race_set_matches_protocol": (
            prediction.get("evaluation_race_set_sha256") == expected_hash
            and bankroll.get("evaluation_race_set_sha256") == expected_hash
            and int(prediction.get("evaluated_races") or 0) == expected_races
            and int(bankroll.get("evaluated_races") or 0) == expected_races
        ),
        "entry_log_loss_not_worse": ready and (
            float(prediction["entry_log_loss"])
            <= float(baseline_prediction["entry_log_loss"])
        ),
        "winner_top1_not_worse": ready and (
            float(prediction["winner_top1_accuracy"])
            >= float(baseline_prediction["winner_top1_accuracy"])
        ),
        "trifecta_top5_not_worse": ready and (
            float(prediction["trifecta_top5_hit_rate"])
            >= float(baseline_prediction["trifecta_top5_hit_rate"])
        ),
        "roi_at_least_one": float(bankroll.get("roi") or 0.0) >= 1.0,
        "profit_positive": int(bankroll.get("profit_yen") or 0) > 0,
    }
    return {
        "baseline_model": "no_odds_v8",
        "checks": checks,
        "promotion_eligible": all(checks.values()),
        "baseline": {
            "entry_log_loss": (
                baseline_prediction.get("entry_log_loss")
                if baseline_prediction else None
            ),
            "winner_top1_accuracy": (
                baseline_prediction.get("winner_top1_accuracy")
                if baseline_prediction else None
            ),
            "trifecta_top5_hit_rate": (
                baseline_prediction.get("trifecta_top5_hit_rate")
                if baseline_prediction else None
            ),
            "roi": baseline_bankroll.get("roi") if baseline_bankroll else None,
            "profit_yen": (
                baseline_bankroll.get("profit_yen")
                if baseline_bankroll else None
            ),
        },
    }


def evaluate(
    conn: Any,
    *,
    output_path: Path,
    evaluation_date: date,
    model_dir: Path,
) -> dict[str, Any]:
    protocol = standard_evaluation.build_protocol(
        conn,
        as_of_date=evaluation_date + timedelta(days=1),
    )
    training_races = int(protocol["training_races"])
    if training_races < 1:
        raise ValueError("standard protocol has no training races")

    model_path = output_path.with_suffix(".joblib")
    prediction_path = output_path.with_name(f".{output_path.name}.prediction.json")
    bankroll_path = output_path.with_name(f".{output_path.name}.bankroll.json")
    previous_max_date = os.environ.get("BOATRACE_EVAL_MAX_RACE_DATE")
    os.environ["BOATRACE_EVAL_MAX_RACE_DATE"] = evaluation_date.isoformat()
    try:
        prediction = historical_model.backtest_model(
            conn,
            output_path=prediction_path,
            folds=1,
            min_train_races=training_races,
            model_output_path=model_path,
            include_research=True,
        )
        bankroll = operational_bankroll.operational_adaptive_bankroll(
            conn,
            output_path=bankroll_path,
            folds=1,
            min_train_races=training_races,
            model_input_path=model_path,
            daily_budget_yen=10_000,
            include_research=True,
        )
    finally:
        if previous_max_date is None:
            os.environ.pop("BOATRACE_EVAL_MAX_RACE_DATE", None)
        else:
            os.environ["BOATRACE_EVAL_MAX_RACE_DATE"] = previous_max_date

    comparison = _comparison(
        protocol=protocol,
        prediction=prediction,
        bankroll=bankroll,
        baseline_prediction=_read_json(
            model_dir / "standardized_365d_v2/raw/no_odds_v8_prediction.json"
        ),
        baseline_bankroll=_read_json(
            model_dir / "standardized_365d_v2/raw/no_odds_v8_bankroll.json"
        ),
    )
    metrics = {
        "entry_log_loss": prediction.get("entry_log_loss"),
        "entry_brier": prediction.get("entry_brier"),
        "winner_top1_accuracy": prediction.get("winner_top1_accuracy"),
        "trifecta_top1_hit_rate": prediction.get("trifecta_top1_hit_rate"),
        "trifecta_top5_hit_rate": prediction.get("trifecta_top5_hit_rate"),
        "evaluated_races": prediction.get("evaluated_races"),
        "roi": bankroll.get("roi"),
        "profit_yen": bankroll.get("profit_yen"),
        "stake_yen": bankroll.get("stake_yen"),
        "return_yen": bankroll.get("return_yen"),
    }
    result = {
        "status": "completed",
        "task_type": TASK_TYPE,
        "model": MODEL_ID,
        "model_path": str(model_path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_date": evaluation_date.isoformat(),
        "include_odds": False,
        "include_beforeinfo": False,
        "include_research": True,
        "feature_set": historical_model.RESEARCH_FEATURE_SET,
        "protocol": protocol,
        "prediction": prediction,
        "bankroll": bankroll,
        "metrics": metrics,
        "comparison": comparison,
        "promotion_eligible": comparison["promotion_eligible"],
        "daily": list(bankroll.get("daily") or []),
    }
    _write_json_atomic(output_path, result)
    prediction_path.unlink(missing_ok=True)
    bankroll_path.unlink(missing_ok=True)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate research interactions in the production logistic model"
    )
    parser.add_argument("--db", required=True)
    parser.add_argument("--evaluation-date", type=_date, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("data/models"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    with connection(args.db) as conn:
        result = evaluate(
            conn,
            output_path=args.output,
            evaluation_date=args.evaluation_date,
            model_dir=args.model_dir,
        )
    print(json.dumps(result["metrics"], ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
