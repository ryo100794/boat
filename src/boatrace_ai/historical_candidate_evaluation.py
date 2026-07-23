from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
import json
import os
from pathlib import Path
from typing import Any

import joblib
from sklearn.metrics import brier_score_loss

from . import historical_model, operational_bankroll, standard_evaluation
from .db import connection
from .feature_tuning import load_complete_race_ids
from .modeling import _race_level_metrics
from .standard_evaluation import race_set_sha256


TASK_TYPE = "historical_coverage_safe"
DAILY_BUDGET_YEN = 10_000
LEGACY_FEATURE_SET = (
    "no_odds_v8_relative_weather_sparse32_scaled_"
    "logreg_C0.20_unweighted"
)


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


def _protocol_race_sets(
    conn: Any,
    protocol: dict[str, Any],
) -> tuple[set[str], set[str]]:
    race_keys = load_complete_race_ids(conn)
    training_count = int(protocol["training_races"])
    prediction_count = int(protocol["prediction_races"])
    if len(race_keys) != training_count + prediction_count:
        raise ValueError(
            "protocol complete race count mismatch: "
            f"expected {training_count + prediction_count}, got {len(race_keys)}"
        )

    training_rows = race_keys[:training_count]
    holdout_rows = race_keys[training_count:]
    training_races = {str(row[0]) for row in training_rows}
    holdout_races = {str(row[0]) for row in holdout_rows}
    if len(training_races) != training_count:
        raise ValueError("protocol training race IDs are not unique")
    if len(holdout_races) != prediction_count:
        raise ValueError("protocol holdout race count mismatch")
    if training_races & holdout_races:
        raise ValueError("protocol training and holdout races overlap")

    holdout_start = str(protocol["holdout_start"])
    holdout_end = str(protocol["holdout_end"])
    if any(str(row[1]) >= holdout_start for row in training_rows):
        raise ValueError("protocol training races cross the holdout boundary")
    if any(
        not holdout_start <= str(row[1]) <= holdout_end
        for row in holdout_rows
    ):
        raise ValueError("protocol holdout races cross the frozen date boundary")
    holdout_hash = race_set_sha256(holdout_races)
    if holdout_hash != protocol.get("race_set_sha256"):
        raise ValueError("protocol holdout race set hash mismatch")
    return training_races, holdout_races


def _training_beforeinfo_rows(conn: Any, *, holdout_start: str) -> int:
    row = conn.execute(
        """
        SELECT COUNT(*)
        FROM beforeinfo b
        JOIN races r ON r.race_id = b.race_id
        WHERE r.race_date < ?
          AND (SELECT COUNT(*) FROM entries e
               WHERE e.race_id = r.race_id) = 6
          AND (SELECT COUNT(*) FROM race_results rr
               WHERE rr.race_id = r.race_id AND rr.rank IS NOT NULL) = 6
        """,
        (holdout_start,),
    ).fetchone()
    return int(row[0])


def _copy_reused_model(
    *,
    source_path: Path,
    output_path: Path,
    training_races: set[str],
    training_beforeinfo_rows: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if training_beforeinfo_rows != 0:
        raise ValueError(
            "source model reuse rejected: training races contain beforeinfo rows"
        )
    if source_path.resolve() == output_path.resolve():
        raise ValueError("source and job model paths must differ")

    source_bundle = joblib.load(source_path)
    if not isinstance(source_bundle, dict) or "pipeline" not in source_bundle:
        raise ValueError("source model pipeline missing")
    source_metadata = source_bundle.get("metadata")
    if not isinstance(source_metadata, dict):
        raise ValueError("source model metadata missing")

    source_train_races = source_metadata.get("train_races")
    if type(source_train_races) is not int or source_train_races != len(training_races):
        raise ValueError("source model training race count mismatch")
    expected_train_hash = race_set_sha256(training_races)
    source_train_hash = source_metadata.get("train_race_set_sha256")
    if source_train_hash != expected_train_hash:
        raise ValueError("source model training race set hash mismatch")
    source_feature_set = source_metadata.get("feature_set")
    if source_feature_set not in {LEGACY_FEATURE_SET, historical_model.FEATURE_SET}:
        raise ValueError("source model feature set is not reusable")
    if source_metadata.get("include_odds") is not False:
        raise ValueError("source model must exclude odds")

    # With zero training beforeinfo rows, the legacy missing-value vectors are
    # identical to explicit include_beforeinfo=False vectors. Only the metadata
    # contract changes; the fitted pipeline remains byte-for-byte equivalent.
    reused_metadata = dict(source_metadata)
    reused_metadata.update(
        {
            "feature_set": historical_model.FEATURE_SET,
            "include_beforeinfo": False,
            "source_feature_set": source_feature_set,
            "training_beforeinfo_rows": 0,
            "training_reuse_equivalent": True,
        }
    )
    reused_bundle = dict(source_bundle)
    reused_bundle["metadata"] = reused_metadata
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    try:
        joblib.dump(reused_bundle, temporary)
        temporary.replace(output_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return reused_bundle, dict(source_metadata)


def _score_holdout(
    conn: Any,
    *,
    bundle: dict[str, Any],
    training_races: set[str],
    holdout_races: set[str],
    holdout_start: str,
    holdout_end: str,
) -> dict[str, Any]:
    probabilities: list[float] = []
    labels: list[int] = []
    race_predictions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    scored = historical_model.iter_scored_entries(
        conn,
        pipeline=bundle["pipeline"],
        include_races=holdout_races,
        from_date=holdout_start,
        through_date=holdout_end,
    )
    for probability, meta in scored:
        race_id = str(meta["race_id"])
        label = 1 if int(meta["rank"]) == 1 else 0
        probability_value = float(probability)
        probabilities.append(probability_value)
        labels.append(label)
        race_predictions[race_id].append(
            {
                "lane": int(meta["lane"]),
                "rank": int(meta["rank"]),
                "probability": probability_value,
            }
        )

    if set(race_predictions) != holdout_races:
        raise ValueError("scored holdout race set mismatch")
    if len(probabilities) != len(holdout_races) * 6:
        raise ValueError("scored holdout entry count mismatch")
    if any(len(rows) != 6 for rows in race_predictions.values()):
        raise ValueError("scored holdout contains incomplete races")

    entry_log_loss = historical_model.base._safe_log_loss(labels, probabilities)
    entry_brier = float(brier_score_loss(labels, probabilities))
    fold = {
        "fold": 1,
        "train_races": len(training_races),
        "test_races": len(holdout_races),
        "entry_log_loss": entry_log_loss,
        "entry_brier": entry_brier,
    }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "folds": [fold],
        "examples": len(probabilities),
        "races": len(holdout_races),
        "training_races": len(training_races),
        "holdout_races": len(holdout_races),
        "include_odds": False,
        "include_beforeinfo": False,
        "feature_set": historical_model.FEATURE_SET,
        "evaluation_race_set_sha256": race_set_sha256(race_predictions),
        "entry_log_loss": entry_log_loss,
        "entry_brier": entry_brier,
        **_race_level_metrics(race_predictions),
    }



def evaluate_historical_candidate(
    conn: Any,
    *,
    output_path: Path,
    evaluation_date: date,
    model_input_path: Path,
) -> dict[str, Any]:
    protocol = standard_evaluation.build_protocol(
        conn,
        as_of_date=evaluation_date + timedelta(days=1),
    )
    if int(protocol["training_races"]) < 1:
        raise ValueError("standard evaluation protocol has no training races")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_output_path = output_path.with_suffix(".joblib")
    bankroll_output_path = output_path.with_name(
        f".{output_path.name}.bankroll.json"
    )
    holdout_start = str(protocol["holdout_start"])
    holdout_end = str(protocol["holdout_end"])

    previous_max_race_date = os.environ.get("BOATRACE_EVAL_MAX_RACE_DATE")
    os.environ["BOATRACE_EVAL_MAX_RACE_DATE"] = evaluation_date.isoformat()
    try:
        training_races, holdout_races = _protocol_race_sets(conn, protocol)
        beforeinfo_rows = _training_beforeinfo_rows(
            conn,
            holdout_start=holdout_start,
        )
        reused_bundle, source_metadata = _copy_reused_model(
            source_path=model_input_path,
            output_path=model_output_path,
            training_races=training_races,
            training_beforeinfo_rows=beforeinfo_rows,
        )
        prediction = _score_holdout(
            conn,
            bundle=reused_bundle,
            training_races=training_races,
            holdout_races=holdout_races,
            holdout_start=holdout_start,
            holdout_end=holdout_end,
        )
        bankroll = operational_bankroll.operational_adaptive_bankroll(
            conn,
            output_path=bankroll_output_path,
            folds=1,
            min_train_races=len(training_races),
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
    source_train_hash = str(source_metadata["train_race_set_sha256"])
    result = {
        "status": "completed",
        "task_type": TASK_TYPE,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_date": evaluation_date.isoformat(),
        "model": "win_model_no_odds_v8_beforeinfo_excluded",
        "model_path": str(model_output_path),
        "source_model_path": str(model_input_path),
        "source_train_hash": source_train_hash,
        "training_beforeinfo_rows": 0,
        "training_reuse_equivalent": True,
        "training_reuse_note": (
            "The legacy artifact and historical-only FEATURE_SET are equivalent "
            "because the verified training set has zero beforeinfo rows; only a "
            "copied metadata contract was changed."
        ),
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
    bankroll_output_path.unlink(missing_ok=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a no-odds historical candidate on a frozen holdout"
    )
    parser.add_argument("--db", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-input", type=Path, required=True)
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
            model_input_path=args.model_input,
        )
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
