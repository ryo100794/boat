from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable, Mapping

import numpy as np

from .closing_odds import MAX_ODDS, MIN_ODDS
from .closing_odds_multihorizon_v11 import (
    _snapshot_odds,
    _teacher_selection,
    build_checkpoint_feature_vector,
    normalize_labeled_checkpoints,
)
from .closing_odds_t300_nonlinear_v12 import (
    CHECKPOINT_LABEL,
    CHECKPOINT_OFFSET_SECONDS,
    _fit_estimator,
    _predict_log_ratio,
    _preferred_engine,
)
from .direct_context_market_residual_v25 import direct_context_probabilities
from .v25_top1_narrow_policy_v33 import simulate_v25_top1_narrow_v33


MODEL_NAME = "v25_top1_specific_closing_v36"
RANDOM_STATE = 20260801


def _top1_example(
    race: Mapping[str, Any], probability_artifact: Mapping[str, Any]
) -> dict[str, Any] | None:
    checkpoints = normalize_labeled_checkpoints(
        race, as_of_offset_seconds=CHECKPOINT_OFFSET_SECONDS
    )
    snapshot = checkpoints.get(CHECKPOINT_LABEL)
    if not isinstance(snapshot, Mapping):
        return None
    current = _snapshot_odds(snapshot)
    teacher, teacher_source, teacher_incomplete = _teacher_selection(race)
    if (
        len(current) != 120
        or len(teacher) != 120
        or set(current) != set(teacher)
    ):
        return None
    probabilities = direct_context_probabilities(dict(race), probability_artifact)
    if len(probabilities) != 120 or set(probabilities) != set(current):
        return None
    combination = min(
        probabilities,
        key=lambda value: (-float(probabilities[value]), value),
    )
    vector, trace = build_checkpoint_feature_vector(
        race,
        checkpoint=CHECKPOINT_LABEL,
        combination=combination,
        as_of_offset_seconds=CHECKPOINT_OFFSET_SECONDS,
    )
    if trace.get("future_checkpoint_offsets_used"):
        raise ValueError("V36 feature vector used a post-T-5 checkpoint")
    target = math.log(float(teacher[combination]) / float(current[combination]))
    if not math.isfinite(target):
        return None
    return {
        "race_id": str(race["race_id"]),
        "race_date": str(race["race_date"]),
        "combination": combination,
        "features": np.asarray(vector, dtype=np.float64),
        "target_log_ratio": float(np.clip(target, -8.0, 8.0)),
        "raw_target_log_ratio": target,
        "current_odds": float(current[combination]),
        "closing_odds": float(teacher[combination]),
        "teacher_source": teacher_source,
        "teacher_incomplete": bool(teacher_incomplete),
        "used_checkpoint_offsets": list(trace["used_checkpoint_offsets"]),
    }


def fit_v25_top1_closing_v36(
    races: Iterable[dict[str, Any]],
    *,
    probability_artifact: Mapping[str, Any],
    prediction_date: str,
    minimum_training_days: int = 2,
    minimum_training_races: int = 200,
    engine: str | None = None,
) -> dict[str, Any]:
    source = [
        race for race in races if str(race.get("race_date") or "") < prediction_date
    ]
    examples = [
        example
        for race in source
        if (example := _top1_example(race, probability_artifact)) is not None
    ]
    training_dates = sorted({str(row["race_date"]) for row in examples})
    ready = bool(
        len(training_dates) >= minimum_training_days
        and len(examples) >= minimum_training_races
    )
    fitted = (
        _fit_estimator(
            examples,
            engine or _preferred_engine(),
            RANDOM_STATE + len(training_dates),
        )
        if ready
        else None
    )
    return {
        "model": MODEL_NAME,
        "role": "V25_top1_closing_odds_only",
        "teacher": "log(official_closing_odds/real_t5_odds)_for_V25_top1",
        "uses_outcome_teacher": False,
        "uses_payout_teacher": False,
        "decision_checkpoint": CHECKPOINT_LABEL,
        "prediction_date": prediction_date,
        "trained_through_date": training_dates[-1] if training_dates else None,
        "training_dates": training_dates,
        "training_days": len(training_dates),
        "training_races": len(examples),
        "minimum_training_days": minimum_training_days,
        "minimum_training_races": minimum_training_races,
        "ready": bool(ready and fitted is not None),
        "fitted": fitted,
        "strict_prior_boundary": all(day < prediction_date for day in training_dates),
    }


def attach_v25_top1_closing_v36(
    races: list[dict[str, Any]],
    *,
    probability_artifact: Mapping[str, Any],
    closing_artifact: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not closing_artifact.get("ready"):
        return [], {
            "eligible_races": 0,
            "excluded_races": len(races),
            "baseline_top1_log_mae": None,
            "v36_top1_log_mae": None,
        }
    fitted = closing_artifact.get("fitted")
    if not isinstance(fitted, Mapping):
        raise ValueError("V36 fitted model is missing")
    transformed: list[dict[str, Any]] = []
    baseline_errors: list[float] = []
    model_errors: list[float] = []
    for race in races:
        example = _top1_example(race, probability_artifact)
        if example is None:
            continue
        checkpoints = normalize_labeled_checkpoints(
            race, as_of_offset_seconds=CHECKPOINT_OFFSET_SECONDS
        )
        current = _snapshot_odds(checkpoints[CHECKPOINT_LABEL])
        predicted_log_ratio = _predict_log_ratio(example["features"], fitted)
        combination = str(example["combination"])
        predicted = min(
            MAX_ODDS,
            max(
                MIN_ODDS,
                float(example["current_odds"]) * math.exp(predicted_log_ratio),
            ),
        )
        estimated = dict(current)
        estimated[combination] = predicted
        item = dict(race)
        item["odds"] = current
        item["estimated_final_odds"] = estimated
        item["closing_odds_forecast_target"] = "V25_top1_conditional_median"
        item["closing_odds_model_trained_through_date"] = closing_artifact[
            "trained_through_date"
        ]
        transformed.append(item)
        baseline_errors.append(abs(float(example["raw_target_log_ratio"])))
        model_errors.append(
            abs(float(example["raw_target_log_ratio"]) - predicted_log_ratio)
        )
    return transformed, {
        "eligible_races": len(transformed),
        "excluded_races": len(races) - len(transformed),
        "baseline_top1_log_mae": (
            float(np.mean(baseline_errors)) if baseline_errors else None
        ),
        "v36_top1_log_mae": (
            float(np.mean(model_errors)) if model_errors else None
        ),
        "top1_relative_mae_improvement": (
            1.0 - float(np.mean(model_errors)) / float(np.mean(baseline_errors))
            if baseline_errors and float(np.mean(baseline_errors)) > 0.0
            else None
        ),
    }


def walk_forward_v25_top1_closing_v36(
    races: list[dict[str, Any]],
    *,
    probability_artifact: Mapping[str, Any],
    evaluation_dates: Iterable[str],
    minimum_training_days: int = 2,
    minimum_training_races: int = 200,
    initial_bankroll_yen: int = 10_000,
) -> dict[str, Any]:
    by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for race in races:
        by_day[str(race["race_date"])].append(race)
    transformed: list[dict[str, Any]] = []
    folds: list[dict[str, Any]] = []
    for evaluation_date in sorted(set(evaluation_dates)):
        holdout = by_day.get(evaluation_date, [])
        if not holdout:
            continue
        artifact = fit_v25_top1_closing_v36(
            races,
            probability_artifact=probability_artifact,
            prediction_date=evaluation_date,
            minimum_training_days=minimum_training_days,
            minimum_training_races=minimum_training_races,
        )
        augmented, metrics = attach_v25_top1_closing_v36(
            holdout,
            probability_artifact=probability_artifact,
            closing_artifact=artifact,
        )
        transformed.extend(augmented)
        folds.append(
            {
                "evaluation_date": evaluation_date,
                "status": (
                    "strict_prior_top1_forecast"
                    if artifact["ready"]
                    else "no_bet_insufficient_strict_prior_data"
                ),
                "trained_through_date": artifact["trained_through_date"],
                "training_days": artifact["training_days"],
                "training_races": artifact["training_races"],
                "engine": (
                    artifact["fitted"].get("engine")
                    if isinstance(artifact.get("fitted"), Mapping)
                    else None
                ),
                **metrics,
            }
        )
    bankroll = simulate_v25_top1_narrow_v33(
        transformed,
        probability_artifact=probability_artifact,
        initial_bankroll_yen=initial_bankroll_yen,
    )
    return {
        **bankroll,
        "model": MODEL_NAME,
        "odds_head": "strict_prior_V25_top1_specific_closing_regression",
        "evaluation_dates": [fold["evaluation_date"] for fold in folds],
        "folds": folds,
        "promotion_evidence": False,
        "status": "retrospective_diagnostic_only",
        "real_betting_enabled": False,
    }
