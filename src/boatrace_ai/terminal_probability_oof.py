from __future__ import annotations

from collections import defaultdict
from datetime import date
import hashlib
import json
from math import fsum, isfinite
from typing import Any, Mapping, Sequence

from .joint_scenario_model import (
    JointScenarioObservation,
    TEACHER_KIND,
    outcome_schema_fingerprint,
    terminal_probability_prediction_fingerprint,
)
from .joint_market_value import TRIFECTA_OUTCOMES, validate_probability_simplex
from .listwise.market_residual import (
    fit_log_pool_newton,
    log_pool_probabilities,
    residual_probability_metrics,
)


ARTIFACT_VERSION = "terminal_probability_strict_oof_v1"
FEATURE_CUTOFF_SECONDS = 0


def _canonical_sha256(value: object) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _normalized_reciprocal_odds(
    odds: Mapping[str, object],
    *,
    expected_outcomes: Sequence[str],
) -> dict[str, float]:
    if set(odds) != set(expected_outcomes):
        raise ValueError("official closing odds do not match the outcome schema")
    reciprocal = {}
    for outcome in expected_outcomes:
        try:
            value = float(odds[outcome])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("official closing odds must be finite and positive") from exc
        if not isfinite(value) or value <= 0.0:
            raise ValueError("official closing odds must be finite and positive")
        reciprocal[outcome] = 1.0 / value
    total = fsum(reciprocal.values())
    return {outcome: value / total for outcome, value in reciprocal.items()}


def _terminal_row(
    race: Mapping[str, Any], outcomes: Sequence[str]
) -> dict[str, Any]:
    required = {
        "race_id",
        "race_date",
        "jcd",
        "actual_combination",
        "model_probabilities",
        "official_closing_odds",
    }
    missing = required - set(race)
    if missing:
        raise ValueError("terminal teacher race is missing: " + ", ".join(sorted(missing)))
    try:
        race_date = date.fromisoformat(str(race["race_date"])).isoformat()
    except ValueError as exc:
        raise ValueError("terminal teacher race_date must be an ISO date") from exc
    raw_model = {
        str(key): float(value)
        for key, value in race["model_probabilities"].items()
    }
    model = validate_probability_simplex(
        raw_model,
        expected_outcomes=outcomes,
    )
    actual = str(race["actual_combination"])
    if actual not in outcomes:
        raise ValueError("actual combination is outside the outcome schema")
    return {
        "race_id": str(race["race_id"]),
        "race_date": race_date,
        "jcd": str(race["jcd"]),
        "actual_combination": actual,
        "model_probabilities": model,
        "market_probabilities": _normalized_reciprocal_odds(
            race["official_closing_odds"], expected_outcomes=outcomes
        ),
    }


def build_terminal_probability_oof_artifact(
    scored_races: Sequence[Mapping[str, Any]],
    *,
    minimum_training_days: int = 2,
    regularization: float = 1.0,
    expected_outcomes: Sequence[str] = TRIFECTA_OUTCOMES,
) -> dict[str, Any]:
    """Build soft F_T probability teachers with expanding prior-day folds."""
    if (
        isinstance(minimum_training_days, bool)
        or not isinstance(minimum_training_days, int)
        or minimum_training_days < 1
    ):
        raise ValueError("minimum_training_days must be positive")
    if not isfinite(regularization) or regularization < 0.0:
        raise ValueError("regularization must be finite and non-negative")
    if not scored_races:
        raise ValueError("scored_races must not be empty")
    outcomes = tuple(str(value) for value in expected_outcomes)
    if not outcomes or len(set(outcomes)) != len(outcomes):
        raise ValueError("expected_outcomes must be unique and non-empty")
    rows = [_terminal_row(race, outcomes) for race in scored_races]
    race_ids = [row["race_id"] for row in rows]
    if len(set(race_ids)) != len(race_ids):
        raise ValueError("terminal teacher races must have unique race IDs")
    by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_day[row["race_date"]].append(row)
    dates = sorted(by_day)
    if len(dates) <= minimum_training_days:
        raise ValueError("insufficient dates for strict OOF terminal predictions")
    schema_sha = outcome_schema_fingerprint(outcomes)
    predictions = []
    folds = []
    oof_metric_rows = []
    for index in range(minimum_training_days, len(dates)):
        training_dates = dates[:index]
        evaluation_date = dates[index]
        training = [row for day in training_dates for row in by_day[day]]
        evaluation = by_day[evaluation_date]
        calibrator = fit_log_pool_newton(
            training, regularization=float(regularization)
        )
        training_ids = sorted(row["race_id"] for row in training)
        evaluation_ids = sorted(row["race_id"] for row in evaluation)
        fold_id = f"terminal-oof-{evaluation_date}"
        fold_manifest = {
            "fold_id": fold_id,
            "training_dates": training_dates,
            "trained_through_date": training_dates[-1],
            "evaluation_date": evaluation_date,
            "training_race_ids": training_ids,
            "evaluation_race_ids": evaluation_ids,
            "feature_cutoff_seconds": FEATURE_CUTOFF_SECONDS,
            "outcome_schema_sha256": schema_sha,
        }
        fold_manifest_sha = _canonical_sha256(fold_manifest)
        fold_model_sha = _canonical_sha256({
            "artifact_version": ARTIFACT_VERSION,
            "fold_manifest_sha256": fold_manifest_sha,
            "calibrator": calibrator,
            "regularization": float(regularization),
        })
        for row in evaluation:
            probabilities = log_pool_probabilities(
                row["model_probabilities"],
                row["market_probabilities"],
                model_coefficient=float(calibrator["model_coefficient"]),
                market_coefficient=float(calibrator["market_coefficient"]),
            )
            prediction_sha = terminal_probability_prediction_fingerprint(
                race_id=row["race_id"],
                probabilities=probabilities,
                artifact_sha256=fold_model_sha,
                fold_id=fold_id,
                fold_manifest_sha256=fold_manifest_sha,
                feature_cutoff_seconds=FEATURE_CUTOFF_SECONDS,
                outcomes=outcomes,
            )
            predictions.append({
                "race_id": row["race_id"],
                "race_date": evaluation_date,
                "jcd": row["jcd"],
                "teacher_trained_through_date": training_dates[-1],
                "fold_id": fold_id,
                "fold_manifest_sha256": fold_manifest_sha,
                "fold_model_sha256": fold_model_sha,
                "prediction_sha256": prediction_sha,
                "probabilities": probabilities,
            })
            oof_metric_rows.append({
                "actual_combination": row["actual_combination"],
                "model_probabilities": probabilities,
                "market_probabilities": row["market_probabilities"],
            })
        folds.append({
            **fold_manifest,
            "fold_manifest_sha256": fold_manifest_sha,
            "fold_model_sha256": fold_model_sha,
            "calibrator": calibrator,
            "metrics": residual_probability_metrics(evaluation, calibrator),
        })
    predictions.sort(key=lambda row: row["race_id"])
    artifact_contract = {
        "version": ARTIFACT_VERSION,
        "teacher_kind": TEACHER_KIND,
        "feature_cutoff_seconds": FEATURE_CUTOFF_SECONDS,
        "outcomes": outcomes,
        "outcome_schema_sha256": schema_sha,
        "minimum_training_days": minimum_training_days,
        "regularization": float(regularization),
        "prediction_hashes": [row["prediction_sha256"] for row in predictions],
    }
    aggregate_metrics = residual_probability_metrics(
        oof_metric_rows,
        {
            "model_coefficient": 1.0,
            "market_coefficient": 0.0,
        },
    )
    aggregate_metrics["trifecta_log_loss_delta_vs_market"] = (
        aggregate_metrics["trifecta_log_loss"]
        - aggregate_metrics["market_trifecta_log_loss"]
    )
    aggregate_metrics["trifecta_brier_delta_vs_market"] = (
        aggregate_metrics["trifecta_brier_score"]
        - aggregate_metrics["market_trifecta_brier_score"]
    )
    aggregate_metrics["trifecta_top5_delta_vs_market"] = (
        aggregate_metrics["trifecta_top5_hit_rate"]
        - aggregate_metrics["market_trifecta_top5_hit_rate"]
    )
    return {
        **artifact_contract,
        "artifact_contract_sha256": _canonical_sha256(artifact_contract),
        "role": "strict_oof_soft_terminal_probability_teacher",
        "selection_data": "strictly_prior_dates_only",
        "actual_one_hot_exported_as_probability": False,
        "deployment_eligible": False,
        "predictions": predictions,
        "folds": folds,
        "predicted_races": len(predictions),
        "prediction_dates": sorted({row["race_date"] for row in predictions}),
        "strict_oof_metrics": aggregate_metrics,
    }


def _verify_artifact_contract(artifact: Mapping[str, Any]) -> tuple[str, ...]:
    outcomes = tuple(str(value) for value in artifact.get("outcomes") or ())
    if not outcomes or len(set(outcomes)) != len(outcomes):
        raise ValueError("terminal artifact outcomes are empty or duplicated")
    if artifact.get("outcome_schema_sha256") != outcome_schema_fingerprint(outcomes):
        raise ValueError("terminal artifact outcome schema hash mismatch")
    predictions = artifact.get("predictions") or []
    if not isinstance(predictions, list):
        raise ValueError("terminal artifact predictions must be a list")
    folds = artifact.get("folds") or []
    if not isinstance(folds, list):
        raise ValueError("terminal artifact folds must be a list")
    folds_by_id = {}
    for fold in folds:
        fold_id = str(fold.get("fold_id") or "")
        if not fold_id or fold_id in folds_by_id:
            raise ValueError("terminal artifact fold IDs are empty or duplicated")
        manifest = {
            "fold_id": fold_id,
            "training_dates": fold.get("training_dates"),
            "trained_through_date": fold.get("trained_through_date"),
            "evaluation_date": fold.get("evaluation_date"),
            "training_race_ids": fold.get("training_race_ids"),
            "evaluation_race_ids": fold.get("evaluation_race_ids"),
            "feature_cutoff_seconds": fold.get("feature_cutoff_seconds"),
            "outcome_schema_sha256": fold.get("outcome_schema_sha256"),
        }
        manifest_sha = _canonical_sha256(manifest)
        if fold.get("fold_manifest_sha256") != manifest_sha:
            raise ValueError("terminal artifact fold manifest hash mismatch")
        model_sha = _canonical_sha256({
            "artifact_version": ARTIFACT_VERSION,
            "fold_manifest_sha256": manifest_sha,
            "calibrator": fold.get("calibrator"),
            "regularization": artifact.get("regularization"),
        })
        if fold.get("fold_model_sha256") != model_sha:
            raise ValueError("terminal artifact fold model hash mismatch")
        folds_by_id[fold_id] = fold
    for prediction in predictions:
        fold_id = str(prediction.get("fold_id") or "")
        fold = folds_by_id.get(fold_id)
        if fold is None:
            raise ValueError("terminal prediction references an unknown fold")
        if prediction.get("race_date") != fold.get("evaluation_date"):
            raise ValueError("terminal prediction evaluation date mismatch")
        if (
            prediction.get("teacher_trained_through_date")
            != fold.get("trained_through_date")
        ):
            raise ValueError("terminal prediction training boundary mismatch")
        probabilities = validate_probability_simplex(
            prediction.get("probabilities"),
            expected_outcomes=outcomes,
        )
        prediction_sha = terminal_probability_prediction_fingerprint(
            race_id=str(prediction.get("race_id") or ""),
            probabilities=probabilities,
            artifact_sha256=str(fold["fold_model_sha256"]),
            fold_id=fold_id,
            fold_manifest_sha256=str(fold["fold_manifest_sha256"]),
            feature_cutoff_seconds=FEATURE_CUTOFF_SECONDS,
            outcomes=outcomes,
        )
        if prediction.get("prediction_sha256") != prediction_sha:
            raise ValueError("terminal artifact prediction hash mismatch")
    prediction_hashes = [str(row.get("prediction_sha256")) for row in predictions]
    contract = {
        "version": artifact.get("version"),
        "teacher_kind": artifact.get("teacher_kind"),
        "feature_cutoff_seconds": artifact.get("feature_cutoff_seconds"),
        "outcomes": outcomes,
        "outcome_schema_sha256": artifact.get("outcome_schema_sha256"),
        "minimum_training_days": artifact.get("minimum_training_days"),
        "regularization": artifact.get("regularization"),
        "prediction_hashes": prediction_hashes,
    }
    if artifact.get("artifact_contract_sha256") != _canonical_sha256(contract):
        raise ValueError("terminal artifact contract hash mismatch")
    return outcomes


def _popularity_band(market_probabilities: Mapping[str, float]) -> str:
    maximum = max(float(value) for value in market_probabilities.values())
    if maximum >= 0.25:
        return "favorite_share_ge_025"
    if maximum >= 0.12:
        return "favorite_share_012_025"
    return "favorite_share_lt_012"


def joint_observations_from_terminal_oof(
    scored_races: Sequence[Mapping[str, Any]],
    artifact: Mapping[str, Any],
) -> list[JointScenarioObservation]:
    if artifact.get("version") != ARTIFACT_VERSION:
        raise ValueError("unsupported terminal probability artifact")
    if artifact.get("teacher_kind") != TEACHER_KIND:
        raise ValueError("terminal artifact teacher kind is invalid")
    outcomes = _verify_artifact_contract(artifact)
    predictions = {
        str(row["race_id"]): row for row in artifact.get("predictions") or []
    }
    if len(predictions) != len(artifact.get("predictions") or []):
        raise ValueError("terminal artifact has duplicate race predictions")
    result = []
    for race in scored_races:
        race_id = str(race.get("race_id") or "")
        prediction = predictions.get(race_id)
        if prediction is None:
            continue
        if str(race.get("race_date")) != str(prediction["race_date"]):
            raise ValueError("terminal prediction race date mismatch")
        decision_probability = race.get("model_probabilities")
        decision_market = race.get("market_probabilities")
        closing_odds = race.get("official_closing_odds")
        if not all(isinstance(value, Mapping) for value in (
            decision_probability, decision_market, closing_odds
        )):
            raise ValueError("scored race is missing joint scenario inputs")
        result.append(JointScenarioObservation(
            race_date=str(race["race_date"]),
            race_id=race_id,
            teacher_trained_through_date=str(
                prediction["teacher_trained_through_date"]
            ),
            terminal_probability_teacher_kind=TEACHER_KIND,
            terminal_probability_teacher_source=ARTIFACT_VERSION,
            terminal_probability_artifact_sha256=str(
                prediction["fold_model_sha256"]
            ),
            terminal_probability_fold_id=str(prediction["fold_id"]),
            terminal_probability_fold_manifest_sha256=str(
                prediction["fold_manifest_sha256"]
            ),
            terminal_probability_prediction_sha256=str(
                prediction["prediction_sha256"]
            ),
            terminal_probability_outcome_schema_sha256=str(
                artifact["outcome_schema_sha256"]
            ),
            terminal_probability_feature_cutoff_seconds=FEATURE_CUTOFF_SECONDS,
            venue=str(race["jcd"]),
            decision_horizon_seconds=300,
            popularity_band=_popularity_band(decision_market),
            decision_probabilities=dict(decision_probability),
            terminal_probability_teacher=dict(prediction["probabilities"]),
            decision_market_shares=dict(decision_market),
            final_market_shares=_normalized_reciprocal_odds(
                closing_odds, expected_outcomes=outcomes
            ),
        ))
    return result


__all__ = [
    "ARTIFACT_VERSION",
    "FEATURE_CUTOFF_SECONDS",
    "build_terminal_probability_oof_artifact",
    "joint_observations_from_terminal_oof",
]
