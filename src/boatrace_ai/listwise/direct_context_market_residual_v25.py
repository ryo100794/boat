from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize

from .contextual_market_residual_v24 import (
    EPSILON,
    FEATURE_DIMENSION as BASE_FEATURE_DIMENSION,
    PreparedRace,
    _lanes,
    prepare_race,
)


REGULARIZATIONS = (0.01, 0.1, 1.0)
CONTEXT_FEATURES = (
    "class_rank",
    "national_win_rate",
    "national_2_rate",
    "local_win_rate",
    "local_2_rate",
    "motor_2_rate",
    "boat_2_rate",
    "national_win_rate_rank",
    "local_win_rate_rank",
    "motor_2_rate_rank",
    "boat_2_rate_rank",
    "research_racer_strength",
    "research_racer_strength_rank",
    "research_equipment_strength",
    "research_local_vs_national_win",
    "research_home_branch",
    "hist_racer_win_rate_s",
    "hist_racer_venue_win_rate_s",
    "hist_motor_win_rate_s",
    "hist_boat_win_rate_s",
)
CONTEXT_COLUMNS = len(CONTEXT_FEATURES) * 2
STAGES = 3
FEATURE_DIMENSION = BASE_FEATURE_DIMENSION + STAGES * CONTEXT_COLUMNS


@dataclass(frozen=True)
class PreparedContextRace:
    base: PreparedRace
    stage_lanes: np.ndarray
    lane_context: np.ndarray


def extract_lane_context(
    feature_rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for row in feature_rows:
        meta = row.get("meta") or {}
        lane = int(meta.get("lane", 0))
        if not 1 <= lane <= 6:
            raise ValueError("feature row lane must be between one and six")
        features = row.get("features") or {}
        selected: dict[str, float] = {}
        for name in CONTEXT_FEATURES:
            try:
                value = float(features[name])
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
            if math.isfinite(value):
                selected[name] = value
        result[str(lane)] = selected
    if set(result) != {str(lane) for lane in range(1, 7)}:
        raise ValueError("exactly six lane feature rows are required")
    return result


def _lane_context_matrix(race: Mapping[str, Any]) -> np.ndarray:
    source = race.get("lane_context")
    if not isinstance(source, Mapping):
        return np.zeros((6, CONTEXT_COLUMNS), dtype=np.float64)
    raw = np.full((6, len(CONTEXT_FEATURES)), np.nan, dtype=np.float64)
    for lane in range(1, 7):
        values = source.get(str(lane), source.get(lane))
        if not isinstance(values, Mapping):
            raise ValueError(f"race is missing lane_context for lane {lane}")
        for column, name in enumerate(CONTEXT_FEATURES):
            try:
                value = float(values[name])
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
            if math.isfinite(value):
                raw[lane - 1, column] = value

    present = np.isfinite(raw)
    standardized = np.zeros_like(raw)
    for column in range(raw.shape[1]):
        mask = present[:, column]
        if int(mask.sum()) < 2:
            continue
        values = raw[mask, column]
        scale = float(np.std(values))
        if scale > 1e-9:
            standardized[mask, column] = (values - float(np.mean(values))) / scale
    coverage = present.astype(np.float64)
    coverage -= np.mean(coverage, axis=0, keepdims=True)
    return np.column_stack((standardized, coverage))


def prepare_context_race(race: dict[str, Any]) -> PreparedContextRace:
    base = prepare_race(race)
    stage_lanes = np.asarray(
        [_lanes(combination) for combination in base.combinations],
        dtype=np.int8,
    )
    return PreparedContextRace(
        base=base,
        stage_lanes=stage_lanes,
        lane_context=_lane_context_matrix(race),
    )


def _probability_vector(
    race: PreparedContextRace,
    coefficients: np.ndarray,
) -> np.ndarray:
    base_coefficients = coefficients[:BASE_FEATURE_DIMENSION]
    context_coefficients = coefficients[BASE_FEATURE_DIMENSION:].reshape(
        STAGES, CONTEXT_COLUMNS
    )
    logits = race.base.market_log + np.sum(
        base_coefficients[race.base.indices] * race.base.values,
        axis=1,
    )
    for stage in range(STAGES):
        logits += (
            race.lane_context[race.stage_lanes[:, stage]]
            @ context_coefficients[stage]
        )
    logits -= float(np.max(logits))
    probabilities = np.exp(logits)
    probabilities /= float(np.sum(probabilities))
    return probabilities


def direct_context_probabilities(
    race: dict[str, Any],
    artifact: Mapping[str, Any],
) -> dict[str, float]:
    prepared = prepare_context_race(race)
    coefficients = np.asarray(artifact["coefficients"], dtype=np.float64)
    if coefficients.shape != (FEATURE_DIMENSION,):
        raise ValueError("direct-context residual coefficient shape mismatch")
    probabilities = _probability_vector(prepared, coefficients)
    return {
        combination: float(value)
        for combination, value in zip(prepared.base.combinations, probabilities)
    }


def _objective_gradient(
    coefficients: np.ndarray,
    races: list[PreparedContextRace],
    *,
    regularization: float,
) -> tuple[float, np.ndarray]:
    loss = 0.0
    gradient = np.zeros(FEATURE_DIMENSION, dtype=np.float64)
    base_gradient = gradient[:BASE_FEATURE_DIMENSION]
    context_gradient = gradient[BASE_FEATURE_DIMENSION:].reshape(
        STAGES, CONTEXT_COLUMNS
    )
    for race in races:
        probabilities = _probability_vector(race, coefficients)
        loss -= math.log(
            max(EPSILON, float(probabilities[race.base.actual_index]))
        )
        errors = probabilities
        errors[race.base.actual_index] -= 1.0
        np.add.at(
            base_gradient,
            race.base.indices.reshape(-1),
            (errors[:, None] * race.base.values).reshape(-1),
        )
        for stage in range(STAGES):
            lane_errors = np.bincount(
                race.stage_lanes[:, stage],
                weights=errors,
                minlength=6,
            )
            context_gradient[stage] += race.lane_context.T @ lane_errors
    scale = 1.0 / len(races)
    loss *= scale
    gradient *= scale
    loss += 0.5 * regularization * float(coefficients @ coefficients)
    gradient += regularization * coefficients
    return loss, gradient


def fit_direct_context_residual(
    races: list[dict[str, Any]],
    *,
    regularization: float,
    max_iterations: int = 50,
) -> dict[str, Any]:
    if not races:
        raise ValueError("at least one race is required")
    if regularization <= 0.0 or not math.isfinite(regularization):
        raise ValueError("regularization must be finite and positive")
    prepared = [prepare_context_race(race) for race in races]

    def objective(coefficients: np.ndarray) -> tuple[float, np.ndarray]:
        return _objective_gradient(
            coefficients,
            prepared,
            regularization=regularization,
        )

    result = minimize(
        objective,
        np.zeros(FEATURE_DIMENSION, dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": int(max_iterations), "ftol": 1e-10, "gtol": 1e-6},
    )
    objective_value, gradient = objective(np.asarray(result.x, dtype=np.float64))
    return {
        "model": "direct_context_market_residual_v25",
        "feature_dimension": FEATURE_DIMENSION,
        "base_feature_dimension": BASE_FEATURE_DIMENSION,
        "context_features": list(CONTEXT_FEATURES),
        "context_design": "within-race z-score and availability by finish stage",
        "regularization": float(regularization),
        "coefficients": [float(value) for value in result.x],
        "objective": float(objective_value),
        "gradient_norm": float(np.linalg.norm(gradient)),
        "iterations": int(result.nit),
        "converged": bool(result.success),
        "message": str(result.message),
        "training_races": len(races),
    }


def direct_context_metrics(
    races: list[dict[str, Any]],
    artifact: Mapping[str, Any],
) -> dict[str, Any]:
    loss = market_loss = raw_model_loss = 0.0
    top5_hits = market_top5_hits = 0
    for race in races:
        probabilities = direct_context_probabilities(race, artifact)
        market = race["market_probabilities"]
        model = race["model_probabilities"]
        actual = str(race["actual_combination"])
        loss -= math.log(max(EPSILON, float(probabilities.get(actual, 0.0))))
        market_loss -= math.log(max(EPSILON, float(market.get(actual, 0.0))))
        raw_model_loss -= math.log(max(EPSILON, float(model.get(actual, 0.0))))
        top5_hits += int(
            actual in sorted(probabilities, key=probabilities.get, reverse=True)[:5]
        )
        market_top5_hits += int(
            actual in sorted(market, key=market.get, reverse=True)[:5]
        )
    count = len(races)
    return {
        "evaluated_races": count,
        "trifecta_log_loss": loss / count if count else None,
        "market_trifecta_log_loss": market_loss / count if count else None,
        "raw_model_trifecta_log_loss": raw_model_loss / count if count else None,
        "trifecta_top5_hit_rate": top5_hits / count if count else None,
        "market_trifecta_top5_hit_rate": market_top5_hits / count if count else None,
    }


def fit_temporal_direct_context_residual(
    calibration: list[dict[str, Any]],
    evaluation: list[dict[str, Any]],
    *,
    regularizations: Iterable[float] = REGULARIZATIONS,
) -> dict[str, Any]:
    dates = sorted({str(race["race_date"]) for race in calibration})
    if len(dates) < 2:
        raise ValueError("at least two calibration days are required")
    candidates_values = tuple(float(value) for value in regularizations)
    if not candidates_values:
        raise ValueError("at least one regularization is required")
    split_index = max(1, min(len(dates) - 1, int(len(dates) * 0.8)))
    fit_dates = set(dates[:split_index])
    validation_dates = set(dates[split_index:])
    inner_fit = [race for race in calibration if str(race["race_date"]) in fit_dates]
    inner_validation = [
        race for race in calibration if str(race["race_date"]) in validation_dates
    ]
    candidates = []
    for regularization in candidates_values:
        artifact = fit_direct_context_residual(
            inner_fit,
            regularization=regularization,
            max_iterations=30,
        )
        candidates.append({
            "regularization": regularization,
            "metrics": direct_context_metrics(inner_validation, artifact),
            "converged": artifact["converged"],
        })
    selected = min(
        candidates,
        key=lambda row: (
            float(row["metrics"]["trifecta_log_loss"]),
            -float(row["regularization"]),
        ),
    )
    artifact = fit_direct_context_residual(
        calibration,
        regularization=float(selected["regularization"]),
        max_iterations=50,
    )
    return {
        "validation_design": (
            "Regularization is selected on the latest inner prior-day block; "
            "coefficients are refit on all prior days and scored on outer days"
        ),
        "inner_fit_through": dates[split_index - 1],
        "inner_validation_from": dates[split_index],
        "regularization_candidates": candidates,
        "artifact": artifact,
        "metrics": direct_context_metrics(evaluation, artifact),
    }
