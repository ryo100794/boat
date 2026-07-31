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
from .direct_context_market_residual_v25 import CONTEXT_FEATURES, REGULARIZATIONS


MODEL_NAME = "pruned_direct_context_market_residual_v27"
SELECTION_TOLERANCE = 0.0005
STAGES = 3
FEATURE_VARIANTS: dict[str, tuple[str, ...]] = {
    "market_model_only": (),
    "ability_raw": ("class_rank", "national_win_rate", "national_2_rate"),
    "ability_local": (
        "national_win_rate",
        "local_win_rate",
        "research_local_vs_national_win",
        "research_home_branch",
        "hist_racer_venue_win_rate_s",
    ),
    "equipment_history": (
        "motor_2_rate",
        "boat_2_rate",
        "hist_motor_win_rate_s",
        "hist_boat_win_rate_s",
    ),
    "independent_core": (
        "class_rank",
        "national_win_rate",
        "research_local_vs_national_win",
        "research_home_branch",
        "motor_2_rate",
        "boat_2_rate",
        "hist_racer_win_rate_s",
        "hist_racer_venue_win_rate_s",
        "hist_motor_win_rate_s",
        "hist_boat_win_rate_s",
    ),
    "all_context": tuple(CONTEXT_FEATURES),
}


@dataclass(frozen=True)
class PreparedPrunedRace:
    base: PreparedRace
    stage_lanes: np.ndarray
    lane_context: np.ndarray


def _validate_active_features(active_features: Iterable[str]) -> tuple[str, ...]:
    active = tuple(str(name) for name in active_features)
    if len(active) != len(set(active)):
        raise ValueError("active V27 context features must be unique")
    unknown = set(active).difference(CONTEXT_FEATURES)
    if unknown:
        raise ValueError(f"unknown V27 context features: {sorted(unknown)}")
    return active


def _lane_context_matrix(
    race: Mapping[str, Any], active_features: Sequence[str]
) -> np.ndarray:
    columns = len(active_features)
    if columns == 0:
        return np.zeros((6, 0), dtype=np.float64)
    source = race.get("lane_context")
    if not isinstance(source, Mapping):
        return np.zeros((6, columns * 2), dtype=np.float64)
    raw = np.full((6, columns), np.nan, dtype=np.float64)
    for lane in range(1, 7):
        values = source.get(str(lane), source.get(lane))
        if not isinstance(values, Mapping):
            raise ValueError(f"race is missing lane_context for lane {lane}")
        for column, name in enumerate(active_features):
            try:
                value = float(values[name])
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
            if math.isfinite(value):
                raw[lane - 1, column] = value

    present = np.isfinite(raw)
    standardized = np.zeros_like(raw)
    for column in range(columns):
        mask = present[:, column]
        if int(mask.sum()) < 2:
            continue
        values = raw[mask, column]
        scale = float(np.std(values))
        if scale > 1e-9:
            standardized[mask, column] = (
                values - float(np.mean(values))
            ) / scale
    coverage = present.astype(np.float64)
    coverage -= np.mean(coverage, axis=0, keepdims=True)
    return np.column_stack((standardized, coverage))


def _prepare_race(
    race: dict[str, Any], active_features: Sequence[str]
) -> PreparedPrunedRace:
    base = prepare_race(race)
    return PreparedPrunedRace(
        base=base,
        stage_lanes=np.asarray(
            [_lanes(combination) for combination in base.combinations],
            dtype=np.int8,
        ),
        lane_context=_lane_context_matrix(race, active_features),
    )


def _feature_dimension(active_features: Sequence[str]) -> int:
    return BASE_FEATURE_DIMENSION + STAGES * len(active_features) * 2


def _probability_vector(
    race: PreparedPrunedRace, coefficients: np.ndarray
) -> np.ndarray:
    base_coefficients = coefficients[:BASE_FEATURE_DIMENSION]
    context_coefficients = coefficients[BASE_FEATURE_DIMENSION:].reshape(
        STAGES, race.lane_context.shape[1]
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


def _objective_gradient(
    coefficients: np.ndarray,
    races: list[PreparedPrunedRace],
    *,
    regularization: float,
) -> tuple[float, np.ndarray]:
    loss = 0.0
    gradient = np.zeros_like(coefficients)
    base_gradient = gradient[:BASE_FEATURE_DIMENSION]
    context_gradient = gradient[BASE_FEATURE_DIMENSION:].reshape(
        STAGES, races[0].lane_context.shape[1]
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
                race.stage_lanes[:, stage], weights=errors, minlength=6
            )
            context_gradient[stage] += race.lane_context.T @ lane_errors
    scale = 1.0 / len(races)
    loss *= scale
    gradient *= scale
    loss += 0.5 * regularization * float(coefficients @ coefficients)
    gradient += regularization * coefficients
    return loss, gradient


def fit_pruned_residual(
    races: list[dict[str, Any]],
    *,
    variant: str,
    regularization: float,
    max_iterations: int,
) -> dict[str, Any]:
    if not races:
        raise ValueError("at least one race is required")
    if variant not in FEATURE_VARIANTS:
        raise ValueError(f"unknown V27 feature variant: {variant}")
    if regularization <= 0.0 or not math.isfinite(regularization):
        raise ValueError("regularization must be finite and positive")
    active = _validate_active_features(FEATURE_VARIANTS[variant])
    prepared = [_prepare_race(race, active) for race in races]
    dimension = _feature_dimension(active)

    def objective(coefficients: np.ndarray) -> tuple[float, np.ndarray]:
        return _objective_gradient(
            coefficients, prepared, regularization=regularization
        )

    result = minimize(
        objective,
        np.zeros(dimension, dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": int(max_iterations), "ftol": 1e-10, "gtol": 1e-6},
    )
    objective_value, gradient = objective(np.asarray(result.x, dtype=np.float64))
    return {
        "model": MODEL_NAME,
        "feature_variant": variant,
        "active_context_features": list(active),
        "active_context_feature_count": len(active),
        "feature_dimension": dimension,
        "base_feature_dimension": BASE_FEATURE_DIMENSION,
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


def pruned_probabilities(
    race: dict[str, Any], artifact: Mapping[str, Any]
) -> dict[str, float]:
    active = _validate_active_features(artifact.get("active_context_features") or ())
    prepared = _prepare_race(race, active)
    coefficients = np.asarray(artifact["coefficients"], dtype=np.float64)
    if coefficients.shape != (_feature_dimension(active),):
        raise ValueError("pruned direct-context residual coefficient shape mismatch")
    probabilities = _probability_vector(prepared, coefficients)
    return {
        combination: float(value)
        for combination, value in zip(prepared.base.combinations, probabilities)
    }


def pruned_metrics(
    races: list[dict[str, Any]], artifact: Mapping[str, Any]
) -> dict[str, Any]:
    loss = market_loss = raw_model_loss = 0.0
    top5_hits = market_top5_hits = 0
    for race in races:
        probabilities = pruned_probabilities(race, artifact)
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


def fit_temporal_pruned_residual(
    calibration: list[dict[str, Any]],
    evaluation: list[dict[str, Any]],
    *,
    variants: Mapping[str, tuple[str, ...]] = FEATURE_VARIANTS,
    regularizations: Iterable[float] = REGULARIZATIONS,
    selection_tolerance: float = SELECTION_TOLERANCE,
) -> dict[str, Any]:
    dates = sorted({str(race["race_date"]) for race in calibration})
    if len(dates) < 2:
        raise ValueError("at least two calibration days are required")
    if selection_tolerance < 0.0 or not math.isfinite(selection_tolerance):
        raise ValueError("selection tolerance must be finite and non-negative")
    regularization_values = tuple(float(value) for value in regularizations)
    if not regularization_values:
        raise ValueError("at least one regularization is required")
    split_index = max(1, min(len(dates) - 1, int(len(dates) * 0.8)))
    fit_dates = set(dates[:split_index])
    validation_dates = set(dates[split_index:])
    inner_fit = [race for race in calibration if str(race["race_date"]) in fit_dates]
    inner_validation = [
        race for race in calibration if str(race["race_date"]) in validation_dates
    ]

    candidates = []
    for variant, declared_active in variants.items():
        if variant not in FEATURE_VARIANTS:
            raise ValueError(f"unknown V27 feature variant: {variant}")
        active = _validate_active_features(declared_active)
        if active != FEATURE_VARIANTS[variant]:
            raise ValueError("variant features differ from the preregistered design")
        for regularization in regularization_values:
            artifact = fit_pruned_residual(
                inner_fit,
                variant=variant,
                regularization=regularization,
                max_iterations=40,
            )
            candidates.append({
                "variant": variant,
                "active_context_feature_count": len(active),
                "feature_dimension": artifact["feature_dimension"],
                "regularization": regularization,
                "converged": artifact["converged"],
                "metrics": pruned_metrics(inner_validation, artifact),
            })
    converged = [row for row in candidates if row["converged"]]
    eligible = converged or candidates
    best_loss = min(float(row["metrics"]["trifecta_log_loss"]) for row in eligible)
    near_best = [
        row
        for row in eligible
        if float(row["metrics"]["trifecta_log_loss"])
        <= best_loss + float(selection_tolerance)
    ]
    selected = min(
        near_best,
        key=lambda row: (
            int(row["active_context_feature_count"]),
            -float(row["regularization"]),
            float(row["metrics"]["trifecta_log_loss"]),
        ),
    )
    artifact = fit_pruned_residual(
        calibration,
        variant=str(selected["variant"]),
        regularization=float(selected["regularization"]),
        max_iterations=100,
    )
    return {
        "model": MODEL_NAME,
        "validation_design": (
            "Feature variant and regularization are selected on the latest inner "
            "prior-day block; converged candidates within a fixed LogLoss tolerance "
            "prefer fewer direct features and stronger regularization; outer days "
            "remain untouched"
        ),
        "inner_fit_through": dates[split_index - 1],
        "inner_validation_from": dates[split_index],
        "selection_tolerance": float(selection_tolerance),
        "best_inner_log_loss": best_loss,
        "selected_candidate": selected,
        "candidates": candidates,
        "artifact": artifact,
        "metrics": pruned_metrics(evaluation, artifact),
    }
