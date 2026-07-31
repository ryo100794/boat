from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from scipy.optimize import minimize


EPSILON = 1e-12
RANK_LIMITS = (1, 2, 5, 10, 20, 40, 120)
REGULARIZATIONS = (0.01, 0.1, 1.0)
FIELD_COUNT = 12

GLOBAL_RESIDUAL = 0
GLOBAL_MARKET = 1
RESIDUAL_FIRST = 2
FIRST_BIAS = RESIDUAL_FIRST + 6
SECOND_BIAS = FIRST_BIAS + 6
THIRD_BIAS = SECOND_BIAS + 6
VENUE_FIRST = THIRD_BIAS + 6
MODEL_RANK = VENUE_FIRST + 24 * 6
MARKET_RANK = MODEL_RANK + len(RANK_LIMITS)
FIRST_MARKET_RANK = MARKET_RANK + len(RANK_LIMITS)
RESIDUAL_MARKET_RANK = FIRST_MARKET_RANK + 6 * len(RANK_LIMITS)
FIRST_SECOND = RESIDUAL_MARKET_RANK + len(RANK_LIMITS)
FEATURE_DIMENSION = FIRST_SECOND + 36


@dataclass(frozen=True)
class PreparedRace:
    combinations: tuple[str, ...]
    indices: np.ndarray
    values: np.ndarray
    market_log: np.ndarray
    actual_index: int


def _rank_bucket(rank: int) -> int:
    for index, limit in enumerate(RANK_LIMITS):
        if rank <= limit:
            return index
    return len(RANK_LIMITS) - 1


def _lanes(combination: str) -> tuple[int, int, int]:
    parts = combination.split("-")
    if len(parts) != 3:
        raise ValueError(f"invalid trifecta combination: {combination}")
    lanes = tuple(int(part) - 1 for part in parts)
    if len(set(lanes)) != 3 or any(lane < 0 or lane >= 6 for lane in lanes):
        raise ValueError(f"invalid trifecta combination: {combination}")
    return lanes  # type: ignore[return-value]


def _venue_index(value: Any) -> int:
    try:
        venue = int(str(value))
    except (TypeError, ValueError):
        venue = 1
    return min(23, max(0, venue - 1))


def prepare_race(race: dict[str, Any]) -> PreparedRace:
    model = race["model_probabilities"]
    market = race["market_probabilities"]
    combinations = tuple(sorted(set(model) & set(market)))
    actual = str(race["actual_combination"])
    if not combinations or actual not in combinations:
        raise ValueError("race has no complete market or actual combination")

    model_values = np.asarray(
        [max(EPSILON, float(model[key])) for key in combinations],
        dtype=np.float64,
    )
    market_values = np.asarray(
        [max(EPSILON, float(market[key])) for key in combinations],
        dtype=np.float64,
    )
    model_log = np.log(model_values)
    market_log = np.log(market_values)
    residual = model_log - market_log
    residual -= float(np.mean(residual))
    centered_market = market_log - float(np.mean(market_log))

    model_order = np.argsort(-model_values, kind="stable")
    market_order = np.argsort(-market_values, kind="stable")
    model_ranks = np.empty(len(combinations), dtype=np.int16)
    market_ranks = np.empty(len(combinations), dtype=np.int16)
    model_ranks[model_order] = np.arange(1, len(combinations) + 1)
    market_ranks[market_order] = np.arange(1, len(combinations) + 1)
    venue = _venue_index(race.get("jcd"))

    indices = np.empty((len(combinations), FIELD_COUNT), dtype=np.int16)
    values = np.ones((len(combinations), FIELD_COUNT), dtype=np.float64)
    for row, combination in enumerate(combinations):
        first, second, third = _lanes(combination)
        model_bucket = _rank_bucket(int(model_ranks[row]))
        market_bucket = _rank_bucket(int(market_ranks[row]))
        indices[row] = (
            GLOBAL_RESIDUAL,
            GLOBAL_MARKET,
            RESIDUAL_FIRST + first,
            FIRST_BIAS + first,
            SECOND_BIAS + second,
            THIRD_BIAS + third,
            VENUE_FIRST + venue * 6 + first,
            MODEL_RANK + model_bucket,
            MARKET_RANK + market_bucket,
            FIRST_MARKET_RANK + first * len(RANK_LIMITS) + market_bucket,
            RESIDUAL_MARKET_RANK + market_bucket,
            FIRST_SECOND + first * 6 + second,
        )
        values[row, 0] = residual[row]
        values[row, 1] = centered_market[row]
        values[row, 2] = residual[row]
        values[row, 10] = residual[row]

    return PreparedRace(
        combinations=combinations,
        indices=indices,
        values=values,
        market_log=market_log,
        actual_index=combinations.index(actual),
    )


def _probability_vector(
    prepared: PreparedRace,
    coefficients: np.ndarray,
) -> np.ndarray:
    logits = prepared.market_log + np.sum(
        coefficients[prepared.indices] * prepared.values,
        axis=1,
    )
    logits -= float(np.max(logits))
    probabilities = np.exp(logits)
    probabilities /= float(np.sum(probabilities))
    return probabilities


def contextual_probabilities(
    race: dict[str, Any],
    artifact: dict[str, Any],
) -> dict[str, float]:
    prepared = prepare_race(race)
    coefficients = np.asarray(artifact["coefficients"], dtype=np.float64)
    if coefficients.shape != (FEATURE_DIMENSION,):
        raise ValueError("contextual residual coefficient shape mismatch")
    probabilities = _probability_vector(prepared, coefficients)
    return {
        combination: float(value)
        for combination, value in zip(prepared.combinations, probabilities)
    }


def _objective_gradient(
    coefficients: np.ndarray,
    races: list[PreparedRace],
    *,
    regularization: float,
) -> tuple[float, np.ndarray]:
    loss = 0.0
    gradient = np.zeros(FEATURE_DIMENSION, dtype=np.float64)
    for race in races:
        probabilities = _probability_vector(race, coefficients)
        loss -= math.log(max(EPSILON, float(probabilities[race.actual_index])))
        errors = probabilities
        errors[race.actual_index] -= 1.0
        np.add.at(
            gradient,
            race.indices.reshape(-1),
            (errors[:, None] * race.values).reshape(-1),
        )
    scale = 1.0 / len(races)
    loss *= scale
    gradient *= scale
    loss += 0.5 * regularization * float(coefficients @ coefficients)
    gradient += regularization * coefficients
    return loss, gradient


def fit_contextual_residual(
    races: list[dict[str, Any]],
    *,
    regularization: float,
    max_iterations: int = 50,
) -> dict[str, Any]:
    if not races:
        raise ValueError("at least one race is required")
    if regularization <= 0.0 or not math.isfinite(regularization):
        raise ValueError("regularization must be finite and positive")
    prepared = [prepare_race(race) for race in races]

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
        options={
            "maxiter": int(max_iterations),
            "ftol": 1e-10,
            "gtol": 1e-6,
            "maxls": 30,
        },
    )
    objective_value, gradient = objective(np.asarray(result.x, dtype=np.float64))
    return {
        "model": "contextual_market_residual_v24",
        "feature_dimension": FEATURE_DIMENSION,
        "feature_groups": [
            "model_market_log_residual",
            "market_temperature",
            "residual_by_first_lane",
            "lane_position_biases",
            "venue_by_first_lane",
            "model_rank_band",
            "market_rank_band",
            "residual_by_market_rank_band",
            "first_lane_by_market_rank_band",
            "first_second_lane_pair",
        ],
        "regularization": float(regularization),
        "coefficients": [float(value) for value in result.x],
        "objective": float(objective_value),
        "gradient_norm": float(np.linalg.norm(gradient)),
        "iterations": int(result.nit),
        "converged": bool(result.success),
        "message": str(result.message),
        "training_races": len(races),
    }


def contextual_metrics(
    races: list[dict[str, Any]],
    artifact: dict[str, Any],
) -> dict[str, Any]:
    loss = market_loss = raw_model_loss = 0.0
    top5_hits = market_top5_hits = 0
    for race in races:
        probabilities = contextual_probabilities(race, artifact)
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
        "market_trifecta_top5_hit_rate": (
            market_top5_hits / count if count else None
        ),
    }


def fit_temporal_contextual_residual(
    calibration: list[dict[str, Any]],
    evaluation: list[dict[str, Any]],
    *,
    regularizations: Iterable[float] = REGULARIZATIONS,
) -> dict[str, Any]:
    dates = sorted({str(race["race_date"]) for race in calibration})
    if len(dates) < 2:
        raise ValueError("at least two calibration days are required")
    regularizations = tuple(float(value) for value in regularizations)
    if not regularizations:
        raise ValueError("at least one regularization is required")
    split_index = max(1, min(len(dates) - 1, int(len(dates) * 0.8)))
    fit_dates = set(dates[:split_index])
    validation_dates = set(dates[split_index:])
    inner_fit = [
        race for race in calibration if str(race["race_date"]) in fit_dates
    ]
    inner_validation = [
        race for race in calibration if str(race["race_date"]) in validation_dates
    ]
    candidates = []
    for regularization in regularizations:
        artifact = fit_contextual_residual(
            inner_fit,
            regularization=float(regularization),
            max_iterations=30,
        )
        metrics = contextual_metrics(inner_validation, artifact)
        candidates.append({
            "regularization": float(regularization),
            "metrics": metrics,
            "converged": artifact["converged"],
        })
    selected = min(
        candidates,
        key=lambda row: (
            float(row["metrics"]["trifecta_log_loss"]),
            -float(row["regularization"]),
        ),
    )
    artifact = fit_contextual_residual(
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
        "metrics": contextual_metrics(evaluation, artifact),
    }
