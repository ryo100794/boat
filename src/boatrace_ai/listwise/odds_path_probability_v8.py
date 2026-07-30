from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable

import numpy as np


EPSILON = 1e-12
MODEL_NAME = "odds_path_market_offset_probability_v8"
DEFAULT_REGULARIZATIONS = (0.01, 0.1, 1.0, 10.0)
MIN_NESTED_TRAINING_DAYS = 2
MIN_NESTED_VALIDATION_DAYS = 2

FEATURE_NAMES = (
    "log_base_to_market_probability_ratio",
    "base_minus_market_rank",
    "recent_log_probability_slope",
    "long_log_probability_slope",
    "slope_acceleration",
    "path_volatility",
)


def _race_sort_key(race: dict[str, Any]) -> tuple[str, str, int, str]:
    return (
        str(race.get("race_date") or ""),
        str(race.get("jcd") or ""),
        int(race.get("rno") or 0),
        str(race.get("race_id") or ""),
    )


def _normalized_descending_ranks(
    values: dict[str, float],
) -> dict[str, float]:
    ordered = sorted(values, key=lambda key: (-float(values[key]), key))
    denominator = max(1, len(ordered) - 1)
    return {key: index / denominator for index, key in enumerate(ordered)}


def _normalized_values(
    values: dict[str, float], combinations: list[str]
) -> np.ndarray:
    result = np.asarray([float(values[key]) for key in combinations], dtype=np.float64)
    if not np.all(np.isfinite(result)) or np.any(result <= 0.0):
        raise ValueError("v8 probability inputs must be finite and positive")
    total = float(result.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("v8 probability inputs must have positive mass")
    return result / total


def _trend_features(
    race: dict[str, Any], combination: str
) -> tuple[float, float, float, float]:
    values: list[tuple[float, float]] = []
    for point in race.get("odds_path") or []:
        probability = (point.get("market_probabilities") or {}).get(combination)
        if probability is None or float(probability) <= 0.0:
            continue
        values.append((
            float(point.get("minutes_before_decision") or 0.0),
            math.log(float(probability)),
        ))
    values.sort(reverse=True)
    if len(values) < 2:
        return 0.0, 0.0, 0.0, 0.0
    slopes = [
        (values[index][1] - values[index - 1][1])
        / max(EPSILON, values[index - 1][0] - values[index][0])
        for index in range(1, len(values))
    ]
    recent = slopes[-1]
    long = (values[-1][1] - values[0][1]) / max(
        EPSILON, values[0][0] - values[-1][0]
    )
    previous = (
        sum(slopes[:-1]) / len(slopes[:-1]) if len(slopes) > 1 else long
    )
    volatility = float(np.std(slopes)) if len(slopes) > 1 else 0.0
    return recent, long, recent - previous, volatility


def _race_design(
    race: dict[str, Any],
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    base = race.get("base_model_probabilities") or race.get("model_probabilities")
    market = race.get("market_probabilities")
    if not isinstance(base, dict) or not isinstance(market, dict):
        raise ValueError("v8 probability model requires base and market probabilities")
    if len(base) != 120 or len(market) != 120 or set(base) != set(market):
        raise ValueError("v8 probability model requires the same 120 combinations")
    combinations = sorted(base)
    base_values = _normalized_values(base, combinations)
    market_values = _normalized_values(market, combinations)
    normalized_base = dict(zip(combinations, base_values.tolist()))
    normalized_market = dict(zip(combinations, market_values.tolist()))
    base_ranks = _normalized_descending_ranks(normalized_base)
    market_ranks = _normalized_descending_ranks(normalized_market)
    features = []
    for index, combination in enumerate(combinations):
        recent, long, acceleration, volatility = _trend_features(
            race, combination
        )
        features.append((
            float(np.clip(
                math.log(base_values[index] / market_values[index]), -8.0, 8.0
            )),
            base_ranks[combination] - market_ranks[combination],
            float(np.clip(recent * 10.0, -3.0, 3.0)),
            float(np.clip(long * 20.0, -3.0, 3.0)),
            float(np.clip(acceleration * 10.0, -3.0, 3.0)),
            float(np.clip(volatility * 10.0, 0.0, 3.0)),
        ))
    return (
        combinations,
        np.log(np.maximum(EPSILON, market_values)),
        np.asarray(features, dtype=np.float64),
        market_values,
    )


def _prepare_training_tensors(
    races: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not races:
        raise ValueError("v8 probability model requires races")
    offsets: list[np.ndarray] = []
    features: list[np.ndarray] = []
    actual_indices: list[int] = []
    for race in sorted(races, key=_race_sort_key):
        combinations, offset, raw_features, _market = _race_design(race)
        actual = str(race.get("actual_combination") or "")
        if actual not in combinations:
            raise ValueError("v8 actual combination is missing")
        offsets.append(offset)
        features.append(raw_features)
        actual_indices.append(combinations.index(actual))
    return (
        np.stack(offsets),
        np.stack(features),
        np.asarray(actual_indices, dtype=np.int64),
    )


def _fit_feature_scaler(raw_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(raw_features, dtype=np.float64)
    if values.ndim != 3 or values.shape[2] != len(FEATURE_NAMES):
        raise ValueError("v8 raw feature tensor has an invalid shape")
    flattened = values.reshape(-1, values.shape[2])
    mean = flattened.mean(axis=0)
    scale = flattened.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return mean, scale


def _objective_gradient_hessian(
    market_offsets: np.ndarray,
    standardized_features: np.ndarray,
    actual_indices: np.ndarray,
    coefficients: np.ndarray,
    *,
    regularization: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    offsets = np.asarray(market_offsets, dtype=np.float64)
    features = np.asarray(standardized_features, dtype=np.float64)
    actual = np.asarray(actual_indices, dtype=np.int64)
    weights = np.asarray(coefficients, dtype=np.float64)
    if offsets.ndim != 2 or features.shape[:2] != offsets.shape:
        raise ValueError("v8 offset and feature tensors do not align")
    if features.ndim != 3 or features.shape[2] != len(weights):
        raise ValueError("v8 feature and coefficient tensors do not align")
    if actual.shape != (len(offsets),):
        raise ValueError("v8 actual indices do not align")
    if regularization <= 0.0 or not math.isfinite(regularization):
        raise ValueError("v8 regularization must be finite and positive")

    logits = offsets + features @ weights
    maximum = np.max(logits, axis=1, keepdims=True)
    shifted = logits - maximum
    exp_logits = np.exp(shifted)
    probabilities = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    row_indices = np.arange(len(offsets))
    losses = maximum[:, 0] + np.log(exp_logits.sum(axis=1))
    losses -= logits[row_indices, actual]
    means = np.einsum("rc,rcf->rf", probabilities, features, optimize=True)
    actual_features = features[row_indices, actual]
    gradient = np.mean(means - actual_features, axis=0)
    hessian = np.einsum(
        "rc,rcf,rcg->fg",
        probabilities,
        features,
        features,
        optimize=True,
    ) / len(offsets)
    hessian -= np.einsum("rf,rg->fg", means, means, optimize=True) / len(offsets)
    objective = float(losses.mean()) + 0.5 * regularization * float(weights @ weights)
    gradient += regularization * weights
    hessian += regularization * np.eye(len(weights), dtype=np.float64)
    return objective, gradient, hessian


def _fit_fixed_regularization(
    races: list[dict[str, Any]],
    *,
    regularization: float,
    max_iterations: int,
    tolerance: float,
) -> dict[str, Any]:
    market_offsets, raw_features, actual_indices = _prepare_training_tensors(races)
    feature_mean, feature_scale = _fit_feature_scaler(raw_features)
    features = (raw_features - feature_mean) / feature_scale
    coefficients = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    converged = False
    objective = math.inf
    iteration = 0
    for iteration in range(1, max_iterations + 1):
        objective, gradient, hessian = _objective_gradient_hessian(
            market_offsets,
            features,
            actual_indices,
            coefficients,
            regularization=regularization,
        )
        damped = hessian + 1e-9 * np.eye(len(coefficients), dtype=np.float64)
        try:
            step = np.linalg.solve(damped, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(damped, gradient, rcond=None)[0]
        scale = 1.0
        accepted = False
        candidate = coefficients
        candidate_objective = objective
        while scale >= 1e-8:
            trial = coefficients - scale * step
            trial_objective, _gradient, _hessian = _objective_gradient_hessian(
                market_offsets,
                features,
                actual_indices,
                trial,
                regularization=regularization,
            )
            if trial_objective <= objective + 1e-12:
                candidate = trial
                candidate_objective = trial_objective
                accepted = True
                break
            scale *= 0.5
        if not accepted:
            converged = True
            break
        change = float(np.max(np.abs(candidate - coefficients)))
        coefficients = candidate
        objective = candidate_objective
        if change <= tolerance:
            converged = True
            break

    dates = sorted({str(race["race_date"]) for race in races})
    return {
        "model_type": MODEL_NAME,
        "architecture": "fixed_market_log_offset_plus_standardized_residual",
        "teacher": "actual_120_class_trifecta",
        "loss": "multinomial_cross_entropy_plus_zero_centered_l2",
        "feature_names": list(FEATURE_NAMES),
        "feature_mean": feature_mean.tolist(),
        "feature_scale": feature_scale.tolist(),
        "feature_standardization": "ticket_level_training_only_zscore",
        "coefficients": coefficients.tolist(),
        "fixed_market_log_coefficient": 1.0,
        "regularization": float(regularization),
        "training_races": len(races),
        "training_days": len(dates),
        "training_dates": dates,
        "trained_through_date": dates[-1],
        "iterations": iteration,
        "converged": converged,
        "objective": float(objective),
        "uses_return_multiplier": False,
        "uses_historical_hit_lift": False,
    }


def attach_odds_path_probability_v8(
    races: list[dict[str, Any]], model: dict[str, Any]
) -> list[dict[str, Any]]:
    if tuple(model.get("feature_names") or ()) != FEATURE_NAMES:
        raise ValueError("v8 probability feature contract mismatch")
    if float(model.get("fixed_market_log_coefficient") or 0.0) != 1.0:
        raise ValueError("v8 requires a fixed market log coefficient of one")
    coefficients = np.asarray(model["coefficients"], dtype=np.float64)
    mean = np.asarray(model["feature_mean"], dtype=np.float64)
    scale = np.asarray(model["feature_scale"], dtype=np.float64)
    if coefficients.shape != (len(FEATURE_NAMES),):
        raise ValueError("v8 probability coefficients have an invalid shape")
    if mean.shape != coefficients.shape or scale.shape != coefficients.shape:
        raise ValueError("v8 probability scaler has an invalid shape")
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ValueError("v8 probability feature scales must be positive")

    result = []
    zero_residual = bool(np.all(coefficients == 0.0))
    for race in races:
        combinations, offsets, raw_features, market_values = _race_design(race)
        if zero_residual:
            probabilities = market_values
        else:
            logits = offsets + ((raw_features - mean) / scale) @ coefficients
            logits -= float(np.max(logits))
            probabilities = np.exp(logits)
            probabilities /= float(probabilities.sum())
        item = dict(race)
        item["base_model_probabilities"] = dict(
            race.get("base_model_probabilities")
            or race["model_probabilities"]
        )
        item["model_probabilities"] = dict(
            zip(combinations, probabilities.tolist())
        )
        item.pop("historical_return_multipliers", None)
        item["operational_probability_source"] = MODEL_NAME
        result.append(item)
    return result


def _mean_log_loss(
    races: list[dict[str, Any]], model: dict[str, Any]
) -> float:
    transformed = attach_odds_path_probability_v8(races, model)
    losses = [
        -math.log(max(
            EPSILON,
            float(race["model_probabilities"][str(race["actual_combination"])]),
        ))
        for race in transformed
    ]
    return float(np.mean(losses))


def select_regularization_nested(
    races: list[dict[str, Any]],
    *,
    regularizations: Iterable[float] = DEFAULT_REGULARIZATIONS,
    min_training_days: int = MIN_NESTED_TRAINING_DAYS,
    min_validation_days: int = MIN_NESTED_VALIDATION_DAYS,
    max_iterations: int = 40,
    tolerance: float = 1e-7,
) -> dict[str, Any]:
    candidates = sorted({float(value) for value in regularizations})
    if not candidates or any(
        value <= 0.0 or not math.isfinite(value) for value in candidates
    ):
        raise ValueError("v8 regularization candidates must be positive")
    if min_training_days < 1 or min_validation_days < 1:
        raise ValueError("v8 nested day requirements must be positive")
    by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for race in races:
        by_day[str(race["race_date"])].append(race)
    for rows in by_day.values():
        rows.sort(key=_race_sort_key)
    dates = sorted(by_day)
    validation_dates = dates[min_training_days:]
    if len(validation_dates) < min_validation_days:
        selected = max(candidates)
        return {
            "method": "forward_daily_nested_log_loss",
            "status": "conservative_fallback",
            "reason": "insufficient_nested_validation_days",
            "selected_regularization": selected,
            "selection_basis": "largest_candidate_keeps_residual_near_market",
            "training_dates": dates,
            "validation_dates": validation_dates,
            "minimum_training_days": min_training_days,
            "minimum_validation_days": min_validation_days,
            "candidates": [],
        }

    evaluated = []
    for regularization in candidates:
        folds = []
        weighted_loss = 0.0
        evaluated_races = 0
        for validation_date in validation_dates:
            training_dates = [date for date in dates if date < validation_date]
            training = [
                race for date in training_dates for race in by_day[date]
            ]
            holdout = by_day[validation_date]
            fold_model = _fit_fixed_regularization(
                training,
                regularization=regularization,
                max_iterations=max_iterations,
                tolerance=tolerance,
            )
            loss = _mean_log_loss(holdout, fold_model)
            weighted_loss += loss * len(holdout)
            evaluated_races += len(holdout)
            folds.append({
                "training_dates": training_dates,
                "trained_through_date": training_dates[-1],
                "validation_date": validation_date,
                "validation_races": len(holdout),
                "validation_log_loss": loss,
                "feature_mean": fold_model["feature_mean"],
                "feature_scale": fold_model["feature_scale"],
                "converged": fold_model["converged"],
            })
        evaluated.append({
            "regularization": regularization,
            "validation_days": len(folds),
            "validation_races": evaluated_races,
            "prequential_log_loss": weighted_loss / evaluated_races,
            "folds": folds,
        })
    selected_row = min(
        evaluated,
        key=lambda row: (
            float(row["prequential_log_loss"]),
            -float(row["regularization"]),
        ),
    )
    return {
        "method": "forward_daily_nested_log_loss",
        "status": "selected",
        "reason": "minimum_prequential_trifecta_log_loss",
        "selected_regularization": float(selected_row["regularization"]),
        "selection_basis": (
            "race_weighted forward-day trifecta log loss; stronger L2 wins ties"
        ),
        "training_dates": dates,
        "validation_dates": validation_dates,
        "minimum_training_days": min_training_days,
        "minimum_validation_days": min_validation_days,
        "candidates": evaluated,
    }


def fit_odds_path_probability_v8(
    races: list[dict[str, Any]],
    *,
    regularizations: Iterable[float] = DEFAULT_REGULARIZATIONS,
    min_training_days: int = MIN_NESTED_TRAINING_DAYS,
    min_validation_days: int = MIN_NESTED_VALIDATION_DAYS,
    max_iterations: int = 40,
    tolerance: float = 1e-7,
) -> dict[str, Any]:
    if not races:
        raise ValueError("v8 probability model requires races")
    selection = select_regularization_nested(
        races,
        regularizations=regularizations,
        min_training_days=min_training_days,
        min_validation_days=min_validation_days,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    model = _fit_fixed_regularization(
        races,
        regularization=float(selection["selected_regularization"]),
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    model["regularization_selection"] = selection
    model["selection_basis"] = {
        "method": selection["method"],
        "status": selection["status"],
        "reason": selection["reason"],
        "selected_regularization": selection["selected_regularization"],
        "validation_dates": selection["validation_dates"],
        "holdout_contract": (
            "each nested validation date is scored by a model and scaler fitted "
            "only on strictly earlier dates"
        ),
    }
    return model
