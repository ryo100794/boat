from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable

import numpy as np


EPSILON = 1e-12
DEFAULT_REGULARIZATION = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)


def _raw_identity_calibrator(
    *,
    training_races: int,
    regularization: float,
    reason: str,
) -> dict[str, Any]:
    return {
        "projection_beta": 1.0,
        "model_market_correlation": 0.0,
        "residual_variance_fraction": 1.0,
        "residual_coefficient": 1.0,
        "model_coefficient": 1.0,
        "market_coefficient": 0.0,
        "model_weight": 1.0,
        "temperature": 1.0,
        "regularization": float(regularization),
        "iterations": 0,
        "converged": True,
        "training_races": int(training_races),
        "identity_fallback": True,
        "selection_reason": reason,
    }


def _race_vectors(race: dict[str, Any]) -> tuple[list[str], np.ndarray, np.ndarray, int]:
    combinations = sorted(
        set(race["model_probabilities"]) & set(race["market_probabilities"])
    )
    actual = str(race["actual_combination"])
    if actual not in combinations:
        raise ValueError(f"actual combination {actual} is missing")
    model = np.log(
        np.maximum(
            EPSILON,
            [float(race["model_probabilities"][key]) for key in combinations],
        )
    )
    market = np.log(
        np.maximum(
            EPSILON,
            [float(race["market_probabilities"][key]) for key in combinations],
        )
    )
    return combinations, model, market, combinations.index(actual)


def fit_market_projection(races: list[dict[str, Any]]) -> dict[str, float]:
    if not races:
        raise ValueError("at least one race is required")
    cross = 0.0
    market_square = 0.0
    model_square = 0.0
    for race in races:
        _keys, model, market, _actual = _race_vectors(race)
        model = model - float(np.mean(model))
        market = market - float(np.mean(market))
        cross += float(model @ market)
        market_square += float(market @ market)
        model_square += float(model @ model)
    beta = cross / market_square if market_square > EPSILON else 0.0
    residual_square = max(
        0.0,
        model_square - 2.0 * beta * cross + beta * beta * market_square,
    )
    correlation = (
        cross / math.sqrt(model_square * market_square)
        if model_square > EPSILON and market_square > EPSILON
        else 0.0
    )
    return {
        "projection_beta": float(beta),
        "model_market_correlation": float(np.clip(correlation, -1.0, 1.0)),
        "residual_variance_fraction": (
            residual_square / model_square if model_square > EPSILON else 0.0
        ),
    }


def orthogonal_probabilities(
    model: dict[str, float],
    market: dict[str, float],
    *,
    projection_beta: float,
    residual_coefficient: float,
) -> dict[str, float]:
    combinations = sorted(set(model) & set(market))
    if not combinations:
        return {}
    model_log = np.log(
        np.maximum(EPSILON, [float(model[key]) for key in combinations])
    )
    market_log = np.log(
        np.maximum(EPSILON, [float(market[key]) for key in combinations])
    )
    residual = model_log - float(projection_beta) * market_log
    logits = market_log + float(residual_coefficient) * residual
    logits -= float(np.max(logits))
    values = np.exp(logits)
    values /= float(np.sum(values))
    return {key: float(value) for key, value in zip(combinations, values)}


def fit_orthogonal_residual(
    races: list[dict[str, Any]],
    *,
    regularization: float,
    max_iterations: int = 50,
    tolerance: float = 1e-8,
) -> dict[str, Any]:
    if not races:
        raise ValueError("at least one race is required")
    if regularization <= 0.0 or not math.isfinite(regularization):
        raise ValueError("regularization must be finite and positive")
    projection = fit_market_projection(races)
    beta = float(projection["projection_beta"])
    upper = 4.0 if beta <= 0.0 else min(4.0, 0.95 / beta)
    coefficient = 0.0
    converged = False
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        gradient = regularization * coefficient
        hessian = regularization
        for race in races:
            _keys, model, market, actual = _race_vectors(race)
            residual = model - beta * market
            logits = market + coefficient * residual
            logits -= float(np.max(logits))
            probabilities = np.exp(logits)
            probabilities /= float(np.sum(probabilities))
            mean = float(probabilities @ residual)
            gradient += (mean - float(residual[actual])) / len(races)
            hessian += (
                float(probabilities @ (residual * residual)) - mean * mean
            ) / len(races)
        step = gradient / max(hessian, EPSILON)
        candidate = float(np.clip(coefficient - step, 0.0, upper))
        if abs(candidate - coefficient) <= tolerance:
            coefficient = candidate
            converged = True
            break
        coefficient = candidate
    market_coefficient = 1.0 - coefficient * beta
    coefficient_sum = coefficient + market_coefficient
    return {
        **projection,
        "residual_coefficient": coefficient,
        "model_coefficient": coefficient,
        "market_coefficient": market_coefficient,
        "model_weight": coefficient / coefficient_sum,
        "temperature": 1.0 / coefficient_sum,
        "regularization": float(regularization),
        "iterations": iterations,
        "converged": converged,
        "training_races": len(races),
    }


def probability_metrics(
    races: list[dict[str, Any]], calibrator: dict[str, Any]
) -> dict[str, Any]:
    loss = 0.0
    market_loss = 0.0
    raw_model_loss = 0.0
    top5_hits = 0
    for race in races:
        probabilities = orthogonal_probabilities(
            race["model_probabilities"],
            race["market_probabilities"],
            projection_beta=float(calibrator["projection_beta"]),
            residual_coefficient=float(calibrator["residual_coefficient"]),
        )
        actual = str(race["actual_combination"])
        loss -= math.log(max(EPSILON, probabilities.get(actual, 0.0)))
        market_loss -= math.log(
            max(EPSILON, float(race["market_probabilities"].get(actual, 0.0)))
        )
        raw_model_loss -= math.log(
            max(EPSILON, float(race["model_probabilities"].get(actual, 0.0)))
        )
        top5_hits += int(
            actual in sorted(probabilities, key=probabilities.get, reverse=True)[:5]
        )
    count = len(races)
    return {
        "evaluated_races": count,
        "trifecta_log_loss": loss / count if count else None,
        "raw_model_trifecta_log_loss": raw_model_loss / count if count else None,
        "market_trifecta_log_loss": market_loss / count if count else None,
        "trifecta_top5_hit_rate": top5_hits / count if count else None,
    }


def select_regularization_prequential(
    races: list[dict[str, Any]],
    *,
    regularizations: Iterable[float] = DEFAULT_REGULARIZATION,
) -> dict[str, Any]:
    by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for race in races:
        by_day[str(race["race_date"])].append(race)
    dates = sorted(by_day)
    if len(dates) < 2:
        raise ValueError("at least two dates are required for prequential selection")
    candidates = []
    for regularization in regularizations:
        weighted_loss = 0.0
        weighted_raw_model_loss = 0.0
        total = 0
        folds = []
        for index in range(1, len(dates)):
            training = [race for day in dates[:index] for race in by_day[day]]
            holdout = by_day[dates[index]]
            calibrator = fit_orthogonal_residual(
                training, regularization=float(regularization)
            )
            metrics = probability_metrics(holdout, calibrator)
            count = int(metrics["evaluated_races"])
            weighted_loss += float(metrics["trifecta_log_loss"]) * count
            weighted_raw_model_loss += (
                float(metrics["raw_model_trifecta_log_loss"]) * count
            )
            total += count
            folds.append(
                {
                    "training_dates": dates[:index],
                    "evaluation_date": dates[index],
                    "calibrator": calibrator,
                    "metrics": metrics,
                }
            )
        candidates.append(
            {
                "regularization": float(regularization),
                "prequential_races": total,
                "prequential_log_loss": weighted_loss / total,
                "raw_model_prequential_log_loss": (
                    weighted_raw_model_loss / total
                ),
                "folds": folds,
            }
        )
    selected = min(
        candidates,
        key=lambda row: (row["prequential_log_loss"], -row["regularization"]),
    )
    fitted_calibrator = fit_orthogonal_residual(
        races, regularization=float(selected["regularization"])
    )
    fitted_metrics = probability_metrics(races, fitted_calibrator)
    prequential_regression = (
        float(selected["prequential_log_loss"])
        > float(selected["raw_model_prequential_log_loss"])
    )
    refit_regression = (
        float(fitted_metrics["trifecta_log_loss"])
        > float(fitted_metrics["raw_model_trifecta_log_loss"])
    )
    fallback_reason = (
        "calibrated_prequential_log_loss_worse_than_raw"
        if prequential_regression
        else "calibrated_prior_refit_log_loss_worse_than_raw"
        if refit_regression
        else None
    )
    final_calibrator = (
        _raw_identity_calibrator(
            training_races=len(races),
            regularization=float(selected["regularization"]),
            reason=str(fallback_reason),
        )
        if fallback_reason is not None
        else fitted_calibrator
    )
    return {
        "validation_design": (
            "Market projection is label-free; residual shrinkage is selected on "
            "forward-only full-day folds; raw identity fallback is selected "
            "without access to the outer holdout"
        ),
        "dates": dates,
        "selected_regularization": selected["regularization"],
        "prequential_log_loss": min(
            float(selected["prequential_log_loss"]),
            float(selected["raw_model_prequential_log_loss"]),
        ),
        "candidate_prequential_log_loss": selected["prequential_log_loss"],
        "raw_model_prequential_log_loss": selected[
            "raw_model_prequential_log_loss"
        ],
        "calibration_nonregression": {
            "selection_data": "strict_prior_prequential_and_prior_refit_only",
            "outer_holdout_used": False,
            "candidate_trifecta_log_loss": selected["prequential_log_loss"],
            "raw_trifecta_log_loss": selected[
                "raw_model_prequential_log_loss"
            ],
            "prior_refit_candidate_trifecta_log_loss": fitted_metrics[
                "trifecta_log_loss"
            ],
            "prior_refit_raw_trifecta_log_loss": fitted_metrics[
                "raw_model_trifecta_log_loss"
            ],
            "identity_fallback_applied": fallback_reason is not None,
            "reason": fallback_reason,
        },
        "final_calibrator": final_calibrator,
        "candidates": candidates,
    }


def fit_fixed_regularization(
    races: list[dict[str, Any]], *, regularization: float = 1.0
) -> dict[str, Any]:
    fitted_calibrator = fit_orthogonal_residual(
        races, regularization=float(regularization)
    )
    metrics = probability_metrics(races, fitted_calibrator)
    fallback = (
        float(metrics["trifecta_log_loss"])
        > float(metrics["raw_model_trifecta_log_loss"])
    )
    final_calibrator = (
        _raw_identity_calibrator(
            training_races=len(races),
            regularization=regularization,
            reason="calibrated_training_log_loss_worse_than_raw",
        )
        if fallback
        else fitted_calibrator
    )
    return {
        "validation_design": "Fixed residual shrinkage for a single calibration day",
        "dates": sorted({str(race["race_date"]) for race in races}),
        "selected_regularization": float(regularization),
        "calibration_nonregression": {
            "selection_data": "single_prior_training_day_only",
            "outer_holdout_used": False,
            "candidate_trifecta_log_loss": metrics["trifecta_log_loss"],
            "raw_trifecta_log_loss": metrics["raw_model_trifecta_log_loss"],
            "identity_fallback_applied": fallback,
            "reason": (
                "calibrated_training_log_loss_worse_than_raw" if fallback else None
            ),
        },
        "final_calibrator": final_calibrator,
        "candidates": [],
    }
