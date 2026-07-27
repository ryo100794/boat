from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable

import numpy as np


EPSILON = 1e-12
DEFAULT_REGULARIZATION = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)


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
        top5_hits += int(
            actual in sorted(probabilities, key=probabilities.get, reverse=True)[:5]
        )
    count = len(races)
    return {
        "evaluated_races": count,
        "trifecta_log_loss": loss / count if count else None,
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
                "folds": folds,
            }
        )
    selected = min(
        candidates,
        key=lambda row: (row["prequential_log_loss"], -row["regularization"]),
    )
    return {
        "validation_design": (
            "Market projection is label-free; residual shrinkage is selected on "
            "forward-only full-day folds"
        ),
        "dates": dates,
        "selected_regularization": selected["regularization"],
        "prequential_log_loss": selected["prequential_log_loss"],
        "final_calibrator": fit_orthogonal_residual(
            races, regularization=float(selected["regularization"])
        ),
        "candidates": candidates,
    }


def fit_fixed_regularization(
    races: list[dict[str, Any]], *, regularization: float = 1.0
) -> dict[str, Any]:
    return {
        "validation_design": "Fixed residual shrinkage for a single calibration day",
        "dates": sorted({str(race["race_date"]) for race in races}),
        "selected_regularization": float(regularization),
        "final_calibrator": fit_orthogonal_residual(
            races, regularization=float(regularization)
        ),
        "candidates": [],
    }
