from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable

import numpy as np


EPSILON = 1e-12
DEFAULT_REGULARIZATION = (0.0, 0.001, 0.01, 0.1, 1.0, 10.0)


def log_pool_probabilities(
    model: dict[str, float],
    market: dict[str, float],
    *,
    model_coefficient: float,
    market_coefficient: float,
) -> dict[str, float]:
    combinations = sorted(set(model) & set(market))
    if not combinations:
        return {}
    features = np.asarray(
        [
            [
                math.log(max(EPSILON, float(model[combination]))),
                math.log(max(EPSILON, float(market[combination]))),
            ]
            for combination in combinations
        ],
        dtype=np.float64,
    )
    logits = features @ np.asarray(
        [model_coefficient, market_coefficient], dtype=np.float64
    )
    logits -= float(np.max(logits))
    values = np.exp(logits)
    values /= float(np.sum(values))
    return {
        combination: float(value)
        for combination, value in zip(combinations, values)
    }


def _objective_gradient_hessian(
    races: list[dict[str, Any]],
    coefficients: np.ndarray,
    *,
    regularization: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    objective = 0.0
    gradient = np.zeros(2, dtype=np.float64)
    hessian = np.zeros((2, 2), dtype=np.float64)
    for race in races:
        combinations = sorted(
            set(race["model_probabilities"])
            & set(race["market_probabilities"])
        )
        actual = str(race["actual_combination"])
        if actual not in combinations:
            raise ValueError(f"actual combination {actual} is missing")
        features = np.asarray(
            [
                [
                    math.log(
                        max(
                            EPSILON,
                            float(race["model_probabilities"][combination]),
                        )
                    ),
                    math.log(
                        max(
                            EPSILON,
                            float(race["market_probabilities"][combination]),
                        )
                    ),
                ]
                for combination in combinations
            ],
            dtype=np.float64,
        )
        logits = features @ coefficients
        maximum = float(np.max(logits))
        exp_logits = np.exp(logits - maximum)
        probabilities = exp_logits / float(np.sum(exp_logits))
        actual_index = combinations.index(actual)
        log_partition = maximum + math.log(float(np.sum(exp_logits)))
        objective += log_partition - float(logits[actual_index])
        mean = probabilities @ features
        gradient += mean - features[actual_index]
        second_moment = (features.T * probabilities) @ features
        hessian += second_moment - np.outer(mean, mean)

    count = len(races)
    objective /= count
    gradient /= count
    hessian /= count
    prior = np.asarray([0.0, 1.0], dtype=np.float64)
    delta = coefficients - prior
    objective += 0.5 * regularization * float(delta @ delta)
    gradient += regularization * delta
    hessian += regularization * np.eye(2, dtype=np.float64)
    return objective, gradient, hessian


def fit_log_pool_newton(
    races: list[dict[str, Any]],
    *,
    regularization: float,
    max_iterations: int = 50,
    tolerance: float = 1e-8,
) -> dict[str, Any]:
    if not races:
        raise ValueError("at least one race is required")
    if regularization < 0.0 or not math.isfinite(regularization):
        raise ValueError("regularization must be finite and non-negative")
    coefficients = np.asarray([0.0, 1.0], dtype=np.float64)
    lower_bounds = np.asarray([0.0, 0.05], dtype=np.float64)
    converged = False
    objective = math.inf
    gradient = np.zeros(2, dtype=np.float64)
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        objective, gradient, hessian = _objective_gradient_hessian(
            races, coefficients, regularization=regularization
        )
        damped = hessian + 1e-9 * np.eye(2, dtype=np.float64)
        try:
            step = np.linalg.solve(damped, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(damped, gradient, rcond=None)[0]
        scale = 1.0
        accepted = False
        next_objective = objective
        next_coefficients = coefficients
        while scale >= 1e-8:
            candidate = np.maximum(coefficients - scale * step, lower_bounds)
            candidate_objective, _, _ = _objective_gradient_hessian(
                races, candidate, regularization=regularization
            )
            if candidate_objective <= objective + 1e-12:
                next_coefficients = candidate
                next_objective = candidate_objective
                accepted = True
                break
            scale *= 0.5
        parameter_change = float(np.max(np.abs(next_coefficients - coefficients)))
        objective_change = abs(objective - next_objective)
        coefficients = next_coefficients
        objective = next_objective
        if not accepted or (
            parameter_change <= tolerance and objective_change <= tolerance
        ):
            converged = True
            break
    objective, gradient, _ = _objective_gradient_hessian(
        races, coefficients, regularization=regularization
    )
    coefficient_sum = float(coefficients[0] + coefficients[1])
    return {
        "model_coefficient": float(coefficients[0]),
        "market_coefficient": float(coefficients[1]),
        "model_weight": float(coefficients[0]) / coefficient_sum,
        "temperature": 1.0 / coefficient_sum,
        "regularization": float(regularization),
        "objective": float(objective),
        "gradient_norm": float(np.linalg.norm(gradient)),
        "iterations": iterations,
        "converged": converged,
        "training_races": len(races),
    }


def residual_probability_metrics(
    races: list[dict[str, Any]], calibrator: dict[str, Any]
) -> dict[str, Any]:
    loss = 0.0
    market_loss = 0.0
    top5_hits = 0
    market_top5_hits = 0
    for race in races:
        probabilities = log_pool_probabilities(
            race["model_probabilities"],
            race["market_probabilities"],
            model_coefficient=float(calibrator["model_coefficient"]),
            market_coefficient=float(calibrator["market_coefficient"]),
        )
        actual = str(race["actual_combination"])
        loss -= math.log(max(EPSILON, probabilities.get(actual, 0.0)))
        market = race["market_probabilities"]
        market_loss -= math.log(max(EPSILON, float(market.get(actual, 0.0))))
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
        "trifecta_top5_hit_rate": top5_hits / count if count else None,
        "market_trifecta_top5_hit_rate": market_top5_hits / count if count else None,
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
        folds = []
        weighted_loss = 0.0
        total = 0
        for index in range(1, len(dates)):
            training = [
                race for race_date in dates[:index] for race in by_day[race_date]
            ]
            holdout = by_day[dates[index]]
            calibrator = fit_log_pool_newton(
                training, regularization=float(regularization)
            )
            metrics = residual_probability_metrics(holdout, calibrator)
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
        key=lambda row: (
            row["prequential_log_loss"],
            -row["regularization"],
        ),
    )
    final_calibrator = fit_log_pool_newton(
        races, regularization=float(selected["regularization"])
    )
    return {
        "validation_design": (
            "Regularization is selected on forward-only daily folds; final coefficients "
            "are refit on all calibration days"
        ),
        "dates": dates,
        "selected_regularization": selected["regularization"],
        "prequential_log_loss": selected["prequential_log_loss"],
        "final_calibrator": final_calibrator,
        "candidates": candidates,
    }


def fit_fixed_regularization(
    races: list[dict[str, Any]], *, regularization: float = 1.0
) -> dict[str, Any]:
    dates = sorted({str(race["race_date"]) for race in races})
    calibrator = fit_log_pool_newton(races, regularization=regularization)
    return {
        "validation_design": (
            "Regularization is preregistered because fewer than two calibration "
            "days are available; no holdout selection is performed"
        ),
        "dates": dates,
        "selected_regularization": float(regularization),
        "prequential_log_loss": None,
        "final_calibrator": calibrator,
        "candidates": [],
    }
