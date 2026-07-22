from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable

import numpy as np


EPSILON = 1e-12
DEFAULT_REGULARIZATION = (0.001, 0.01, 0.1, 1.0, 10.0)


def normalized_entropy(probabilities: dict[str, float]) -> float:
    values = [float(value) for value in probabilities.values() if float(value) > 0.0]
    if len(values) <= 1:
        return 0.0
    total = sum(values)
    entropy = -sum((value / total) * math.log(value / total) for value in values)
    return entropy / math.log(len(values))


def _entropy_stats(races: list[dict[str, Any]]) -> tuple[float, float]:
    values = [normalized_entropy(race["market_probabilities"]) for race in races]
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return mean, max(0.02, math.sqrt(variance))


def _race_design(
    race: dict[str, Any], *, entropy_mean: float, entropy_scale: float
) -> tuple[list[str], np.ndarray]:
    model = race["model_probabilities"]
    market = race["market_probabilities"]
    combinations = sorted(set(model) & set(market))
    if not combinations:
        raise ValueError("model and market probabilities do not overlap")
    entropy_z = (
        normalized_entropy(market) - float(entropy_mean)
    ) / float(entropy_scale)
    rows = []
    for combination in combinations:
        log_market = math.log(max(EPSILON, float(market[combination])))
        log_residual = math.log(max(EPSILON, float(model[combination]))) - log_market
        rows.append((log_market, log_residual, entropy_z * log_residual))
    return combinations, np.asarray(rows, dtype=np.float64)


def entropy_residual_probabilities(
    race: dict[str, Any], calibrator: dict[str, Any]
) -> dict[str, float]:
    combinations, features = _race_design(
        race,
        entropy_mean=float(calibrator["entropy_mean"]),
        entropy_scale=float(calibrator["entropy_scale"]),
    )
    coefficients = np.asarray(
        [
            float(calibrator["market_coefficient"]),
            float(calibrator["residual_coefficient"]),
            float(calibrator["entropy_interaction_coefficient"]),
        ],
        dtype=np.float64,
    )
    logits = features @ coefficients
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
    entropy_mean: float,
    entropy_scale: float,
    regularization: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    objective = 0.0
    gradient = np.zeros(3, dtype=np.float64)
    hessian = np.zeros((3, 3), dtype=np.float64)
    for race in races:
        combinations, features = _race_design(
            race,
            entropy_mean=entropy_mean,
            entropy_scale=entropy_scale,
        )
        actual = str(race["actual_combination"])
        if actual not in combinations:
            raise ValueError(f"actual combination {actual} is missing")
        logits = features @ coefficients
        maximum = float(np.max(logits))
        exp_logits = np.exp(logits - maximum)
        probabilities = exp_logits / float(np.sum(exp_logits))
        actual_index = combinations.index(actual)
        objective += maximum + math.log(float(np.sum(exp_logits))) - float(
            logits[actual_index]
        )
        mean = probabilities @ features
        gradient += mean - features[actual_index]
        second_moment = (features.T * probabilities) @ features
        hessian += second_moment - np.outer(mean, mean)
    count = len(races)
    objective /= count
    gradient /= count
    hessian /= count
    prior = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    delta = coefficients - prior
    objective += 0.5 * regularization * float(delta @ delta)
    gradient += regularization * delta
    hessian += regularization * np.eye(3, dtype=np.float64)
    return objective, gradient, hessian


def fit_entropy_residual_newton(
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
    entropy_mean, entropy_scale = _entropy_stats(races)
    coefficients = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    lower = np.asarray([0.05, -4.0, -4.0], dtype=np.float64)
    upper = np.asarray([4.0, 4.0, 4.0], dtype=np.float64)
    converged = False
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        objective, gradient, hessian = _objective_gradient_hessian(
            races,
            coefficients,
            entropy_mean=entropy_mean,
            entropy_scale=entropy_scale,
            regularization=regularization,
        )
        damped = hessian + 1e-9 * np.eye(3, dtype=np.float64)
        try:
            step = np.linalg.solve(damped, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(damped, gradient, rcond=None)[0]
        scale = 1.0
        accepted = False
        next_coefficients = coefficients
        next_objective = objective
        while scale >= 1e-8:
            candidate = np.clip(coefficients - scale * step, lower, upper)
            candidate_objective, _, _ = _objective_gradient_hessian(
                races,
                candidate,
                entropy_mean=entropy_mean,
                entropy_scale=entropy_scale,
                regularization=regularization,
            )
            if candidate_objective <= objective + 1e-12:
                next_coefficients = candidate
                next_objective = candidate_objective
                accepted = True
                break
            scale *= 0.5
        parameter_change = float(
            np.max(np.abs(next_coefficients - coefficients))
        )
        objective_change = abs(next_objective - objective)
        coefficients = next_coefficients
        if not accepted or (
            parameter_change <= tolerance and objective_change <= tolerance
        ):
            converged = True
            break
    objective, gradient, _ = _objective_gradient_hessian(
        races,
        coefficients,
        entropy_mean=entropy_mean,
        entropy_scale=entropy_scale,
        regularization=regularization,
    )
    return {
        "market_coefficient": float(coefficients[0]),
        "residual_coefficient": float(coefficients[1]),
        "entropy_interaction_coefficient": float(coefficients[2]),
        "entropy_mean": entropy_mean,
        "entropy_scale": entropy_scale,
        "regularization": float(regularization),
        "objective": float(objective),
        "gradient_norm": float(np.linalg.norm(gradient)),
        "iterations": iterations,
        "converged": converged,
        "training_races": len(races),
    }


def entropy_residual_metrics(
    races: list[dict[str, Any]], calibrator: dict[str, Any]
) -> dict[str, Any]:
    loss = market_loss = 0.0
    top5 = market_top5 = 0
    for race in races:
        probabilities = entropy_residual_probabilities(race, calibrator)
        market = race["market_probabilities"]
        actual = str(race["actual_combination"])
        loss -= math.log(max(EPSILON, probabilities.get(actual, 0.0)))
        market_loss -= math.log(max(EPSILON, float(market.get(actual, 0.0))))
        top5 += int(actual in sorted(probabilities, key=probabilities.get, reverse=True)[:5])
        market_top5 += int(actual in sorted(market, key=market.get, reverse=True)[:5])
    count = len(races)
    return {
        "evaluated_races": count,
        "trifecta_log_loss": loss / count if count else None,
        "market_trifecta_log_loss": market_loss / count if count else None,
        "trifecta_top5_hit_rate": top5 / count if count else None,
        "market_trifecta_top5_hit_rate": market_top5 / count if count else None,
    }


def select_entropy_regularization_prequential(
    races: list[dict[str, Any]],
    *,
    regularizations: Iterable[float] = DEFAULT_REGULARIZATION,
) -> dict[str, Any]:
    by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for race in races:
        by_day[str(race["race_date"])].append(race)
    dates = sorted(by_day)
    if len(dates) < 2:
        raise ValueError("at least two dates are required")
    candidates = []
    for regularization in regularizations:
        weighted_loss = 0.0
        total = 0
        folds = []
        for index in range(1, len(dates)):
            training = [race for day in dates[:index] for race in by_day[day]]
            holdout = by_day[dates[index]]
            calibrator = fit_entropy_residual_newton(
                training, regularization=float(regularization)
            )
            metrics = entropy_residual_metrics(holdout, calibrator)
            weighted_loss += float(metrics["trifecta_log_loss"]) * len(holdout)
            total += len(holdout)
            folds.append(
                {
                    "training_dates": dates[:index],
                    "evaluation_date": dates[index],
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
        "selected_regularization": float(selected["regularization"]),
        "candidates": candidates,
        "final_calibrator": fit_entropy_residual_newton(
            races, regularization=float(selected["regularization"])
        ),
    }


def entropy_residual_walk_forward(
    races: list[dict[str, Any]], *, min_calibration_days: int = 1
) -> dict[str, Any]:
    by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for race in races:
        by_day[str(race["race_date"])].append(race)
    dates = sorted(by_day)
    folds = []
    for index in range(min_calibration_days, len(dates)):
        training_dates = dates[:index]
        training = [race for day in training_dates for race in by_day[day]]
        if len(training_dates) >= 2:
            selection = select_entropy_regularization_prequential(training)
        else:
            calibrator = fit_entropy_residual_newton(training, regularization=1.0)
            selection = {
                "selected_regularization": 1.0,
                "candidates": [],
                "final_calibrator": calibrator,
            }
        holdout = by_day[dates[index]]
        folds.append(
            {
                "training_dates": training_dates,
                "evaluation_date": dates[index],
                "training_races": len(training),
                "evaluation_races": len(holdout),
                "selection": selection,
                "metrics": entropy_residual_metrics(
                    holdout, selection["final_calibrator"]
                ),
            }
        )
    total = sum(int(fold["evaluation_races"]) for fold in folds)
    names = (
        "trifecta_log_loss",
        "market_trifecta_log_loss",
        "trifecta_top5_hit_rate",
        "market_trifecta_top5_hit_rate",
    )
    metrics = {
        name: (
            sum(
                float(fold["metrics"][name]) * int(fold["evaluation_races"])
                for fold in folds
            )
            / total
            if total
            else None
        )
        for name in names
    }
    return {"evaluation_races": total, "folds": folds, "metrics": metrics}
