from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable

import numpy as np


EPSILON = 1e-12
DEFAULT_REGULARIZATION = (0.001, 0.01, 0.1, 1.0, 10.0)


def first_place_marginals(probabilities: dict[str, float]) -> dict[str, float]:
    marginals: dict[str, float] = defaultdict(float)
    for combination, probability in probabilities.items():
        first = str(combination).split("-", 1)[0]
        marginals[first] += float(probability)
    total = sum(marginals.values())
    if total <= 0.0:
        return {}
    return {lane: value / total for lane, value in marginals.items()}


def _race_design(race: dict[str, Any]) -> tuple[list[str], np.ndarray]:
    model = race["model_probabilities"]
    market = race["market_probabilities"]
    combinations = sorted(set(model) & set(market))
    if not combinations:
        raise ValueError("model and market probabilities do not overlap")
    model_first = first_place_marginals(model)
    market_first = first_place_marginals(market)
    features = np.asarray(
        [
            [
                math.log(max(EPSILON, float(market[combination]))),
                math.log(
                    max(EPSILON, model_first[combination.split("-", 1)[0]])
                    / max(EPSILON, market_first[combination.split("-", 1)[0]])
                ),
            ]
            for combination in combinations
        ],
        dtype=np.float64,
    )
    return combinations, features


def position_residual_probabilities(
    race: dict[str, Any], calibrator: dict[str, Any]
) -> dict[str, float]:
    combinations, features = _race_design(race)
    coefficients = np.asarray(
        [
            float(calibrator["market_coefficient"]),
            float(calibrator["winner_residual_coefficient"]),
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
    regularization: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    objective = 0.0
    gradient = np.zeros(2, dtype=np.float64)
    hessian = np.zeros((2, 2), dtype=np.float64)
    for race in races:
        combinations, features = _race_design(race)
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
    prior = np.asarray([1.0, 0.0], dtype=np.float64)
    delta = coefficients - prior
    objective += 0.5 * regularization * float(delta @ delta)
    gradient += regularization * delta
    hessian += regularization * np.eye(2, dtype=np.float64)
    return objective, gradient, hessian


def fit_position_residual_newton(
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
    coefficients = np.asarray([1.0, 0.0], dtype=np.float64)
    converged = False
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        objective, gradient, hessian = _objective_gradient_hessian(
            races, coefficients, regularization=regularization
        )
        try:
            step = np.linalg.solve(
                hessian + 1e-9 * np.eye(2, dtype=np.float64), gradient
            )
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        scale = 1.0
        accepted = False
        candidate = coefficients
        candidate_objective = objective
        while scale >= 1e-8:
            proposal = coefficients - scale * step
            proposal[0] = max(0.05, proposal[0])
            proposal_objective, _, _ = _objective_gradient_hessian(
                races, proposal, regularization=regularization
            )
            if proposal_objective <= objective + 1e-12:
                candidate = proposal
                candidate_objective = proposal_objective
                accepted = True
                break
            scale *= 0.5
        change = float(np.max(np.abs(candidate - coefficients)))
        objective_change = abs(objective - candidate_objective)
        coefficients = candidate
        if not accepted or (change <= tolerance and objective_change <= tolerance):
            converged = True
            break
    objective, gradient, _ = _objective_gradient_hessian(
        races, coefficients, regularization=regularization
    )
    return {
        "market_coefficient": float(coefficients[0]),
        "winner_residual_coefficient": float(coefficients[1]),
        "regularization": float(regularization),
        "objective": float(objective),
        "gradient_norm": float(np.linalg.norm(gradient)),
        "iterations": iterations,
        "converged": converged,
        "training_races": len(races),
    }


def position_residual_metrics(
    races: list[dict[str, Any]], calibrator: dict[str, Any]
) -> dict[str, Any]:
    model_loss = market_loss = 0.0
    model_top5 = market_top5 = model_winner = market_winner = 0
    for race in races:
        probabilities = position_residual_probabilities(race, calibrator)
        market = race["market_probabilities"]
        actual = str(race["actual_combination"])
        model_loss -= math.log(max(EPSILON, probabilities.get(actual, 0.0)))
        market_loss -= math.log(max(EPSILON, float(market.get(actual, 0.0))))
        model_top5 += int(
            actual in sorted(probabilities, key=probabilities.get, reverse=True)[:5]
        )
        market_top5 += int(
            actual in sorted(market, key=market.get, reverse=True)[:5]
        )
        actual_first = actual.split("-", 1)[0]
        model_first = first_place_marginals(probabilities)
        market_first = first_place_marginals(market)
        model_winner += int(max(model_first, key=model_first.get) == actual_first)
        market_winner += int(max(market_first, key=market_first.get) == actual_first)
    count = len(races)
    return {
        "evaluated_races": count,
        "trifecta_log_loss": model_loss / count if count else None,
        "market_trifecta_log_loss": market_loss / count if count else None,
        "trifecta_top5_hit_rate": model_top5 / count if count else None,
        "market_trifecta_top5_hit_rate": market_top5 / count if count else None,
        "winner_top1_accuracy": model_winner / count if count else None,
        "market_winner_top1_accuracy": market_winner / count if count else None,
    }


def select_position_regularization_prequential(
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
        for index in range(1, len(dates)):
            training = [race for day in dates[:index] for race in by_day[day]]
            holdout = by_day[dates[index]]
            calibrator = fit_position_residual_newton(
                training, regularization=float(regularization)
            )
            metrics = position_residual_metrics(holdout, calibrator)
            weighted_loss += float(metrics["trifecta_log_loss"]) * len(holdout)
            total += len(holdout)
        candidates.append(
            {
                "regularization": float(regularization),
                "prequential_races": total,
                "prequential_log_loss": weighted_loss / total,
            }
        )
    selected = min(
        candidates,
        key=lambda row: (row["prequential_log_loss"], -row["regularization"]),
    )
    return {
        "selected_regularization": float(selected["regularization"]),
        "candidates": candidates,
        "final_calibrator": fit_position_residual_newton(
            races, regularization=float(selected["regularization"])
        ),
    }


def position_residual_walk_forward(
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
            selection = select_position_regularization_prequential(training)
            calibrator = selection["final_calibrator"]
        else:
            calibrator = fit_position_residual_newton(training, regularization=1.0)
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
                "metrics": position_residual_metrics(holdout, calibrator),
            }
        )
    total = sum(int(fold["evaluation_races"]) for fold in folds)
    metric_names = (
        "trifecta_log_loss",
        "market_trifecta_log_loss",
        "trifecta_top5_hit_rate",
        "market_trifecta_top5_hit_rate",
        "winner_top1_accuracy",
        "market_winner_top1_accuracy",
    )
    aggregate = {
        name: (
            sum(
                float(fold["metrics"][name]) * int(fold["evaluation_races"])
                for fold in folds
            )
            / total
            if total
            else None
        )
        for name in metric_names
    }
    return {"evaluation_races": total, "folds": folds, "metrics": aggregate}
