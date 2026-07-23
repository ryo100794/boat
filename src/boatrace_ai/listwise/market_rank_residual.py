from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable

import numpy as np


EPSILON = 1e-12
PARAMETER_NAMES = (
    "market_log_probability",
    "top5_bias",
    "rank6_20_bias",
    "top5_model_market_residual",
    "rank6_20_model_market_residual",
    "rank21_120_model_market_residual",
)
PARAMETER_COUNT = len(PARAMETER_NAMES)
DEFAULT_REGULARIZATION = (0.01, 0.1, 1.0, 10.0, 100.0)


def rank_residual_probabilities(
    model: dict[str, float],
    market: dict[str, float],
    *,
    coefficients: Iterable[float],
) -> dict[str, float]:
    combinations = sorted(set(model) & set(market))
    if not combinations:
        return {}
    values = np.asarray(tuple(coefficients), dtype=np.float64)
    if values.shape != (PARAMETER_COUNT,):
        raise ValueError(f"coefficients must contain {PARAMETER_COUNT} values")
    features = _combination_features(combinations, model=model, market=market)
    logits = features @ values
    logits -= float(np.max(logits))
    probabilities = np.exp(logits)
    probabilities /= float(np.sum(probabilities))
    return {
        combination: float(probability)
        for combination, probability in zip(combinations, probabilities)
    }


def fit_rank_residual_newton(
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
    prior = np.zeros(PARAMETER_COUNT, dtype=np.float64)
    prior[0] = 1.0
    coefficients = prior.copy()
    lower_bounds = np.full(PARAMETER_COUNT, -2.0, dtype=np.float64)
    upper_bounds = np.full(PARAMETER_COUNT, 2.0, dtype=np.float64)
    lower_bounds[0] = 0.05
    upper_bounds[0] = 4.0
    converged = False
    iterations = 0

    for iterations in range(1, max_iterations + 1):
        objective, gradient, hessian = _objective_gradient_hessian(
            races,
            coefficients,
            regularization=regularization,
            prior=prior,
        )
        damped = hessian + 1e-9 * np.eye(PARAMETER_COUNT, dtype=np.float64)
        try:
            step = np.linalg.solve(damped, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(damped, gradient, rcond=None)[0]
        scale = 1.0
        accepted = False
        next_coefficients = coefficients
        next_objective = objective
        while scale >= 1e-8:
            candidate = np.clip(
                coefficients - scale * step,
                lower_bounds,
                upper_bounds,
            )
            candidate_objective, _, _ = _objective_gradient_hessian(
                races,
                candidate,
                regularization=regularization,
                prior=prior,
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
        objective_change = abs(objective - next_objective)
        coefficients = next_coefficients
        if not accepted or (
            parameter_change <= tolerance and objective_change <= tolerance
        ):
            converged = True
            break

    objective, gradient, _ = _objective_gradient_hessian(
        races,
        coefficients,
        regularization=regularization,
        prior=prior,
    )
    return {
        "coefficients": coefficients.tolist(),
        "coefficient_map": {
            name: float(value)
            for name, value in zip(PARAMETER_NAMES, coefficients)
        },
        "regularization": float(regularization),
        "objective": float(objective),
        "gradient_norm": float(np.linalg.norm(gradient)),
        "iterations": iterations,
        "converged": converged,
        "training_races": len(races),
        "parameter_count": PARAMETER_COUNT,
    }


def rank_residual_metrics(
    races: list[dict[str, Any]], calibrator: dict[str, Any]
) -> dict[str, Any]:
    loss = 0.0
    market_loss = 0.0
    top5_hits = 0
    market_top5_hits = 0
    for race in races:
        probabilities = rank_residual_probabilities(
            race["model_probabilities"],
            race["market_probabilities"],
            coefficients=calibrator["coefficients"],
        )
        market = race["market_probabilities"]
        actual = str(race["actual_combination"])
        loss -= math.log(max(EPSILON, probabilities.get(actual, 0.0)))
        market_loss -= math.log(max(EPSILON, float(market.get(actual, 0.0))))
        top5_hits += int(
            actual
            in sorted(probabilities, key=probabilities.get, reverse=True)[:5]
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
        "market_trifecta_top5_hit_rate": (
            market_top5_hits / count if count else None
        ),
    }


def select_rank_regularization_prequential(
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
            training = [
                race for race_date in dates[:index] for race in by_day[race_date]
            ]
            holdout = by_day[dates[index]]
            calibrator = fit_rank_residual_newton(
                training,
                regularization=float(regularization),
            )
            metrics = rank_residual_metrics(holdout, calibrator)
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
    final_calibrator = fit_rank_residual_newton(
        races,
        regularization=float(selected["regularization"]),
    )
    return {
        "validation_design": (
            "Regularization is selected on forward-only full-day folds; market "
            "rank buckets and six parameters are fixed before evaluation"
        ),
        "dates": dates,
        "selected_regularization": selected["regularization"],
        "prequential_log_loss": selected["prequential_log_loss"],
        "final_calibrator": final_calibrator,
        "candidates": candidates,
    }


def _combination_features(
    combinations: list[str],
    *,
    model: dict[str, float],
    market: dict[str, float],
) -> np.ndarray:
    ranked = sorted(combinations, key=lambda key: (-float(market[key]), key))
    ranks = {combination: index + 1 for index, combination in enumerate(ranked)}
    features = np.zeros((len(combinations), PARAMETER_COUNT), dtype=np.float64)
    for row_index, combination in enumerate(combinations):
        log_market = math.log(max(EPSILON, float(market[combination])))
        disagreement = (
            math.log(max(EPSILON, float(model[combination]))) - log_market
        )
        rank = ranks[combination]
        features[row_index, 0] = log_market
        if rank <= 5:
            features[row_index, 1] = 1.0
            features[row_index, 3] = disagreement
        elif rank <= 20:
            features[row_index, 2] = 1.0
            features[row_index, 4] = disagreement
        else:
            features[row_index, 5] = disagreement
    return features


def _objective_gradient_hessian(
    races: list[dict[str, Any]],
    coefficients: np.ndarray,
    *,
    regularization: float,
    prior: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    objective = 0.0
    gradient = np.zeros(PARAMETER_COUNT, dtype=np.float64)
    hessian = np.zeros((PARAMETER_COUNT, PARAMETER_COUNT), dtype=np.float64)
    for race in races:
        combinations = sorted(
            set(race["model_probabilities"])
            & set(race["market_probabilities"])
        )
        actual = str(race["actual_combination"])
        if actual not in combinations:
            raise ValueError(f"actual combination {actual} is missing")
        features = _combination_features(
            combinations,
            model=race["model_probabilities"],
            market=race["market_probabilities"],
        )
        logits = features @ coefficients
        maximum = float(np.max(logits))
        exp_logits = np.exp(logits - maximum)
        probabilities = exp_logits / float(np.sum(exp_logits))
        actual_index = combinations.index(actual)
        objective += (
            maximum
            + math.log(float(np.sum(exp_logits)))
            - float(logits[actual_index])
        )
        mean = probabilities @ features
        gradient += mean - features[actual_index]
        second_moment = (features.T * probabilities) @ features
        hessian += second_moment - np.outer(mean, mean)

    count = len(races)
    objective /= count
    gradient /= count
    hessian /= count
    delta = coefficients - prior
    objective += 0.5 * regularization * float(delta @ delta)
    gradient += regularization * delta
    hessian += regularization * np.eye(PARAMETER_COUNT, dtype=np.float64)
    return objective, gradient, hessian
