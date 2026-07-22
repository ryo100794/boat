from __future__ import annotations

import math
from collections import defaultdict
from itertools import combinations
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


EPSILON = 1e-12
DEFAULT_REGULARIZATION = (0.0, 0.001, 0.01, 0.1, 1.0, 10.0)


def align_scored_races(
    named_races: Mapping[str, Sequence[dict[str, Any]]],
) -> list[dict[str, Any]]:
    if not named_races:
        raise ValueError("at least one scored model is required")
    indexed = {
        name: {str(row["race_id"]): row for row in rows}
        for name, rows in named_races.items()
    }
    common_ids = set.intersection(*(set(rows) for rows in indexed.values()))
    if not common_ids:
        raise ValueError("scored models have no common races")
    first_name = next(iter(indexed))
    aligned = []
    for race_id in sorted(
        common_ids,
        key=lambda value: (
            str(indexed[first_name][value]["race_date"]),
            value,
        ),
    ):
        rows = {name: values[race_id] for name, values in indexed.items()}
        base = rows[first_name]
        actual = str(base["actual_combination"])
        race_date = str(base["race_date"])
        for name, row in rows.items():
            if str(row["actual_combination"]) != actual:
                raise ValueError(f"actual result differs for {race_id}: {name}")
            if str(row["race_date"]) != race_date:
                raise ValueError(f"race date differs for {race_id}: {name}")
        aligned.append(
            {
                **base,
                "source_probabilities": {
                    name: row["model_probabilities"] for name, row in rows.items()
                },
            }
        )
    return aligned


def log_pool_probabilities(
    race: dict[str, Any],
    *,
    source_names: Sequence[str],
    coefficients: Sequence[float],
) -> dict[str, float]:
    names = ("market", *source_names)
    values = {
        "market": race["market_probabilities"],
        **{
            name: race["source_probabilities"][name]
            for name in source_names
        },
    }
    combinations_in_all = sorted(
        set.intersection(*(set(values[name]) for name in names))
    )
    if not combinations_in_all:
        return {}
    feature_matrix = np.asarray(
        [
            [
                math.log(max(EPSILON, float(values[name][combination])))
                for name in names
            ]
            for combination in combinations_in_all
        ],
        dtype=np.float64,
    )
    logits = feature_matrix @ np.asarray(coefficients, dtype=np.float64)
    logits -= float(np.max(logits))
    probabilities = np.exp(logits)
    probabilities /= float(np.sum(probabilities))
    return {
        combination: float(probability)
        for combination, probability in zip(combinations_in_all, probabilities)
    }


def _race_features(
    race: dict[str, Any], source_names: Sequence[str]
) -> tuple[list[str], np.ndarray, int]:
    values = [
        race["market_probabilities"],
        *(race["source_probabilities"][name] for name in source_names),
    ]
    combinations_in_all = sorted(set.intersection(*(set(value) for value in values)))
    actual = str(race["actual_combination"])
    if actual not in combinations_in_all:
        raise ValueError(f"actual combination {actual} is missing")
    features = np.asarray(
        [
            [
                math.log(max(EPSILON, float(value[combination])))
                for value in values
            ]
            for combination in combinations_in_all
        ],
        dtype=np.float64,
    )
    return combinations_in_all, features, combinations_in_all.index(actual)


def _objective_gradient_hessian(
    races: Sequence[dict[str, Any]],
    source_names: Sequence[str],
    coefficients: np.ndarray,
    *,
    regularization: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    size = len(source_names) + 1
    objective = 0.0
    gradient = np.zeros(size, dtype=np.float64)
    hessian = np.zeros((size, size), dtype=np.float64)
    for race in races:
        _combinations, features, actual_index = _race_features(race, source_names)
        logits = features @ coefficients
        maximum = float(np.max(logits))
        exp_logits = np.exp(logits - maximum)
        probabilities = exp_logits / float(np.sum(exp_logits))
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
    prior = np.zeros(size, dtype=np.float64)
    prior[0] = 1.0
    delta = coefficients - prior
    objective += 0.5 * regularization * float(delta @ delta)
    gradient += regularization * delta
    hessian += regularization * np.eye(size, dtype=np.float64)
    return objective, gradient, hessian


def fit_log_pool_newton(
    races: Sequence[dict[str, Any]],
    *,
    source_names: Sequence[str],
    regularization: float,
    max_iterations: int = 50,
    tolerance: float = 1e-8,
) -> dict[str, Any]:
    if not races:
        raise ValueError("at least one race is required")
    names = tuple(dict.fromkeys(str(name) for name in source_names))
    if not names:
        raise ValueError("at least one source model is required")
    if regularization < 0.0 or not math.isfinite(regularization):
        raise ValueError("regularization must be finite and non-negative")
    coefficients = np.zeros(len(names) + 1, dtype=np.float64)
    coefficients[0] = 1.0
    lower_bounds = np.zeros_like(coefficients)
    lower_bounds[0] = 0.05
    converged = False
    objective = math.inf
    gradient = np.zeros_like(coefficients)
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        objective, gradient, hessian = _objective_gradient_hessian(
            races,
            names,
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
        next_objective = objective
        next_coefficients = coefficients
        while scale >= 1e-8:
            candidate = np.maximum(coefficients - scale * step, lower_bounds)
            candidate_objective, _, _ = _objective_gradient_hessian(
                races,
                names,
                candidate,
                regularization=regularization,
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
        races,
        names,
        coefficients,
        regularization=regularization,
    )
    coefficient_map = {
        name: float(value)
        for name, value in zip(("market", *names), coefficients)
    }
    coefficient_sum = float(np.sum(coefficients))
    return {
        "source_names": list(names),
        "coefficients": coefficient_map,
        "weights": {
            name: value / coefficient_sum
            for name, value in coefficient_map.items()
        },
        "temperature": 1.0 / coefficient_sum,
        "regularization": float(regularization),
        "objective": float(objective),
        "gradient_norm": float(np.linalg.norm(gradient)),
        "iterations": iterations,
        "converged": converged,
        "training_races": len(races),
    }


def probability_metrics(
    races: Sequence[dict[str, Any]],
    *,
    source_names: Sequence[str],
    calibrator: Mapping[str, Any],
) -> dict[str, Any]:
    coefficient_map = calibrator["coefficients"]
    coefficients = [
        float(coefficient_map[name]) for name in ("market", *source_names)
    ]
    loss = 0.0
    top5_hits = 0
    for race in races:
        probabilities = log_pool_probabilities(
            race,
            source_names=source_names,
            coefficients=coefficients,
        )
        actual = str(race["actual_combination"])
        loss -= math.log(max(EPSILON, probabilities.get(actual, 0.0)))
        top5 = sorted(probabilities, key=probabilities.get, reverse=True)[:5]
        top5_hits += int(actual in top5)
    count = len(races)
    return {
        "evaluated_races": count,
        "trifecta_log_loss": loss / count if count else None,
        "trifecta_top5_hit_rate": top5_hits / count if count else None,
    }


def select_regularization_prequential(
    races: Sequence[dict[str, Any]],
    *,
    source_names: Sequence[str],
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
        weighted_top5_hits = 0.0
        total = 0
        folds = []
        for index in range(1, len(dates)):
            training = [
                race for race_date in dates[:index] for race in by_day[race_date]
            ]
            holdout = by_day[dates[index]]
            calibrator = fit_log_pool_newton(
                training,
                source_names=source_names,
                regularization=float(regularization),
            )
            metrics = probability_metrics(
                holdout,
                source_names=source_names,
                calibrator=calibrator,
            )
            count = int(metrics["evaluated_races"])
            weighted_loss += float(metrics["trifecta_log_loss"]) * count
            weighted_top5_hits += float(metrics["trifecta_top5_hit_rate"]) * count
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
                "prequential_top5_hit_rate": weighted_top5_hits / total,
                "folds": folds,
            }
        )
    selected = min(
        candidates,
        key=lambda row: (
            row["prequential_log_loss"],
            -row["prequential_top5_hit_rate"],
            -row["regularization"],
        ),
    )
    return {
        "source_names": list(source_names),
        "dates": dates,
        "selected_regularization": selected["regularization"],
        "prequential_log_loss": selected["prequential_log_loss"],
        "prequential_top5_hit_rate": selected["prequential_top5_hit_rate"],
        "final_calibrator": fit_log_pool_newton(
            races,
            source_names=source_names,
            regularization=float(selected["regularization"]),
        ),
        "candidates": candidates,
    }


def select_source_subset_prequential(
    races: Sequence[dict[str, Any]],
    *,
    available_sources: Sequence[str],
) -> dict[str, Any]:
    names = tuple(dict.fromkeys(str(name) for name in available_sources))
    candidates = []
    for size in range(1, len(names) + 1):
        for subset in combinations(names, size):
            selection = select_regularization_prequential(
                races,
                source_names=subset,
            )
            candidates.append(selection)
    selected = min(
        candidates,
        key=lambda row: (
            row["prequential_log_loss"],
            -row["prequential_top5_hit_rate"],
            len(row["source_names"]),
            row["source_names"],
        ),
    )
    return {
        "validation_design": (
            "Source subset and regularization are selected only on forward daily folds"
        ),
        "selected": selected,
        "candidates": candidates,
    }
