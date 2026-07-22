from __future__ import annotations

import math
from typing import Any

import numpy as np


MIN_ODDS = 1.0
MAX_ODDS = 999.9


def fit_closing_odds_model(
    races: list[dict[str, Any]], *, regularization: float = 0.001
) -> dict[str, Any]:
    if regularization < 0.0 or not math.isfinite(regularization):
        raise ValueError("regularization must be finite and non-negative")
    features = []
    targets = []
    race_count = 0
    for race in races:
        current = race.get("odds") or {}
        closing = race.get("closing_odds") or {}
        combinations = sorted(set(current) & set(closing))
        if len(combinations) != 120:
            continue
        race_count += 1
        for combination in combinations:
            current_odds = float(current[combination])
            closing_odds = float(closing[combination])
            if current_odds <= 0.0 or closing_odds <= 0.0:
                continue
            features.append((1.0, math.log(current_odds)))
            targets.append(math.log(closing_odds))
    if not targets:
        raise ValueError("closing odds calibration requires complete paired snapshots")
    matrix = np.asarray(features, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    prior = np.asarray([0.0, 1.0], dtype=np.float64)
    gram = matrix.T @ matrix / len(target)
    rhs = matrix.T @ target / len(target)
    coefficients = np.linalg.solve(
        gram + regularization * np.eye(2, dtype=np.float64),
        rhs + regularization * prior,
    )
    predicted = matrix @ coefficients
    baseline = matrix @ prior
    return {
        "intercept": float(coefficients[0]),
        "log_odds_coefficient": float(coefficients[1]),
        "regularization": float(regularization),
        "training_races": race_count,
        "training_tickets": len(targets),
        "training_mean_absolute_log_error": float(np.mean(np.abs(target - predicted))),
        "baseline_mean_absolute_log_error": float(np.mean(np.abs(target - baseline))),
    }


def forecast_closing_odds(
    odds: dict[str, float], model: dict[str, Any]
) -> dict[str, float]:
    intercept = float(model["intercept"])
    coefficient = float(model["log_odds_coefficient"])
    return {
        combination: min(
            MAX_ODDS,
            max(MIN_ODDS, math.exp(intercept + coefficient * math.log(float(value)))),
        )
        for combination, value in odds.items()
        if float(value) > 0.0
    }


def attach_forecast_closing_odds(
    races: list[dict[str, Any]], model: dict[str, Any]
) -> list[dict[str, Any]]:
    result = []
    for race in races:
        item = dict(race)
        item["estimated_final_odds"] = forecast_closing_odds(race["odds"], model)
        result.append(item)
    return result


def decision_odds(race: dict[str, Any]) -> dict[str, float]:
    forecast = race.get("estimated_final_odds")
    return forecast if isinstance(forecast, dict) and forecast else race["odds"]


def closing_odds_metrics(
    races: list[dict[str, Any]], model: dict[str, Any]
) -> dict[str, Any]:
    baseline_errors = []
    forecast_errors = []
    race_count = 0
    for race in races:
        current = race.get("odds") or {}
        closing = race.get("closing_odds") or {}
        combinations = sorted(set(current) & set(closing))
        if len(combinations) != 120:
            continue
        race_count += 1
        forecast = forecast_closing_odds(current, model)
        for combination in combinations:
            target = math.log(float(closing[combination]))
            baseline_errors.append(abs(target - math.log(float(current[combination]))))
            forecast_errors.append(abs(target - math.log(float(forecast[combination]))))
    return {
        "evaluation_races": race_count,
        "evaluation_tickets": len(forecast_errors),
        "baseline_mean_absolute_log_error": (
            sum(baseline_errors) / len(baseline_errors) if baseline_errors else None
        ),
        "forecast_mean_absolute_log_error": (
            sum(forecast_errors) / len(forecast_errors) if forecast_errors else None
        ),
    }
