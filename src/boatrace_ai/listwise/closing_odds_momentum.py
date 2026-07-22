from __future__ import annotations

import math
from typing import Any

import numpy as np

from .closing_odds import MAX_ODDS, MIN_ODDS


def _momentum_features(
    race: dict[str, Any], combination: str
) -> tuple[float, float, float]:
    current_odds = float(race["odds"][combination])
    current_market = float(race["market_probabilities"][combination])
    earlier_market = float(race["earlier_market_probabilities"][combination])
    scale = float(race.get("momentum_scale") or 1.0)
    return (
        1.0,
        math.log(current_odds),
        scale * (math.log(current_market) - math.log(earlier_market)),
    )


def fit_momentum_closing_odds_model(
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
        current_market = race.get("market_probabilities") or {}
        earlier_market = race.get("earlier_market_probabilities") or {}
        combinations = sorted(
            set(current) & set(closing) & set(current_market) & set(earlier_market)
        )
        if len(combinations) != 120:
            continue
        race_count += 1
        for combination in combinations:
            if min(
                float(current[combination]),
                float(closing[combination]),
                float(current_market[combination]),
                float(earlier_market[combination]),
            ) <= 0.0:
                continue
            features.append(_momentum_features(race, combination))
            targets.append(math.log(float(closing[combination])))
    if not targets:
        raise ValueError("momentum closing-odds calibration requires complete snapshots")
    matrix = np.asarray(features, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    prior = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    gram = matrix.T @ matrix / len(targets)
    rhs = matrix.T @ target / len(targets)
    coefficients = np.linalg.solve(
        gram + regularization * np.eye(3, dtype=np.float64),
        rhs + regularization * prior,
    )
    predicted = matrix @ coefficients
    baseline = matrix @ prior
    return {
        "intercept": float(coefficients[0]),
        "log_odds_coefficient": float(coefficients[1]),
        "momentum_coefficient": float(coefficients[2]),
        "regularization": float(regularization),
        "training_races": race_count,
        "training_tickets": len(targets),
        "training_mean_absolute_log_error": float(
            np.mean(np.abs(target - predicted))
        ),
        "baseline_mean_absolute_log_error": float(
            np.mean(np.abs(target - baseline))
        ),
    }


def forecast_momentum_closing_odds(
    race: dict[str, Any], model: dict[str, Any]
) -> dict[str, float]:
    coefficients = np.asarray(
        [
            float(model["intercept"]),
            float(model["log_odds_coefficient"]),
            float(model["momentum_coefficient"]),
        ],
        dtype=np.float64,
    )
    combinations = sorted(
        set(race["odds"])
        & set(race["market_probabilities"])
        & set(race["earlier_market_probabilities"])
    )
    return {
        combination: min(
            MAX_ODDS,
            max(
                MIN_ODDS,
                math.exp(
                    float(
                        np.asarray(_momentum_features(race, combination))
                        @ coefficients
                    )
                ),
            ),
        )
        for combination in combinations
    }


def attach_momentum_closing_odds(
    races: list[dict[str, Any]], model: dict[str, Any]
) -> list[dict[str, Any]]:
    result = []
    for race in races:
        item = dict(race)
        item["estimated_final_odds"] = forecast_momentum_closing_odds(race, model)
        result.append(item)
    return result


def momentum_closing_odds_metrics(
    races: list[dict[str, Any]], model: dict[str, Any]
) -> dict[str, Any]:
    baseline_errors = []
    forecast_errors = []
    race_count = 0
    for race in races:
        closing = race.get("closing_odds") or {}
        forecast = forecast_momentum_closing_odds(race, model)
        combinations = sorted(set(closing) & set(forecast))
        if len(combinations) != 120:
            continue
        race_count += 1
        for combination in combinations:
            target = math.log(float(closing[combination]))
            baseline_errors.append(
                abs(target - math.log(float(race["odds"][combination])))
            )
            forecast_errors.append(
                abs(target - math.log(float(forecast[combination])))
            )
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
