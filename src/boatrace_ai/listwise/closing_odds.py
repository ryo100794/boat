from __future__ import annotations

import math
from typing import Any

import numpy as np


MIN_ODDS = 1.0
MAX_ODDS = 999.9
EXPECTED_ODDS_CONFIDENCE_Z = 1.96
MIN_EXPECTED_ODDS_MULTIPLIER = 0.5
MAX_EXPECTED_ODDS_MULTIPLIER = 2.0


def expected_odds_correction(
    targets: np.ndarray,
    predicted_log_odds: np.ndarray,
    race_offsets: list[tuple[int, int]],
) -> dict[str, Any]:
    if targets.shape != predicted_log_odds.shape or targets.ndim != 1:
        raise ValueError("closing-odds correction arrays must align")
    race_means = np.asarray(
        [
            float(np.mean(np.exp(targets[start:stop] - predicted_log_odds[start:stop])))
            for start, stop in race_offsets
            if stop > start
        ],
        dtype=np.float64,
    )
    if not len(race_means):
        raise ValueError("closing-odds correction requires race clusters")
    point = float(np.mean(race_means))
    standard_error = (
        float(np.std(race_means, ddof=1) / math.sqrt(len(race_means)))
        if len(race_means) > 1
        else 0.0
    )
    lower95 = point - EXPECTED_ODDS_CONFIDENCE_Z * standard_error
    multiplier = min(
        MAX_EXPECTED_ODDS_MULTIPLIER,
        max(MIN_EXPECTED_ODDS_MULTIPLIER, lower95),
    )
    return {
        "expected_odds_multiplier": float(multiplier),
        "expected_odds_multiplier_point": point,
        "expected_odds_multiplier_standard_error": standard_error,
        "expected_odds_multiplier_lower95": float(lower95),
        "expected_odds_race_clusters": int(len(race_means)),
        "expected_odds_method": (
            "race-cluster mean closing/median ratio, conservative 95% lower bound"
        ),
    }


def fit_closing_odds_model(
    races: list[dict[str, Any]], *, regularization: float = 0.001
) -> dict[str, Any]:
    if regularization < 0.0 or not math.isfinite(regularization):
        raise ValueError("regularization must be finite and non-negative")
    features = []
    targets = []
    race_offsets: list[tuple[int, int]] = []
    race_count = 0
    for race in races:
        current = race.get("odds") or {}
        closing = race.get("closing_odds") or {}
        combinations = sorted(set(current) & set(closing))
        if len(combinations) != 120:
            continue
        race_start = len(targets)
        for combination in combinations:
            current_odds = float(current[combination])
            closing_odds = float(closing[combination])
            if current_odds <= 0.0 or closing_odds <= 0.0:
                continue
            features.append((1.0, math.log(current_odds)))
            targets.append(math.log(closing_odds))
        if len(targets) > race_start:
            race_offsets.append((race_start, len(targets)))
            race_count += 1
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
    correction = expected_odds_correction(target, predicted, race_offsets)
    return {
        "intercept": float(coefficients[0]),
        "log_odds_coefficient": float(coefficients[1]),
        "regularization": float(regularization),
        "training_races": race_count,
        "training_tickets": len(targets),
        "training_mean_absolute_log_error": float(np.mean(np.abs(target - predicted))),
        "baseline_mean_absolute_log_error": float(np.mean(np.abs(target - baseline))),
        **correction,
    }


def forecast_closing_odds(
    odds: dict[str, float],
    model: dict[str, Any],
    *,
    expected_value: bool = False,
) -> dict[str, float]:
    intercept = float(model["intercept"])
    coefficient = float(model["log_odds_coefficient"])
    multiplier = (
        float(model.get("expected_odds_multiplier") or 1.0)
        if expected_value
        else 1.0
    )
    return {
        combination: min(
            MAX_ODDS,
            max(
                MIN_ODDS,
                multiplier
                * math.exp(intercept + coefficient * math.log(float(value))),
            ),
        )
        for combination, value in odds.items()
        if float(value) > 0.0
    }


def attach_forecast_closing_odds(
    races: list[dict[str, Any]],
    model: dict[str, Any],
    *,
    expected_value: bool = True,
) -> list[dict[str, Any]]:
    result = []
    for race in races:
        item = dict(race)
        item["estimated_final_odds"] = forecast_closing_odds(
            race["odds"], model, expected_value=expected_value
        )
        item["closing_odds_forecast_target"] = (
            "conservative_expected_odds" if expected_value else "median_log_odds"
        )
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
