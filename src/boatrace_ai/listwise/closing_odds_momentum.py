from __future__ import annotations

import math
from typing import Any

import numpy as np

from .closing_odds import (
    MAX_ODDS,
    MIN_ODDS,
    fit_closing_odds_model,
    forecast_closing_odds,
)


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



def momentum_price_eligible(race: dict[str, Any]) -> bool:
    return bool(
        len(race.get("odds") or {}) == 120
        and len(race.get("earlier_market_probabilities") or {}) == 120
        and len(race.get("market_probabilities") or {}) == 120
    )


def select_closing_odds_model(
    races: list[dict[str, Any]], *, minimum_relative_improvement: float = 0.01
) -> dict[str, Any]:
    if not 0.0 <= minimum_relative_improvement < 1.0:
        raise ValueError("minimum relative improvement must be in [0, 1)")
    baseline_model = fit_closing_odds_model(races)
    eligible = [
        race
        for race in races
        if momentum_price_eligible(race)
        and len(race.get("closing_odds") or {}) == 120
    ]
    by_day: dict[str, list[dict[str, Any]]] = {}
    for race in eligible:
        by_day.setdefault(str(race["race_date"]), []).append(race)
    dates = sorted(by_day)
    folds = []
    baseline_error_sum = momentum_error_sum = 0.0
    evaluation_tickets = 0
    for index in range(1, len(dates)):
        training = [race for day in dates[:index] for race in by_day[day]]
        holdout = by_day[dates[index]]
        fold_baseline = fit_closing_odds_model(training)
        fold_momentum = fit_momentum_closing_odds_model(training)
        baseline_metrics = _forecast_metrics(
            holdout, baseline_model=fold_baseline, momentum_model=None
        )
        momentum_metrics = _forecast_metrics(
            holdout, baseline_model=None, momentum_model=fold_momentum
        )
        tickets = int(momentum_metrics["evaluation_tickets"])
        baseline_error_sum += float(
            baseline_metrics["forecast_mean_absolute_log_error"]
        ) * tickets
        momentum_error_sum += float(
            momentum_metrics["forecast_mean_absolute_log_error"]
        ) * tickets
        evaluation_tickets += tickets
        folds.append(
            {
                "training_dates": dates[:index],
                "evaluation_date": dates[index],
                "training_races": len(training),
                "evaluation_races": len(holdout),
                "baseline_metrics": baseline_metrics,
                "momentum_metrics": momentum_metrics,
            }
        )
    baseline_mae = (
        baseline_error_sum / evaluation_tickets if evaluation_tickets else None
    )
    momentum_mae = (
        momentum_error_sum / evaluation_tickets if evaluation_tickets else None
    )
    relative_improvement = (
        1.0 - momentum_mae / baseline_mae
        if baseline_mae is not None and baseline_mae > 0.0 and momentum_mae is not None
        else None
    )
    use_momentum = bool(
        relative_improvement is not None
        and relative_improvement >= minimum_relative_improvement
    )
    momentum_model = (
        fit_momentum_closing_odds_model(eligible) if eligible else None
    )
    return {
        "selected": "momentum" if use_momentum else "baseline",
        "minimum_relative_improvement": minimum_relative_improvement,
        "eligible_momentum_races": len(eligible),
        "eligible_momentum_days": len(dates),
        "prequential_evaluation_races": sum(
            int(fold["evaluation_races"]) for fold in folds
        ),
        "prequential_evaluation_tickets": evaluation_tickets,
        "prequential_baseline_mae": baseline_mae,
        "prequential_momentum_mae": momentum_mae,
        "prequential_relative_improvement": relative_improvement,
        "baseline_model": baseline_model,
        "momentum_model": momentum_model,
        "folds": folds,
    }


def attach_selected_closing_odds(
    races: list[dict[str, Any]], selection: dict[str, Any]
) -> list[dict[str, Any]]:
    result = []
    use_momentum = selection.get("selected") == "momentum"
    momentum_model = selection.get("momentum_model")
    baseline_model = selection["baseline_model"]
    for race in races:
        item = dict(race)
        if use_momentum and momentum_model and momentum_price_eligible(race):
            forecast = forecast_momentum_closing_odds(race, momentum_model)
            source = "momentum"
        else:
            forecast = forecast_closing_odds(race["odds"], baseline_model)
            source = "baseline"
        item["estimated_final_odds"] = forecast
        item["closing_odds_forecast_source"] = source
        result.append(item)
    return result


def selected_closing_odds_metrics(
    races: list[dict[str, Any]], selection: dict[str, Any]
) -> dict[str, Any]:
    augmented = attach_selected_closing_odds(races, selection)
    baseline_errors = []
    forecast_errors = []
    sources: dict[str, int] = {}
    evaluated_races = 0
    for race in augmented:
        closing = race.get("closing_odds") or {}
        forecast = race.get("estimated_final_odds") or {}
        combinations = sorted(set(closing) & set(forecast))
        if len(combinations) != 120:
            continue
        evaluated_races += 1
        source = str(race.get("closing_odds_forecast_source") or "baseline")
        sources[source] = sources.get(source, 0) + 1
        for combination in combinations:
            target = math.log(float(closing[combination]))
            baseline_errors.append(
                abs(target - math.log(float(race["odds"][combination])))
            )
            forecast_errors.append(
                abs(target - math.log(float(forecast[combination])))
            )
    return {
        "evaluation_races": evaluated_races,
        "evaluation_tickets": len(forecast_errors),
        "forecast_sources": sources,
        "baseline_mean_absolute_log_error": (
            sum(baseline_errors) / len(baseline_errors) if baseline_errors else None
        ),
        "forecast_mean_absolute_log_error": (
            sum(forecast_errors) / len(forecast_errors) if forecast_errors else None
        ),
    }


def _forecast_metrics(
    races: list[dict[str, Any]],
    *,
    baseline_model: dict[str, Any] | None,
    momentum_model: dict[str, Any] | None,
) -> dict[str, Any]:
    errors = []
    evaluated_races = 0
    for race in races:
        closing = race.get("closing_odds") or {}
        forecast = (
            forecast_momentum_closing_odds(race, momentum_model)
            if momentum_model is not None
            else forecast_closing_odds(race["odds"], baseline_model or {})
        )
        combinations = sorted(set(closing) & set(forecast))
        if len(combinations) != 120:
            continue
        evaluated_races += 1
        errors.extend(
            abs(
                math.log(float(closing[combination]))
                - math.log(float(forecast[combination]))
            )
            for combination in combinations
        )
    return {
        "evaluation_races": evaluated_races,
        "evaluation_tickets": len(errors),
        "forecast_mean_absolute_log_error": (
            sum(errors) / len(errors) if errors else None
        ),
    }
