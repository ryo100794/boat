from __future__ import annotations

import math
from typing import Any

import numpy as np

from .closing_odds import MAX_ODDS, MIN_ODDS


QUANTILE_LEVELS = (0.10, 0.50, 0.90)
QUANTILE_NAMES = ("q10", "q50", "q90")


def _paired_race(race: dict[str, Any]) -> tuple[list[str], np.ndarray, np.ndarray] | None:
    current = race.get("odds") or {}
    closing = race.get("closing_odds") or {}
    combinations = sorted(set(current) & set(closing))
    if len(combinations) != 120:
        return None
    current_values = np.asarray([float(current[key]) for key in combinations])
    closing_values = np.asarray([float(closing[key]) for key in combinations])
    if not (
        np.all(np.isfinite(current_values))
        and np.all(np.isfinite(closing_values))
        and np.all(current_values > 0.0)
        and np.all(closing_values > 0.0)
    ):
        return None
    return combinations, current_values, closing_values


def fit_closing_odds_quantile_model(
    races: list[dict[str, Any]], *, regularization: float = 0.001
) -> dict[str, Any]:
    """Fit a log-odds location model and calibrate residual quantiles.

    Residual quantiles are estimated only from the supplied training races. The caller
    must enforce the temporal boundary; ``walk_forward_closing_odds_quantiles`` does
    this at whole-day granularity.
    """
    if regularization < 0.0 or not math.isfinite(regularization):
        raise ValueError("regularization must be finite and non-negative")
    features: list[tuple[float, float]] = []
    targets: list[float] = []
    race_count = 0
    for race in races:
        paired = _paired_race(race)
        if paired is None:
            continue
        _, current, closing = paired
        features.extend((1.0, math.log(value)) for value in current)
        targets.extend(math.log(value) for value in closing)
        race_count += 1
    if not targets:
        raise ValueError("closing odds quantiles require complete paired snapshots")

    matrix = np.asarray(features, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    prior = np.asarray([0.0, 1.0], dtype=np.float64)
    gram = matrix.T @ matrix / len(target)
    rhs = matrix.T @ target / len(target)
    coefficients = np.linalg.solve(
        gram + regularization * np.eye(2, dtype=np.float64),
        rhs + regularization * prior,
    )
    location = matrix @ coefficients
    residuals = target - location
    residual_quantiles = np.quantile(residuals, QUANTILE_LEVELS)
    predicted_median = location + float(residual_quantiles[1])
    return {
        "model_type": "ridge_log_location_conformal_residual_quantiles",
        "intercept": float(coefficients[0]),
        "log_odds_coefficient": float(coefficients[1]),
        "residual_q10": float(residual_quantiles[0]),
        "residual_q50": float(residual_quantiles[1]),
        "residual_q90": float(residual_quantiles[2]),
        "regularization": float(regularization),
        "training_races": race_count,
        "training_tickets": len(target),
        "training_log_mae": float(np.mean(np.abs(target - predicted_median))),
        "training_interval_coverage": float(
            np.mean(
                (target >= location + residual_quantiles[0])
                & (target <= location + residual_quantiles[2])
            )
        ),
    }


def forecast_closing_odds_quantiles(
    race: dict[str, Any], model: dict[str, Any]
) -> dict[str, dict[str, float]]:
    current = race.get("odds") or {}
    intercept = float(model["intercept"])
    coefficient = float(model["log_odds_coefficient"])
    residuals = [float(model[f"residual_{name}"]) for name in QUANTILE_NAMES]
    result: dict[str, dict[str, float]] = {name: {} for name in QUANTILE_NAMES}
    for combination, raw_odds in current.items():
        odds = float(raw_odds)
        if odds <= 0.0 or not math.isfinite(odds):
            continue
        location = intercept + coefficient * math.log(odds)
        values = [
            min(MAX_ODDS, max(MIN_ODDS, math.exp(location + residual)))
            for residual in residuals
        ]
        # Clipping is monotone, but this also protects artifacts loaded from an
        # older or manually edited model file.
        values.sort()
        for name, value in zip(QUANTILE_NAMES, values):
            result[name][str(combination)] = value
    return result


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0
        start = stop
    return ranks


def _rank_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    left_rank = _average_ranks(left)
    right_rank = _average_ranks(right)
    if float(np.std(left_rank)) == 0.0 or float(np.std(right_rank)) == 0.0:
        return None
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def closing_odds_quantile_metrics(
    races: list[dict[str, Any]], model: dict[str, Any]
) -> dict[str, Any]:
    errors: list[float] = []
    baseline_errors: list[float] = []
    covered: list[bool] = []
    interval_widths: list[float] = []
    rank_correlations: list[float] = []
    snapshot_ages: list[float] = []
    race_count = 0
    for race in races:
        paired = _paired_race(race)
        if paired is None:
            continue
        combinations, current, closing = paired
        forecast = forecast_closing_odds_quantiles(race, model)
        if any(len(forecast[name]) != 120 for name in QUANTILE_NAMES):
            continue
        q10 = np.asarray([forecast["q10"][key] for key in combinations])
        q50 = np.asarray([forecast["q50"][key] for key in combinations])
        q90 = np.asarray([forecast["q90"][key] for key in combinations])
        target_log = np.log(closing)
        errors.extend(np.abs(target_log - np.log(q50)))
        baseline_errors.extend(np.abs(target_log - np.log(current)))
        covered.extend((closing >= q10) & (closing <= q90))
        interval_widths.extend(np.log(q90) - np.log(q10))
        correlation = _rank_correlation(q50, closing)
        if correlation is not None:
            rank_correlations.append(correlation)
        age = race.get("closing_snapshot_age_seconds")
        if age is None:
            age = race.get("snapshot_age_seconds")
        if age is not None:
            numeric_age = float(age)
            if math.isfinite(numeric_age) and numeric_age >= 0.0:
                snapshot_ages.append(numeric_age)
        race_count += 1
    return {
        "evaluation_races": race_count,
        "evaluation_tickets": len(errors),
        "closing_odds_log_mae": float(np.mean(errors)) if errors else None,
        "baseline_closing_odds_log_mae": (
            float(np.mean(baseline_errors)) if baseline_errors else None
        ),
        "closing_odds_rank_correlation": (
            float(np.mean(rank_correlations)) if rank_correlations else None
        ),
        "closing_odds_interval_coverage": (
            float(np.mean(covered)) if covered else None
        ),
        "closing_odds_interval_mean_log_width": (
            float(np.mean(interval_widths)) if interval_widths else None
        ),
        "closing_snapshot_age_seconds": (
            float(np.mean(snapshot_ages)) if snapshot_ages else None
        ),
        "closing_snapshot_age_seconds_median": (
            float(np.median(snapshot_ages)) if snapshot_ages else None
        ),
        "closing_snapshot_age_seconds_p90": (
            float(np.quantile(snapshot_ages, 0.90)) if snapshot_ages else None
        ),
        "closing_snapshot_age_p90_seconds": (
            float(np.quantile(snapshot_ages, 0.90)) if snapshot_ages else None
        ),
        "snapshot_age_races": len(snapshot_ages),
    }


def walk_forward_closing_odds_quantiles(
    races: list[dict[str, Any]],
    *,
    minimum_training_days: int = 1,
    regularization: float = 0.001,
) -> dict[str, Any]:
    if minimum_training_days < 1:
        raise ValueError("minimum_training_days must be positive")
    by_day: dict[str, list[dict[str, Any]]] = {}
    excluded_races = 0
    for race in races:
        race_date = race.get("race_date")
        if race_date is None or _paired_race(race) is None:
            excluded_races += 1
            continue
        by_day.setdefault(str(race_date), []).append(race)
    dates = sorted(by_day)
    folds: list[dict[str, Any]] = []
    evaluated_races: list[dict[str, Any]] = []
    weighted: dict[str, float] = {
        "error": 0.0,
        "baseline_error": 0.0,
        "coverage": 0.0,
        "width": 0.0,
        "rank": 0.0,
        "age": 0.0,
    }
    tickets = rank_races = age_races = 0
    for index in range(minimum_training_days, len(dates)):
        training = [race for day in dates[:index] for race in by_day[day]]
        holdout = by_day[dates[index]]
        model = fit_closing_odds_quantile_model(
            training, regularization=regularization
        )
        metrics = closing_odds_quantile_metrics(holdout, model)
        fold_tickets = int(metrics["evaluation_tickets"])
        fold_races = int(metrics["evaluation_races"])
        fold_age_races = int(metrics["snapshot_age_races"])
        weighted["error"] += float(metrics["closing_odds_log_mae"]) * fold_tickets
        weighted["baseline_error"] += (
            float(metrics["baseline_closing_odds_log_mae"]) * fold_tickets
        )
        weighted["coverage"] += (
            float(metrics["closing_odds_interval_coverage"]) * fold_tickets
        )
        weighted["width"] += (
            float(metrics["closing_odds_interval_mean_log_width"]) * fold_tickets
        )
        if metrics["closing_odds_rank_correlation"] is not None:
            weighted["rank"] += (
                float(metrics["closing_odds_rank_correlation"]) * fold_races
            )
            rank_races += fold_races
        if metrics["closing_snapshot_age_seconds"] is not None:
            weighted["age"] += (
                float(metrics["closing_snapshot_age_seconds"]) * fold_age_races
            )
            age_races += fold_age_races
        tickets += fold_tickets
        evaluated_races.extend(holdout)
        folds.append(
            {
                "race_date": dates[index],
                "training_dates": dates[:index],
                "training_days": index,
                "training_max_date": dates[index - 1],
                "evaluation_date": dates[index],
                "training_races": len(training),
                "evaluation_races": fold_races,
                "metrics": metrics,
            }
        )
    aggregate_metrics = {
        "evaluation_races": len(evaluated_races),
        "evaluation_tickets": tickets,
        "closing_odds_log_mae": weighted["error"] / tickets if tickets else None,
        "baseline_closing_odds_log_mae": (
            weighted["baseline_error"] / tickets if tickets else None
        ),
        "closing_odds_rank_correlation": (
            weighted["rank"] / rank_races if rank_races else None
        ),
        "closing_odds_interval_coverage": (
            weighted["coverage"] / tickets if tickets else None
        ),
        "closing_odds_interval_mean_log_width": (
            weighted["width"] / tickets if tickets else None
        ),
        "closing_snapshot_age_seconds": (
            weighted["age"] / age_races if age_races else None
        ),
        "snapshot_age_races": age_races,
    }
    return {
        "evaluation_method": "expanding_daily_walk_forward",
        "minimum_training_days": minimum_training_days,
        "eligible_days": len(dates),
        "evaluation_days": len(folds),
        "evaluation_races": len(evaluated_races),
        "evaluation_tickets": tickets,
        "excluded_races": excluded_races,
        "closing_odds_log_mae": weighted["error"] / tickets if tickets else None,
        "baseline_closing_odds_log_mae": (
            weighted["baseline_error"] / tickets if tickets else None
        ),
        "closing_odds_rank_correlation": (
            weighted["rank"] / rank_races if rank_races else None
        ),
        "closing_odds_interval_coverage": (
            weighted["coverage"] / tickets if tickets else None
        ),
        "closing_odds_interval_mean_log_width": (
            weighted["width"] / tickets if tickets else None
        ),
        "closing_snapshot_age_seconds": (
            weighted["age"] / age_races if age_races else None
        ),
        "snapshot_age_races": age_races,
        "metrics": aggregate_metrics,
        "folds": folds,
    }
