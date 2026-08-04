from __future__ import annotations

import math
from typing import Any

import numpy as np

from .closing_odds import MAX_ODDS, MIN_ODDS


QUANTILE_LEVELS = (0.10, 0.50, 0.90)
QUANTILE_NAMES = ("q10", "q50", "q90")
TREND_FEATURE_NAMES = (
    "intercept",
    "log_current_odds",
    "log_current_odds_squared",
    "current_odds_rank",
    "recent_log_probability_slope",
    "long_log_probability_slope",
    "log_probability_slope_acceleration",
    "log_probability_slope_volatility",
    "log_path_points",
)
CONTEXT_TREND_FEATURE_NAMES = TREND_FEATURE_NAMES + (
    "race_number",
    "first_lane",
    "log_model_market_probability_ratio",
    "log_model_market_probability_ratio_squared",
    "log_model_market_ratio_x_log_current_odds",
    "log_model_market_ratio_x_current_odds_rank",
    "log_model_market_ratio_x_race_number",
)
CROSS_CONFORMAL_METHOD = "leave_one_training_day_out_cross_conformal"
SINGLE_DAY_FALLBACK_METHOD = "in_sample_residual_quantiles_single_training_day"
ADAPTIVE_CONFORMAL_METHOD = "online_adaptive_conformal_miscoverage_control"
INITIAL_ADAPTIVE_ALPHA = 0.20
TARGET_INTERVAL_COVERAGE = 0.80
MIN_ADAPTIVE_ALPHA = 0.02
MAX_ADAPTIVE_ALPHA = 0.40


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
        "calibration_method": "in_sample_residual_quantiles",
        "crossfit_days": 0,
        "crossfit_tickets": 0,
    }


def _path_trend_features(
    race: dict[str, Any], combination: str
) -> tuple[float, float, float, float]:
    values: list[tuple[float, float]] = []
    for point in race.get("odds_path") or []:
        probability = (point.get("market_probabilities") or {}).get(combination)
        if probability is None or float(probability) <= 0.0:
            continue
        values.append((
            float(point.get("minutes_before_decision") or 0.0),
            math.log(float(probability)),
        ))
    values.sort(reverse=True)
    if len(values) < 2:
        return 0.0, 0.0, 0.0, 0.0
    slopes = [
        (values[index][1] - values[index - 1][1])
        / max(1e-12, values[index - 1][0] - values[index][0])
        for index in range(1, len(values))
    ]
    recent = slopes[-1]
    long = (values[-1][1] - values[0][1]) / max(
        1e-12, values[0][0] - values[-1][0]
    )
    previous = (
        sum(slopes[:-1]) / len(slopes[:-1])
        if len(slopes) > 1
        else long
    )
    volatility = float(np.std(slopes)) if len(slopes) > 1 else 0.0
    return recent, long, recent - previous, volatility


def _trend_design_matrix(
    race: dict[str, Any],
    combinations: list[str],
    current: np.ndarray,
    *,
    context_features: bool = False,
) -> np.ndarray:
    log_odds = np.log(current)
    ranks = _average_ranks(current) / max(1, len(current) - 1)
    trends = np.asarray([
        _path_trend_features(race, combination)
        for combination in combinations
    ])
    path_points = math.log1p(float(
        race.get("odds_path_points") or len(race.get("odds_path") or [])
    ))
    base = np.column_stack((
        np.ones(len(combinations), dtype=np.float64),
        log_odds,
        log_odds * log_odds,
        ranks,
        np.clip(trends[:, 0] * 10.0, -3.0, 3.0),
        np.clip(trends[:, 1] * 20.0, -3.0, 3.0),
        np.clip(trends[:, 2] * 10.0, -3.0, 3.0),
        np.clip(trends[:, 3] * 10.0, 0.0, 3.0),
        np.full(len(combinations), path_points, dtype=np.float64),
    ))
    if not context_features:
        return base
    model_probabilities = race.get("model_probabilities") or {}
    market_probabilities = race.get("market_probabilities") or {}
    model_market_ratio = np.asarray([
        np.clip(
            math.log(
                max(1e-12, float(model_probabilities.get(key, 1e-12)))
                / max(1e-12, float(market_probabilities.get(key, 1e-12)))
            ),
            -4.0,
            4.0,
        )
        for key in combinations
    ])
    race_number = (
        float(race.get("rno") or race.get("race_no") or 6.5) - 6.5
    ) / 5.5
    first_lane = np.asarray([
        (int(key.split("-", 1)[0]) - 3.5) / 2.5
        for key in combinations
    ])
    return np.column_stack((
        base,
        np.full(len(combinations), race_number, dtype=np.float64),
        first_lane,
        model_market_ratio,
        model_market_ratio * model_market_ratio,
        model_market_ratio * log_odds,
        model_market_ratio * ranks,
        model_market_ratio * race_number,
    ))


def fit_closing_odds_trend_quantile_model(
    races: list[dict[str, Any]],
    *,
    regularization: float = 0.001,
    context_features: bool = False,
) -> dict[str, Any]:
    if regularization < 0.0 or not math.isfinite(regularization):
        raise ValueError("regularization must be finite and non-negative")
    matrices: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    race_count = 0
    for race in races:
        paired = _paired_race(race)
        if paired is None:
            continue
        combinations, current, closing = paired
        matrices.append(_trend_design_matrix(
            race,
            combinations,
            current,
            context_features=context_features,
        ))
        targets.append(np.log(closing))
        race_count += 1
    if not targets:
        raise ValueError("closing odds trends require complete paired snapshots")

    matrix = np.vstack(matrices)
    target = np.concatenate(targets)
    feature_mean = np.zeros(matrix.shape[1], dtype=np.float64)
    feature_scale = np.ones(matrix.shape[1], dtype=np.float64)
    feature_mean[1:] = matrix[:, 1:].mean(axis=0)
    feature_scale[1:] = matrix[:, 1:].std(axis=0)
    feature_scale[feature_scale < 1e-8] = 1.0
    standardized = (matrix - feature_mean) / feature_scale
    penalty = np.eye(matrix.shape[1], dtype=np.float64) * regularization
    penalty[0, 0] = 0.0
    gram = standardized.T @ standardized / len(target)
    rhs = standardized.T @ target / len(target)
    coefficients = np.linalg.solve(
        gram + penalty + 1e-10 * np.eye(matrix.shape[1]), rhs
    )
    location = standardized @ coefficients
    residuals = target - location
    residual_quantiles = np.quantile(residuals, QUANTILE_LEVELS)
    predicted_median = location + float(residual_quantiles[1])
    return {
        "model_type": (
            "ridge_log_location_odds_path_context_v3"
            if context_features
            else "ridge_log_location_odds_path_v2"
        ),
        "feature_names": list(
            CONTEXT_TREND_FEATURE_NAMES
            if context_features
            else TREND_FEATURE_NAMES
        ),
        "feature_mean": feature_mean.tolist(),
        "feature_scale": feature_scale.tolist(),
        "coefficients": coefficients.tolist(),
        "intercept": float(coefficients[0]),
        "log_odds_coefficient": float(coefficients[1] / feature_scale[1]),
        "residual_q10": float(residual_quantiles[0]),
        "residual_q50": float(residual_quantiles[1]),
        "residual_q90": float(residual_quantiles[2]),
        "regularization": float(regularization),
        "training_races": race_count,
        "training_tickets": len(target),
        "training_log_mae": float(np.mean(np.abs(target - predicted_median))),
        "training_interval_coverage": float(np.mean(
            (target >= location + residual_quantiles[0])
            & (target <= location + residual_quantiles[2])
        )),
        "calibration_method": "in_sample_residual_quantiles",
        "crossfit_days": 0,
        "crossfit_tickets": 0,
    }


def _location_values(
    race: dict[str, Any],
    combinations: list[str],
    current: np.ndarray,
    model: dict[str, Any],
) -> np.ndarray:
    model_type = model.get("model_type")
    if model_type not in {
        "ridge_log_location_odds_path_v2",
        "ridge_log_location_odds_path_context_v3",
    }:
        return (
            float(model["intercept"])
            + float(model["log_odds_coefficient"]) * np.log(current)
        )
    matrix = _trend_design_matrix(
        race,
        combinations,
        current,
        context_features=(
            model_type == "ridge_log_location_odds_path_context_v3"
        ),
    )
    mean = np.asarray(model["feature_mean"], dtype=np.float64)
    scale = np.asarray(model["feature_scale"], dtype=np.float64)
    coefficients = np.asarray(model["coefficients"], dtype=np.float64)
    if not (
        matrix.shape[1:] == mean.shape == scale.shape == coefficients.shape
    ):
        raise ValueError("closing odds trend model feature contract mismatch")
    return ((matrix - mean) / scale) @ coefficients


def _fit_daily_cross_conformal_model(
    by_day: dict[str, list[dict[str, Any]]],
    training_dates: list[str],
    *,
    regularization: float,
    alpha: float = INITIAL_ADAPTIVE_ALPHA,
    use_trend_features: bool = True,
    trend_context_features: bool = False,
) -> dict[str, Any]:
    """Fit location on all training days and calibrate on daily OOF residuals."""
    training = [race for day in training_dates for race in by_day[day]]
    fitter = (
        fit_closing_odds_trend_quantile_model
        if use_trend_features
        else fit_closing_odds_quantile_model
    )
    fit_kwargs = {"regularization": regularization}
    if use_trend_features:
        fit_kwargs["context_features"] = trend_context_features
    model = fitter(training, **fit_kwargs)
    if len(training_dates) == 1:
        residuals = _model_residuals(training, model)
        _set_residual_quantiles(model, residuals, alpha=alpha)
        model.update(
            calibration_method=SINGLE_DAY_FALLBACK_METHOD,
            crossfit_days=0,
            crossfit_tickets=0,
        )
        return model

    residuals: list[float] = []
    for held_out_date in training_dates:
        fit_races = [
            race
            for day in training_dates
            if day != held_out_date
            for race in by_day[day]
        ]
        fold_model = fitter(fit_races, **fit_kwargs)
        for race in by_day[held_out_date]:
            paired = _paired_race(race)
            if paired is None:
                continue
            combinations, current, closing = paired
            location = _location_values(
                race, combinations, current, fold_model
            )
            residuals.extend(np.log(closing) - location)
    _set_residual_quantiles(model, residuals, alpha=alpha)
    model.update(
        calibration_method=CROSS_CONFORMAL_METHOD,
        crossfit_days=len(training_dates),
        crossfit_tickets=len(residuals),
    )
    return model


def _model_residuals(
    races: list[dict[str, Any]], model: dict[str, Any]
) -> list[float]:
    residuals: list[float] = []
    for race in races:
        paired = _paired_race(race)
        if paired is None:
            continue
        combinations, current, closing = paired
        location = _location_values(race, combinations, current, model)
        residuals.extend(np.log(closing) - location)
    return residuals


def _set_residual_quantiles(
    model: dict[str, Any], residuals: list[float], *, alpha: float
) -> None:
    levels = (alpha / 2.0, 0.50, 1.0 - alpha / 2.0)
    residual_quantiles = np.quantile(
        np.asarray(residuals, dtype=np.float64), levels
    )
    model.update(
        residual_q10=float(residual_quantiles[0]),
        residual_q50=float(residual_quantiles[1]),
        residual_q90=float(residual_quantiles[2]),
        interval_alpha=float(alpha),
        interval_quantile_levels=[float(level) for level in levels],
    )


def forecast_closing_odds_quantiles(
    race: dict[str, Any], model: dict[str, Any]
) -> dict[str, dict[str, float]]:
    current = race.get("odds") or {}
    residuals = [float(model[f"residual_{name}"]) for name in QUANTILE_NAMES]
    result: dict[str, dict[str, float]] = {name: {} for name in QUANTILE_NAMES}
    combinations = sorted(current)
    odds = np.asarray([float(current[key]) for key in combinations])
    valid = np.isfinite(odds) & (odds > 0.0)
    valid_combinations = [
        combination for combination, keep in zip(combinations, valid) if keep
    ]
    valid_odds = odds[valid]
    locations = _location_values(race, valid_combinations, valid_odds, model)
    for combination, location in zip(valid_combinations, locations):
        values = [
            min(MAX_ODDS, max(MIN_ODDS, math.exp(float(location) + residual)))
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
    adaptive_rate: float = 0.5,
    include_policy_forecasts: bool = False,
    use_trend_features: bool = True,
    trend_context_features: bool = False,
) -> dict[str, Any]:
    if minimum_training_days < 1:
        raise ValueError("minimum_training_days must be positive")
    if not math.isfinite(adaptive_rate) or not 0.0 <= adaptive_rate <= 1.0:
        raise ValueError("adaptive_rate must be finite and between 0 and 1")
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
    policy_forecasts_by_race_id: dict[str, dict[str, Any]] = {}
    point_policy_forecasts_by_race_id: dict[str, dict[str, Any]] = {}
    alpha = INITIAL_ADAPTIVE_ALPHA
    for index in range(minimum_training_days, len(dates)):
        training = [race for day in dates[:index] for race in by_day[day]]
        holdout = by_day[dates[index]]
        model = _fit_daily_cross_conformal_model(
            by_day,
            dates[:index],
            regularization=regularization,
            alpha=alpha,
            use_trend_features=use_trend_features,
            trend_context_features=trend_context_features,
        )
        metrics = closing_odds_quantile_metrics(holdout, model)
        if include_policy_forecasts:
            for race in holdout:
                race_id = str(race.get("race_id") or "")
                forecast = forecast_closing_odds_quantiles(race, model)
                lower_odds = forecast["q10"]
                if not race_id or len(lower_odds) != 120:
                    continue
                policy_forecasts_by_race_id[race_id] = {
                    "estimated_final_odds": lower_odds,
                    "closing_odds_forecast_target": "adaptive_conformal_lower_bound",
                    "closing_odds_model_training_days": len(dates[:index]),
                    "closing_odds_model_training_races": len(training),
                    "closing_odds_model_trained_through_date": dates[index - 1],
                    "closing_odds_model_type": model["model_type"],
                    "closing_odds_interval_alpha": float(alpha),
                    "closing_odds_lower_quantile": float(alpha / 2.0),
                }
                point_odds = forecast["q50"]
                if len(point_odds) == 120:
                    point_policy_forecasts_by_race_id[race_id] = {
                        "estimated_final_odds": point_odds,
                        "closing_odds_forecast_target": "conditional_median",
                        "closing_odds_model_training_days": len(dates[:index]),
                        "closing_odds_model_training_races": len(training),
                        "closing_odds_model_trained_through_date": dates[index - 1],
                        "closing_odds_model_type": model["model_type"],
                    }
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
        alpha_before = alpha
        observed_coverage = metrics["closing_odds_interval_coverage"]
        if observed_coverage is not None:
            alpha = min(
                MAX_ADAPTIVE_ALPHA,
                max(
                    MIN_ADAPTIVE_ALPHA,
                    alpha_before
                    + adaptive_rate
                    * (float(observed_coverage) - TARGET_INTERVAL_COVERAGE),
                ),
            )
        folds.append(
            {
                "race_date": dates[index],
                "training_dates": dates[:index],
                "training_days": index,
                "training_max_date": dates[index - 1],
                "evaluation_date": dates[index],
                "training_races": len(training),
                "evaluation_races": fold_races,
                "calibration_method": model["calibration_method"],
                "crossfit_days": model["crossfit_days"],
                "crossfit_tickets": model["crossfit_tickets"],
                "alpha_before": alpha_before,
                "observed_coverage": observed_coverage,
                "alpha_after": alpha,
                "interval_quantile_levels": model["interval_quantile_levels"],
                "model_type": model["model_type"],
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
    calibration_methods = sorted(
        {str(fold["calibration_method"]) for fold in folds}
    )
    crossfit_folds = [
        fold
        for fold in folds
        if fold["calibration_method"] == CROSS_CONFORMAL_METHOD
    ]
    result = {
        "evaluation_method": "expanding_daily_walk_forward",
        "adaptive_conformal_method": ADAPTIVE_CONFORMAL_METHOD,
        "adaptive_conformal_target_coverage": TARGET_INTERVAL_COVERAGE,
        "target_coverage": TARGET_INTERVAL_COVERAGE,
        "adaptive_rate": adaptive_rate,
        "initial_alpha": INITIAL_ADAPTIVE_ALPHA,
        "alpha_bounds": [MIN_ADAPTIVE_ALPHA, MAX_ADAPTIVE_ALPHA],
        "calibration_method": (
            calibration_methods[0]
            if len(calibration_methods) == 1
            else "mixed_daily_cross_conformal_with_single_day_fallback"
            if calibration_methods
            else None
        ),
        "crossfit_days": sum(
            int(fold["crossfit_days"]) for fold in crossfit_folds
        ),
        "crossfit_tickets": sum(
            int(fold["crossfit_tickets"]) for fold in crossfit_folds
        ),
        "minimum_training_days": minimum_training_days,
        "closing_odds_model_type": (
            folds[-1]["model_type"] if folds else None
        ),
        "uses_odds_path_features": bool(use_trend_features),
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
    if include_policy_forecasts:
        result["policy_forecasts_by_race_id"] = policy_forecasts_by_race_id
        result["point_policy_forecasts_by_race_id"] = (
            point_policy_forecasts_by_race_id
        )
    return result
