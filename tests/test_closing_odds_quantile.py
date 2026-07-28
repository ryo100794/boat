import copy
import math
from itertools import permutations

import pytest

from boatrace_ai.listwise.closing_odds_quantile import (
    closing_odds_quantile_metrics,
    fit_closing_odds_quantile_model,
    forecast_closing_odds_quantiles,
    walk_forward_closing_odds_quantiles,
)


COMBINATIONS = tuple(
    "-".join(map(str, values)) for values in permutations(range(1, 7), 3)
)


def _race(
    race_date: str,
    race_id: str,
    *,
    log_residual: float = 0.0,
    snapshot_age_seconds: float = 15.0,
) -> dict:
    odds = {
        combination: 8.0 + index
        for index, combination in enumerate(COMBINATIONS)
    }
    return {
        "race_date": race_date,
        "race_id": race_id,
        "odds": odds,
        "closing_odds": {
            combination: value * math.exp(log_residual)
            for combination, value in odds.items()
        },
        "closing_snapshot_age_seconds": snapshot_age_seconds,
    }


def _training_races() -> list[dict]:
    return [
        _race("2026-07-20", f"train-{index}", log_residual=residual)
        for index, residual in enumerate((-0.40, -0.20, 0.0, 0.20, 0.40))
    ]


def test_fit_requires_a_complete_120_combination_pair() -> None:
    incomplete = _race("2026-07-20", "incomplete")
    incomplete["closing_odds"].pop(COMBINATIONS[-1])

    with pytest.raises(ValueError, match="complete"):
        fit_closing_odds_quantile_model([incomplete])


def test_forecast_returns_complete_ordered_quantiles() -> None:
    model = fit_closing_odds_quantile_model(
        _training_races(), regularization=0.001
    )
    forecast = forecast_closing_odds_quantiles(
        _race("2026-07-21", "holdout"), model
    )

    assert set(forecast) == {"q10", "q50", "q90"}
    assert all(set(forecast[name]) == set(COMBINATIONS) for name in forecast)
    assert all(
        forecast["q10"][combination]
        <= forecast["q50"][combination]
        <= forecast["q90"][combination]
        for combination in COMBINATIONS
    )


def test_known_residual_is_covered_by_the_fitted_interval() -> None:
    model = fit_closing_odds_quantile_model(_training_races())
    holdout = _race("2026-07-21", "covered", log_residual=0.0)
    forecast = forecast_closing_odds_quantiles(holdout, model)

    assert all(
        forecast["q10"][combination]
        <= holdout["closing_odds"][combination]
        <= forecast["q90"][combination]
        for combination in COMBINATIONS
    )

    metrics = closing_odds_quantile_metrics([holdout], model)
    assert metrics["evaluation_races"] == 1
    assert metrics["evaluation_tickets"] == 120
    assert metrics["closing_odds_interval_coverage"] == pytest.approx(1.0)


def test_metrics_include_within_race_rank_correlation() -> None:
    model = fit_closing_odds_quantile_model(_training_races())
    holdout = _race("2026-07-21", "ranked", log_residual=0.05)

    metrics = closing_odds_quantile_metrics([holdout], model)

    assert metrics["closing_odds_log_mae"] >= 0.0
    assert metrics["closing_odds_rank_correlation"] == pytest.approx(1.0)


def test_metrics_aggregate_snapshot_age_for_eligible_races_only() -> None:
    model = fit_closing_odds_quantile_model(_training_races())
    recent = _race(
        "2026-07-21", "recent", snapshot_age_seconds=12.0
    )
    stale = _race(
        "2026-07-21", "stale", snapshot_age_seconds=48.0
    )
    incomplete = _race(
        "2026-07-21", "excluded", snapshot_age_seconds=999.0
    )
    incomplete["closing_odds"].pop(COMBINATIONS[0])

    metrics = closing_odds_quantile_metrics([recent, stale, incomplete], model)

    assert metrics["evaluation_races"] == 2
    assert metrics["closing_snapshot_age_seconds"] == pytest.approx(30.0)
    assert metrics["closing_snapshot_age_p90_seconds"] == pytest.approx(44.4)


def test_daily_walk_forward_does_not_learn_from_future_targets() -> None:
    races = [
        _race("2026-07-20", "day-1-a", log_residual=-0.10),
        _race("2026-07-20", "day-1-b", log_residual=0.10),
        _race("2026-07-21", "day-2", log_residual=0.05),
        _race("2026-07-22", "day-3", log_residual=-0.05),
    ]
    changed_future = copy.deepcopy(races)
    changed_future[-1]["closing_odds"] = {
        combination: value * 20.0
        for combination, value in changed_future[-1]["closing_odds"].items()
    }

    baseline = walk_forward_closing_odds_quantiles(
        races, minimum_training_days=1
    )
    mutated = walk_forward_closing_odds_quantiles(
        changed_future, minimum_training_days=1
    )
    baseline_fold = next(
        fold for fold in baseline["folds"] if fold["evaluation_date"] == "2026-07-21"
    )
    mutated_fold = next(
        fold for fold in mutated["folds"] if fold["evaluation_date"] == "2026-07-21"
    )

    assert baseline_fold["training_dates"] == ["2026-07-20"]
    assert baseline_fold == mutated_fold
    assert baseline["evaluation_races"] == 2
    assert baseline["evaluation_tickets"] == 240


def test_policy_forecasts_are_strictly_prior_lower_bounds() -> None:
    races = [
        _race("2026-07-20", "day-1", log_residual=-0.20),
        _race("2026-07-21", "day-2", log_residual=0.10),
        _race("2026-07-22", "day-3", log_residual=0.30),
    ]
    changed_future = copy.deepcopy(races)
    changed_future[-1]["closing_odds"] = {
        combination: value * 20.0
        for combination, value in changed_future[-1]["closing_odds"].items()
    }

    baseline = walk_forward_closing_odds_quantiles(
        races, minimum_training_days=1, include_policy_forecasts=True
    )
    mutated = walk_forward_closing_odds_quantiles(
        changed_future,
        minimum_training_days=1,
        include_policy_forecasts=True,
    )
    baseline_forecast = baseline["policy_forecasts_by_race_id"]["day-2"]
    mutated_forecast = mutated["policy_forecasts_by_race_id"]["day-2"]

    assert baseline_forecast == mutated_forecast
    assert baseline_forecast["closing_odds_model_trained_through_date"] < (
        "2026-07-21"
    )
    assert baseline_forecast["closing_odds_lower_quantile"] == pytest.approx(0.10)
    assert len(baseline_forecast["estimated_final_odds"]) == 120


def test_walk_forward_uses_daily_cross_conformal_residuals() -> None:
    races = [
        _race("2026-07-20", "day-1", log_residual=-0.40),
        _race("2026-07-21", "day-2", log_residual=0.40),
        _race("2026-07-22", "day-3", log_residual=0.70),
    ]

    result = walk_forward_closing_odds_quantiles(
        races, minimum_training_days=1, regularization=0.0
    )
    fallback_fold, crossfit_fold = result["folds"]

    assert fallback_fold["calibration_method"] == (
        "in_sample_residual_quantiles_single_training_day"
    )
    assert fallback_fold["crossfit_days"] == 0
    assert fallback_fold["crossfit_tickets"] == 0
    assert crossfit_fold["calibration_method"] == (
        "leave_one_training_day_out_cross_conformal"
    )
    assert crossfit_fold["crossfit_days"] == 2
    assert crossfit_fold["crossfit_tickets"] == 240
    assert crossfit_fold["metrics"]["closing_odds_interval_coverage"] == pytest.approx(
        1.0
    )
    assert result["calibration_method"] == (
        "mixed_daily_cross_conformal_with_single_day_fallback"
    )
    assert result["crossfit_days"] == 2
    assert result["crossfit_tickets"] == 240


def test_cross_conformal_fold_does_not_learn_from_future_day() -> None:
    races = [
        _race("2026-07-20", "day-1", log_residual=-0.30),
        _race("2026-07-21", "day-2", log_residual=0.30),
        _race("2026-07-22", "day-3", log_residual=0.20),
        _race("2026-07-23", "day-4", log_residual=-0.10),
    ]
    changed_future = copy.deepcopy(races)
    changed_future[-1]["closing_odds"] = {
        combination: value * 50.0
        for combination, value in changed_future[-1]["closing_odds"].items()
    }

    baseline = walk_forward_closing_odds_quantiles(
        races, minimum_training_days=1, regularization=0.0
    )
    mutated = walk_forward_closing_odds_quantiles(
        changed_future, minimum_training_days=1, regularization=0.0
    )
    baseline_fold = next(
        fold for fold in baseline["folds"] if fold["evaluation_date"] == "2026-07-22"
    )
    mutated_fold = next(
        fold for fold in mutated["folds"] if fold["evaluation_date"] == "2026-07-22"
    )

    assert baseline_fold["calibration_method"] == (
        "leave_one_training_day_out_cross_conformal"
    )
    assert baseline_fold == mutated_fold


def test_public_fit_api_reports_legacy_calibration_metadata() -> None:
    model = fit_closing_odds_quantile_model(_training_races())

    assert model["calibration_method"] == "in_sample_residual_quantiles"
    assert model["crossfit_days"] == 0
    assert model["crossfit_tickets"] == 0
