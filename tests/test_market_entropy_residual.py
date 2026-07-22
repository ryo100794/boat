import pytest

from boatrace_ai.listwise.market_entropy_residual import (
    entropy_residual_metrics,
    entropy_residual_probabilities,
    entropy_residual_walk_forward,
    fit_entropy_residual_newton,
    normalized_entropy,
)


def _race(day: str, actual: str) -> dict:
    return {
        "race_date": day,
        "actual_combination": actual,
        "model_probabilities": {"1-2-3": 0.8, "2-1-3": 0.2},
        "market_probabilities": {"1-2-3": 0.45, "2-1-3": 0.55},
    }


def test_entropy_residual_has_exact_market_endpoint() -> None:
    race = _race("2026-07-20", "1-2-3")
    market = race["market_probabilities"]
    probabilities = entropy_residual_probabilities(
        race,
        {
            "market_coefficient": 1.0,
            "residual_coefficient": 0.0,
            "entropy_interaction_coefficient": 0.0,
            "entropy_mean": normalized_entropy(market),
            "entropy_scale": 1.0,
        },
    )

    assert probabilities == pytest.approx(market)


def test_entropy_residual_newton_learns_model_residual() -> None:
    races = [_race("2026-07-20", "1-2-3") for _ in range(8)]
    calibrator = fit_entropy_residual_newton(races, regularization=0.01)
    metrics = entropy_residual_metrics(races, calibrator)

    assert calibrator["converged"] is True
    assert calibrator["residual_coefficient"] > 0.0
    assert metrics["trifecta_log_loss"] < metrics["market_trifecta_log_loss"]


def test_entropy_residual_walk_forward_uses_prior_days_only() -> None:
    races = [
        _race(day, "1-2-3")
        for day in ("2026-07-20", "2026-07-21", "2026-07-22")
        for _ in range(3)
    ]
    report = entropy_residual_walk_forward(races)

    assert report["evaluation_races"] == 6
    assert [fold["evaluation_date"] for fold in report["folds"]] == [
        "2026-07-21",
        "2026-07-22",
    ]
    assert report["folds"][0]["training_dates"] == ["2026-07-20"]
    assert report["folds"][1]["training_dates"] == [
        "2026-07-20",
        "2026-07-21",
    ]
