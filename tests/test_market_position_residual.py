import pytest

from boatrace_ai.listwise.market_position_residual import (
    first_place_marginals,
    fit_position_residual_newton,
    position_residual_metrics,
    position_residual_probabilities,
    position_residual_walk_forward,
)


def _race(day: str, actual: str, model_first: float, market_first: float):
    return {
        "race_date": day,
        "actual_combination": actual,
        "model_probabilities": {
            "1-2-3": model_first,
            "2-1-3": 1.0 - model_first,
        },
        "market_probabilities": {
            "1-2-3": market_first,
            "2-1-3": 1.0 - market_first,
        },
    }


def test_first_place_marginals_normalizes_probabilities() -> None:
    marginals = first_place_marginals(
        {"1-2-3": 0.3, "1-3-2": 0.2, "2-1-3": 0.5}
    )
    assert marginals == pytest.approx({"1": 0.5, "2": 0.5})


def test_position_residual_newton_learns_winner_signal() -> None:
    races = [
        _race("2026-07-20", "1-2-3", 0.85, 0.45),
        _race("2026-07-20", "2-1-3", 0.15, 0.55),
        _race("2026-07-20", "1-2-3", 0.80, 0.40),
        _race("2026-07-20", "2-1-3", 0.20, 0.60),
    ]
    calibrator = fit_position_residual_newton(races, regularization=0.01)
    probabilities = position_residual_probabilities(races[0], calibrator)
    metrics = position_residual_metrics(races, calibrator)

    assert calibrator["converged"] is True
    assert calibrator["winner_residual_coefficient"] > 0.0
    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert metrics["trifecta_log_loss"] < metrics["market_trifecta_log_loss"]


def test_position_residual_walk_forward_uses_prior_days_only() -> None:
    races = []
    for day in ("2026-07-20", "2026-07-21", "2026-07-22"):
        races.extend(
            [
                _race(day, "1-2-3", 0.8, 0.45),
                _race(day, "2-1-3", 0.2, 0.55),
            ]
        )
    report = position_residual_walk_forward(races, min_calibration_days=1)

    assert report["evaluation_races"] == 4
    assert [fold["evaluation_date"] for fold in report["folds"]] == [
        "2026-07-21",
        "2026-07-22",
    ]
    assert report["folds"][0]["training_dates"] == ["2026-07-20"]
    assert report["folds"][1]["training_dates"] == [
        "2026-07-20",
        "2026-07-21",
    ]
