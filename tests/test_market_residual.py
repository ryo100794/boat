import math

from boatrace_ai.listwise.market_residual import (
    fit_log_pool_newton,
    log_pool_probabilities,
    residual_probability_metrics,
    select_regularization_prequential,
)


def _race(race_date: str, actual: str, model_a: float, market_a: float) -> dict:
    return {
        "race_date": race_date,
        "actual_combination": actual,
        "model_probabilities": {"1-2-3": model_a, "1-3-2": 1.0 - model_a},
        "market_probabilities": {"1-2-3": market_a, "1-3-2": 1.0 - market_a},
    }


def test_log_pool_probabilities_are_normalized() -> None:
    probabilities = log_pool_probabilities(
        {"1-2-3": 0.7, "1-3-2": 0.3},
        {"1-2-3": 0.6, "1-3-2": 0.4},
        model_coefficient=0.5,
        market_coefficient=1.0,
    )

    assert set(probabilities) == {"1-2-3", "1-3-2"}
    assert math.isclose(sum(probabilities.values()), 1.0)


def test_newton_fit_uses_model_signal_and_converges() -> None:
    races = [
        _race("2026-07-20", "1-2-3", 0.9, 0.55),
        _race("2026-07-20", "1-3-2", 0.1, 0.45),
    ] * 10

    calibrator = fit_log_pool_newton(races, regularization=0.1)
    metrics = residual_probability_metrics(races, calibrator)

    assert calibrator["converged"] is True
    assert calibrator["model_coefficient"] > 0.0
    assert metrics["trifecta_log_loss"] < metrics["market_trifecta_log_loss"]


def test_regularization_selection_is_forward_only() -> None:
    races = [
        _race("2026-07-20", "1-2-3", 0.8, 0.55),
        _race("2026-07-20", "1-3-2", 0.2, 0.45),
        _race("2026-07-21", "1-2-3", 0.75, 0.55),
        _race("2026-07-21", "1-3-2", 0.25, 0.45),
    ]

    result = select_regularization_prequential(
        races, regularizations=(0.01, 1.0)
    )

    assert result["dates"] == ["2026-07-20", "2026-07-21"]
    assert result["selected_regularization"] in {0.01, 1.0}
    assert all(
        fold["training_dates"] == ["2026-07-20"]
        and fold["evaluation_date"] == "2026-07-21"
        for candidate in result["candidates"]
        for fold in candidate["folds"]
    )
