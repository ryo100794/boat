import math

from boatrace_ai.listwise.market_structure import (
    PARAMETER_COUNT,
    fit_structured_log_pool_newton,
    select_structured_regularization_prequential,
    structured_log_pool_probabilities,
    structured_probability_metrics,
)


def _race(race_date: str, actual: str) -> dict:
    combinations = {
        "1-2-3": 0.35,
        "1-3-2": 0.25,
        "2-1-3": 0.22,
        "2-3-1": 0.18,
    }
    return {
        "race_date": race_date,
        "actual_combination": actual,
        "model_probabilities": dict(combinations),
        "market_probabilities": dict(combinations),
    }


def test_structured_probabilities_are_normalized() -> None:
    race = _race("2026-07-20", "1-2-3")
    coefficients = [0.0, 1.0] + [0.0] * (PARAMETER_COUNT - 2)
    probabilities = structured_log_pool_probabilities(
        race["model_probabilities"],
        race["market_probabilities"],
        coefficients=coefficients,
    )

    assert set(probabilities) == set(race["market_probabilities"])
    assert math.isclose(sum(probabilities.values()), 1.0)
    assert all(
        math.isclose(probabilities[key], value)
        for key, value in race["market_probabilities"].items()
    )


def test_structured_newton_learns_position_lane_bias() -> None:
    races = [
        _race("2026-07-20", "2-1-3"),
        _race("2026-07-20", "2-3-1"),
    ] * 20

    calibrator = fit_structured_log_pool_newton(races, regularization=0.1)
    metrics = structured_probability_metrics(races, calibrator)

    assert calibrator["converged"] is True
    assert calibrator["parameter_count"] == PARAMETER_COUNT
    assert calibrator["position_lane_coefficients"]["position_1_lane_2"] > 0.0
    assert metrics["trifecta_log_loss"] < metrics["market_trifecta_log_loss"]


def test_structured_regularization_selection_is_forward_only() -> None:
    races = [
        _race("2026-07-20", "1-2-3"),
        _race("2026-07-20", "2-1-3"),
        _race("2026-07-21", "1-3-2"),
        _race("2026-07-21", "2-3-1"),
    ]

    result = select_structured_regularization_prequential(
        races,
        regularizations=(0.1, 10.0),
    )

    assert result["dates"] == ["2026-07-20", "2026-07-21"]
    assert result["selected_regularization"] in {0.1, 10.0}
    assert all(
        fold["training_dates"] == ["2026-07-20"]
        and fold["evaluation_date"] == "2026-07-21"
        for candidate in result["candidates"]
        for fold in candidate["folds"]
    )
