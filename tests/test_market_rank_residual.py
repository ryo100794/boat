from __future__ import annotations

import pytest

from boatrace_ai.listwise.market_rank_residual import (
    PARAMETER_COUNT,
    fit_rank_residual_newton,
    rank_residual_metrics,
    rank_residual_probabilities,
    select_rank_regularization_prequential,
)


def probability_row(actual: str, race_date: str) -> dict:
    combinations = [f"1-2-{lane}" for lane in range(3, 123)]
    market_weights = {
        combination: float(len(combinations) - index)
        for index, combination in enumerate(combinations)
    }
    model_weights = {
        combination: float(index + 1)
        for index, combination in enumerate(combinations)
    }
    market_total = sum(market_weights.values())
    model_total = sum(model_weights.values())
    return {
        "race_date": race_date,
        "actual_combination": actual,
        "market_probabilities": {
            key: value / market_total for key, value in market_weights.items()
        },
        "model_probabilities": {
            key: value / model_total for key, value in model_weights.items()
        },
    }


def test_rank_residual_probabilities_are_normalized() -> None:
    row = probability_row("1-2-3", "2026-07-20")
    values = rank_residual_probabilities(
        row["model_probabilities"],
        row["market_probabilities"],
        coefficients=(1.0, 0.1, -0.1, 0.2, 0.0, -0.2),
    )
    assert len(values) == 120
    assert sum(values.values()) == pytest.approx(1.0)
    assert all(value > 0.0 for value in values.values())


def test_rank_residual_requires_six_coefficients() -> None:
    row = probability_row("1-2-3", "2026-07-20")
    with pytest.raises(ValueError, match=str(PARAMETER_COUNT)):
        rank_residual_probabilities(
            row["model_probabilities"],
            row["market_probabilities"],
            coefficients=(1.0, 0.0),
        )


def test_rank_residual_newton_converges_and_reports_metrics() -> None:
    rows = [
        probability_row("1-2-3", "2026-07-20"),
        probability_row("1-2-4", "2026-07-20"),
        probability_row("1-2-5", "2026-07-20"),
    ]
    fitted = fit_rank_residual_newton(rows, regularization=1.0)
    metrics = rank_residual_metrics(rows, fitted)
    assert fitted["parameter_count"] == 6
    assert fitted["converged"] is True
    assert fitted["gradient_norm"] < 1e-6
    assert metrics["evaluated_races"] == 3
    assert metrics["trifecta_log_loss"] is not None


def test_regularization_selection_uses_forward_only_days() -> None:
    rows = [
        probability_row("1-2-3", "2026-07-20"),
        probability_row("1-2-4", "2026-07-21"),
        probability_row("1-2-5", "2026-07-22"),
    ]
    selected = select_rank_regularization_prequential(
        rows,
        regularizations=(1.0,),
    )
    folds = selected["candidates"][0]["folds"]
    assert [fold["evaluation_date"] for fold in folds] == [
        "2026-07-21",
        "2026-07-22",
    ]
    assert folds[0]["training_dates"] == ["2026-07-20"]
    assert folds[1]["training_dates"] == ["2026-07-20", "2026-07-21"]
