from __future__ import annotations

import math

import pytest

from boatrace_ai.listwise.contextual_market_residual_v24 import (
    FEATURE_DIMENSION,
    contextual_metrics,
    contextual_probabilities,
    fit_contextual_residual,
    prepare_race,
)


def _race(day: int, actual: str = "1-2-3") -> dict:
    return {
        "race_id": f"2026-01-{day:02d}-01-01",
        "race_date": f"2026-01-{day:02d}",
        "jcd": "01",
        "actual_combination": actual,
        "model_probabilities": {
            "1-2-3": 0.5,
            "2-1-3": 0.5,
        },
        "market_probabilities": {
            "1-2-3": 0.5,
            "2-1-3": 0.5,
        },
    }


def test_zero_contextual_residual_is_exact_market() -> None:
    race = _race(1)
    probabilities = contextual_probabilities(
        race,
        {"coefficients": [0.0] * FEATURE_DIMENSION},
    )
    assert probabilities == pytest.approx(race["market_probabilities"])
    assert sum(probabilities.values()) == pytest.approx(1.0)


def test_prepared_context_features_are_bounded() -> None:
    prepared = prepare_race(_race(1))
    assert prepared.indices.shape == (2, 12)
    assert prepared.values.shape == (2, 12)
    assert int(prepared.indices.min()) >= 0
    assert int(prepared.indices.max()) < FEATURE_DIMENSION


def test_contextual_residual_learns_first_lane_bias() -> None:
    training = [_race(day) for day in range(1, 13)]
    artifact = fit_contextual_residual(
        training,
        regularization=0.01,
        max_iterations=40,
    )
    metrics = contextual_metrics([_race(13)], artifact)
    assert artifact["feature_dimension"] == FEATURE_DIMENSION
    assert artifact["training_races"] == 12
    assert artifact["gradient_norm"] < 1e-4
    assert metrics["trifecta_log_loss"] < math.log(2.0)
    probabilities = contextual_probabilities(_race(13), artifact)
    assert probabilities["1-2-3"] > probabilities["2-1-3"]


def test_contextual_residual_rejects_invalid_combination() -> None:
    race = _race(1)
    race["model_probabilities"] = {"1-1-2": 1.0}
    race["market_probabilities"] = {"1-1-2": 1.0}
    race["actual_combination"] = "1-1-2"
    with pytest.raises(ValueError, match="invalid trifecta"):
        prepare_race(race)
