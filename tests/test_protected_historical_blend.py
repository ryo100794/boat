from __future__ import annotations

import pytest

from boatrace_ai.protected_historical_blend import (
    blend_predictions,
    prediction_metrics,
    select_protected_blend,
)


def race(race_id: str, winner: int, probabilities: list[float]):
    return [
        {
            "race_id": race_id,
            "race_date": "2026-01-01",
            "jcd": "01",
            "rno": 1,
            "lane": lane,
            "rank": 1 if lane == winner else lane + 1,
            "probability": probability,
        }
        for lane, probability in enumerate(probabilities, start=1)
    ]


def test_blend_predictions_preserves_baseline_at_zero_weight() -> None:
    baseline = {"r1": race("r1", 1, [0.4, 0.2, 0.15, 0.1, 0.08, 0.07])}
    candidate = {"r1": race("r1", 1, [0.1, 0.2, 0.2, 0.2, 0.2, 0.1])}

    blended = blend_predictions(baseline, candidate, candidate_weight=0.0)

    assert [row["probability"] for row in blended["r1"]] == pytest.approx(
        [row["probability"] for row in baseline["r1"]]
    )


def test_selection_uses_improving_candidate_without_holdout_input() -> None:
    baseline = {
        "r1": race("r1", 1, [0.3, 0.2, 0.15, 0.13, 0.12, 0.1]),
        "r2": race("r2", 2, [0.3, 0.25, 0.15, 0.12, 0.1, 0.08]),
    }
    candidate = {
        "r1": race("r1", 1, [0.6, 0.12, 0.1, 0.07, 0.06, 0.05]),
        "r2": race("r2", 2, [0.12, 0.58, 0.1, 0.08, 0.07, 0.05]),
    }

    result = select_protected_blend(baseline, candidate, weights=(0.0, 0.5, 1.0))

    assert result["candidate_weight"] == 1.0
    assert result["selected_metrics"]["entry_log_loss"] < result["baseline_metrics"]["entry_log_loss"]
    assert result["selection_scope"] == "training-only calibration; holdout untouched"


def test_selection_falls_back_to_baseline_when_candidate_degrades_accuracy() -> None:
    baseline = {
        "r1": race("r1", 1, [0.6, 0.12, 0.1, 0.07, 0.06, 0.05]),
        "r2": race("r2", 2, [0.12, 0.58, 0.1, 0.08, 0.07, 0.05]),
    }
    candidate = {
        "r1": race("r1", 1, [0.1, 0.55, 0.1, 0.1, 0.08, 0.07]),
        "r2": race("r2", 2, [0.55, 0.1, 0.1, 0.1, 0.08, 0.07]),
    }

    result = select_protected_blend(baseline, candidate, weights=(0.0, 0.5, 1.0))

    assert result["candidate_weight"] == 0.0
    assert result["protected_candidate_count"] == 1
    assert prediction_metrics(
        blend_predictions(baseline, candidate, candidate_weight=result["candidate_weight"])
    ) == pytest.approx(result["baseline_metrics"])


def test_blend_rejects_mismatched_race_sets() -> None:
    with pytest.raises(ValueError, match="race sets differ"):
        blend_predictions(
            {"r1": race("r1", 1, [0.4, 0.2, 0.15, 0.1, 0.08, 0.07])},
            {"r2": race("r2", 1, [0.4, 0.2, 0.15, 0.1, 0.08, 0.07])},
            candidate_weight=0.5,
        )
