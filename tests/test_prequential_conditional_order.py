from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from boatrace_ai.listwise.prequential_conditional_order import (
    apply_prequential_conditional_order,
)
from boatrace_ai.listwise.stagewise_mlp import COMBINATION_LANES


def _probabilities() -> dict[str, float]:
    return {
        "-".join(str(int(lane) + 1) for lane in combination): (
            1.0 / len(COMBINATION_LANES)
        )
        for combination in COMBINATION_LANES
    }


def _races() -> list[dict]:
    result = []
    actuals = ("1-2-3", "2-1-3")
    for day in range(1, 6):
        for race_number in range(1, 3):
            result.append({
                "race_id": f"2026-01-0{day}-01-{race_number:02d}",
                "race_date": f"2026-01-0{day}",
                "jcd": "01",
                "rno": race_number,
                "actual_combination": actuals[race_number - 1],
                "model_probabilities": _probabilities(),
            })
    return result


@dataclass(frozen=True)
class _Model:
    training_races: int


def test_transform_uses_only_prior_dates_and_preserves_warmup(
    monkeypatch,
) -> None:
    fit_sizes = []

    def fake_fit(scores, orders, *, regularization, max_iterations=100):
        assert len(scores) == len(orders)
        fit_sizes.append(len(scores))
        return _Model(len(scores)), {
            "success": True,
            "status": 0,
            "message": "ok",
            "iterations": 1,
            "function_evaluations": 1,
            "objective": 0.0,
            "gradient_norm": 0.0,
            "elapsed_seconds": 0.0,
        }

    def fake_probabilities(scores, model):
        output = np.full(
            (len(scores), len(COMBINATION_LANES)),
            0.9 / (len(COMBINATION_LANES) - 1),
        )
        output[:, 0] = 0.1
        return output

    monkeypatch.setattr(
        "boatrace_ai.listwise.prequential_conditional_order."
        "fit_conditional_order",
        fake_fit,
    )
    monkeypatch.setattr(
        "boatrace_ai.listwise.prequential_conditional_order."
        "conditional_probabilities",
        fake_probabilities,
    )
    source = _races()
    transformed, report = apply_prequential_conditional_order(source)

    assert fit_sizes == [6, 6, 6, 6, 8]
    assert report["transformed_days"] == 1
    assert report["transformed_races"] == 2
    assert report["daily"][-1]["fit_through"] == "2026-01-03"
    assert report["daily"][-1]["validation_date"] == "2026-01-04"
    assert report["daily"][-1]["fit_races"] == 6
    assert report["daily"][-1]["validation_races"] == 2
    assert all(
        "model_probability_transform" not in race
        for race in transformed[:8]
    )
    assert all(
        race["model_probability_transform"]
        == "strict_prior_conditional_order"
        for race in transformed[8:]
    )
    assert source[8]["model_probabilities"]["1-2-3"] == pytest.approx(
        1.0 / 120.0
    )
    assert transformed[8]["model_probabilities"]["1-2-3"] > 1.0 / 120.0
    assert sum(transformed[8]["model_probabilities"].values()) == pytest.approx(
        1.0
    )


def test_transform_rejects_incomplete_probability_space() -> None:
    races = _races()
    races[0]["model_probabilities"].pop("1-2-3")

    with pytest.raises(ValueError, match="120 model probabilities"):
        apply_prequential_conditional_order(races)


def test_transform_validates_selection_grid() -> None:
    with pytest.raises(ValueError, match="regularizations"):
        apply_prequential_conditional_order(
            _races(), regularizations=()
        )
    with pytest.raises(ValueError, match="blends"):
        apply_prequential_conditional_order(
            _races(), blends=(0.0,)
        )
