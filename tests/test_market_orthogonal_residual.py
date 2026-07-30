import pytest

import boatrace_ai.listwise.market_orthogonal_residual as orthogonal_residual
from boatrace_ai.listwise.market_calibration import blend_probabilities
from boatrace_ai.listwise.market_orthogonal_residual import (
    fit_market_projection,
    fit_orthogonal_residual,
    orthogonal_probabilities,
    select_regularization_prequential,
)


def _race(day: str, actual: str, model: tuple[float, float]) -> dict:
    return {
        "race_date": day,
        "actual_combination": actual,
        "model_probabilities": {"1-2-3": model[0], "2-1-3": model[1]},
        "market_probabilities": {"1-2-3": 0.6, "2-1-3": 0.4},
    }


def test_projection_removes_market_collinearity() -> None:
    races = [_race("2026-07-01", "1-2-3", (0.7, 0.3))]
    projection = fit_market_projection(races)

    assert projection["projection_beta"] > 0.0
    assert projection["model_market_correlation"] == pytest.approx(1.0)
    assert projection["residual_variance_fraction"] == pytest.approx(0.0, abs=1e-12)


def test_zero_residual_coefficient_reproduces_market() -> None:
    probabilities = orthogonal_probabilities(
        {"1-2-3": 0.8, "2-1-3": 0.2},
        {"1-2-3": 0.6, "2-1-3": 0.4},
        projection_beta=1.5,
        residual_coefficient=0.0,
    )

    assert probabilities == pytest.approx({"1-2-3": 0.6, "2-1-3": 0.4})


def test_fit_returns_standard_blend_contract() -> None:
    races = [
        _race("2026-07-01", "1-2-3", (0.8, 0.2)),
        _race("2026-07-01", "2-1-3", (0.2, 0.8)),
    ]
    result = fit_orthogonal_residual(races, regularization=1.0)

    assert 0.0 <= result["model_weight"] <= 1.0
    assert result["temperature"] > 0.0
    assert result["market_coefficient"] >= 0.05
    model = races[0]["model_probabilities"]
    market = races[0]["market_probabilities"]
    direct = orthogonal_probabilities(
        model,
        market,
        projection_beta=result["projection_beta"],
        residual_coefficient=result["residual_coefficient"],
    )
    compatible = blend_probabilities(
        model,
        market,
        model_weight=result["model_weight"],
        temperature=result["temperature"],
    )
    assert compatible == pytest.approx(direct)


def test_orthogonal_prequential_falls_back_to_exact_raw_identity(
    monkeypatch,
) -> None:
    races = [
        _race("2026-07-01", "1-2-3", (0.9, 0.1)),
        _race("2026-07-02", "1-2-3", (0.9, 0.1)),
    ]

    def market_only(training, *, regularization):
        return {
            "projection_beta": 1.0,
            "residual_coefficient": 0.0,
            "model_coefficient": 0.0,
            "market_coefficient": 1.0,
            "model_weight": 0.0,
            "temperature": 1.0,
            "regularization": float(regularization),
            "training_races": len(training),
        }

    monkeypatch.setattr(
        orthogonal_residual, "fit_orthogonal_residual", market_only
    )

    result = select_regularization_prequential(races, regularizations=(1.0,))

    assert result["calibration_nonregression"]["outer_holdout_used"] is False
    assert result["calibration_nonregression"]["identity_fallback_applied"] is True
    calibrator = result["final_calibrator"]
    assert calibrator["model_weight"] == 1.0
    assert calibrator["temperature"] == 1.0
    for race in races:
        assert orthogonal_residual.orthogonal_probabilities(
            race["model_probabilities"],
            race["market_probabilities"],
            projection_beta=calibrator["projection_beta"],
            residual_coefficient=calibrator["residual_coefficient"],
        ) == pytest.approx(race["model_probabilities"])


def test_regularization_is_selected_on_forward_days() -> None:
    races = [
        _race("2026-07-01", "1-2-3", (0.8, 0.2)),
        _race("2026-07-02", "2-1-3", (0.3, 0.7)),
        _race("2026-07-03", "1-2-3", (0.75, 0.25)),
    ]
    result = select_regularization_prequential(
        races, regularizations=(0.1, 1.0)
    )

    assert result["selected_regularization"] in {0.1, 1.0}
    assert result["final_calibrator"]["training_races"] == 3
    assert result["candidates"][0]["folds"][0]["evaluation_date"] == "2026-07-02"
