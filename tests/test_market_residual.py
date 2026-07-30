import math

import pytest

import boatrace_ai.listwise.market_residual as market_residual
from boatrace_ai.listwise.market_residual import (
    fit_fixed_regularization,
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


def _market_identity_fit(races, *, regularization):
    return {
        "model_coefficient": 0.0,
        "market_coefficient": 1.0,
        "model_weight": 0.0,
        "temperature": 1.0,
        "regularization": float(regularization),
        "converged": True,
        "training_races": len(races),
    }


def test_prequential_calibration_preserves_fitted_calibrator_by_default(
    monkeypatch,
) -> None:
    prior_races = [
        _race("2026-07-20", "1-2-3", 0.9, 0.5),
        _race("2026-07-21", "1-2-3", 0.9, 0.5),
    ]
    monkeypatch.setattr(
        market_residual, "fit_log_pool_newton", _market_identity_fit
    )

    result = select_regularization_prequential(
        prior_races, regularizations=(1.0,)
    )

    assert result["final_calibrator"]["model_weight"] == 0.0
    assert result["final_calibrator"]["temperature"] == 1.0
    assert "calibration_nonregression" not in result
    assert "raw_model_prequential_log_loss" not in result["candidates"][0]
    assert "raw_model_trifecta_log_loss" not in (
        result["candidates"][0]["folds"][0]["metrics"]
    )


def test_prequential_calibration_falls_back_to_raw_without_outer_holdout(
    monkeypatch,
) -> None:
    prior_races = [
        _race("2026-07-20", "1-2-3", 0.9, 0.5),
        _race("2026-07-21", "1-2-3", 0.9, 0.5),
    ]
    monkeypatch.setattr(
        market_residual, "fit_log_pool_newton", _market_identity_fit
    )

    result = select_regularization_prequential(
        prior_races,
        regularizations=(1.0,),
        enforce_raw_nonregression=True,
    )

    audit = result["calibration_nonregression"]
    calibrator = result["final_calibrator"]
    assert audit["selection_data"] == "strict_prior_prequential_and_prior_refit_only"
    assert audit["outer_holdout_used"] is False
    assert audit["identity_fallback_applied"] is True
    assert audit["reason"] == "calibrated_prequential_log_loss_worse_than_raw"
    assert calibrator["model_weight"] == 1.0
    assert calibrator["temperature"] == 1.0
    metrics = residual_probability_metrics(
        prior_races, calibrator, include_raw_model=True
    )
    assert metrics["trifecta_log_loss"] == pytest.approx(
        metrics["raw_model_trifecta_log_loss"]
    )
    assert all(
        fold["evaluation_date"] <= "2026-07-21"
        for fold in result["candidates"][0]["folds"]
    )


def test_single_prior_day_calibration_falls_back_using_training_only(
    monkeypatch,
) -> None:
    prior_races = [_race("2026-07-20", "1-2-3", 0.9, 0.5)]
    monkeypatch.setattr(
        market_residual, "fit_log_pool_newton", _market_identity_fit
    )

    result = fit_fixed_regularization(
        prior_races, enforce_raw_nonregression=True
    )

    audit = result["calibration_nonregression"]
    assert audit["selection_data"] == "single_prior_training_day_only"
    assert audit["outer_holdout_used"] is False
    assert audit["identity_fallback_applied"] is True
    assert result["final_calibrator"]["model_coefficient"] == 1.0
    assert result["final_calibrator"]["market_coefficient"] == 0.0


def test_single_day_calibration_uses_preregistered_regularization() -> None:
    races = [
        _race("2026-07-20", "1-2-3", 0.8, 0.55),
        _race("2026-07-20", "1-3-2", 0.2, 0.45),
    ]

    result = fit_fixed_regularization(races)

    assert result["dates"] == ["2026-07-20"]
    assert result["selected_regularization"] == 1.0
    assert result["prequential_log_loss"] is None
    assert result["candidates"] == []
    assert result["final_calibrator"]["converged"] is True
