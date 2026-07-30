from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from boatrace_ai.listwise.closing_odds_t300_nonlinear_v12 import (
    CHECKPOINT_LABEL,
    FORBIDDEN_FEATURE_TOKENS,
    MODEL_NAME,
    closing_odds_t300_nonlinear_v12_metrics,
    fit_closing_odds_t300_nonlinear_v12,
    forecast_closing_odds_t300_nonlinear_v12,
)


COMBINATIONS = [
    f"{first}{second}{third}"
    for first in range(1, 7)
    for second in range(1, 7)
    if second != first
    for third in range(1, 7)
    if third not in (first, second)
]


def _race(day: int, race_no: int, *, with_t300: bool = True) -> dict[str, object]:
    venue = 1 + race_no % 2
    current = {
        combination: 4.0 + index * 0.17 + race_no * 0.03
        for index, combination in enumerate(COMBINATIONS)
    }
    # A deliberately nonlinear, learnable relationship with current price and venue.
    closing = {
        combination: value
        * math.exp(
            0.10 * math.sin(math.log(value) * 2.8)
            + (0.035 if venue == 1 and index % 3 == 0 else -0.015)
        )
        for index, (combination, value) in enumerate(current.items())
    }
    checkpoints: dict[str, object] = {}
    if with_t300:
        checkpoints[CHECKPOINT_LABEL] = {
            "target_offset_seconds": 300,
            "captured_age_seconds": 302,
            "captured_at": f"2026-07-{day:02d}T03:00:00+00:00",
            "odds": current,
        }
    return {
        "race_id": f"202607{day:02d}{venue:02d}{race_no:02d}",
        "race_date": f"2026-07-{day:02d}",
        "jcd": venue,
        "rno": race_no,
        "odds_checkpoints": checkpoints,
        "official_closing_odds": closing,
        "result": {"trifecta": COMBINATIONS[race_no % len(COMBINATIONS)]},
        "payouts": {"trifecta": 999999},
    }


def _dataset(days: range = range(1, 7), races_per_day: int = 1) -> list[dict[str, object]]:
    return [
        _race(day, race_no)
        for day in days
        for race_no in range(1, races_per_day + 1)
    ]


def _fit(races: list[dict[str, object]], **kwargs: object) -> dict[str, object]:
    return fit_closing_odds_t300_nonlinear_v12(
        races,
        prediction_date="2026-07-09",
        minimum_training_days=3,
        minimum_training_races=4,
        minimum_training_examples=480,
        calibration_warmup_days=2,
        minimum_calibration_clusters=2,
        engine="sklearn_hist_gradient_boosting",
        **kwargs,
    )


@pytest.fixture(scope="module")
def baseline_model() -> dict[str, object]:
    return _fit(_dataset(), minimum_relative_mae_improvement=0.99)


@pytest.fixture(scope="module")
def permissive_model() -> dict[str, object]:
    return _fit(_dataset(), minimum_relative_mae_improvement=0.0)


def test_strict_prior_group_day_cv_and_teacher_provenance(baseline_model) -> None:
    model = baseline_model

    assert model["model_name"] == MODEL_NAME
    assert model["trained_through_date"] == "2026-07-06"
    assert model["boundary_audit"]["strict_training_boundary"] is True
    assert model["boundary_audit"]["strict_outer_day_boundaries"] is True
    assert model["boundary_audit"]["group_day_unit"] == "race_date_x_venue"
    for fold in model["boundary_audit"]["outer_day_folds"]:
        assert fold["trained_through_date"] < fold["evaluation_date"]
        assert fold["evaluation_group_unit"] == "race_date_x_venue"
    provenance = model["teacher_provenance"]
    assert provenance["selection_policy"].startswith("official_closing_odds")
    assert provenance["training_examples_by_source"]["official_closing_odds"] > 0
    assert provenance["robustization"]["cluster_unit"] == (
        "race_date_x_venue_x_horizon"
    )


def test_adoption_requires_one_percent_strict_prior_improvement(
    permissive_model, baseline_model
) -> None:
    races = _dataset()
    permissive, impossible = permissive_model, baseline_model

    assert permissive["strict_prior_relative_mae_improvement"] is not None
    assert impossible["challenger_adopted"] is False
    assert impossible["selected_mode"] == "current_odds_baseline"
    assert impossible["minimum_relative_mae_improvement"] == 0.99
    if permissive["strict_prior_relative_mae_improvement"] >= 0.01:
        gated = _fit(races)
        assert gated["challenger_adopted"] is True
        assert gated["selected_mode"] == "nonlinear_model"


def test_day_venue_cluster_conformal_lower_bound(baseline_model) -> None:
    model = baseline_model
    conformal = model["lower_quantile_model"]

    assert conformal["ready"] is True
    assert conformal["cluster_unit"] == "race_date_x_venue"
    assert conformal["calibrated_point_source"] == "current_odds_baseline"
    assert conformal["effective_sample_clusters"] >= 2
    assert conformal["residual_log_ratio_adjustment"] <= 0.0
    assert all("|" in key for key in conformal["cluster_lower_residuals"])


def test_missing_t300_returns_no_forecast(baseline_model) -> None:
    model = baseline_model
    missing = _race(9, 1, with_t300=False)

    forecast = forecast_closing_odds_t300_nonlinear_v12(missing, model)

    assert forecast["ready"] is False
    assert forecast["reason"] == "missing_t300_checkpoint"
    assert forecast["point_final_odds"] == {}
    assert forecast["lower_final_odds"] == {}


def test_future_checkpoint_is_never_used(baseline_model) -> None:
    model = baseline_model
    race = _race(9, 1)
    race["odds_checkpoints"]["t120"] = {
        "target_offset_seconds": 120,
        "captured_age_seconds": 120,
        "odds": {key: 999.0 for key in COMBINATIONS},
    }

    forecast = forecast_closing_odds_t300_nonlinear_v12(race, model)

    assert forecast["ready"] is True
    assert forecast["used_checkpoint_offsets"] == [300]
    assert forecast["future_checkpoint_offsets_used"] == []
    assert forecast["future_checkpoint_imputation"] is False


def test_result_and_payout_are_not_features_or_decision_inputs(baseline_model) -> None:
    races = _dataset()
    changed = copy.deepcopy(races)
    for race in changed:
        race["result"] = {"trifecta": "654", "finish_order": [6, 5, 4]}
        race["payouts"] = {"trifecta": -12345, "refund": 777}
    first = baseline_model
    second = _fit(changed, minimum_relative_mae_improvement=0.99)

    assert first["strict_prior_challenger_mae"] == pytest.approx(
        second["strict_prior_challenger_mae"]
    )
    assert first["feature_names"] == second["feature_names"]
    assert first["boundary_audit"]["result_or_payout_features"] == []
    assert not any(
        token in name.lower()
        for name in first["feature_names"]
        for token in FORBIDDEN_FEATURE_TOKENS
    )


def test_forecast_is_invariant_to_future_checkpoint_result_and_payout(baseline_model) -> None:
    model = baseline_model
    original = _race(9, 1)
    changed = copy.deepcopy(original)
    changed["odds_checkpoints"]["t10"] = {
        "target_offset_seconds": 10,
        "captured_age_seconds": 10,
        "odds": {key: 1.1 for key in COMBINATIONS},
    }
    changed["result"] = {"trifecta": "654"}
    changed["payouts"] = {"trifecta": 1234567}

    first = forecast_closing_odds_t300_nonlinear_v12(original, model)
    second = forecast_closing_odds_t300_nonlinear_v12(changed, model)

    assert first["point_final_odds"] == second["point_final_odds"]
    assert first["lower_final_odds"] == second["lower_final_odds"]


def test_metrics_report_baseline_and_lower_coverage(baseline_model) -> None:
    training = _dataset()
    model = baseline_model
    evaluation = _dataset(range(9, 10), races_per_day=2)

    metrics = closing_odds_t300_nonlinear_v12_metrics(evaluation, model)

    assert metrics["evaluation_races"] == 2
    assert metrics["evaluation_tickets"] == 240
    assert metrics["baseline_current_log_mae"] is not None
    assert metrics["selected_point_log_mae"] is not None
    assert 0.0 <= metrics["lower_bound_coverage"] <= 1.0


def test_lightgbm_is_preferred_when_available() -> None:
    pytest.importorskip("lightgbm")
    model = fit_closing_odds_t300_nonlinear_v12(
        _dataset(),
        prediction_date="2026-07-09",
        minimum_training_days=3,
        minimum_training_races=4,
        minimum_training_examples=480,
        calibration_warmup_days=2,
        minimum_calibration_clusters=2,
        minimum_relative_mae_improvement=0.99,
    )

    assert model["requested_engine"] == "lightgbm"
    assert model["actual_engine"] == "lightgbm"


def test_non_past_training_rows_are_excluded() -> None:
    races = _dataset()
    races.append(_race(9, 1))
    model = _fit(races, minimum_relative_mae_improvement=0.99)

    assert model["boundary_audit"]["excluded_non_past_races"] == 1
    assert model["trained_through_date"] == "2026-07-06"
