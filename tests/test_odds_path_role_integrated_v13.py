from __future__ import annotations

import json
from typing import Any

from boatrace_ai.listwise import market_calibration
from boatrace_ai.listwise import odds_path_role_integrated_v13 as v13


class _Estimator:
    pass


def _calibration(candidate_count: int = 3) -> dict[str, Any]:
    return {
        "evaluation_days": 1,
        "candidate_count": candidate_count,
        "raw_expected_hits": 1.2,
        "adjusted_expected_hits": 0.7,
        "observed_hits": 1,
        "raw_overprediction_hits": 0.2,
        "adjusted_overprediction_hits": 0.0,
        "overprediction_reduction_hits": 0.2,
        "adjusted_expected_vs_hit_ratio": 1 / 0.7,
        "daily_lower_bound_covered": True,
        "missing_t300_races": 0,
    }


def _divergence() -> dict[str, Any]:
    return {
        "evaluated_races": 1,
        "missing_t300_races": 0,
        "bands": [{
            "divergence_band": "d_ge_100",
            "races": 1,
            "tickets": 2,
            "sum_predicted_probability": 0.4,
            "hits": 1,
            "hit_to_expected_ratio": 2.5,
            "stake_yen": 200,
            "return_yen": 800,
            "actual_payout_roi": 4.0,
        }],
    }


def test_v13_reuses_v12_stack_and_removes_estimators_from_json(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_v12(races, **kwargs):
        captured.update(kwargs)
        assert kwargs["probability_lcb_fit"] is v13.fit_edge_conditional_probability_lcb
        assert kwargs["probability_lcb_metrics"] is v13._settlement_probability_metrics
        estimator = {"model_name": "closing_v12", "estimator": _Estimator()}
        return {
            "model": "odds_path_role_integrated_t300_nonlinear_v12",
            "calibrator_strategy": "odds_path_role_integrated_t300_nonlinear_v12",
            "fixed_policy": {"name": "v12"},
            "folds": [{
                "evaluation_date": "2026-07-30",
                "selected_policy": {"name": "v12"},
                "leakage_guard": {"pass": True},
                "closing_model": estimator,
                "closing_v12_model": estimator,
                "probability_lcb_metrics": {
                    "conditional_calibration": _calibration(),
                    "strict_prior_divergence_bands": _divergence(),
                },
            }],
            "prospective_role_integrated_v12_walk_forward": {
                "roi": 1.1,
                "roi_without_largest_hit": 1.02,
                "daily_cluster_bootstrap_roi_lower_95": 0.91,
            },
            "deployment_configuration": {
                "calibrator_strategy": "odds_path_role_integrated_t300_nonlinear_v12",
                "closing_t300_v12_model": estimator,
            },
        }

    monkeypatch.setattr(v13, "walk_forward_evaluate_v12", fake_v12)
    result = v13.walk_forward_evaluate_v13(
        [], daily_budget_yen=10_000, min_calibration_days=2
    )

    assert result["model"] == v13.MODEL_NAME
    assert result["calibrator_strategy"] == v13.STRATEGY_NAME
    assert "prospective_role_integrated_v12_walk_forward" not in result
    prospective = result[v13.PROSPECTIVE_OUTPUT_KEY]
    assert prospective["roi"] == 1.1
    assert prospective["roi_without_largest_hit"] == 1.02
    assert prospective["daily_cluster_bootstrap_roi_lower_95"] == 0.91
    assert prospective["conditional_calibration"]["candidate_count"] == 3
    assert result["strict_prior_divergence_bands"]["bands"]
    fold = result["folds"][0]
    assert "closing_model" not in fold
    assert "closing_v12_model" not in fold
    assert fold["closing_model_artifact_audit"]["closing_model"] == {
        "model_name": "closing_v12"
    }
    assert fold["selected_policy"]["name"].startswith("v13_")
    guard = fold["leakage_guard"]
    assert guard["probability_lcb_crossfit_unit"] == "whole_race_day"
    assert guard["result_payout_in_purchase_features"] is False
    assert guard["closing_odds_in_probability_lcb_features"] is False
    deployment = result["deployment_configuration"]
    assert "closing_t300_v12_model" not in deployment
    json.dumps(result)
    assert captured["closing_fallback_policy"] == "v11"


def test_market_calibration_dispatches_v13_and_keeps_v12_available(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        v13,
        "walk_forward_evaluate_v13",
        lambda *args, **kwargs: {"model": v13.MODEL_NAME, "kwargs": kwargs},
    )

    result = market_calibration.walk_forward_evaluate(
        [],
        calibrator_strategy=v13.STRATEGY_NAME,
        daily_budget_yen=10_000,
        min_calibration_days=2,
        v12_closing_fallback_policy="no_bet",
    )

    assert result["model"] == v13.MODEL_NAME
    assert result["kwargs"]["closing_fallback_policy"] == "no_bet"
    assert market_calibration.odds_path_model_name(v13.STRATEGY_NAME) == (
        v13.MODEL_NAME
    )
    choices = next(
        action.choices
        for action in market_calibration.build_parser()._actions
        if action.dest == "calibrator_strategy"
    )
    assert v13.STRATEGY_NAME in choices
    assert "odds_path_role_integrated_t300_nonlinear_v12" in choices
