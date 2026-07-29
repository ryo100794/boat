from __future__ import annotations

import pytest

from boatrace_ai.listwise import odds_path_role_integrated_v12 as v12
from boatrace_ai.listwise import odds_path_role_integrated_v16 as v16
from boatrace_ai.listwise.v16_fixed_band_ranking_diagnostics import RULES


def _base_v12_result() -> dict:
    envelope = {
        "ready": True,
        "haircut": 0.9,
        "training_days": 5,
        "training_races": 30,
        "training_observations": 3600,
        "trained_through_date": "2026-07-29",
        "missing_audit": {
            "input_races": 30,
            "accepted_races": 30,
            "rejected_races": 0,
        },
    }
    ranking_diagnostics = {
        "candidate_population_fingerprint": "population",
        "decision_information_fingerprint": "decision",
        "rules": {
            rule: {
                "aggregate": {
                    "evaluation_days": 1,
                    "evaluated_races": 12,
                    "tickets": 10,
                    "hits": 1,
                    "stake_yen": 1_000,
                    "return_yen": 1_200,
                    "profit_yen": 200,
                    "roi": 1.2,
                    "largest_hit_return_yen": 1_200,
                    "return_excluding_largest_hit_yen": 0,
                    "roi_excluding_largest_hit": 0.0,
                }
            } for rule in RULES
        },
    }
    return {
        "fixed_policy": {"name": "v12-policy"},
        "selection_conformal": {},
        "selection_conformal_artifacts_by_date": {"2026-07-30": envelope},
        "folds": [{
            "evaluation_date": "2026-07-30",
            "probability_lcb_metrics": ranking_diagnostics,
            "selected_policy": {"name": "v12-policy"},
            "selection_conformal": envelope,
            "selection_observations_appended_after_decision": 12,
            "leakage_guard": {
                "selection_conformal_trained_through": "2026-07-29"
            },
            "bankroll": {"selection_conformal": envelope},
        }],
        v16.V12_PROSPECTIVE_OUTPUT_KEY: {
            "promotion_gate": {"base_roi_pass": True},
            "promotion_eligible": True,
        },
        "deployment_configuration": {},
    }


def test_v16_injects_passthrough_and_reuses_v15_envelope(monkeypatch) -> None:
    captured = {}

    def fake_v12(races, **kwargs):
        captured.update(kwargs)
        return _base_v12_result()

    monkeypatch.setattr(v16, "walk_forward_evaluate_v12", fake_v12)
    result = v16.walk_forward_evaluate_v16(
        [], daily_budget_yen=10_000, min_calibration_days=5
    )

    assert captured["closing_fallback_policy"] == v12.CLOSING_FALLBACK_NO_BET
    assert captured["closing_forecast_field"] == "point_final_odds"
    assert (
        captured["probability_lcb_fit"]
        is v16.fit_strict_prior_t300_divergence_passthrough_v16
    )
    assert captured["selection_conformal_fit"] is v16._fit_closing_envelope
    assert (
        captured["probability_lcb_metrics"]
        is v16.compare_v16_fixed_band_ranking_rules
    )
    assert (
        "probability_lcb_metrics_use_preallocation_population" not in captured
    )
    assert (
        captured["selection_observation_append"]
        is v16.append_closing_envelope_observations_v15
    )
    assert result["model"] == v16.MODEL_NAME
    assert result["registered_after"] == "2026-07-29"
    assert result["real_betting_enabled"] is False
    assert result["missing_real_t300_action"] == "no_bet"
    assert result["fixed_policy"]["conditional_lcb"] is False
    assert result["fixed_policy"]["raw_model_probability_inside_fixed_band"] is True
    assert result["closing_envelope_conformal"]["ready_folds"] == 1
    diagnostics = result["fixed_band_ranking_diagnostics"]
    assert diagnostics["real_betting_enabled"] is False
    assert (
        diagnostics["post_hoc_best_rule_is_promotion_evidence"]
        is False
    )
    guard = result["folds"][0]["leakage_guard"]
    assert set(diagnostics["rules"]) == set(RULES)
    assert diagnostics["rules"]["safe_ev_desc"]["days"] == 1
    assert diagnostics["rules"]["safe_ev_desc"]["races"] == 12
    assert len(result["folds"][0]["probability_lcb_metrics"]["rules"]) == 5
    assert guard["probability_artifact_uses_result"] is False
    assert guard["probability_artifact_uses_payout"] is False
    assert guard["result_payout_in_purchase_features"] is False
    assert result["deployment_configuration"]["selected_policy"] == {
        "name": "no_bet",
        "no_bet": True,
    }


def test_v16_rejects_any_closing_fallback_other_than_no_bet() -> None:
    with pytest.raises(ValueError, match="requires closing_fallback_policy='no_bet'"):
        v16.walk_forward_evaluate_v16(
            [],
            daily_budget_yen=10_000,
            min_calibration_days=5,
            closing_fallback_policy=v12.CLOSING_FALLBACK_V11,
        )
