from __future__ import annotations

from boatrace_ai.evaluation_queue import summarize_result


def test_v38_summary_exposes_probability_and_fixed_strength_purchase_roles() -> None:
    payload = {
        "model": "archive_closing_market_oracle_v1",
        "temporal_residual_diagnostic": {
            "calibration_from": "2026-05-10",
            "calibration_through": "2026-06-30",
            "evaluation_from": "2026-07-01",
            "evaluation_through": "2026-07-19",
            "nonlinear_market_offset_residual_v38": {
                "market_is_exact_nested_null": True,
                "selected_tree_preset": "compact",
                "selected_shrinkage": 0.25,
                "inner_fit_through": "2026-06-20",
                "inner_validation_from": "2026-06-21",
                "artifact": {
                    "feature_dimension": 121,
                    "objective": (
                        "grouped_multinomial_logloss_with_fixed_market_offset"
                    ),
                    "training_races": 6972,
                    "booster_sha256": "a" * 64,
                },
                "metrics": {
                    "evaluated_races": 2744,
                    "evaluated_days": 19,
                    "trifecta_log_loss": 3.69,
                    "market_trifecta_log_loss": 3.70,
                    "log_loss_delta_vs_market": -0.01,
                    "days_better_than_market": 13,
                    "trifecta_top5_hit_rate": 0.374,
                    "market_trifecta_top5_hit_rate": 0.372,
                },
                "purchase_diagnostics": [
                    {
                        "role": "fixed_full_residual_research_control",
                        "shrinkage": 1.0,
                        "policy": {"name": "fixed-policy"},
                        "simulation": {
                            "tickets": 195,
                            "stake_yen": 19500,
                            "return_yen": 20510,
                            "profit_yen": 1010,
                            "roi": 1.0518,
                        },
                        "bootstrap": {"roi_ci95_lower": 0.71},
                    }
                ],
            },
        },
    }

    summary = summarize_result(payload)

    assert summary["model"] == "nonlinear_market_offset_residual_v38"
    assert summary["trifecta_log_loss"] == 3.69
    assert summary["market_trifecta_log_loss"] == 3.70
    assert summary["residual_selected_shrinkage"] == 0.25
    assert summary["residual_selected_tree_preset"] == "compact"
    assert summary["residual_log_loss_delta_vs_market"] == -0.01
    assert summary["residual_days_better_than_market"] == 13
    assert summary["residual_booster_sha256"] == "a" * 64
    assert summary["residual_objective"] == (
        "grouped_multinomial_logloss_with_fixed_market_offset"
    )
    policy = summary["residual_purchase_policies"][0]
    assert policy["role"] == "fixed_full_residual_research_control"
    assert policy["shrinkage"] == 1.0
    assert policy["roi"] == 1.0518
    assert summary["promotion_eligible"] is False
