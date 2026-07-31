from __future__ import annotations

from boatrace_ai.evaluation_queue import summarize_result


def test_v29_probability_ranking_and_bankroll_roles_are_reported() -> None:
    payload = {
        "model": "archive_closing_market_oracle_v1",
        "temporal_residual_diagnostic": {
            "calibration_from": "2026-05-10",
            "calibration_through": "2026-07-17",
            "evaluation_from": "2026-07-18",
            "evaluation_through": "2026-07-30",
            "course_interaction_market_residual_v28": {
                "metrics": {"trifecta_log_loss": 3.66},
                "artifact": {},
            },
            "payout_weighted_role_model_v29": {
                "selected_candidate": {
                    "payout_weight_exponent": 0.3,
                    "converged": True,
                },
                "probability_artifact": {
                    "structure_variant": "shared_independent_core",
                    "feature_dimension": 329,
                    "regularization": 0.03,
                    "converged": True,
                },
                "probability_metrics": {
                    "evaluated_races": 1944,
                    "trifecta_log_loss": 3.64,
                    "market_trifecta_log_loss": 3.67,
                    "raw_model_trifecta_log_loss": 3.78,
                    "trifecta_top5_hit_rate": 0.37,
                    "market_trifecta_top5_hit_rate": 0.36,
                },
                "ranking_metrics": {
                    "evaluated_races": 1944,
                    "trifecta_log_loss": 3.65,
                    "trifecta_top5_hit_rate": 0.38,
                    "top5_flat_stake_yen": 972000,
                    "top5_flat_return_yen": 1010000,
                    "top5_flat_profit_yen": 38000,
                    "top5_flat_roi": 1.039,
                },
                "bankroll": {
                    "evaluated_races": 1944,
                    "evaluation_days": 13,
                    "tickets": 55,
                    "hit_tickets": 2,
                    "stake_yen": 5500,
                    "return_yen": 6200,
                    "profit_yen": 700,
                    "roi": 1.127,
                    "max_drawdown_yen": 2100,
                    "status": "completed",
                },
                "promotion_eligible": True,
                "empirical_ev_calibration": {"ready": True, "tickets": 820},
            },
        },
    }

    summary = summarize_result(payload)

    assert summary["model"] == "payout_weighted_role_model_v29"
    assert summary["trifecta_log_loss"] == 3.64
    assert summary["trifecta_top5_hit_rate"] == 0.37
    assert summary["residual_selection"]["payout_weight_exponent"] == 0.3
    assert summary["residual_ranking_metrics"]["trifecta_top5_hit_rate"] == 0.38
    assert summary["residual_ranking_metrics"]["top5_flat_roi"] == 1.039
    assert summary["roi"] == 1.127
    assert summary["profit_yen"] == 700
    assert summary["promotion_eligible"] is True
