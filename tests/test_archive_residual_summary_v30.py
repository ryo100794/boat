from __future__ import annotations

from boatrace_ai.evaluation_queue import summarize_result


def test_v30_conditional_structure_and_bankroll_are_reported() -> None:
    payload = {
        "model": "archive_closing_market_oracle_v1",
        "temporal_residual_diagnostic": {
            "calibration_from": "2026-05-10",
            "calibration_through": "2026-07-17",
            "evaluation_from": "2026-07-18",
            "evaluation_through": "2026-07-30",
            "payout_weighted_role_model_v29": {
                "probability_metrics": {"trifecta_log_loss": 3.665},
                "probability_artifact": {},
            },
            "conditional_ticket_residual_v30": {
                "selected_candidate": {
                    "feature_variant": "race_shape_number",
                    "regularization": 0.03,
                    "converged": True,
                },
                "artifact": {
                    "feature_variant": "race_shape_number",
                    "active_ticket_feature_count": 17,
                    "feature_dimension": 346,
                    "regularization": 0.03,
                    "converged": True,
                },
                "metrics": {
                    "evaluated_races": 1944,
                    "trifecta_log_loss": 3.64,
                    "market_trifecta_log_loss": 3.675,
                    "raw_model_trifecta_log_loss": 3.76,
                    "trifecta_top5_hit_rate": 0.37,
                    "market_trifecta_top5_hit_rate": 0.36,
                },
                "bankroll": {
                    "evaluated_races": 1944,
                    "evaluation_days": 13,
                    "tickets": 60,
                    "hit_tickets": 3,
                    "stake_yen": 6000,
                    "return_yen": 6800,
                    "profit_yen": 800,
                    "roi": 1.1333,
                    "max_drawdown_yen": 1900,
                    "status": "ready",
                },
                "promotion_eligible": True,
            },
        },
    }

    summary = summarize_result(payload)

    assert summary["model"] == "conditional_ticket_residual_v30"
    assert summary["trifecta_log_loss"] == 3.64
    assert summary["residual_feature_variant"] == "race_shape_number"
    assert summary["residual_active_ticket_feature_count"] == 17
    assert summary["residual_selection"]["feature_variant"] == "race_shape_number"
    assert summary["roi"] == 1.1333
    assert summary["promotion_eligible"] is True
