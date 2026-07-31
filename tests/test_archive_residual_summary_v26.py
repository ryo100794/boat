from __future__ import annotations

from boatrace_ai.evaluation_queue import summarize_result


def test_v26_probability_and_bankroll_are_reported_as_headlines() -> None:
    payload = {
        "model": "archive_closing_market_oracle_v1",
        "status": "completed",
        "temporal_residual_diagnostic": {
            "calibration_from": "2026-05-10",
            "calibration_through": "2026-07-17",
            "evaluation_from": "2026-07-18",
            "evaluation_through": "2026-07-30",
            "direct_context_empirical_lcb_v26": {
                "promotion_eligible": False,
                "probability_metrics": {
                    "evaluated_races": 1944,
                    "trifecta_log_loss": 3.65,
                    "market_trifecta_log_loss": 3.67,
                    "raw_model_trifecta_log_loss": 3.78,
                    "trifecta_top5_hit_rate": 0.37,
                    "market_trifecta_top5_hit_rate": 0.36,
                },
                "final_probability_artifact": {
                    "feature_dimension": 389,
                    "regularization": 0.1,
                    "converged": True,
                },
                "empirical_ev_calibration": {
                    "ready": True,
                    "training_days": 30,
                },
                "bankroll": {
                    "status": "ready",
                    "evaluated_races": 1944,
                    "evaluation_days": 13,
                    "tickets": 20,
                    "hit_tickets": 2,
                    "stake_yen": 2000,
                    "return_yen": 1800,
                    "profit_yen": -200,
                    "roi": 0.9,
                    "max_drawdown_yen": 500,
                },
            },
        },
    }
    summary = summarize_result(payload)
    assert summary["model"] == "direct_context_empirical_lcb_v26"
    assert summary["trifecta_log_loss"] == 3.65
    assert summary["tickets"] == 20
    assert summary["roi"] == 0.9
    assert summary["residual_empirical_ev_calibration"]["training_days"] == 30
    assert summary["residual_purchase_policies"][0]["name"] == (
        "empirical_ev_lcb95_adaptive_kelly"
    )
