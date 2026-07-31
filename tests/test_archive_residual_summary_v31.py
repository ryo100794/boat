from __future__ import annotations

from boatrace_ai.evaluation_queue import summarize_result


def test_v31_probability_ranking_and_bankroll_roles_are_separate() -> None:
    payload = {
        "model": "archive_closing_market_oracle_v1",
        "temporal_residual_diagnostic": {
            "calibration_from": "2026-05-10",
            "calibration_through": "2026-07-17",
            "evaluation_from": "2026-07-18",
            "evaluation_through": "2026-07-30",
            "conditional_ticket_residual_v30": {
                "metrics": {"trifecta_log_loss": 3.64},
                "artifact": {},
            },
            "ticket_utility_meta_ranking_v31": {
                "selected_candidate": {
                    "label_scheme": "payout_bucket",
                    "tree_preset": "balanced",
                    "top_k": 3,
                },
                "probability_artifact": {
                    "structure_variant": "shared_independent_core",
                    "feature_dimension": 329,
                    "converged": True,
                },
                "probability_metrics": {
                    "evaluated_races": 1944,
                    "trifecta_log_loss": 3.665,
                    "market_trifecta_log_loss": 3.675,
                    "raw_model_trifecta_log_loss": 3.76,
                    "trifecta_top5_hit_rate": 0.364,
                    "market_trifecta_top5_hit_rate": 0.361,
                },
                "ranking_metrics": {
                    "evaluated_races": 1944,
                    "selected_top_k": 3,
                    "selected_top_k_metrics": {
                        "hit_rate": 0.21,
                        "stake_yen": 583200,
                        "return_yen": 620000,
                        "profit_yen": 36800,
                        "roi": 1.0631,
                        "roi_ci95_lower": 1.01,
                        "probability_roi_above_one": 0.97,
                    },
                },
                "bankroll": {
                    "evaluation_days": 13,
                    "evaluated_races": 1944,
                    "tickets": 70,
                    "hit_tickets": 4,
                    "stake_yen": 7000,
                    "return_yen": 8000,
                    "profit_yen": 1000,
                    "roi": 1.1429,
                    "max_drawdown_yen": 1700,
                    "status": "ready",
                },
                "promotion_eligible": True,
            },
        },
    }

    summary = summarize_result(payload)

    assert summary["model"] == "ticket_utility_meta_ranking_v31"
    assert summary["trifecta_log_loss"] == 3.665
    assert summary["residual_selection"]["label_scheme"] == "payout_bucket"
    assert summary["residual_selection"]["tree_preset"] == "balanced"
    assert summary["residual_selection"]["top_k"] == 3
    assert summary["residual_ranking_metrics"]["roi"] == 1.0631
    assert summary["residual_ranking_metrics"]["roi_ci95_lower"] == 1.01
    assert summary["roi"] == 1.1429
    assert summary["promotion_eligible"] is True
