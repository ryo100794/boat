from __future__ import annotations

from boatrace_ai.evaluation_queue import summarize_result


def _residual_payload(*, include_v25: bool) -> dict:
    residual = {
        "artifact": {
            "feature_dimension": 389,
            "regularization": 0.1,
            "gradient_norm": 1e-6,
            "converged": True,
            "training_races": 9000,
        },
        "metrics": {
            "evaluated_races": 1900,
            "trifecta_log_loss": 3.64,
            "market_trifecta_log_loss": 3.67,
            "raw_model_trifecta_log_loss": 3.78,
            "trifecta_top5_hit_rate": 0.38,
            "market_trifecta_top5_hit_rate": 0.36,
        },
        "purchase_diagnostics": [
            {
                "policy": {"name": "fixed-policy"},
                "simulation": {
                    "tickets": 12,
                    "races_bet": 8,
                    "hit_tickets": 2,
                    "stake_yen": 1200,
                    "return_yen": 1500,
                    "profit_yen": 300,
                    "roi": 1.25,
                },
                "bootstrap": {
                    "roi_ci95_lower": 0.72,
                    "probability_roi_above_one": 0.61,
                },
            }
        ],
    }
    temporal = {
        "calibration_from": "2026-05-10",
        "calibration_through": "2026-07-17",
        "evaluation_from": "2026-07-18",
        "evaluation_through": "2026-07-30",
        "contextual_market_residual_v24": residual,
    }
    if include_v25:
        temporal["direct_context_market_residual_v25"] = residual
    return {
        "model": "archive_closing_market_oracle_v1",
        "status": "completed",
        "probability_metrics": {"trifecta_log_loss": 3.9},
        "temporal_residual_diagnostic": temporal,
    }


def test_v25_residual_metrics_replace_generic_oracle_headline() -> None:
    summary = summarize_result(_residual_payload(include_v25=True))
    assert summary["model"] == "direct_context_market_residual_v25"
    assert summary["trifecta_log_loss"] == 3.64
    assert summary["market_trifecta_log_loss"] == 3.67
    assert summary["trifecta_top5_hit_rate"] == 0.38
    assert summary["residual_feature_dimension"] == 389
    assert summary["residual_evaluation_from"] == "2026-07-18"
    assert summary["residual_purchase_policies"] == [
        {
            "name": "fixed-policy",
            "tickets": 12,
            "races_bet": 8,
            "hit_tickets": 2,
            "stake_yen": 1200,
            "return_yen": 1500,
            "profit_yen": 300,
            "roi": 1.25,
            "roi_ci95_lower": 0.72,
            "probability_roi_above_one": 0.61,
        }
    ]


def test_v24_is_used_when_v25_is_absent() -> None:
    summary = summarize_result(_residual_payload(include_v25=False))
    assert summary["model"] == "contextual_market_residual_v24"
    assert summary["trifecta_log_loss"] == 3.64
