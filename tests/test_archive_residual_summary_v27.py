from __future__ import annotations

from boatrace_ai.evaluation_queue import summarize_result


def test_v27_pruned_structure_and_performance_are_reported_as_headlines() -> None:
    payload = {
        "model": "archive_closing_market_oracle_v1",
        "temporal_residual_diagnostic": {
            "calibration_from": "2026-05-10",
            "calibration_through": "2026-07-17",
            "evaluation_from": "2026-07-18",
            "evaluation_through": "2026-07-30",
            "direct_context_empirical_lcb_v26": {
                "probability_metrics": {"trifecta_log_loss": 3.67},
                "final_probability_artifact": {},
            },
            "pruned_direct_context_market_residual_v27": {
                "selected_candidate": {
                    "variant": "ability_raw",
                    "active_context_feature_count": 3,
                    "feature_dimension": 287,
                    "regularization": 0.1,
                    "converged": True,
                },
                "artifact": {
                    "feature_variant": "ability_raw",
                    "active_context_feature_count": 3,
                    "feature_dimension": 287,
                    "regularization": 0.1,
                    "converged": True,
                },
                "metrics": {
                    "evaluated_races": 1944,
                    "trifecta_log_loss": 3.66,
                    "market_trifecta_log_loss": 3.67,
                    "raw_model_trifecta_log_loss": 3.78,
                    "trifecta_top5_hit_rate": 0.365,
                    "market_trifecta_top5_hit_rate": 0.36,
                },
                "purchase_diagnostics": [],
            },
        },
    }
    summary = summarize_result(payload)
    assert summary["model"] == "pruned_direct_context_market_residual_v27"
    assert summary["trifecta_log_loss"] == 3.66
    assert summary["residual_feature_variant"] == "ability_raw"
    assert summary["residual_feature_dimension"] == 287
    assert summary["residual_selection"]["regularization"] == 0.1
