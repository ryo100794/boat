from __future__ import annotations

from boatrace_ai.evaluation_queue import summarize_result


def test_v28_course_structure_and_performance_are_reported_as_headlines() -> None:
    payload = {
        "model": "archive_closing_market_oracle_v1",
        "temporal_residual_diagnostic": {
            "calibration_from": "2026-05-10",
            "calibration_through": "2026-07-17",
            "evaluation_from": "2026-07-18",
            "evaluation_through": "2026-07-30",
            "pruned_direct_context_market_residual_v27": {
                "metrics": {"trifecta_log_loss": 3.67},
                "artifact": {},
            },
            "course_interaction_market_residual_v28": {
                "selected_candidate": {
                    "structure_variant": "course_independent_core",
                    "architecture": "finish_stage_by_starting_lane_context",
                    "feature_variant": "independent_core",
                    "active_context_feature_count": 10,
                    "feature_dimension": 629,
                    "regularization": 0.03,
                    "converged": True,
                },
                "artifact": {
                    "structure_variant": "course_independent_core",
                    "architecture": "finish_stage_by_starting_lane_context",
                    "feature_variant": "independent_core",
                    "active_context_feature_count": 10,
                    "feature_dimension": 629,
                    "regularization": 0.03,
                    "converged": True,
                },
                "metrics": {
                    "evaluated_races": 1944,
                    "trifecta_log_loss": 3.65,
                    "market_trifecta_log_loss": 3.67,
                    "raw_model_trifecta_log_loss": 3.78,
                    "trifecta_top5_hit_rate": 0.37,
                    "market_trifecta_top5_hit_rate": 0.36,
                },
                "purchase_diagnostics": [],
            },
        },
    }
    summary = summarize_result(payload)
    assert summary["model"] == "course_interaction_market_residual_v28"
    assert summary["trifecta_log_loss"] == 3.65
    assert summary["residual_architecture"] == (
        "finish_stage_by_starting_lane_context"
    )
    assert summary["residual_structure_variant"] == "course_independent_core"
    assert summary["residual_selection"]["feature_variant"] == "independent_core"
