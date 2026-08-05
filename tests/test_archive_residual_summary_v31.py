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
                    "label_scheme": "payout_weighted",
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
                    "roi_ci95_lower": 1.02,
                    "probability_roi_above_one": 0.96,
                    "max_drawdown_yen": 1700,
                    "status": "ready",
                },
                "promotion_eligible": True,
            },
        },
    }

    summary = summarize_result(payload)

    assert summary["model"] == "ticket_utility_meta_ranking_v31"
    assert summary["evaluation_from"] == "2026-07-18"
    assert summary["evaluation_through"] == "2026-07-30"
    assert summary["trifecta_log_loss"] == 3.665
    assert summary["residual_selection"]["label_scheme"] == "payout_weighted"
    assert summary["residual_selection"]["tree_preset"] == "balanced"
    assert summary["residual_selection"]["top_k"] == 3
    assert summary["residual_ranking_metrics"]["roi"] == 1.0631
    assert summary["residual_ranking_metrics"]["roi_ci95_lower"] == 1.01
    assert summary["roi"] == 1.1429
    assert summary["roi_ci95_lower"] == 1.02
    assert summary["probability_roi_above_one"] == 0.96
    assert summary["promotion_eligible"] is True


def test_v33_summary_preserves_selection_and_generator_transport_audit() -> None:
    payload = {
        "model": "archive_closing_market_oracle_v1",
        "temporal_residual_diagnostic": {
            "calibration_from": "2026-01-01",
            "calibration_through": "2026-06-30",
            "evaluation_from": "2026-07-01",
            "evaluation_through": "2026-08-05",
            "ticket_utility_calibration_aligned_v33": {
                "validation_design": "frozen calibration generators",
                "selection_rule": "family-wise temporal floor",
                "selection_robustness_gate": {
                    "day_block_familywise_roi_lcb_above_one": False,
                },
                "selection_robustness_passed": False,
                "candidate_family_size": 18,
                "selection_lower_quantile": 0.05 / 18,
                "familywise_selection_alpha": 0.05,
                "selection_bootstrap_samples": 20_000,
                "calibration_generator_transport": {
                    "frozen": True,
                    "ranking_sha256_match": True,
                    "probability_artifact_match": True,
                },
                "selected_candidate": {
                    "label_scheme": "gross_return_poisson_c50",
                    "tree_preset": "compact",
                    "top_k": 3,
                },
                "probability_artifact": {
                    "structure_variant": "shared_independent_core",
                },
                "probability_metrics": {
                    "evaluated_races": 4659,
                    "trifecta_log_loss": 3.67,
                },
                "ranking_metrics": {
                    "selected_top_k": 3,
                    "selected_top_k_metrics": {
                        "evaluated_races": 4659,
                        "roi": 0.91,
                        "roi_ci95_lower": 0.82,
                        "roi_lower_quantile": 0.05,
                        "largest_hit_return_yen": 42000,
                        "roi_excluding_largest_hit": 0.88,
                        "effective_hit_count": 17.5,
                        "temporal_block_count": 3,
                        "temporal_block_rois": [0.89, 0.95, 0.90],
                        "minimum_temporal_block_roi": 0.89,
                        "roi_bootstrap_samples": 2000,
                        "roi_bootstrap_valid_samples": 2000,
                    },
                },
                "bankroll": {
                    "evaluation_days": 32,
                    "stake_yen": 0,
                    "profit_yen": 0,
                    "status": "purchase_gate_disabled",
                },
                "promotion_eligible": False,
            },
        },
    }

    summary = summarize_result(payload)

    assert summary["model"] == "ticket_utility_calibration_aligned_v33"
    assert summary["residual_candidate_family_size"] == 18
    assert summary["residual_selection_lower_quantile"] == 0.05 / 18
    assert summary["residual_selection_bootstrap_samples"] == 20_000
    assert summary["residual_selection_robustness_passed"] is False
    assert summary["residual_calibration_generator_transport"] == {
        "frozen": True,
        "ranking_sha256_match": True,
        "probability_artifact_match": True,
    }
    ranking = summary["residual_ranking_metrics"]
    assert ranking["roi_excluding_largest_hit"] == 0.88
    assert ranking["minimum_temporal_block_roi"] == 0.89
    assert ranking["roi_bootstrap_samples"] == 2000
    assert "roi" not in summary
    assert summary["roi_status"] == "not_applicable"
    assert summary["roi_not_applicable_reason"] == "purchase_gate_no_authorization"
