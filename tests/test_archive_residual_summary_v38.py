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
            "nested_nonlinear_value_calibration_v40": {
                "model": "nested_nonlinear_value_calibration_v40",
                "status": "completed",
                "model_training_from": "2026-05-10",
                "model_training_through": "2026-05-31",
                "model_training_days": 22,
                "value_calibration_from": "2026-06-01",
                "value_calibration_through": "2026-06-30",
                "value_calibration_days": 30,
                "evaluation_from": "2026-07-01",
                "evaluation_through": "2026-07-19",
                "evaluation_probability_metrics": {
                    "evaluated_races": 2744,
                    "trifecta_log_loss": 3.71,
                    "market_trifecta_log_loss": 3.70,
                    "log_loss_delta_vs_market": 0.01,
                },
                "empirical_ev_calibration": {
                    "ready": True,
                    "ready_reasons": [],
                    "training_days": 30,
                    "tickets": 20000,
                    "candidate_days": 30,
                    "candidate_min_raw_ev": 0.0,
                    "bins": [{
                        "bin_index": 0,
                        "lower": float("-inf"),
                        "upper": 1.0,
                        "support": 100,
                        "empirical_ev": 0.8,
                        "empirical_ev_lcb95": 0.7,
                    }],
                },
                "candidate_population": "all_stacked_probability_top5_before_purchase_gate",
                "calibration_ledger_candidates": 20000,
                "evaluation_ledger_candidates": 13720,
                "value_decile_audit": {
                    "edge_source": "value_calibration_only",
                    "evaluation_used_for_edges": False,
                    "calibration": [{"decile": 1, "realized_roi": 0.7}],
                    "evaluation": [{"decile": 1, "realized_roi": 0.8}],
                },
                "bankroll": {
                    "tickets": 0,
                    "stake_yen": 0,
                    "return_yen": 0,
                    "profit_yen": 0,
                    "roi": None,
                    "roi_display": "N/A",
                    "roi_ci95_lower": None,
                    "probability_roi_above_one": None,
                },
                "promotion_eligible": False,
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
    assert summary["nested_value_model_training_days"] == 22
    assert summary["nested_value_calibration_days"] == 30
    assert summary["nested_value_calibration_ready"] is True
    assert summary["nested_value_calibration_training_days"] == 30
    assert summary["nested_value_calibration_tickets"] == 20000
    assert summary["nested_value_calibration_candidate_days"] == 30
    assert summary["nested_value_calibration_candidate_min_raw_ev"] == 0.0
    assert summary["nested_value_candidate_population"] == (
        "all_stacked_probability_top5_before_purchase_gate"
    )
    assert summary["nested_value_calibration_bins"][0][
        "empirical_ev_lcb95"
    ] == 0.7
    assert summary["nested_value_calibration_bins"][0]["lower"] is None
    assert "roi" not in summary
    assert summary["roi_status"] == "not_applicable"
    assert summary["nested_value_evaluation_candidates"] == 13720
    assert summary["nested_value_decile_audit"][
        "evaluation_used_for_edges"
    ] is False
    assert summary["nested_value_roi_display"] == "N/A"
    assert summary["nested_value_promotion_eligible"] is False


def test_v41_summary_exposes_inner_selected_context_contract() -> None:
    payload = {
        "model": "archive_closing_market_oracle_v1",
        "temporal_residual_diagnostic": {
            "calibration_from": "2026-05-10",
            "calibration_through": "2026-06-30",
            "evaluation_from": "2026-07-01",
            "evaluation_through": "2026-07-19",
            "nonlinear_market_offset_context_search_v41": {
                "market_is_exact_nested_null": True,
                "outer_period_used_for_selection": False,
                "selected_context_variant": "full_context_20",
                "selected_context_features": ["class_rank", "national_win_rate"],
                "selected_tree_preset": "compact",
                "selected_shrinkage": 0.5,
                "artifact": {
                    "feature_dimension": 181,
                    "context_features": ["class_rank", "national_win_rate"],
                    "booster_sha256": "b" * 64,
                },
                "metrics": {
                    "evaluated_races": 2744,
                    "trifecta_log_loss": 3.68,
                    "market_trifecta_log_loss": 3.70,
                    "log_loss_delta_vs_market": -0.02,
                    "trifecta_top5_hit_rate": 0.38,
                    "market_trifecta_top5_hit_rate": 0.372,
                },
                "purchase_diagnostics": [],
            },
        },
    }

    summary = summarize_result(payload)

    assert summary["model"] == "nonlinear_market_offset_context_search_v41"
    assert summary["residual_outer_period_used_for_selection"] is False
    assert summary["residual_selected_context_variant"] == "full_context_20"
    assert summary["residual_selected_context_features"] == [
        "class_rank",
        "national_win_rate",
    ]
    assert summary["residual_context_features"] == [
        "class_rank",
        "national_win_rate",
    ]


def test_v42_and_v43_are_public_summary_heads() -> None:
    payload = {
        "model": "archive_closing_market_oracle_v1",
        "temporal_residual_diagnostic": {
            "calibration_from": "2026-05-10",
            "calibration_through": "2026-06-30",
            "evaluation_from": "2026-07-01",
            "evaluation_through": "2026-07-19",
            "stacked_market_residual_v42": {
                "market_is_exact_nested_null": True,
                "outer_period_used_for_selection": False,
                "base_training_through": "2026-06-19",
                "stack_validation_from": "2026-06-20",
                "selected_stack": "market50_linear50",
                "selected_weights": {
                    "market": 0.5,
                    "linear": 0.5,
                    "nonlinear": 0.0,
                },
                "artifact": {"artifact_sha256": "c" * 64},
                "metrics": {
                    "evaluated_races": 2744,
                    "trifecta_log_loss": 3.67,
                    "market_trifecta_log_loss": 3.70,
                    "log_loss_delta_vs_market": -0.03,
                    "trifecta_top5_hit_rate": 0.38,
                    "market_trifecta_top5_hit_rate": 0.37,
                },
                "purchase_diagnostics": [],
            },
            "nested_stacked_value_calibration_v43": {
                "model": "nested_stacked_value_calibration_v43",
                "status": "completed",
                "model_training_days": 22,
                "value_calibration_days": 30,
                "calibration_ledger_candidates": 20660,
                "evaluation_ledger_candidates": 13720,
                "evaluation_probability_metrics": {
                    "evaluated_races": 2744,
                    "trifecta_log_loss": 3.68,
                    "market_trifecta_log_loss": 3.70,
                    "log_loss_delta_vs_market": -0.02,
                },
                "empirical_ev_calibration": {
                    "ready": True,
                    "ready_reasons": [],
                    "bins": [],
                },
                "bankroll": {
                    "tickets": 0,
                    "stake_yen": 0,
                    "return_yen": 0,
                    "profit_yen": 0,
                    "roi": None,
                    "roi_display": "N/A",
                },
                "promotion_eligible": False,
            },
        },
    }

    summary = summarize_result(payload)

    assert summary["model"] == "stacked_market_residual_v42"
    assert summary["residual_selected_stack"] == "market50_linear50"
    assert summary["residual_selected_weights"]["linear"] == 0.5
    assert summary["residual_artifact_sha256"] == "c" * 64
    assert summary["nested_value_model"] == (
        "nested_stacked_value_calibration_v43"
    )
    assert summary["nested_value_evaluation_candidates"] == 13720


def test_mature_nested_value_is_preferred_only_when_completed() -> None:
    base = {
        "model": "archive_closing_market_oracle_v1",
        "dataset": {
            "official_eligible_target_races": 1000,
            "official_excluded_non_six_boat_races": 10,
            "official_expected_six_boat_races": 990,
            "official_snapshot_races": 980,
            "official_unresolved_races": 10,
            "official_invalid_attempt_races": 9,
            "official_fetch_failure_attempt_races": 1,
            "official_coverage_ratio": 980 / 990,
            "official_minimum_required_coverage": 0.995,
            "official_coverage_ready": False,
            "official_monthly_coverage": [
                {
                    "month": "2026-01",
                    "coverage_ratio": 980 / 990,
                    "unresolved_races": 10,
                }
            ],
        },
        "temporal_residual_diagnostic": {
            "targeted_temporal_component": "mature_stacked_contextual_value",
            "stacked_market_residual_v42": {
                "metrics": {"evaluated_races": 20},
                "artifact": {},
            },
            "nested_stacked_value_calibration_v43": {
                "model": "nested_stacked_value_calibration_v43",
                "status": "completed",
                "evaluation_probability_metrics": {"evaluated_races": 10},
                "empirical_ev_calibration": {"ready": True, "bins": []},
                "bankroll": {"tickets": 0, "stake_yen": 0, "roi": None},
                "promotion_eligible": False,
            },
            "mature_stacked_contextual_value": {
                "model": "mature_stacked_contextual_value",
                "status": "insufficient_nested_days",
                "required_days": 180,
                "promotion_eligible": False,
            },
        },
    }

    summary = summarize_result(base)
    assert summary["official_eligible_target_races"] == 1000
    assert summary["official_expected_six_boat_races"] == 990
    assert summary["official_snapshot_races"] == 980
    assert summary["official_unresolved_races"] == 10
    assert summary["official_coverage_ready"] is False
    assert summary["official_monthly_coverage"][0]["month"] == "2026-01"
    assert summary["nested_value_model"] == (
        "nested_stacked_value_calibration_v43"
    )

    base["temporal_residual_diagnostic"][
        "mature_stacked_contextual_value"
    ] = {
        "model": "mature_stacked_contextual_value",
        "status": "completed",
        "stack_selection_calibration_disjoint": True,
        "search_validation_draw_sets_disjoint": True,
        "value_stack_selection_from": "2026-04-01",
        "value_stack_selection_through": "2026-05-30",
        "value_stack_selection_days": 60,
        "value_stack_selection_races": 8000,
        "evaluation_probability_metrics": {"evaluated_races": 20},
        "empirical_ev_calibration": {"ready": True, "cells": []},
        "probability_selection": {
            "raw_selected_stack": "linear",
            "selected_stack": "market",
            "component_selection": {
                "linear_context_variant": "official_history_core",
                "linear_context_feature_count": 11,
                "linear_regularization": 0.1,
                "nonlinear_context_variant": "independent_core_10",
                "nonlinear_tree_preset": "compact",
            },
            "stack_selection_gate": {
                "policy_id": "logloss_daily_majority_top5_noninferiority_v1",
                "status": "fallback_market",
                "fallback_reasons": ["validation_top5_below_market"],
                "required_conditions": ["validation_top5_not_below_market"],
            },
        },
        "value_aligned_stack_selection": {
            "policy_id": (
                "top20_max_raw_ev_familywise_q01_top5_noninferiority_v1"
            ),
            "status": "selected",
            "base_selected_stack": "market",
            "selected_stack": "linear",
            "selection_lower_quantile": 0.01,
            "outer_period_used": False,
            "candidates": [
                {
                    "name": "linear",
                    "tickets": 1200,
                    "candidate_days": 120,
                    "roi": 1.02,
                    "roi_lcb95": 0.91,
                    "selected": True,
                }
            ],
        },
        "context_value_audit": {
            "status": "completed",
            "evaluation": [{"rank_group": "top5", "odds_band": "<20"}],
        },
        "research_sidecar": {
            "sha256": "d" * 64,
            "bytes": 1234,
            "format": "joblib",
            "candidate_decision_count": 456,
        },
        "formal_roi_gate": {
            "cluster_unit": "complete_race_date",
            "lower_quantile": 0.05,
            "quantile_method": "inverted_cdf",
        },
        "bankroll": {
            "tickets": 0,
            "stake_yen": 0,
            "roi": None,
            "roi_lower_quantile": 0.05,
            "roi_quantile_method": "inverted_cdf",
        },
        "promotion_eligible": False,
    }
    summary = summarize_result(base)
    assert summary["nested_value_model"] == "mature_stacked_contextual_value"
    assert summary["nested_value_evaluated_races"] == 20
    assert summary["nested_value_stack_selection_calibration_disjoint"] is True
    assert summary["nested_value_search_validation_draw_sets_disjoint"] is True
    assert summary["nested_value_stack_selection_from"] == "2026-04-01"
    assert summary["nested_value_stack_selection_through"] == "2026-05-30"
    assert summary["nested_value_stack_selection_days"] == 60
    assert summary["nested_value_stack_selection_races"] == 8000
    assert summary["nested_value_raw_selected_stack"] == "linear"
    assert summary["nested_value_selected_stack"] == "market"
    assert summary["nested_value_stack_selection_gate_status"] == (
        "fallback_market"
    )
    assert summary["nested_value_linear_context_variant"] == (
        "official_history_core"
    )
    assert summary["nested_value_linear_context_feature_count"] == 11
    assert summary["nested_value_nonlinear_context_variant"] == (
        "independent_core_10"
    )
    assert summary["nested_value_stack_selection_policy_id"] == (
        "logloss_daily_majority_top5_noninferiority_v1"
    )
    assert summary["nested_value_stack_selection_fallback_reasons"] == [
        "validation_top5_below_market"
    ]
    aligned = summary["nested_value_value_aligned_stack_selection"]
    assert aligned["selection_lower_quantile"] == 0.01
    assert aligned["selected_stack"] == "linear"
    assert aligned["outer_period_used"] is False
    assert aligned["candidates"][0]["roi_lcb95"] == 0.91
    assert summary["nested_value_roi_lower_quantile"] == 0.05
    assert summary["nested_value_roi_quantile_method"] == "inverted_cdf"
    assert summary["nested_value_formal_roi_gate"]["cluster_unit"] == (
        "complete_race_date"
    )
    assert summary["nested_value_context_audit"]["evaluation"][0] == {
        "rank_group": "top5",
        "odds_band": "<20",
    }
    assert summary["nested_value_research_sidecar_sha256"] == "d" * 64
    assert summary["nested_value_research_sidecar_bytes"] == 1234
    assert summary["nested_value_full_candidate_decision_count"] == 456
    assert summary["targeted_temporal_component"] == (
        "mature_stacked_contextual_value"
    )
