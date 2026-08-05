from __future__ import annotations

from boatrace_ai.evaluation_queue import result_decision, summarize_result


def test_v38_summary_exposes_frozen_probability_selection_audit() -> None:
    payload = {
        "model": "decision_time_nonlinear_market_residual_v38",
        "training_status": "ready",
        "official_closing_fields_used": False,
        "market_probability_source": "decision_snapshot_odds",
        "minimum_decision_lead_seconds": 300.0,
        "required_minimum_decision_lead_seconds": 300.0,
        "training_from": "2026-07-20",
        "training_through": "2026-08-18",
        "training_days": 30,
        "training_races": 4300,
        "minimum_training_days": 30,
        "minimum_training_races": 3000,
        "evaluation_from": "2026-08-19",
        "evaluation_through": "2026-08-25",
        "market_is_exact_nested_null": True,
        "selected_tree_preset": "compact",
        "selected_shrinkage": 0.25,
        "inner_fit_through": "2026-08-12",
        "inner_validation_from": "2026-08-13",
        "source_scored_cache_sha256": "b" * 64,
        "artifact": {"booster_sha256": "a" * 64},
        "holdout_metrics": {
            "evaluated_races": 1000,
            "evaluated_days": 7,
            "trifecta_log_loss": 3.61,
            "market_trifecta_log_loss": 3.62,
            "log_loss_delta_vs_market": -0.01,
            "days_better_than_market": 5,
            "trifecta_top5_hit_rate": 0.371,
            "market_trifecta_top5_hit_rate": 0.372,
        },
    }

    summary = summarize_result(payload)

    assert summary["model"] == payload["model"]
    assert summary["official_closing_fields_used"] is False
    assert summary["minimum_decision_lead_seconds"] == 300.0
    assert summary["required_minimum_decision_lead_seconds"] == 300.0
    assert summary["training_days"] == 30
    assert summary["evaluation_days"] == 7
    assert summary["log_loss_delta_vs_market"] == -0.01
    assert summary["booster_sha256"] == "a" * 64
    assert summary["challenger_selection_gate_pass"] is True
    assert summary["promotion_eligible"] is False
    assert result_decision("decision_market_residual_v38", summary) == (
        "freeze_for_prospective_value_calibration"
    )


def test_v44_summary_exposes_stack_selection_audit() -> None:
    payload = {
        "model": "decision_time_stacked_market_residual_v44",
        "training_status": "ready",
        "official_closing_fields_used": False,
        "market_probability_source": "decision_snapshot_odds",
        "training_days": 30,
        "training_races": 4300,
        "evaluation_from": "2026-08-19",
        "evaluation_through": "2026-08-25",
        "market_is_exact_nested_null": True,
        "base_training_through": "2026-08-12",
        "stack_validation_from": "2026-08-13",
        "selected_stack": "market50_linear50",
        "selected_weights": {
            "market": 0.5,
            "linear": 0.5,
            "nonlinear": 0.0,
        },
        "artifact": {"artifact_sha256": "c" * 64},
        "holdout_metrics": {
            "evaluated_races": 1000,
            "evaluated_days": 7,
            "trifecta_log_loss": 3.61,
            "market_trifecta_log_loss": 3.62,
            "log_loss_delta_vs_market": -0.01,
            "days_better_than_market": 5,
            "trifecta_top5_hit_rate": 0.371,
            "market_trifecta_top5_hit_rate": 0.372,
        },
    }

    summary = summarize_result(payload)

    assert summary["selected_stack"] == "market50_linear50"
    assert summary["selected_weights"]["linear"] == 0.5
    assert summary["base_training_through"] < summary["stack_validation_from"]
    assert summary["challenger_selection_gate_pass"] is True
    assert result_decision("decision_stacked_market_v44", summary) == (
        "freeze_for_prospective_value_calibration"
    )


def test_v39_summary_exposes_strict_prior_lcb_and_na_roi() -> None:
    payload = {
        "model": "decision_stack_contextual_strict_prior_lcb_v45",
        "frozen_probability_model": (
            "decision_time_stacked_market_residual_v44"
        ),
        "registered_after": "2026-08-26",
        "frozen_model_training_through": "2026-08-18",
        "selection_evaluation_through": "2026-08-25",
        "frozen_model_hash": "a" * 64,
        "settlement_engine_hash": "b" * 64,
        "candidate_population": "all_probability_top5_before_purchase_gate",
        "purchase_residual_shrinkage": 1.0,
        "purchase_max_probability_rank": 5,
        "calibration_target": "gross ROI including principal",
        "purchase_threshold": "empirical_ROI_LCB95 > 1.0",
        "range_policy": "deny outside local isotonic block support",
        "bootstrap_cluster_unit": "race_date",
        "ticket_level_independence_assumed": False,
        "warmup": {
            "logical_operator": "AND",
            "minimum_training_calendar_days": 30,
            "minimum_pregate_candidates": 300,
            "minimum_candidate_days": 20,
        },
        "latest_calibrator": {
            "ready": False,
            "ready_reasons": ["training_days_below_minimum"],
            "training_days": 3,
            "tickets": 150,
            "candidate_days": 3,
            "isotonic_block_count": 2,
            "context_ready_cells": 1,
            "context_cells": 12,
            "cells": [{"rank_group": "top5", "odds_band": "<20"}],
        },
        "fold_audit": [{
            "calibration_ready": False,
            "authorized_tickets": 0,
            "stake_yen": 0,
            "strict_prior_check": True,
            "max_training_settlement_date": "2026-08-28",
            "calibration_cutoff_date": "2026-08-28",
            "calibrator_hash": "e" * 64,
            "calibration_ledger_hash": "d" * 64,
            "frozen_model_hash": "a" * 64,
            "settlement_engine_hash": "b" * 64,
            "decision_contract_hash": "c" * 64,
            "candidate_decisions": 15,
            "purchase_gate_approved_candidates": 0,
            "purchase_gate_denied_candidates": 15,
            "denial_reason_counts": {
                "calibrated_roi_lcb95_not_above_one": 15
            },
            "maximum_raw_estimated_ev": 1.42,
            "maximum_calibrated_roi": 1.08,
            "maximum_calibrated_roi_lcb95": 0.94,
            "buy_threshold": 1.0,
            "approval_rule": (
                "local_support_ready_and_calibrated_roi_lcb95_above_one"
            ),
        }],
        "candidate_decision_audit": [{
            "race_id": "r1",
            "combination": "1-2-3",
            "cell_support": 80,
            "cell_support_days": 18,
            "context_local_support_ready": False,
            "context_local_support_reasons": [
                "insufficient_context_bin_support",
                "insufficient_context_bin_support_days",
            ],
            "required_context_local_candidates": 100,
            "required_context_local_candidate_days": 20,
            "denial_reason": "context_local_bin_not_ready",
        }],
        "ledger_candidates": 150,
        "ledger_hash": "d" * 64,
        "bankroll": {
            "evaluation_days": 3,
            "tickets": 0,
            "hit_tickets": 0,
            "stake_yen": 0,
            "return_yen": 0,
            "profit_yen": 0,
            "roi": None,
            "roi_display": "N/A",
            "roi_ci95_lower": None,
            "probability_roi_above_one": None,
            "max_drawdown_yen": 0,
        },
        "promotion_eligible": False,
        "real_betting_enabled": False,
    }

    summary = summarize_result(payload)

    assert summary["model"] == payload["model"]
    assert summary["calibration_warmup_logical_operator"] == "AND"
    assert summary["frozen_probability_model"] == (
        "decision_time_stacked_market_residual_v44"
    )
    assert summary["calibration_context_ready_cells"] == 1
    assert summary["calibration_context_cells"] == 12
    assert summary["calibration_strict_prior_all_folds"] is True
    assert summary["calibration_max_training_settlement_date"] == "2026-08-28"
    assert summary["calibration_decision_contract_hashes"] == 1
    assert summary["calibration_ticket_level_independence_assumed"] is False
    assert summary["warmup_days"] == 3
    assert summary["required_days"] == 30
    assert summary["prior_candidates"] == 150
    assert summary["required_candidates"] == 300
    assert summary["calibration_warmup_no_purchases_before_ready"] is True
    assert summary["calibration_warmup_pre_ready_stake_yen"] == 0
    assert summary["calibration_lcb_cluster_unit"] == "race_date"
    assert summary["calibration_lcb_ticket_independence_assumed"] is False
    assert summary["decision_model_sha256"] == "a" * 64
    assert summary["decision_hash_bundle_sha256"] == "c" * 64
    assert summary["candidate_decision_count"] == 15
    assert summary["approved_candidate_count"] == 0
    assert summary["denied_candidate_count"] == 15
    assert summary["maximum_calibrated_roi_lcb95"] == 0.94
    assert summary["buy_threshold"] == 1.0
    assert summary["context_local_support_ready"] is False
    assert summary["context_local_candidates"] == 80
    assert summary["context_local_candidate_days"] == 18
    assert summary["required_context_local_candidates"] == 100
    assert summary["required_context_local_candidate_days"] == 20
    assert "roi" not in summary
    assert summary["roi_display"] == "N/A"
    assert summary["roi_status"] == "not_applicable_no_stake"
    assert summary["roi_not_applicable_reason"] == (
        "warmup_or_no_authorized_purchases"
    )
    assert summary["promotion_eligible"] is False
    assert summary["real_betting_enabled"] is False
    assert result_decision("decision_v38_empirical_lcb", summary) == (
        "accumulate_strict_prior_value_calibration"
    )


def test_v38_v39_decisions_distinguish_data_accumulation_and_promotion() -> None:
    assert result_decision("decision_market_residual_v38", {
        "training_status": "insufficient_training_history",
    }) == "accumulate_decision_training_history"
    assert result_decision("decision_v38_empirical_lcb", {
        "calibration_ready": True,
        "promotion_eligible": False,
    }) == "accumulate_prospective_bankroll_evidence"
    assert result_decision("decision_v38_empirical_lcb", {
        "calibration_ready": True,
        "promotion_eligible": True,
    }) == "promotion_candidate"
