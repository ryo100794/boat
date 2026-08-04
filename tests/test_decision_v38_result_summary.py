from __future__ import annotations

from boatrace_ai.evaluation_queue import result_decision, summarize_result


def test_v38_summary_exposes_frozen_probability_selection_audit() -> None:
    payload = {
        "model": "decision_time_nonlinear_market_residual_v38",
        "training_status": "ready",
        "official_closing_fields_used": False,
        "market_probability_source": "decision_snapshot_odds",
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
    assert summary["training_days"] == 30
    assert summary["evaluation_days"] == 7
    assert summary["log_loss_delta_vs_market"] == -0.01
    assert summary["booster_sha256"] == "a" * 64
    assert summary["challenger_selection_gate_pass"] is True
    assert summary["promotion_eligible"] is False
    assert result_decision("decision_market_residual_v38", summary) == (
        "freeze_for_prospective_value_calibration"
    )


def test_v39_summary_exposes_strict_prior_lcb_and_na_roi() -> None:
    payload = {
        "model": "decision_v38_strict_prior_empirical_lcb_v39",
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
