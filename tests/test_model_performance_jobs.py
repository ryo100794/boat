import json
import sqlite3
from datetime import datetime, timedelta, timezone

from boatrace_ai.web.dashboard import (
    MODEL_REPORT_HTML,
    _database_evaluation_artifacts,
    _database_evaluation_status,
    _nested_checkpoint_fold_progress,
    _remote_evaluation_job_summaries,
    genetic_evolution_report,
)


def test_remote_job_summary_reports_fold_progress_and_metrics() -> None:
    remote = {
        "jobs": [
            {
                "name": "kelly-sweep",
                "milestone": "M6",
                "kind": "bankroll_norm",
                "status": "実行中",
                "running": True,
                "process": {"elapsed": "00:10:00", "cmd": "runner --folds 5 --epochs 1"},
                "log_tail": [
                    '{"fold": 1, "evaluated_races": 100}',
                    '{"fold": 4, "evaluated_races": 400}',
                ],
                "result": None,
            },
            {
                "name": "baseline",
                "milestone": "M4",
                "kind": "backtest",
                "status": "完了",
                "running": False,
                "process": None,
                "result": {
                    "metrics": {
                        "roi": 0.91,
                        "profit_yen": -900,
                        "evaluated_races": 1000,
                    }
                },
            },
        ]
    }

    rows = _remote_evaluation_job_summaries(remote)

    assert rows[0]["completed_folds"] == 4
    assert rows[0]["expected_folds"] == 5
    assert rows[0]["elapsed"] == "00:10:00"
    assert rows[1]["roi"] == 0.91
    assert rows[1]["profit_yen"] == -900


def test_nested_checkpoint_progress_counts_only_complete_valid_folds(tmp_path) -> None:
    checkpoint = (
        tmp_path / "data/models/evaluation_cache/nested_annual/job-00000077"
    )
    checkpoint.mkdir(parents=True)
    (checkpoint / "fold-01.npz").write_bytes(b"arrays")
    (checkpoint / "fold-01.json").write_text(json.dumps({
        "checkpoint_version": 1,
        "complete": True,
        "fold": 1,
        "npz_file": "fold-01.npz",
        "boundary_audit": {"passed": True},
    }), encoding="utf-8")
    (checkpoint / "fold-02.json").write_text(json.dumps({
        "checkpoint_version": 1,
        "complete": True,
        "fold": 2,
        "npz_file": "missing.npz",
        "boundary_audit": {"passed": True},
    }), encoding="utf-8")
    (checkpoint / "fold-03.json").write_text("not-json", encoding="utf-8")

    assert _nested_checkpoint_fold_progress(77, root=tmp_path) == 1


def test_model_report_contains_live_evaluation_table() -> None:
    assert 'id="evaluationRows"' in MODEL_REPORT_HTML
    assert 'id="candidateRows"' in MODEL_REPORT_HTML
    assert "基準1着" in MODEL_REPORT_HTML
    assert "evaluation_jobs" in MODEL_REPORT_HTML
    assert "<th>判定</th>" in MODEL_REPORT_HTML
    assert "<th>予測</th>" in MODEL_REPORT_HTML
    assert "<th>資金診断</th>" in MODEL_REPORT_HTML
    assert "top5_flat_roi" in MODEL_REPORT_HTML
    assert "<th>EV帯 証拠/診断</th>" in MODEL_REPORT_HTML
    assert "policyEvidence" in MODEL_REPORT_HTML
    assert 'policyEvidence(x,"registered_ev_band","R")' in MODEL_REPORT_HTML
    assert '_roi_without_largest_hit"' in MODEL_REPORT_HTML
    assert '_daily_cluster_bootstrap_roi_lower_95"' in MODEL_REPORT_HTML
    assert '"prospective_top5_narrow_ev","T5"' in MODEL_REPORT_HTML
    assert 'policyEvidence(x,"prospective_normalized_ev","N")' in MODEL_REPORT_HTML
    assert "registeredSummary" in MODEL_REPORT_HTML
    assert "winner_log_loss" in MODEL_REPORT_HTML
    assert "V_buy" in MODEL_REPORT_HTML
    assert "purchaseValueCalibrationRows" in MODEL_REPORT_HTML
    assert "purchase_value_realization_deciles" in MODEL_REPORT_HTML
    assert "id=\"nestedValueRows\"" in MODEL_REPORT_HTML
    assert "renderNestedValueAudit(jobs)" in MODEL_REPORT_HTML
    assert "nested_value_decile_audit" in MODEL_REPORT_HTML
    assert "id=\"nestedContextRows\"" in MODEL_REPORT_HTML
    assert "nested_value_context_audit" in MODEL_REPORT_HTML
    assert "nestedStackGateEvidence" in MODEL_REPORT_HTML
    assert "nested_value_stack_selection_fallback_reasons" in MODEL_REPORT_HTML
    assert "nested_value_research_sidecar_sha256" in MODEL_REPORT_HTML
    assert "完全証跡" in MODEL_REPORT_HTML
    assert "最大1的中除外ROI" in MODEL_REPORT_HTML
    assert "decisionStackEvidence" in MODEL_REPORT_HTML
    assert "購入余裕" in MODEL_REPORT_HTML
    assert "minimum_decision_lead_seconds" in MODEL_REPORT_HTML
    assert "required_minimum_decision_lead_seconds" in MODEL_REPORT_HTML
    assert "contextualValueEvidence" in MODEL_REPORT_HTML
    assert "局所ready" in MODEL_REPORT_HTML
    assert "candidate_decision_count" in MODEL_REPORT_HTML
    assert "maximum_calibrated_roi_lcb95" in MODEL_REPORT_HTML
    assert "context-local N" in MODEL_REPORT_HTML
    assert "required_context_local_candidates" in MODEL_REPORT_HTML
    assert "denial_reason_counts" in MODEL_REPORT_HTML
    assert "監査snapshot" in MODEL_REPORT_HTML
    assert "N/A / 購入なし" in MODEL_REPORT_HTML
    assert "daily_block_roi_lower_95" in MODEL_REPORT_HTML
    assert "安全余裕" in MODEL_REPORT_HTML
    assert "ROI LCB95" in MODEL_REPORT_HTML
    assert "正式ROI" in MODEL_REPORT_HTML
    assert "P(ROI&gt;1) 補助" in MODEL_REPORT_HTML
    assert "再標本化条件" in MODEL_REPORT_HTML
    assert "formalJointEvidence" in MODEL_REPORT_HTML
    assert 'id="geneticEvolution"' in MODEL_REPORT_HTML
    assert 'id="gaEvolutionChart"' in MODEL_REPORT_HTML
    assert "renderGeneticEvolution(jobs)" in MODEL_REPORT_HTML
    assert "drawGeneticEvolutionChart" in MODEL_REPORT_HTML
    assert "世代別fitness・アイランド内分散" in MODEL_REPORT_HTML
    assert "row.genetic_fitness==null?NaN" in MODEL_REPORT_HTML
    assert "/api/reports/genetic-evolution" in MODEL_REPORT_HTML
    assert "setInterval(refreshGeneticEvolution,10000)" in MODEL_REPORT_HTML
    assert "変異率" in MODEL_REPORT_HTML
    assert "投機fitnessは候補削減専用" in MODEL_REPORT_HTML
    assert "promotion_gate_passed" in MODEL_REPORT_HTML
    assert "pairedAuditText" in MODEL_REPORT_HTML
    assert "bootstrap_daily_profit_difference_lower_yen" in MODEL_REPORT_HTML
    assert "gateTitle" in MODEL_REPORT_HTML
    assert "minimum_fold_roi" in MODEL_REPORT_HTML
    assert "largest_hit_excluded_roi" in MODEL_REPORT_HTML
    assert "probability_roi_above_one" in MODEL_REPORT_HTML
    assert "5F min" in MODEL_REPORT_HTML


def test_database_evaluation_status_exposes_paired_payout_comparison(tmp_path) -> None:
    db_path = tmp_path / "queue.sqlite"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE model_evaluation_jobs (
          job_id INTEGER PRIMARY KEY, task_type TEXT, category TEXT,
          model_key TEXT, status TEXT, parameters TEXT,
          attempt INTEGER, max_attempts INTEGER,
          started_at TEXT, completed_at TEXT, decision TEXT,
          result_summary TEXT, result_path TEXT, error TEXT
        );
        CREATE TABLE model_improvement_candidates (
          job_id INTEGER PRIMARY KEY, metrics TEXT, parameters TEXT,
          created_at TEXT
        );
        """
    )
    metrics = {
        "roi": 0.94,
        "profit_yen": -600,
        "stake_yen": 10_000,
        "return_yen": 9_400,
        "max_drawdown_yen": 1_500,
        "tickets": 100,
        "hit_tickets": 8,
        "residual_purchase_policies": [
            {
                "name": "residual-top5",
                "tickets": 12,
                "hit_tickets": 2,
                "stake_yen": 1_200,
                "roi": 1.25,
            }
        ],
        "residual_selection_robustness_gate": {
            "day_block_familywise_roi_lcb_above_one": False,
        },
        "residual_selection_robustness_passed": False,
        "residual_candidate_family_size": 18,
        "residual_selection_lower_quantile": 0.05 / 18,
        "residual_familywise_selection_alpha": 0.05,
        "residual_selection_bootstrap_samples": 20_000,
        "residual_calibration_generator_transport": {
            "frozen": True,
            "ranking_sha256_match": True,
            "probability_artifact_match": True,
        },
        "residual_ranking_metrics": {
            "roi": 0.91,
            "roi_ci95_lower": 0.82,
            "roi_excluding_largest_hit": 0.88,
            "minimum_temporal_block_roi": 0.89,
        },
        "residual_selected_context_variant": "full_context_20",
        "residual_selected_stack": "linear50_nonlinear50",
        "residual_selected_weights": {
            "market": 0.0,
            "linear": 0.5,
            "nonlinear": 0.5,
        },
        "residual_outer_period_used_for_selection": False,
        "training_status": "ready",
        "market_probability_source": "decision_snapshot_odds",
        "official_closing_fields_used": False,
        "decision_time_boundary_all_passed": True,
        "decision_time_boundary_violations": 0,
        "minimum_decision_lead_seconds": 300.0,
        "required_minimum_decision_lead_seconds": 300.0,
        "maximum_input_snapshot_age_seconds": 60.0,
        "allowed_input_snapshot_age_seconds": 65.0,
        "training_days": 30,
        "training_races": 4300,
        "selected_stack": "market50_linear50",
        "selected_weights": {"market": 0.5, "linear": 0.5, "nonlinear": 0.0},
        "challenger_selection_gate_pass": True,
        "frozen_probability_model": "decision_time_stacked_market_residual_v44",
        "calibration_context_ready_cells": 2,
        "calibration_context_cells": 12,
        "calibration_contextual_hierarchy": (
            "global -> probability-rank -> rank-by-odds"
        ),
        "nested_value_model": "nested_nonlinear_value_calibration_v40",
        "nested_value_status": "completed",
        "nested_value_raw_selected_stack": "linear",
        "nested_value_selected_stack": "market",
        "nested_value_stack_selection_gate_status": "fallback_market",
        "nested_value_stack_selection_fallback_reasons": [
            "validation_top5_below_market"
        ],
        "nested_value_stack_selection_required_conditions": [
            "validation_top5_hit_rate_not_below_market"
        ],
        "nested_value_model_training_from": "2026-05-10",
        "nested_value_model_training_through": "2026-05-31",
        "nested_value_model_training_days": 22,
        "nested_value_calibration_from": "2026-06-01",
        "nested_value_calibration_through": "2026-06-30",
        "nested_value_calibration_days": 30,
        "nested_value_evaluation_from": "2026-07-01",
        "nested_value_evaluation_through": "2026-07-19",
        "nested_value_evaluated_races": 2744,
        "nested_value_calibration_ready": True,
        "nested_value_calibration_bins": [{
            "bin_index": 0,
            "support": 100,
            "empirical_ev": 0.8,
            "empirical_ev_lcb95": 0.7,
        }],
        "nested_value_context_ready_cells": 2,
        "nested_value_context_cells": [{
            "rank_group": "rank_1_5",
            "odds_band": "odds_10_25",
            "ready": True,
            "support": 320,
            "support_days": 90,
            "bins": [],
        }],
        "nested_value_calibration_candidates": 8640,
        "nested_value_evaluation_candidates": 13720,
        "nested_value_decile_audit": {
            "evaluation_used_for_edges": False,
            "calibration": [{"decile": 1, "realized_roi": 0.7}],
            "evaluation": [{"decile": 1, "realized_roi": 0.8}],
        },
        "nested_value_tickets": 0,
        "nested_value_stake_yen": 0,
        "nested_value_roi": None,
        "nested_value_roi_display": "N/A",
        "nested_value_promotion_eligible": False,
        "roi_without_largest_hit": 0.82,
        "trifecta_log_loss": 3.79,
        "winner_log_loss": 1.24,
        "winner_top1_accuracy": 0.53,
        "trifecta_top5_hit_rate": 0.35,
        "payout_feature_candidate_schema": "interactions_v2",
        "payout_feature_legacy_schema": "additive_v1",
        "payout_feature_candidate_roi": 1.03,
        "payout_feature_candidate_profit_yen": 300,
        "payout_feature_candidate_max_drawdown_yen": 1_200,
        "payout_feature_roi_ci95_lower": 1.01,
        "payout_feature_probability_roi_above_one": 0.96,
        "payout_feature_legacy_roi": 0.90,
        "payout_feature_roi_delta": 0.13,
        "payout_feature_roi_delta_ci95_lower": 0.02,
        "payout_feature_roi_delta_ci95_upper": 0.24,
        "payout_feature_probability_roi_delta_above_zero": 0.99,
        "promotion_gate_passed": 7,
        "promotion_gate_total": 10,
        "promotion_gate_failed": ["minimum_betting_days"],
        "holdout_temporal_minimum_roi": 0.94,
        "holdout_temporal_fold_rois": [1.10, 0.94, 1.03],
        "fold_count": 5,
        "fold_rois": [1.10, 1.04, 0.98, 1.02, 1.06],
        "minimum_fold_roi": 0.98,
        "largest_hit_excluded_roi": 1.01,
        "roi_ci95_lower": 0.97,
        "roi_ci95_upper": 1.09,
        "probability_roi_above_one": 0.91,
        "daily_cluster_bootstrap_roi_lower_95": 0.97,
        "joint_purchase_value_minimum": 0.08,
        "joint_purchase_safety_margin": 0.05,
        "joint_purchase_value_minimum_excess": 0.03,
        "joint_purchase_value_selected_portfolios": 12,
        "joint_purchase_value_gate_passed": True,
        "formal_roi_gate_method": "Q0.05_ROI_greater_than_1",
        "formal_roi_gate_passed": False,
        "roi_probability_is_diagnostic_only": True,
        "bootstrap_condition_id": "a" * 64,
        "bootstrap_primary_block": "complete_operating_day",
        "bootstrap_quantile_method": "inverted_cdf",
        "bootstrap_samples": 2000,
        "day_venue_roi_lower_95": 0.92,
        "venue_meeting_roi_lower_95": 0.89,
        "registered_ev_band_status": "evaluating",
        "registered_ev_band_evaluation_days": 3,
        "registered_ev_band_tickets": 15,
        "registered_ev_band_hit_tickets": 5,
        "registered_ev_band_roi": 1.889,
        "registered_ev_band_profit_yen": 1690,
        "registered_ev_band_roi_without_largest_hit": 1.342,
        "registered_ev_band_daily_cluster_bootstrap_roi_lower_95": 0.88,
        "registered_ev_band_probability_roi_above_one": 0.74,
        "prospective_top5_narrow_ev_status": "evaluating",
        "prospective_top5_narrow_ev_evaluation_days": 2,
        "prospective_top5_narrow_ev_tickets": 112,
        "prospective_top5_narrow_ev_hit_tickets": 13,
        "prospective_top5_narrow_ev_roi": 1.298,
        "top5_narrow_retrospective_status": (
            "diagnostic_only_not_promotion_evidence"
        ),
        "top5_narrow_retrospective_evaluation_days": 8,
        "top5_narrow_retrospective_tickets": 609,
        "top5_narrow_retrospective_hit_tickets": 71,
        "top5_narrow_retrospective_roi": 1.312,
        "top5_narrow_retrospective_roi_without_largest_hit": 1.28,
    }
    conn.execute(
        "INSERT INTO model_evaluation_jobs VALUES (273, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "venue_conditional_order", "evaluation", "venue-v1", "completed",
            "{}", 2, 2, "2026-07-23T00:00:00+00:00", "2026-07-23T01:00:00+00:00",
            "confirm_on_new_holdout", json.dumps(metrics), "result.json", None,
        ),
    )
    conn.execute(
        "INSERT INTO model_improvement_candidates VALUES (?, ?, ?, ?)",
        (273, json.dumps(metrics), "{}", "2026-07-23T01:00:00+00:00"),
    )
    conn.commit()
    conn.close()

    status = _database_evaluation_status(db_path)

    assert status["jobs"][0]["status"] == "完了"
    assert status["jobs"][0]["decision"] == "confirm_on_new_holdout"
    assert status["jobs"][0]["winner_log_loss"] == 1.24
    assert status["jobs"][0]["trifecta_log_loss"] == 3.79
    assert status["jobs"][0]["winner_top1_accuracy"] == 0.53
    assert status["jobs"][0]["stake_yen"] == 10_000
    assert status["jobs"][0]["return_yen"] == 9_400
    assert status["jobs"][0]["max_drawdown_yen"] == 1_500
    assert status["jobs"][0]["tickets"] == 100
    assert status["jobs"][0]["hit_tickets"] == 8
    assert status["jobs"][0]["residual_purchase_policies"] == [
        {
            "name": "residual-top5",
            "tickets": 12,
            "hit_tickets": 2,
            "stake_yen": 1_200,
            "roi": 1.25,
        }
    ]
    assert status["jobs"][0]["residual_selected_context_variant"] == (
        "full_context_20"
    )
    assert status["jobs"][0]["residual_selection_robustness_passed"] is False
    assert status["jobs"][0]["residual_candidate_family_size"] == 18
    assert status["jobs"][0]["residual_selection_lower_quantile"] == 0.05 / 18
    assert status["jobs"][0]["residual_selection_bootstrap_samples"] == 20_000
    assert status["jobs"][0]["residual_selection_robustness_gate"] == {
        "day_block_familywise_roi_lcb_above_one": False,
    }
    assert status["jobs"][0]["residual_ranking_metrics"][
        "roi_excluding_largest_hit"
    ] == 0.88
    assert status["jobs"][0]["residual_calibration_generator_transport"] == {
        "frozen": True,
        "ranking_sha256_match": True,
        "probability_artifact_match": True,
    }
    assert status["jobs"][0]["residual_selected_stack"] == (
        "linear50_nonlinear50"
    )
    assert status["jobs"][0]["residual_selected_weights"]["nonlinear"] == 0.5
    assert status["jobs"][0][
        "residual_outer_period_used_for_selection"
    ] is False
    assert status["jobs"][0]["selected_stack"] == "market50_linear50"
    assert status["jobs"][0]["official_closing_fields_used"] is False
    assert status["jobs"][0]["decision_time_boundary_all_passed"] is True
    assert status["jobs"][0]["minimum_decision_lead_seconds"] == 300.0
    assert status["jobs"][0]["required_minimum_decision_lead_seconds"] == 300.0
    assert status["jobs"][0]["calibration_context_ready_cells"] == 2
    assert status["jobs"][0]["nested_value_calibration_days"] == 30
    assert status["jobs"][0]["nested_value_raw_selected_stack"] == "linear"
    assert status["jobs"][0]["nested_value_selected_stack"] == "market"
    assert status["jobs"][0]["nested_value_stack_selection_gate_status"] == (
        "fallback_market"
    )
    assert status["jobs"][0][
        "nested_value_stack_selection_fallback_reasons"
    ] == ["validation_top5_below_market"]
    assert status["jobs"][0]["nested_value_calibration_bins"][0][
        "empirical_ev_lcb95"
    ] == 0.7
    assert status["jobs"][0]["nested_value_context_ready_cells"] == 2
    assert status["jobs"][0]["nested_value_context_cells"][0][
        "rank_group"
    ] == "rank_1_5"
    assert status["jobs"][0]["nested_value_evaluation_candidates"] == 13720
    assert status["jobs"][0]["nested_value_decile_audit"][
        "evaluation_used_for_edges"
    ] is False
    assert status["jobs"][0]["roi_without_largest_hit"] == 0.82
    assert status["jobs"][0]["promotion_gate_passed"] == 7
    assert status["jobs"][0]["promotion_gate_total"] == 10
    assert status["jobs"][0]["promotion_gate_failed"] == ["minimum_betting_days"]
    assert status["jobs"][0]["holdout_temporal_minimum_roi"] == 0.94
    assert status["jobs"][0]["fold_count"] == 5
    assert status["jobs"][0]["minimum_fold_roi"] == 0.98
    assert status["jobs"][0]["largest_hit_excluded_roi"] == 1.01
    assert status["jobs"][0]["roi_ci95_lower"] == 0.97
    assert status["jobs"][0]["probability_roi_above_one"] == 0.91
    assert status["jobs"][0]["joint_purchase_value_minimum"] == 0.08
    assert status["jobs"][0]["joint_purchase_safety_margin"] == 0.05
    assert status["jobs"][0]["joint_purchase_value_gate_passed"] is True
    assert status["jobs"][0]["formal_roi_gate_passed"] is False
    assert status["jobs"][0]["bootstrap_condition_id"] == "a" * 64
    assert status["jobs"][0]["bootstrap_primary_block"] == (
        "complete_operating_day"
    )
    assert status["jobs"][0]["bootstrap_quantile_method"] == "inverted_cdf"
    assert status["jobs"][0]["bootstrap_samples"] == 2000
    assert status["jobs"][0]["registered_ev_band_profit_yen"] == 1690
    assert status["jobs"][0]["registered_ev_band_roi_without_largest_hit"] == 1.342
    assert (
        status["jobs"][0]["registered_ev_band_daily_cluster_bootstrap_roi_lower_95"]
        == 0.88
    )
    assert status["jobs"][0]["prospective_top5_narrow_ev_evaluation_days"] == 2
    assert status["jobs"][0]["prospective_top5_narrow_ev_roi"] == 1.298
    assert status["jobs"][0]["top5_narrow_retrospective_evaluation_days"] == 8
    assert status["jobs"][0]["top5_narrow_retrospective_roi"] == 1.312
    assert (
        status["jobs"][0]["top5_narrow_retrospective_roi_without_largest_hit"]
        == 1.28
    )
    assert status["candidates"][0]["payout_feature_candidate_roi"] == 1.03
    assert status["candidates"][0]["payout_feature_candidate_profit_yen"] == 300
    assert status["candidates"][0]["payout_feature_candidate_max_drawdown_yen"] == 1_200
    assert status["candidates"][0]["payout_feature_roi_ci95_lower"] == 1.01
    assert status["candidates"][0]["payout_feature_probability_roi_above_one"] == 0.96
    assert status["candidates"][0]["payout_feature_roi_delta_ci95_lower"] == 0.02
    assert status["candidates"][0]["joint_purchase_value_minimum"] == 0.08
    assert status["candidates"][0]["joint_purchase_safety_margin"] == 0.05
    assert status["candidates"][0]["joint_purchase_value_gate_passed"] is True
    assert status["candidates"][0]["formal_roi_gate_passed"] is False
    assert status["candidates"][0]["bootstrap_condition_id"] == "a" * 64
    assert status["candidates"][0]["day_venue_roi_lower_95"] == 0.92
    assert status["candidates"][0]["venue_meeting_roi_lower_95"] == 0.89


def test_database_evaluation_status_includes_parent_of_recent_job(tmp_path) -> None:
    db_path = tmp_path / "queue.sqlite"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE model_evaluation_jobs (
          job_id INTEGER PRIMARY KEY, task_type TEXT, category TEXT,
          model_key TEXT, status TEXT, parameters TEXT,
          attempt INTEGER, max_attempts INTEGER,
          started_at TEXT, completed_at TEXT, decision TEXT,
          result_summary TEXT, result_path TEXT, error TEXT,
          parent_job_id INTEGER
        );
        CREATE TABLE model_improvement_candidates (
          job_id INTEGER PRIMARY KEY, metrics TEXT, parameters TEXT,
          created_at TEXT
        );
        """
    )
    rows = []
    for job_id in range(1, 103):
        parent_job_id = 1 if job_id == 102 else None
        summary = json.dumps({"roi": 0.81}) if job_id == 1 else "{}"
        rows.append(
            (
                job_id,
                "listwise_feature_search",
                "evaluation",
                f"job-{job_id}",
                "completed",
                "{}",
                1,
                2,
                None,
                "2026-07-28T00:00:00+00:00",
                None,
                summary,
                None,
                None,
                parent_job_id,
            )
        )
    conn.executemany(
        "INSERT INTO model_evaluation_jobs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()
    conn.close()

    status = _database_evaluation_status(db_path)

    by_name = {row["name"]: row for row in status["jobs"]}
    assert "job-1" in by_name
    assert by_name["job-1"]["roi"] == 0.81
    assert "job-2" not in by_name


def test_database_evaluation_status_normalizes_duplicate_formal_jobs(tmp_path) -> None:
    db_path = tmp_path / "queue.sqlite"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE model_evaluation_jobs (
          job_id INTEGER PRIMARY KEY, task_type TEXT, category TEXT,
          model_key TEXT, status TEXT, parameters TEXT, priority INTEGER,
          attempt INTEGER, max_attempts INTEGER,
          started_at TEXT, completed_at TEXT, decision TEXT,
          result_summary TEXT, result_path TEXT, error TEXT,
          parent_job_id INTEGER
        );
        CREATE TABLE model_improvement_candidates (
          job_id INTEGER PRIMARY KEY, metrics TEXT, parameters TEXT,
          created_at TEXT
        );
        """
    )
    model_key = "triple_head_v21_daily:market_residual:20260718-29"
    metrics = {
        "evaluation_days": 6,
        "evaluated_races": 918,
        "roi": 1.4756097561,
        "stake_yen": 8200,
        "return_yen": 12100,
        "profit_yen": 3900,
        "winner_log_loss": 1.1649013962,
        "calibrated_trifecta_log_loss": 3.7089263912,
        "trifecta_top5_hit_rate": 0.3736383442,
        "comparison_role": "triple_head",
    }
    base_parameters = {
        "from_date": "2026-07-18",
        "through_date": "2026-07-29",
        "calibrator_strategy": "triple_head_v21",
    }
    rows = [
        (8624, base_parameters, 97, "accumulate_formal_evidence", None),
        (8666, base_parameters, 116, "accumulate_formal_evidence", 8458),
        (
            8667,
            {**base_parameters, "calibrator_strategy": "tail_diagnostic"},
            80,
            "accumulate_formal_evidence",
            None,
        ),
        (8668, base_parameters, 10, "research_only", None),
    ]
    for job_id, parameters, priority, decision, parent_job_id in rows:
        conn.execute(
            """
            INSERT INTO model_evaluation_jobs (
              job_id, task_type, category, model_key, status, parameters,
              priority, attempt, max_attempts, decision, result_summary,
              parent_job_id
            ) VALUES (?, 'market_residual_walk_forward', 'evaluation', ?,
                      'completed', ?, ?, 1, 2, ?, ?, ?)
            """,
            (
                job_id,
                model_key,
                json.dumps(parameters),
                priority,
                decision,
                json.dumps(metrics),
                parent_job_id,
            ),
        )
        conn.execute(
            "INSERT INTO model_improvement_candidates VALUES (?, ?, '{}', NULL)",
            (job_id, json.dumps(metrics)),
        )
    conn.commit()
    conn.close()

    status = _database_evaluation_status(db_path)

    assert [row["db_job_id"] for row in status["jobs"]] == [8668, 8667, 8666]
    assert status["jobs"][-1]["priority"] == 116
    assert status["jobs"][-1]["parent_job_id"] == 8458
    assert {row["job_id"] for row in status["candidates"]} == {8666, 8667, 8668}


def test_database_evaluation_status_quarantines_invalid_data_source(tmp_path) -> None:
    db_path = tmp_path / "queue.sqlite"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE model_evaluation_jobs (
          job_id INTEGER PRIMARY KEY, task_type TEXT, category TEXT,
          model_key TEXT, status TEXT, parameters TEXT,
          attempt INTEGER, max_attempts INTEGER,
          started_at TEXT, completed_at TEXT, decision TEXT,
          result_summary TEXT, result_path TEXT, error TEXT
        );
        CREATE TABLE model_improvement_candidates (
          job_id INTEGER PRIMARY KEY, metrics TEXT, parameters TEXT,
          created_at TEXT
        );
        """
    )
    metrics = {
        "roi": 1.25,
        "profit_yen": 2500,
        "winner_top1_accuracy": 0.70,
        "data_source_validation_pass": False,
    }
    conn.execute(
        "INSERT INTO model_evaluation_jobs VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "archive_market_oracle", "evaluation", "invalid-oracle", "completed",
            "{}", 1, 1, "2026-07-28T00:00:00+00:00",
            "2026-07-28T01:00:00+00:00", "invalid_data_source",
            json.dumps(metrics), "invalid.json", None,
        ),
    )
    conn.execute(
        "INSERT INTO model_improvement_candidates VALUES (?, ?, ?, ?)",
        (1, json.dumps(metrics), "{}", "2026-07-28T01:00:00+00:00"),
    )
    conn.commit()
    conn.close()

    status = _database_evaluation_status(db_path)

    assert status["jobs"][0]["status"] == "無効"
    assert status["jobs"][0]["valid_for_comparison"] is False
    assert status["jobs"][0]["roi"] is None
    assert status["jobs"][0]["winner_top1_accuracy"] is None
    assert status["candidates"] == []


def test_database_evaluation_artifact_exposes_daily_and_payout_walk_forward(
    tmp_path,
) -> None:
    model_dir = tmp_path / "models"
    result_path = model_dir / "evaluation_queue" / "job-00000932.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(
        json.dumps(
            {
                "model": "calibrated_mlp_recency_selected",
                "generated_at": "2026-07-24T00:00:00+00:00",
                "entry_log_loss": 0.32,
                "entry_brier": 0.09,
                "winner_top1_accuracy": 0.57,
                "trifecta_top5_hit_rate": 0.31,
                "evaluated_races": 100,
                "bankroll": {
                    "roi": 0.8,
                    "profit_yen": -200,
                    "stake_yen": 1000,
                    "return_yen": 800,
                },
                "daily": [
                    {
                        "race_date": "2026-07-23",
                        "stake_yen": 1000,
                        "return_yen": 800,
                    }
                ],
                "conditional_payout_walk_forward": {
                    "bankroll": {
                        "roi": 1.2,
                        "profit_yen": 200,
                        "stake_yen": 1000,
                        "return_yen": 1200,
                        "daily": [
                            {
                                "race_date": "2026-07-23",
                                "stake_yen": 1000,
                                "return_yen": 1200,
                            }
                        ],
                    },
                    "bankroll_confidence": {
                        "roi_ci95_lower": 1.01,
                        "roi_ci95_upper": 1.4,
                        "roi_delta_ci95_lower": 0.1,
                        "roi_delta_ci95_upper": 0.5,
                    },
                },
                "market_offset_multinomial_kelly_walk_forward": {
                    "evaluation_days": 6,
                    "evaluated_races": 918,
                    "tickets": 28,
                    "hit_tickets": 3,
                    "stake_yen": 3000,
                    "return_yen": 3130,
                    "profit_yen": 130,
                    "roi": 1.0433333333333332,
                    "daily": [{
                        "race_date": "2026-07-23",
                        "stake_yen": 3000,
                        "return_yen": 3130,
                    }],
                },
                "conservative_market_offset_kelly_walk_forward": {
                    "status": "waiting_for_first_unseen_day",
                    "registered_after": "2026-07-28",
                    "evaluation_days": 0,
                    "evaluated_races": 0,
                    "tickets": 0,
                    "stake_yen": 0,
                    "return_yen": 0,
                    "profit_yen": 0,
                    "roi": 0.0,
                },
            }
        ),
        encoding="utf-8",
    )
    queue_status = {
        "candidates": [
            {
                "model_key": "calibrated_mlp_recency_selected",
                "result_path": str(result_path),
            }
        ]
    }

    backtests, bankroll, daily = _database_evaluation_artifacts(
        queue_status,
        model_dir,
    )

    assert [row["name"] for row in backtests] == [
        "calibrated_mlp_recency_selected"
    ]
    assert [row["name"] for row in bankroll] == [
        "calibrated_mlp_recency_selected",
        "calibrated_mlp_recency_selected_conditional_payout_walk_forward",
        "calibrated_mlp_recency_selected_market_offset_multinomial_kelly_walk_forward",
        "calibrated_mlp_recency_selected_conservative_market_offset_kelly_walk_forward",
    ]
    assert daily["calibrated_mlp_recency_selected"][0]["roi_delta"] == -0.2
    assert daily[
        "calibrated_mlp_recency_selected_conditional_payout_walk_forward"
    ][0]["roi_delta"] == 0.2
    assert daily[
        "calibrated_mlp_recency_selected_market_offset_multinomial_kelly_walk_forward"
    ][0]["roi_delta"] == 0.043333333333333335


def test_database_evaluation_artifact_rejects_paths_outside_model_dir(
    tmp_path,
) -> None:
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")

    assert _database_evaluation_artifacts(
        {"candidates": [{"model_key": "outside", "result_path": str(outside)}]},
        tmp_path / "models",
    ) == ([], [], {})


def test_database_evaluation_artifact_prioritizes_bankroll_results(
    tmp_path,
) -> None:
    model_dir = tmp_path / "models"
    queue_dir = model_dir / "evaluation_queue"
    queue_dir.mkdir(parents=True)
    search_path = queue_dir / "job-00000001.json"
    search_path.write_text(
        json.dumps({"entry_log_loss": 0.34, "evaluated_races": 100}),
        encoding="utf-8",
    )
    bankroll_path = queue_dir / "job-00000002.json"
    bankroll_path.write_text(
        json.dumps(
            {
                "entry_log_loss": 0.32,
                "evaluated_races": 100,
                "bankroll": {"roi": 0.8, "stake_yen": 1000},
                "conditional_payout_walk_forward": {
                    "bankroll": {
                        "roi": 0.0,
                        "stake_yen": 0,
                        "policy": {
                            "no_bet": True,
                            "no_bet_reason": "selection_gate_no_bet",
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    queue_status = {
        "candidates": [
            {
                "model_key": "search-only",
                "result_path": str(search_path),
                "roi": None,
            },
            {
                "model_key": "bankroll-model",
                "result_path": str(bankroll_path),
                "roi": 0.8,
                "payout_feature_candidate_roi": 0.0,
            },
        ]
    }

    _, bankroll, _ = _database_evaluation_artifacts(
        queue_status,
        model_dir,
        maximum_artifacts=1,
    )

    assert [row["name"] for row in bankroll] == [
        "bankroll-model",
        "bankroll-model_conditional_payout_walk_forward",
    ]
    assert bankroll[1]["no_bet"] is True
    assert bankroll[1]["no_bet_reason"] == "selection_gate_no_bet"


def test_database_evaluation_status_uses_current_attempt_elapsed(tmp_path) -> None:
    db_path = tmp_path / "queue.sqlite"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE model_evaluation_jobs (
          job_id INTEGER PRIMARY KEY, task_type TEXT, category TEXT,
          model_key TEXT, status TEXT, parameters TEXT,
          attempt INTEGER, max_attempts INTEGER,
          started_at TEXT, completed_at TEXT, decision TEXT,
          result_summary TEXT, result_path TEXT, error TEXT
        );
        CREATE TABLE model_improvement_candidates (
          job_id INTEGER PRIMARY KEY, metrics TEXT, parameters TEXT,
          created_at TEXT
        );
        CREATE TABLE model_evaluation_job_runs (
          job_id INTEGER, attempt INTEGER, status TEXT, started_at TEXT
        );
        """
    )
    original_started = datetime.now(timezone.utc) - timedelta(hours=8)
    current_started = datetime.now(timezone.utc) - timedelta(minutes=5)
    conn.execute(
        "INSERT INTO model_evaluation_jobs VALUES (3564, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "listwise_feature_search",
            "evaluation",
            "retrying-search",
            "running",
            "{}",
            4,
            4,
            original_started.isoformat(),
            None,
            None,
            "{}",
            None,
            None,
        ),
    )
    conn.execute(
        "INSERT INTO model_evaluation_job_runs VALUES (?, ?, ?, ?)",
        (3564, 4, "running", current_started.isoformat()),
    )
    conn.commit()
    conn.close()

    row = _database_evaluation_status(db_path)["jobs"][0]

    assert row["running"] is True
    assert row["elapsed"].startswith("00:")
