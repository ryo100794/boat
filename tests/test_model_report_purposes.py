from __future__ import annotations

from boatrace_ai.web.dashboard import MODEL_REPORT_HTML
from boatrace_ai.web.model_report_purposes import (
    PURPOSE_DECISION_RULES,
    PURPOSE_REQUIREMENTS,
    PURPOSE_SPECS,
    evaluation_purpose_groups,
    evaluation_purpose_keys,
)


def test_probability_only_job_is_not_assigned_bankroll_metrics() -> None:
    job = {
        "db_job_id": 10,
        "name": "listwise_newton_cg_v1",
        "kind": "listwise_newton_refine",
        "status": "完了",
        "winner_log_loss": 1.2,
        "trifecta_top5_hit_rate": 0.31,
    }

    assert evaluation_purpose_keys(job) == ["outcome_probability"]


def test_unknown_completed_job_is_not_misclassified_as_probability() -> None:
    job = {"name": "database-backup", "kind": "backup", "status": "完了"}

    assert evaluation_purpose_keys(job) == []
    assert all(not group["models"] for group in evaluation_purpose_groups([job]))


def test_running_joint_bankroll_job_is_visible_in_its_three_objectives() -> None:
    job = {
        "db_job_id": 11845,
        "name": "joint_bankroll_strict_walk_forward_v1",
        "kind": "joint_bankroll_walk_forward",
        "status": "実行中",
        "running": True,
        "parameters": {"initial_daily_bankroll_yen": 10_000},
    }

    assert evaluation_purpose_keys(job) == [
        "outcome_probability",
        "bankroll_policy",
        "production_validation",
    ]
    groups = {row["key"]: row for row in evaluation_purpose_groups([job])}
    assert groups["bankroll_policy"]["models"][0][
        "purpose_evaluation"
    ]["backtest"]["daily_budget_yen"] == 10_000


def test_multi_head_purchase_job_is_reported_under_each_objective() -> None:
    job = {
        "db_job_id": 11160,
        "name": "four_head_pairwise_rank_v10_a001",
        "kind": "market_residual_walk_forward",
        "status": "実行中",
        "running": True,
        "parameters": {"purchase_loss": "pairwise_contextual_rank_calibrated"},
    }

    keys = evaluation_purpose_keys(job)
    assert keys == [
        "outcome_probability",
        "closing_odds",
        "ticket_value",
        "bankroll_policy",
    ]
    groups = {row["key"]: row for row in evaluation_purpose_groups([job])}
    assert groups["ticket_value"]["active_models"] == 1
    assert groups["bankroll_policy"]["models"][0]["db_job_id"] == 11160


def test_completed_model_is_only_grouped_where_metrics_exist() -> None:
    job = {
        "db_job_id": 11582,
        "name": "four_head_payout_stacked_tweedie_v20",
        "kind": "four_head_learned_value",
        "status": "完了",
        "winner_log_loss": 1.57,
        "purchase_value_pearson_correlation": 0.066,
        "roi": 0.0,
        "promotion_gate_total": 9,
    }

    assert evaluation_purpose_keys(job) == [
        "outcome_probability",
        "ticket_value",
        "bankroll_policy",
        "production_validation",
    ]


def test_purpose_contract_and_page_use_objective_specific_metrics() -> None:
    metrics = {key: set(values) for key, _, _, values, _ in PURPOSE_SPECS}

    assert "winner_log_loss" in metrics["outcome_probability"]
    assert "roi" not in metrics["outcome_probability"]
    assert "closing_odds_log_mae" in metrics["closing_odds"]
    assert "purchase_value_pearson_correlation" in metrics["ticket_value"]
    assert "purchase_probability_temperature" in metrics["ticket_value"]
    assert "purchase_hit_log_loss" in metrics["ticket_value"]
    assert "t5_market_log_loss" in metrics["ticket_value"]
    assert "purchase_residual_scale" in metrics["ticket_value"]
    assert "purchase_payout_residual_scale" in metrics["ticket_value"]
    assert "purchase_oof_scaled_payout_log_mae" in metrics["ticket_value"]
    assert "purchase_payout_log_mae" in metrics["ticket_value"]
    assert "purchase_gross_hit_exponent" in metrics["ticket_value"]
    assert "purchase_gross_payout_exponent" in metrics["ticket_value"]
    assert "purchase_gross_direct_value_exponent" in metrics["ticket_value"]
    assert "daily_cluster_bootstrap_roi_lower_95" in metrics["bankroll_policy"]
    assert 'id="purposeSection"' in MODEL_REPORT_HTML
    assert 'id="purposeGroups"' in MODEL_REPORT_HTML
    assert "renderPurposeGroups(data.evaluation_purposes||[])" in MODEL_REPORT_HTML
    assert "正予測群" in MODEL_REPORT_HTML
    assert "的中LL / 市場 / Δ" in MODEL_REPORT_HTML
    assert "払戻 外側/OOF MAE" in MODEL_REPORT_HTML
    assert "q20 pinball / 下側被覆" in MODEL_REPORT_HTML
    assert "収益指数 H/P" in MODEL_REPORT_HTML
    assert "評価期間" in MODEL_REPORT_HTML
    assert "最大1的中除外" in MODEL_REPORT_HTML



def test_purpose_group_reports_metric_completeness_and_backtest_scope() -> None:
    job = {
        "db_job_id": 22,
        "name": "candidate",
        "status": "完了",
        "winner_log_loss": 1.12,
        "winner_top1_accuracy": 0.58,
        "trifecta_log_loss": 3.7,
        "trifecta_top5_hit_rate": 0.37,
        "evaluation_days": 31,
        "evaluated_races": 1520,
        "parameters": {
            "from_date": "2026-06-01",
            "through_date": "2026-07-01",
            "decision_minutes_before": 5,
            "daily_budget_yen": 10_000,
            "include_odds": True,
        },
    }

    group = evaluation_purpose_groups([job])[0]
    evaluation = group["models"][0]["purpose_evaluation"]
    assert evaluation["complete"] is True
    assert evaluation["metric_count"] == evaluation["metric_total"] == 4
    assert evaluation["backtest"] == {
        "start": "2026-06-01",
        "end": "2026-07-01",
        "days": 31,
        "races": 1520,
        "folds": None,
        "decision_minutes_before": 5,
        "daily_budget_yen": 10_000,
        "odds_mode": "T-5",
        "race_set_sha256": None,
        "protocol_sha256": None,
        "policy_sha256": None,
        "allocation_mode": None,
        "profit_reinvestment": None,
    }
    assert evaluation["comparison_ready"] is True
    assert len(evaluation["comparison_id"]) == 10
    assert group["required_metrics"] == list(
        PURPOSE_REQUIREMENTS["outcome_probability"]
    )
    assert group["decision_rule"] == PURPOSE_DECISION_RULES["outcome_probability"]


def test_incomplete_purpose_metrics_are_explicit() -> None:
    group = evaluation_purpose_groups([
        {
            "name": "closing-candidate",
            "status": "完了",
            "closing_odds_log_mae": 0.1,
        }
    ])[1]
    evaluation = group["models"][0]["purpose_evaluation"]
    assert evaluation["complete"] is False
    assert evaluation["metric_count"] == 1
    assert "closing_odds_rank_correlation" in evaluation["missing_metrics"]


def test_purpose_page_shows_contract_scope_and_objective_specific_diagnostics() -> None:
    assert "評価充足" in MODEL_REPORT_HTML
    assert "バックテスト条件" in MODEL_REPORT_HTML
    assert "ΔLL 日次95%CI" in MODEL_REPORT_HTML
    assert "q20 pinball / 下側被覆" in MODEL_REPORT_HTML
    assert "的中Top5" in MODEL_REPORT_HTML
    assert "最大払戻比 / 有効的中" in MODEL_REPORT_HTML
    assert "失敗ゲート" in MODEL_REPORT_HTML
    assert "purposeCompleteness" in MODEL_REPORT_HTML
    assert "purposeScope" in MODEL_REPORT_HTML



def test_ticket_utility_ranking_uses_role_specific_required_metrics() -> None:
    job = {
        "name": "archive-role-models-v31",
        "kind": "archive_market_oracle",
        "status": "完了",
        "evaluation_from": "2026-07-18",
        "evaluation_through": "2026-08-01",
        "evaluated_races": 2268,
        "residual_selection": {
            "label_scheme": "winner",
            "tree_preset": "balanced",
            "top_k": 3,
        },
        "residual_ranking_metrics": {
            "hit_rate": 0.2478,
            "roi": 0.7720,
            "roi_ci95_lower": 0.7041,
        },
    }

    groups = {group["key"]: group for group in evaluation_purpose_groups([job])}
    evaluation = groups["ticket_value"]["models"][0]["purpose_evaluation"]
    assert evaluation["metric_profile"] == "ticket_utility_ranking"
    assert evaluation["complete"] is True
    assert evaluation["comparison_ready"] is True
    assert "ticket_ranking_roi_ci95_lower" in evaluation["required_metrics"]


def test_v33_ticket_utility_requires_selection_and_transport_audit() -> None:
    job = {
        "name": "archive-v33",
        "model": "ticket_utility_calibration_aligned_v33",
        "kind": "archive_market_oracle",
        "status": "完了",
        "evaluation_from": "2026-07-01",
        "evaluation_through": "2026-08-05",
        "evaluated_races": 4659,
        "residual_selection": {
            "label_scheme": "gross_return_poisson_c50",
            "tree_preset": "compact",
            "top_k": 3,
        },
        "residual_selection_lower_quantile": 0.05 / 18,
        "residual_selection_robustness_passed": False,
        "residual_calibration_generator_transport": {
            "frozen": True,
            "ranking_sha256_match": True,
            "probability_artifact_match": True,
        },
        "residual_ranking_metrics": {
            "hit_rate": 0.2478,
            "roi": 0.91,
            "roi_ci95_lower": 0.82,
            "roi_excluding_largest_hit": 0.88,
            "minimum_temporal_block_roi": 0.89,
        },
    }

    groups = {group["key"]: group for group in evaluation_purpose_groups([job])}
    evaluation = groups["ticket_value"]["models"][0]["purpose_evaluation"]
    assert evaluation["complete"] is True
    assert "selection_lower_quantile" in evaluation["required_metrics"]
    assert "selection_robustness_passed" in evaluation["required_metrics"]
    assert "calibration_generator_transport" in evaluation["required_metrics"]


def test_v33_ticket_utility_is_incomplete_without_generator_transport() -> None:
    job = {
        "model": "ticket_utility_calibration_aligned_v33",
        "residual_selection": {"label_scheme": "winner", "top_k": 1},
        "residual_selection_lower_quantile": 0.05 / 18,
        "residual_selection_robustness_passed": False,
        "residual_ranking_metrics": {
            "hit_rate": 0.2,
            "roi": 0.9,
            "roi_ci95_lower": 0.8,
            "roi_excluding_largest_hit": 0.85,
            "minimum_temporal_block_roi": 0.82,
        },
    }

    groups = {group["key"]: group for group in evaluation_purpose_groups([job])}
    evaluation = groups["ticket_value"]["models"][0]["purpose_evaluation"]
    assert evaluation["complete"] is False
    assert evaluation["missing_metrics"] == ["calibration_generator_transport"]


def test_purpose_page_separates_ranking_diagnostic_from_formal_bankroll() -> None:
    assert "役割 / 教師" in MODEL_REPORT_HTML
    assert "順位診断 的中/ROI/LCB" in MODEL_REPORT_HTML
    assert "正式資金BT" in MODEL_REPORT_HTML
    assert "ticketRankingDiagnostic" in MODEL_REPORT_HTML

    assert "最大除外" in MODEL_REPORT_HTML
    assert "最低block" in MODEL_REPORT_HTML
    assert "選択gate不合格" in MODEL_REPORT_HTML
    assert "校正生成器一致" in MODEL_REPORT_HTML


def test_no_bet_bankroll_does_not_count_roi_as_evaluated() -> None:
    job = {
        "name": "no-bet",
        "status": "完了",
        "roi": 0.0,
        "profit_yen": 0,
        "stake_yen": 0,
        "max_drawdown_yen": 0,
        "roi_without_largest_hit": 0.0,
        "daily_cluster_bootstrap_roi_lower_95": 0.0,
    }

    group = {
        row["key"]: row for row in evaluation_purpose_groups([job])
    }["bankroll_policy"]
    evaluation = group["models"][0]["purpose_evaluation"]
    assert evaluation["complete"] is False
    assert "roi" in evaluation["missing_metrics"]
    assert "roi_without_largest_hit" in evaluation["missing_metrics"]
    assert "daily_cluster_bootstrap_roi_lower_95" in evaluation["missing_metrics"]


def test_purpose_page_labels_no_bet_roi_as_unavailable() -> None:
    assert "ROI算出不能" in MODEL_REPORT_HTML
    assert "formalBankrollResult" in MODEL_REPORT_HTML
    assert "formalBankrollCi" in MODEL_REPORT_HTML
