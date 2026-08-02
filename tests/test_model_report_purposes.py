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
        "calibrated_trifecta_log_loss": 3.7,
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
