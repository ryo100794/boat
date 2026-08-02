from __future__ import annotations

from boatrace_ai.web.dashboard import MODEL_REPORT_HTML
from boatrace_ai.web.model_report_purposes import (
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
    assert "的中LL / 市場LL / Δ" in MODEL_REPORT_HTML
    assert "払戻倍率 / OOF MAE" in MODEL_REPORT_HTML
    assert "外側払戻MAE" in MODEL_REPORT_HTML
    assert "収益指数 H/P" in MODEL_REPORT_HTML
    assert "評価期間" in MODEL_REPORT_HTML
    assert "最大1的中除外" in MODEL_REPORT_HTML
