from __future__ import annotations

from typing import Any


PURPOSE_SPECS = (
    (
        "outcome_probability",
        "着順・3連単確率",
        "着順分布と3連単候補の較正確率",
        (
            "winner_log_loss",
            "winner_top1_accuracy",
            "calibrated_trifecta_log_loss",
            "trifecta_top5_hit_rate",
        ),
        "時系列holdout。購入成績は合否に使用しない",
    ),
    (
        "closing_odds",
        "締切時オッズ",
        "締切直前のオッズ水準・順位・不確実性",
        (
            "closing_odds_log_mae",
            "closing_odds_rank_correlation",
            "closing_odds_interval_coverage",
            "closing_snapshot_age_seconds",
        ),
        "締切窓固定の日付順walk-forward。結果的中率とは分離",
    ),
    (
        "ticket_value",
        "券別収益価値",
        "各3連単券の払戻期待値と購入順位",
        (
            "purchase_value_pearson_correlation",
            "purchase_value_calibration_mae",
            "purchase_probability_temperature",
            "purchase_residual_scale",
            "purchase_oof_market_log_loss",
            "purchase_oof_scaled_log_loss",
            "purchase_hit_log_loss",
            "t5_market_log_loss",
            "purchase_hit_log_loss_delta_vs_market",
            "purchase_hit_top5_rate",
            "purchase_payout_residual_scale",
            "purchase_oof_base_payout_log_mae",
            "purchase_oof_scaled_payout_log_mae",
            "purchase_payout_log_mae",
            "purchase_value_positive_predicted_tickets",
            "purchase_value_positive_observed_capped_roi",
        ),
        "外側未見期間で券別較正。正予測群ROIは診断値で昇格ROIではない",
    ),
    (
        "bankroll_policy",
        "資金配分・購入方策",
        "日額資金を0口以上の券へ配分した損益再現性",
        (
            "roi",
            "profit_yen",
            "max_drawdown_yen",
            "roi_without_largest_hit",
            "daily_cluster_bootstrap_roi_lower_95",
            "profitable_day_fraction",
        ),
        "時系列資金運用。利益再投資、最大1的中除外、日次bootstrapを必須化",
    ),
    (
        "production_validation",
        "本番昇格",
        "確率・市場・資金運用ゲートの同一未見期間検証",
        (
            "promotion_gate_passed",
            "promotion_gate_total",
            "prediction_deployment_eligible",
            "roi",
            "daily_cluster_bootstrap_roi_lower_95",
        ),
        "統一期間または完全未見前進評価。全必須ゲート合格時のみ昇格",
    ),
)


def evaluation_purpose_keys(job: dict[str, Any]) -> list[str]:
    """Return every objective evaluated by a possibly multi-head job."""
    task = str(job.get("kind") or job.get("milestone") or "").lower()
    name = str(job.get("name") or "").lower()
    parameters = (
        job.get("parameters") if isinstance(job.get("parameters"), dict) else {}
    )
    text = f"{task} {name} {parameters.get('purchase_loss', '')}".lower()
    purposes: list[str] = []

    def has(keys: tuple[str, ...]) -> bool:
        return any(job.get(key) is not None for key in keys)

    def tokens(values: tuple[str, ...]) -> bool:
        return any(value in text for value in values)

    if has(
        (
            "entry_log_loss",
            "winner_log_loss",
            "winner_top1_accuracy",
            "trifecta_log_loss",
            "calibrated_trifecta_log_loss",
            "trifecta_top5_hit_rate",
        )
    ) or tokens(
        (
            "listwise",
            "newton",
            "feature_search",
            "genetic",
            "mlp",
            "historical_research",
            "venue_conditional",
            "four_head",
            "standardized_365d",
        )
    ):
        purposes.append("outcome_probability")
    if has(
        (
            "closing_odds_log_mae",
            "closing_odds_rank_correlation",
            "closing_odds_interval_coverage",
            "closing_snapshot_age_seconds",
        )
    ) or tokens(("closing_odds", "odds_path", "market_curvature", "four_head")):
        purposes.append("closing_odds")
    if has(
        (
            "purchase_value_pearson_correlation",
            "purchase_value_calibration_mae",
            "purchase_value_positive_predicted_tickets",
            "purchase_value_positive_observed_capped_roi",
            "purchase_payout_log_mae",
            "purchase_oof_scaled_payout_log_mae",
        )
    ) or parameters.get("purchase_loss") or tokens(
        (
            "four_head",
            "conditional_payout",
            "pairwise",
            "learned_value",
            "contextual",
            "market_residual",
        )
    ):
        purposes.append("ticket_value")
    if has(
        (
            "roi",
            "profit_yen",
            "stake_yen",
            "max_drawdown_yen",
            "roi_without_largest_hit",
            "daily_cluster_bootstrap_roi_lower_95",
        )
    ) or tokens(
        (
            "bankroll_policy",
            "market_residual",
            "four_head",
            "conditional_payout",
            "standardized_365d",
        )
    ):
        purposes.append("bankroll_policy")
    if (
        job.get("promotion_gate_total") is not None
        or job.get("prediction_deployment_eligible") is not None
        or tokens(("standardized_365d", "nested_annual", "promotion", "prospective"))
    ):
        purposes.append("production_validation")
    return purposes or ["outcome_probability"]


def evaluation_purpose_groups(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = {spec[0]: [] for spec in PURPOSE_SPECS}
    status_order = {"実行中": 0, "待機中": 1, "完了": 2, "失敗": 3, "取消": 4}
    for job in jobs:
        keys = evaluation_purpose_keys(job)
        item = {**job, "purpose_keys": keys}
        for key in keys:
            grouped[key].append(item)
    result = []
    for key, label, objective, metrics, protocol in PURPOSE_SPECS:
        models = sorted(
            grouped[key],
            key=lambda row: (
                status_order.get(str(row.get("status") or ""), 5),
                -int(row.get("db_job_id") or 0),
            ),
        )[:16]
        result.append(
            {
                "key": key,
                "label": label,
                "objective": objective,
                "primary_metrics": list(metrics),
                "backtest_protocol": protocol,
                "models": models,
                "active_models": sum(bool(row.get("running")) for row in models),
            }
        )
    return result
