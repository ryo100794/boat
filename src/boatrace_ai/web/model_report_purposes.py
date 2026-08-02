from __future__ import annotations

import hashlib
import json
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
            "purchase_gross_hit_exponent",
            "purchase_gross_payout_exponent",
            "purchase_gross_direct_value_exponent",
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

PURPOSE_REQUIREMENTS = {
    "outcome_probability": (
        "winner_log_loss",
        "winner_top1_accuracy",
        "trifecta_log_loss",
        "trifecta_top5_hit_rate",
    ),
    "closing_odds": (
        "closing_odds_log_mae",
        "closing_odds_rank_correlation",
        "closing_odds_interval_coverage",
        "closing_snapshot_age_seconds",
    ),
    "ticket_value": (
        "purchase_value_pearson_correlation",
        "purchase_value_calibration_mae",
        "purchase_hit_log_loss_delta_vs_market",
        "purchase_payout_log_mae",
    ),
    "bankroll_policy": (
        "roi",
        "profit_yen",
        "stake_yen",
        "max_drawdown_yen",
        "roi_without_largest_hit",
        "daily_cluster_bootstrap_roi_lower_95",
    ),
    "production_validation": (
        "promotion_gate_passed",
        "promotion_gate_total",
        "prediction_deployment_eligible",
        "roi",
        "daily_cluster_bootstrap_roi_lower_95",
    ),
}


PURPOSE_DECISION_RULES = {
    "outcome_probability": (
        "未見時系列でLogLossを市場基準と比較し、日クラスタ95%CI、1着、3T5を併記"
    ),
    "closing_odds": (
        "同一T-5窓でlog MAEを最小化し、順位相関・区間被覆・取得時差を同時監査"
    ),
    "ticket_value": (
        "外側未見券で価値相関>0、的中LLの市場差<0、払戻誤差を確認。ROI単独で採用しない"
    ),
    "bankroll_policy": (
        "日額1万円・利益再投資でROI>1、最大1的中除外ROI>1、日次LCB95>1を全て要求"
    ),
    "production_validation": (
        "30日以上の前進未見評価で必須ゲート全合格。締切後情報の推論利用を禁止"
    ),
}


def _first_value(*values: Any) -> Any:
    return next((value for value in values if value not in (None, "")), None)


def _backtest_scope(job: dict[str, Any]) -> dict[str, Any]:
    parameters = job.get("parameters")
    if not isinstance(parameters, dict):
        parameters = {}
    return {
        "start": _first_value(
            job.get("holdout_start"),
            job.get("evaluation_start"),
            job.get("formal_evaluation_from"),
            job.get("evaluation_from"),
            parameters.get("outer_from"),
            parameters.get("evaluation_from"),
            parameters.get("from_date"),
        ),
        "end": _first_value(
            job.get("holdout_end"),
            job.get("evaluation_end"),
            job.get("market_comparison_date"),
            job.get("evaluation_through"),
            parameters.get("outer_through"),
            parameters.get("evaluation_through"),
            parameters.get("through_date"),
        ),
        "days": _first_value(job.get("evaluation_days"), job.get("race_days")),
        "races": _first_value(job.get("evaluated_races"), job.get("evaluation_races")),
        "folds": _first_value(job.get("fold_count"), job.get("expected_folds")),
        "decision_minutes_before": _first_value(
            parameters.get("decision_minutes_before"),
            parameters.get("minutes_before_deadline"),
            parameters.get("snapshot_minutes_before"),
        ),
        "daily_budget_yen": _first_value(
            job.get("daily_budget_yen"),
            parameters.get("daily_budget_yen"),
            parameters.get("budget_yen"),
            parameters.get("budget"),
        ),
        "odds_mode": _first_value(
            job.get("odds_mode"),
            parameters.get("odds_mode"),
            parameters.get("market_source"),
            "T-5" if parameters.get("include_odds") is True else None,
            "履歴のみ" if parameters.get("include_odds") is False else None,
        ),
        "race_set_sha256": _first_value(
            parameters.get("race_set_sha256"), job.get("race_set_sha256")
        ),
        "protocol_sha256": _first_value(
            parameters.get("protocol_sha256"), job.get("protocol_sha256")
        ),
        "policy_sha256": _first_value(
            parameters.get("policy_sha256"), job.get("policy_sha256")
        ),
        "allocation_mode": _first_value(
            job.get("allocation_mode"), parameters.get("allocation_mode")
        ),
        "profit_reinvestment": _first_value(
            job.get("profit_reinvestment"),
            parameters.get("profit_reinvestment"),
            parameters.get("reinvest_profit"),
        ),
    }


def _purpose_evaluation(job: dict[str, Any], key: str) -> dict[str, Any]:
    ranking = job.get("residual_ranking_metrics")
    if key == "ticket_value" and isinstance(ranking, dict):
        required = (
            "residual_selection",
            "ticket_ranking_hit_rate",
            "ticket_ranking_roi",
            "ticket_ranking_roi_ci95_lower",
        )
        values = {
            "residual_selection": job.get("residual_selection"),
            "ticket_ranking_hit_rate": ranking.get("hit_rate"),
            "ticket_ranking_roi": ranking.get("roi"),
            "ticket_ranking_roi_ci95_lower": ranking.get("roi_ci95_lower"),
        }
        profile = "ticket_utility_ranking"
    else:
        required = PURPOSE_REQUIREMENTS[key]
        values = job
        profile = "direct_expected_value" if key == "ticket_value" else key
    if key in {"bankroll_policy", "production_validation"} and not (
        float(job.get("stake_yen") or 0) > 0
    ):
        values = dict(values)
        for metric in (
            "roi",
            "roi_without_largest_hit",
            "daily_cluster_bootstrap_roi_lower_95",
        ):
            values[metric] = None
    available = [metric for metric in required if values.get(metric) is not None]
    missing = [metric for metric in required if metric not in available]
    backtest = _backtest_scope(job)
    missing_scope = [
        field for field in ("start", "end", "races")
        if backtest.get(field) is None
    ]
    serialized = json.dumps(
        backtest, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    )
    return {
        "required_metrics": list(required),
        "available_metrics": available,
        "missing_metrics": missing,
        "metric_profile": profile,
        "metric_count": len(available),
        "metric_total": len(required),
        "complete": not missing,
        "comparison_ready": not missing and not missing_scope,
        "missing_scope": missing_scope,
        "comparison_id": hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:10],
        "backtest": backtest,
    }


def evaluation_purpose_keys(job: dict[str, Any]) -> list[str]:
    """Return every objective evaluated by a possibly multi-head job."""
    task = str(job.get("kind") or job.get("milestone") or "").lower()
    name = str(job.get("name") or "").lower()
    parameters = (
        job.get("parameters") if isinstance(job.get("parameters"), dict) else {}
    )
    text = f"{task} {name} {parameters.get('purchase_loss', '')}".lower()
    purposes: list[str] = []
    pending = bool(job.get("running")) or str(job.get("status") or "") in {
        "実行中", "待機中"
    }

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
    ) or (pending and tokens(
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
    )):
        purposes.append("outcome_probability")
    if has(
        (
            "closing_odds_log_mae",
            "closing_odds_rank_correlation",
            "closing_odds_interval_coverage",
            "closing_snapshot_age_seconds",
        )
    ) or (
        pending
        and tokens(("closing_odds", "odds_path", "market_curvature", "four_head"))
    ):
        purposes.append("closing_odds")
    if has(
        (
            "residual_ranking_metrics",
            "purchase_value_pearson_correlation",
            "purchase_value_calibration_mae",
            "purchase_value_positive_predicted_tickets",
            "purchase_value_positive_observed_capped_roi",
            "purchase_payout_log_mae",
            "purchase_oof_scaled_payout_log_mae",
        )
    ) or (pending and (parameters.get("purchase_loss") or tokens(
        (
            "four_head",
            "conditional_payout",
            "pairwise",
            "learned_value",
            "contextual",
            "market_residual",
        )))
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
    ) or (pending and tokens(
        (
            "bankroll_policy",
            "market_residual",
            "four_head",
            "conditional_payout",
            "standardized_365d",
        )
    )):
        purposes.append("bankroll_policy")
    if (
        job.get("promotion_gate_total") is not None
        or job.get("prediction_deployment_eligible") is not None
        or tokens(("standardized_365d", "nested_annual", "promotion", "prospective"))
    ):
        purposes.append("production_validation")
    return purposes


def evaluation_purpose_groups(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = {spec[0]: [] for spec in PURPOSE_SPECS}
    status_order = {"実行中": 0, "待機中": 1, "完了": 2, "失敗": 3, "取消": 4}
    for job in jobs:
        keys = evaluation_purpose_keys(job)
        for key in keys:
            item = {
                **job,
                "purpose_keys": keys,
                "purpose_evaluation": _purpose_evaluation(job, key),
            }
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
                "required_metrics": list(PURPOSE_REQUIREMENTS[key]),
                "backtest_protocol": protocol,
                "decision_rule": PURPOSE_DECISION_RULES[key],
                "models": models,
                "active_models": sum(bool(row.get("running")) for row in models),
            }
        )
    return result
