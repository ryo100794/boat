from __future__ import annotations

from collections.abc import Mapping
from typing import Any


PROBABILITY_HEAD_FIELDS = (
    "model_winner_log_loss",
    "model_winner_top1_accuracy",
    "model_trifecta_log_loss",
    "model_trifecta_top5_hit_rate",
    "market_winner_log_loss",
    "market_winner_top1_accuracy",
    "market_trifecta_log_loss",
    "market_trifecta_top5_hit_rate",
    "calibrated_winner_log_loss",
    "calibrated_winner_top1_accuracy",
    "calibrated_trifecta_log_loss",
    "calibrated_trifecta_top5_hit_rate",
)


PROMOTION_GATE_BOOLEAN_FIELDS = (
    "sample_size_pass",
    "effective_hit_count_pass",
    "calibration_pass",
    "market_confidence_pass",
)


MARKET_COMPARISON_FIELDS = (
    "market_comparison_races",
    "market_comparison_days",
    "market_log_loss_delta",
    "market_log_loss_delta_ci95_lower",
    "market_log_loss_delta_ci95_upper",
    "market_improvement_probability",
    "market_top5_delta",
    "market_top5_delta_ci95_lower",
    "market_top5_delta_ci95_upper",
    "market_day_log_loss_delta_ci95_lower",
    "market_day_log_loss_delta_ci95_upper",
    "market_day_top5_delta_ci95_lower",
    "market_day_top5_delta_ci95_upper",
    "market_race_confidence_pass",
    "market_day_confidence_pass",
    "market_confidence_pass",
)


def market_comparison_fields(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {}
    loss = payload.get("log_loss_difference_calibrated_minus_market") or {}
    top5 = payload.get("top5_hit_difference_calibrated_minus_market") or {}
    day_loss = payload.get(
        "day_cluster_log_loss_difference_calibrated_minus_market"
    ) or {}
    day_top5 = payload.get(
        "day_cluster_top5_hit_difference_calibrated_minus_market"
    ) or {}
    if not isinstance(loss, Mapping) or not loss.get("observations"):
        return {}
    return {
        "market_comparison_races": loss.get("observations"),
        "market_comparison_days": day_loss.get("clusters"),
        "market_log_loss_delta": loss.get("mean_difference"),
        "market_log_loss_delta_ci95_lower": loss.get("ci95_lower"),
        "market_log_loss_delta_ci95_upper": loss.get("ci95_upper"),
        "market_improvement_probability": loss.get(
            "probability_less_than_zero"
        ),
        "market_top5_delta": top5.get("mean_difference"),
        "market_top5_delta_ci95_lower": top5.get("ci95_lower"),
        "market_top5_delta_ci95_upper": top5.get("ci95_upper"),
        "market_day_log_loss_delta_ci95_lower": day_loss.get("ci95_lower"),
        "market_day_log_loss_delta_ci95_upper": day_loss.get("ci95_upper"),
        "market_day_top5_delta_ci95_lower": day_top5.get("ci95_lower"),
        "market_day_top5_delta_ci95_upper": day_top5.get("ci95_upper"),
        "market_race_confidence_pass": bool(
            payload.get("race_level_confidence_pass")
        ),
        "market_day_confidence_pass": bool(
            payload.get("day_cluster_confidence_pass")
        ),
        "market_confidence_pass": bool(payload.get("confidence_pass")),
    }


def canonicalize_probability_metrics(
    values: Mapping[str, Any],
    *,
    probability_metrics: Mapping[str, Any] | None = None,
    market_comparison: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result = dict(values)
    probability = (
        probability_metrics
        if probability_metrics is not None
        else values.get("probability_metrics")
    )
    probability = probability if isinstance(probability, Mapping) else {}
    for key in PROBABILITY_HEAD_FIELDS:
        value = probability.get(key)
        if result.get(key) is None and value is not None:
            result[key] = value

    def first(*keys: str) -> Any:
        for key in keys:
            value = result.get(key)
            if value is not None:
                return value
        return None

    headlines = {
        "winner_log_loss": first(
            "winner_log_loss",
            "calibrated_winner_log_loss",
            "model_winner_log_loss",
        ),
        "winner_top1_accuracy": first(
            "winner_top1_accuracy",
            "calibrated_winner_top1_accuracy",
            "model_winner_top1_accuracy",
        ),
        "calibrated_trifecta_log_loss": first(
            "calibrated_trifecta_log_loss",
        ),
        "trifecta_top5_hit_rate": first(
            "trifecta_top5_hit_rate",
            "calibrated_trifecta_top5_hit_rate",
            "model_trifecta_top5_hit_rate",
        ),
    }
    result.update({key: value for key, value in headlines.items() if value is not None})

    comparison = (
        market_comparison
        if market_comparison is not None
        else values.get("market_comparison")
    )
    if isinstance(comparison, Mapping):
        result["market_comparison"] = dict(comparison)
        for key, value in market_comparison_fields(comparison).items():
            if result.get(key) is None:
                result[key] = value

    explicit_gate = result.get("promotion_gate")
    if isinstance(explicit_gate, Mapping):
        for key in PROMOTION_GATE_BOOLEAN_FIELDS:
            value = explicit_gate.get(key)
            if result.get(key) is None and isinstance(value, bool):
                result[key] = value
    failed = {
        str(key) for key in (result.get("promotion_gate_failed") or [])
    }
    if failed.intersection(PROMOTION_GATE_BOOLEAN_FIELDS):
        for key in PROMOTION_GATE_BOOLEAN_FIELDS:
            if result.get(key) is None:
                result[key] = key not in failed
    return result
