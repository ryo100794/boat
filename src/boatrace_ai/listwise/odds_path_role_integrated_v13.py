from __future__ import annotations

from typing import Any, Iterable, Mapping

from .edge_conditional_probability_lcb_v13 import (
    METHOD as PROBABILITY_LCB_METHOD,
    aggregate_conditional_calibration_metrics,
    conditional_calibration_metrics,
    fit_edge_conditional_probability_lcb,
)
from .odds_path_role_integrated_v12 import (
    CLOSING_FALLBACK_V11,
    DISCRETE_POLICY_V12,
    PROSPECTIVE_OUTPUT_KEY as V12_PROSPECTIVE_OUTPUT_KEY,
    walk_forward_evaluate_v12,
)
from .strict_prior_divergence_diagnostics import (
    aggregate_strict_prior_divergence_band_metrics,
    strict_prior_divergence_band_metrics,
)


MODEL_NAME = "odds_path_role_integrated_edge_conditional_lcb_v13"
STRATEGY_NAME = MODEL_NAME
REGISTERED_AFTER = "2026-07-29"
PROSPECTIVE_OUTPUT_KEY = "prospective_role_integrated_v13_walk_forward"
COMPARISON_ROLE = "v12_closing_with_edge_conditional_probability_lcb_v13_shadow"

DISCRETE_POLICY_V13: dict[str, Any] = {
    **DISCRETE_POLICY_V12,
    "name": "v13_edge_conditional_probability_lcb_v12_closing_discrete_log",
    "probability_lcb_method": PROBABILITY_LCB_METHOD,
    "probability_lcb_conditions": (
        "probability_rank_x_probability_band_x_"
        "log_model_probability_over_normalized_t300_market_probability_band"
    ),
    "probability_lcb_bootstrap_unit": "whole_race_day",
    "probability_lcb_sparse_fallback": "conditional_to_rank_to_global",
    "closing_probability_correction_independent": True,
}


def _settlement_probability_metrics(
    races: list[dict[str, Any]],
    *,
    closing_forecasts: Mapping[str, Mapping[str, float]],
    probability_lcb: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "conditional_calibration": conditional_calibration_metrics(
            races,
            closing_forecasts=closing_forecasts,
            probability_lcb=probability_lcb,
        ),
        "strict_prior_divergence_bands": (
            strict_prior_divergence_band_metrics(races)
        ),
        "result_and_payout_usage": "post_purchase_settlement_metrics_only",
    }


def _aggregate_fold_metrics(folds: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = [
        fold.get("probability_lcb_metrics") or {}
        for fold in folds
        if isinstance(fold, Mapping)
    ]
    return {
        "conditional_calibration": aggregate_conditional_calibration_metrics(
            row.get("conditional_calibration") or {}
            for row in metrics
        ),
        "strict_prior_divergence_bands": (
            aggregate_strict_prior_divergence_band_metrics(
                row.get("strict_prior_divergence_bands") or {}
                for row in metrics
            )
        ),
    }


_JSON_AUDIT_KEYS = (
    "model_name", "model_type", "ready", "challenger_adopted",
    "selection_reason", "trained_through_date", "prediction_date",
    "training_days", "training_races", "engine",
)


def _artifact_audit(value: object) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    return {key: value.get(key) for key in _JSON_AUDIT_KEYS if key in value}


def _remove_closing_estimators(container: dict[str, Any]) -> dict[str, Any]:
    audits: dict[str, Any] = {}
    for key in (
        "closing_model", "closing_v12_model", "closing_v11_fallback_model",
        "closing_t300_v12_model",
    ):
        value = container.pop(key, None)
        audit = _artifact_audit(value)
        if audit is not None:
            audits[key] = audit
    if audits:
        container["closing_model_artifact_audit"] = audits
    return container


def _conditional_calibration_gate(summary: Mapping[str, Any]) -> dict[str, Any]:
    candidates = int(summary.get("candidate_count") or 0)
    coverage = summary.get("daily_lower_bound_coverage")
    reduction = float(summary.get("overprediction_reduction_hits") or 0.0)
    return {
        "conditional_calibration_minimum_candidates": 30,
        "conditional_calibration_candidate_count": candidates,
        "conditional_calibration_candidate_count_pass": candidates >= 30,
        "conditional_calibration_coverage_target": 0.80,
        "conditional_calibration_coverage": coverage,
        "conditional_calibration_coverage_pass": (
            coverage is not None and float(coverage) >= 0.80
        ),
        "conditional_overprediction_reduction_hits": reduction,
        "conditional_overprediction_improvement_pass": reduction > 0.0,
    }


def _v13_policy(policy: Mapping[str, Any] | None) -> dict[str, Any]:
    source = dict(policy or {})
    if source.get("no_bet"):
        return source
    return {**source, **DISCRETE_POLICY_V13}


def walk_forward_evaluate_v13(
    races: list[dict[str, Any]],
    *,
    daily_budget_yen: int,
    min_calibration_days: int,
    evaluation_dates: Iterable[str] | None = None,
    closing_fallback_policy: str = CLOSING_FALLBACK_V11,
) -> dict[str, Any]:
    """Evaluate a V12 closing stack with an independent conditional p-LCB."""
    result = walk_forward_evaluate_v12(
        races,
        daily_budget_yen=daily_budget_yen,
        min_calibration_days=min_calibration_days,
        evaluation_dates=evaluation_dates,
        closing_fallback_policy=closing_fallback_policy,
        probability_lcb_fit=fit_edge_conditional_probability_lcb,
        probability_lcb_metrics=_settlement_probability_metrics,
    )
    folds = list(result.get("folds") or [])
    aggregate = _aggregate_fold_metrics(folds)
    for fold in folds:
        _remove_closing_estimators(fold)
        fold["selected_policy"] = _v13_policy(fold.get("selected_policy"))
        guard = fold.get("leakage_guard") or {}
        guard.update({
            "probability_lcb_crossfit_unit": "whole_race_day",
            "probability_lcb_conditions": (
                "rank_probability_band_t300_model_market_divergence"
            ),
            "result_payout_in_purchase_features": False,
            "closing_odds_in_probability_lcb_features": False,
        })
        fold["leakage_guard"] = guard
    prospective = dict(result.pop(V12_PROSPECTIVE_OUTPUT_KEY, {}) or {})
    prospective_dates = {
        str(fold.get("evaluation_date"))
        for fold in folds
        if str(fold.get("evaluation_date")) > REGISTERED_AFTER
    }
    prospective_folds = [
        fold
        for fold in folds
        if str(fold.get("evaluation_date")) in prospective_dates
    ]
    prospective.update(_aggregate_fold_metrics(prospective_folds))
    prospective["comparison_role"] = (
        "pre_registered_strict_outer_day_edge_conditional_lcb_v13_shadow"
    )
    prospective_gate = dict(prospective.get("promotion_gate") or {})
    prospective_gate.update(_conditional_calibration_gate(
        prospective["conditional_calibration"]
    ))
    prospective_checks = [
        value for key, value in prospective_gate.items() if key.endswith("_pass")
    ]
    prospective["promotion_gate"] = prospective_gate
    prospective["promotion_eligible"] = (
        bool(prospective_checks) and all(prospective_checks)
    )
    deployment = dict(result.get("deployment_configuration") or {})
    _remove_closing_estimators(deployment)
    deployment.update({
        "calibrator_strategy": STRATEGY_NAME,
        "candidate_policy": dict(DISCRETE_POLICY_V13),
        "probability_lcb_role": "edge_conditional_only",
        "closing_lower_bound_role": "unchanged_v12_or_explicit_v11_fallback",
        "selected_policy": {"name": "no_bet", "no_bet": True},
        "operational_status": "shadow_only_until_v13_promotion_gate",
    })
    result.update({
        "model": MODEL_NAME,
        "calibrator_strategy": STRATEGY_NAME,
        "comparison_role": COMPARISON_ROLE,
        "validation_design": (
            "V12 closing lower bound is unchanged. Probability v8 is fit on "
            "strict prior days; its whole-day crossfit predictions fit a daily-"
            "cluster hierarchical LCB conditional on rank, probability band and "
            "model/normalized-T300-market divergence. Purchase precedes all result, "
            "payout and final-closing-odds settlement metrics."
        ),
        "registered_after": REGISTERED_AFTER,
        "fixed_policy": dict(DISCRETE_POLICY_V13),
        "folds": folds,
        "edge_conditional_probability_calibration": aggregate[
            "conditional_calibration"
        ],
        "strict_prior_divergence_bands": aggregate[
            "strict_prior_divergence_bands"
        ],
        "promotion_gate": prospective_gate,
        "promotion_eligible": prospective["promotion_eligible"],
        PROSPECTIVE_OUTPUT_KEY: prospective,
        "deployment_configuration": deployment,
    })
    return result
