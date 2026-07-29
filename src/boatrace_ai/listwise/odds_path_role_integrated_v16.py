from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .odds_path_role_integrated_v12 import (
    CLOSING_FALLBACK_NO_BET,
    PROSPECTIVE_OUTPUT_KEY as V12_PROSPECTIVE_OUTPUT_KEY,
    walk_forward_evaluate_v12,
)
from .odds_path_role_integrated_v15 import (
    DISCRETE_POLICY_V15,
    _aggregate_closing_envelopes,
    _closing_envelope_promotion_gate,
    _evaluation_population_hash,
    _fit_closing_envelope,
    _rename_envelope_keys,
    _v15_policy,
    append_closing_envelope_observations_v15,
    build_strict_prior_prewarm_observations_v15,
)
from .strict_prior_t300_divergence_passthrough_v16 import (
    METHOD as PASSTHROUGH_METHOD,
    MODEL_NAME as PASSTHROUGH_MODEL_NAME,
    REGISTERED_DIVERGENCE_LOWER,
    REGISTERED_DIVERGENCE_UPPER,
    fit_strict_prior_t300_divergence_passthrough_v16,
)
from .v16_fixed_band_ranking_diagnostics import (
    aggregate_v16_fixed_band_ranking_diagnostics,
    compare_v16_fixed_band_ranking_rules,
)


MODEL_NAME = "odds_path_role_integrated_fixed_band_passthrough_v16"
STRATEGY_NAME = MODEL_NAME
REGISTERED_AFTER = "2026-07-29"
PROSPECTIVE_OUTPUT_KEY = "prospective_role_integrated_v16_walk_forward"
COMPARISON_ROLE = "fixed_t300_divergence_raw_probability_v15_envelope_shadow"

DISCRETE_POLICY_V16: dict[str, Any] = {
    **DISCRETE_POLICY_V15,
    "name": "v16_fixed_t300_divergence_raw_probability_v15_envelope",
    "probability_artifact": PASSTHROUGH_MODEL_NAME,
    "probability_artifact_method": PASSTHROUGH_METHOD,
    "registered_after": REGISTERED_AFTER,
    "registered_divergence_definition": (
        "log(model_probability / normalized_T300_market_probability)"
    ),
    "registered_divergence_lower_inclusive": REGISTERED_DIVERGENCE_LOWER,
    "registered_divergence_upper_exclusive": REGISTERED_DIVERGENCE_UPPER,
    "conditional_lcb": False,
    "raw_model_probability_inside_fixed_band": True,
    "real_betting_enabled": False,
}


def _v16_policy(policy: Mapping[str, Any] | None) -> dict[str, Any]:
    normalized = _v15_policy(policy)
    if normalized.get("no_bet"):
        return normalized
    return {**normalized, **DISCRETE_POLICY_V16}


def walk_forward_evaluate_v16(
    races: list[dict[str, Any]],
    *,
    daily_budget_yen: int,
    min_calibration_days: int,
    evaluation_dates: Iterable[str] | None = None,
    closing_fallback_policy: str = CLOSING_FALLBACK_NO_BET,
) -> dict[str, Any]:
    """Evaluate fixed T300 divergence selection with the V15 envelope stack."""
    if closing_fallback_policy != CLOSING_FALLBACK_NO_BET:
        raise ValueError("V16 requires closing_fallback_policy='no_bet'")

    prewarm_observations = build_strict_prior_prewarm_observations_v15(
        races, min_calibration_days=min_calibration_days
    )
    result = walk_forward_evaluate_v12(
        races,
        daily_budget_yen=daily_budget_yen,
        min_calibration_days=min_calibration_days,
        evaluation_dates=evaluation_dates,
        closing_fallback_policy=closing_fallback_policy,
        closing_forecast_field="point_final_odds",
        probability_lcb_fit=(
            fit_strict_prior_t300_divergence_passthrough_v16
        ),
        probability_lcb_metrics=compare_v16_fixed_band_ranking_rules,
        selection_conformal_fit=_fit_closing_envelope,
        selection_observation_append=append_closing_envelope_observations_v15,
        initial_selection_observations=prewarm_observations,
    )

    folds = [
        _rename_envelope_keys(fold) for fold in list(result.get("folds") or [])
    ]
    for fold in folds:
        fold["selected_policy"] = _v16_policy(fold.get("selected_policy"))
        guard = dict(fold.get("leakage_guard") or {})
        guard.update({
            "candidate_filter": "fixed_t300_log_divergence_[0.5,1.0)",
            "probability_inside_fixed_band": "raw_model_probability",
            "conditional_probability_lcb_used": False,
            "probability_artifact_uses_result": False,
            "probability_artifact_uses_payout": False,
            "closing_envelope_population": "all_120_complete_combinations",
            "closing_envelope_selection_free": True,
            "closing_teacher_appended_after_purchase_decision": True,
            "result_payout_in_purchase_features": False,
            "missing_real_t300_action": "no_bet",
            "real_betting_enabled": False,
        })
        fold["leakage_guard"] = guard

    prospective = _rename_envelope_keys(
        dict(result.pop(V12_PROSPECTIVE_OUTPUT_KEY, {}) or {})
    )
    prospective_folds = [
        fold
        for fold in folds
        if str(fold.get("evaluation_date") or "") > REGISTERED_AFTER
    ]
    prospective_envelope = _aggregate_closing_envelopes(prospective_folds)
    prospective["closing_envelope_conformal"] = prospective_envelope
    prospective_gate = dict(prospective.get("promotion_gate") or {})
    prospective_gate.update(
        _closing_envelope_promotion_gate(prospective_envelope)
    )
    prospective_gate["fixed_divergence_filter_pass"] = True
    prospective_gate["raw_probability_passthrough_pass"] = True
    prospective_checks = [
        bool(value)
        for key, value in prospective_gate.items()
        if key.endswith("_pass")
    ]
    prospective["promotion_gate"] = prospective_gate
    prospective["promotion_eligible"] = (
        bool(prospective_checks) and all(prospective_checks)
    )
    prospective["comparison_role"] = (
        "pre_registered_fixed_t300_band_v15_envelope_v16_shadow"
    )
    prospective["real_betting_enabled"] = False

    deployment = _rename_envelope_keys(
        dict(result.get("deployment_configuration") or {})
    )
    deployment.update({
        "calibrator_strategy": STRATEGY_NAME,
        "candidate_policy": dict(DISCRETE_POLICY_V16),
        "selected_policy": {"name": "no_bet", "no_bet": True},
        "operational_status": "shadow_only_real_betting_disabled",
        "real_betting_enabled": False,
        "missing_real_t300_action": "no_bet",
    })

    artifacts = result.pop("selection_conformal_artifacts_by_date", {})
    result.pop("selection_conformal", None)
    summary = _aggregate_closing_envelopes(folds)
    fixed_band_ranking_diagnostics = (
        aggregate_v16_fixed_band_ranking_diagnostics(folds)
    )
    result.update({
        "model": MODEL_NAME,
        "calibrator_strategy": STRATEGY_NAME,
        "comparison_role": COMPARISON_ROLE,
        "calibration_input_scope": (
            "all_eligible_races_including_partial_market_days"
        ),
        "evaluation_date_scope": "formal_complete_market_days_only",
        "validation_design": (
            "Strict-prior outer-day V8 probability calibration trained on all "
            "eligible prior races, including T300-complete races from partial "
            "market days; holdout evaluation restricted to supplied formal "
            "complete-market dates; fixed T300 "
            "log-divergence [0.5,1.0) candidate filter with raw probability "
            "passthrough; V15 selection-free 120-way closing envelope; "
            "zero-bet-capable discrete-log allocation"
        ),
        "registered_after": REGISTERED_AFTER,
        "fixed_policy": _v16_policy(result.get("fixed_policy")),
        "probability_artifact": {
            "model_name": PASSTHROUGH_MODEL_NAME,
            "method": PASSTHROUGH_METHOD,
            "uses_result": False,
            "uses_payout": False,
        },
        "closing_envelope_conformal": summary,
        "fixed_band_ranking_diagnostics": fixed_band_ranking_diagnostics,
        "closing_envelope_conformal_artifacts_by_date": artifacts,
        "selection_free_closing_envelope": True,
        "evaluation_population_hash": _evaluation_population_hash(races),
        "evaluation_population_races": len(races),
        "zero_bet_allowed": True,
        "missing_real_t300_action": "no_bet",
        "real_betting_enabled": False,
        "folds": folds,
        PROSPECTIVE_OUTPUT_KEY: prospective,
        "promotion_gate": prospective_gate,
        "promotion_eligible": prospective["promotion_eligible"],
        "deployment_configuration": deployment,
    })
    return result
