from __future__ import annotations

import hashlib
import json

from collections.abc import Iterable, Mapping
from typing import Any

from .closing_envelope_conformal_v15 import (
    METHOD as CLOSING_ENVELOPE_METHOD,
    MODEL_NAME as CLOSING_ENVELOPE_MODEL_NAME,
    fit_closing_envelope_conformal_v15,
)
from .closing_odds_multihorizon_v11 import select_teacher_final_odds
from .odds_path_role_integrated_v12 import (
    CLOSING_FALLBACK_NO_BET,
    DISCRETE_POLICY_V12,
    PROSPECTIVE_OUTPUT_KEY as V12_PROSPECTIVE_OUTPUT_KEY,
    walk_forward_evaluate_v12,
)


MODEL_NAME = "odds_path_role_integrated_selection_free_envelope_v15"
STRATEGY_NAME = MODEL_NAME
REGISTERED_AFTER = "2026-07-29"
PROSPECTIVE_OUTPUT_KEY = "prospective_role_integrated_v15_walk_forward"
COMPARISON_ROLE = "selection_free_closing_envelope_t300_v15_shadow"

DISCRETE_POLICY_V15: dict[str, Any] = {
    **{
        key: value
        for key, value in DISCRETE_POLICY_V12.items()
        if not key.startswith("selection_conformal")
    },
    "name": "v15_selection_free_closing_envelope_t300_discrete_log",
    "closing_envelope_method": CLOSING_ENVELOPE_METHOD,
    "closing_envelope_population": "all_120_complete_combinations_per_race",
    "selection_free": True,
    "zero_bet_allowed": True,
    "missing_real_t300_action": "no_bet",
    "real_betting_enabled": False,
}


def _evaluation_population_hash(races: Iterable[Mapping[str, Any]]) -> str:
    keys = sorted(
        [str(race.get("race_date") or ""), str(race.get("race_id") or "")]
        for race in races
    )
    payload = json.dumps(keys, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _odds_mapping(values: object) -> dict[str, object]:
    if not isinstance(values, Mapping):
        return {}
    return {str(combination): value for combination, value in values.items()}


def append_closing_envelope_observations_v15(
    observations: list[dict[str, Any]],
    races: list[dict[str, Any]],
    *,
    closing_forecasts: dict[str, dict[str, float]],
    probability_lcb: dict[str, Any],
    evaluation_date: str,
) -> int:
    """Append complete 120-way closing teachers after purchase is frozen."""
    del probability_lcb
    appended = 0
    for race in races:
        race_id = str(race.get("race_id") or "")
        predicted = _odds_mapping(closing_forecasts.get(race_id))
        actual_raw, _source = select_teacher_final_odds(race)
        actual = _odds_mapping(actual_raw)
        observations.append({
            "race_date": str(evaluation_date),
            "race_id": race_id,
            "predicted_closing_odds": predicted,
            "actual_closing_odds": actual,
            "teacher_population": "all_120_complete_combinations",
            "teacher_appended_after_purchase_decision": True,
        })
        appended += 1
    return appended


def _fit_closing_envelope(
    observations: Iterable[Mapping[str, Any]], *, evaluation_date: str
) -> dict[str, Any]:
    return fit_closing_envelope_conformal_v15(
        observations,
        evaluation_date=evaluation_date,
    )


def _v15_policy(policy: Mapping[str, Any] | None) -> dict[str, Any]:
    source = {
        key: value
        for key, value in dict(policy or {}).items()
        if not key.startswith("selection_conformal")
    }
    if source.get("no_bet"):
        return {
            **source,
            "zero_bet_allowed": True,
            "real_betting_enabled": False,
        }
    return {**source, **DISCRETE_POLICY_V15}


_V15_KEY_RENAMES = {
    "selection_conformal": "closing_envelope_conformal",
    "selection_conformal_artifacts_by_date": (
        "closing_envelope_conformal_artifacts_by_date"
    ),
    "selection_conformal_trained_through": "closing_envelope_trained_through",
    "selection_observations_appended_after_decision": (
        "closing_envelope_races_appended_after_decision"
    ),
}


def _rename_envelope_keys(value: Any) -> Any:
    if isinstance(value, list):
        return [_rename_envelope_keys(item) for item in value]
    if not isinstance(value, Mapping):
        return value
    normalized: dict[str, Any] = {}
    for raw_key, item in value.items():
        key = _V15_KEY_RENAMES.get(str(raw_key), str(raw_key))
        normalized[key] = _rename_envelope_keys(item)
    return normalized


def _aggregate_closing_envelopes(
    folds: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    artifacts = [
        dict(fold.get("closing_envelope_conformal") or {})
        for fold in folds
    ]
    ready = [artifact for artifact in artifacts if artifact.get("ready")]
    haircuts = [float(artifact["haircut"]) for artifact in ready]
    latest = artifacts[-1] if artifacts else {}
    missing_audits = [
        artifact.get("missing_audit") or {}
        for artifact in artifacts
    ]
    return {
        "model_name": CLOSING_ENVELOPE_MODEL_NAME,
        "method": CLOSING_ENVELOPE_METHOD,
        "selection_free": True,
        "evaluation_folds": len(artifacts),
        "ready_folds": len(ready),
        "not_ready_folds": len(artifacts) - len(ready),
        "training_days_latest": int(latest.get("training_days") or 0),
        "training_races_latest": int(latest.get("training_races") or 0),
        "training_observations_latest": int(
            latest.get("training_observations") or 0
        ),
        "missing_audit_input_races": sum(
            int(audit.get("input_races") or 0) for audit in missing_audits
        ),
        "missing_audit_accepted_races": sum(
            int(audit.get("accepted_races") or 0) for audit in missing_audits
        ),
        "missing_audit_rejected_races": sum(
            int(audit.get("rejected_races") or 0) for audit in missing_audits
        ),
        "trained_through_date_latest": latest.get("trained_through_date"),
        "haircut_latest": latest.get("haircut"),
        "haircut_min": min(haircuts) if haircuts else None,
        "haircut_max": max(haircuts) if haircuts else None,
    }


def _closing_envelope_promotion_gate(
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    evaluation_folds = int(summary.get("evaluation_folds") or 0)
    ready_folds = int(summary.get("ready_folds") or 0)
    input_races = int(summary.get("missing_audit_input_races") or 0)
    rejected_races = int(summary.get("missing_audit_rejected_races") or 0)
    return {
        "closing_envelope_evaluation_folds": evaluation_folds,
        "closing_envelope_ready_folds": ready_folds,
        "closing_envelope_input_races": input_races,
        "closing_envelope_rejected_races": rejected_races,
        "closing_envelope_ready_pass": (
            evaluation_folds > 0 and ready_folds == evaluation_folds
        ),
        "closing_envelope_no_missing_races_pass": (
            input_races > 0 and rejected_races == 0
        ),
    }


def walk_forward_evaluate_v15(
    races: list[dict[str, Any]],
    *,
    daily_budget_yen: int,
    min_calibration_days: int,
    evaluation_dates: Iterable[str] | None = None,
    closing_fallback_policy: str = CLOSING_FALLBACK_NO_BET,
) -> dict[str, Any]:
    """Evaluate the selection-free closing-envelope V15 shadow stack.

    V15 retains V12's continuous V8 probability calibration, real-T300 closing
    forecast, and discrete-log allocation. It changes the closing uncertainty
    teacher to every complete 120-way race and never enables real betting.
    """
    if closing_fallback_policy != CLOSING_FALLBACK_NO_BET:
        raise ValueError("V15 requires closing_fallback_policy='no_bet'")
    evaluation_population_hash = _evaluation_population_hash(races)
    result = walk_forward_evaluate_v12(
        races,
        daily_budget_yen=daily_budget_yen,
        min_calibration_days=min_calibration_days,
        evaluation_dates=evaluation_dates,
        closing_fallback_policy=closing_fallback_policy,
        closing_forecast_field="point_final_odds",
        selection_conformal_fit=_fit_closing_envelope,
        selection_observation_append=append_closing_envelope_observations_v15,
    )

    folds = [
        _rename_envelope_keys(fold) for fold in list(result.get("folds") or [])
    ]
    for fold in folds:
        fold["selected_policy"] = _v15_policy(fold.get("selected_policy"))
        guard = dict(fold.get("leakage_guard") or {})
        guard.update({
            "closing_envelope_population": "all_120_complete_combinations",
            "closing_envelope_selection_free": True,
            "closing_teacher_appended_after_purchase_decision": True,
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
        "pre_registered_strict_outer_day_selection_free_v15_shadow"
    )
    prospective["real_betting_enabled"] = False

    deployment = _rename_envelope_keys(
        dict(result.get("deployment_configuration") or {})
    )
    deployment.update({
        "calibrator_strategy": STRATEGY_NAME,
        "candidate_policy": dict(DISCRETE_POLICY_V15),
        "selected_policy": {"name": "no_bet", "no_bet": True},
        "operational_status": "shadow_only_real_betting_disabled",
        "real_betting_enabled": False,
        "missing_real_t300_action": "no_bet",
    })

    artifacts = result.pop("selection_conformal_artifacts_by_date", {})
    result.pop("selection_conformal", None)
    summary = _aggregate_closing_envelopes(folds)
    fixed_policy = _v15_policy(result.get("fixed_policy"))
    result.update({
        "model": MODEL_NAME,
        "calibrator_strategy": STRATEGY_NAME,
        "comparison_role": COMPARISON_ROLE,
        "validation_design": (
            "Strict-prior outer-day V8 probability calibration, real T300 V12 "
            "closing forecast, selection-free whole-race 120-way daily q20 "
            "closing envelope, then zero-bet-capable discrete-log allocation"
        ),
        "registered_after": REGISTERED_AFTER,
        "fixed_policy": fixed_policy,
        "closing_envelope_conformal": summary,
        "closing_envelope_conformal_artifacts_by_date": artifacts,
        "selection_free": True,
        "evaluation_population_hash": evaluation_population_hash,
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
