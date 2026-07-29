from __future__ import annotations

import hashlib
import json

from collections.abc import Iterable, Mapping
from typing import Any

from .closing_envelope_conformal_v15 import (
    METHOD as CLOSING_ENVELOPE_METHOD,
    MODEL_NAME as CLOSING_ENVELOPE_MODEL_NAME,
    evaluate_closing_envelope_holdout_v15,
    fit_closing_envelope_conformal_v15,
)
from .closing_odds_multihorizon_v11 import (
    normalize_labeled_checkpoints,
    select_teacher_final_odds,
)
from .odds_path_role_integrated_v12 import (
    CLOSING_FALLBACK_NO_BET,
    DECISION_CHECKPOINT,
    DECISION_OFFSET_SECONDS,
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


def _holdout_auditing_observation_append_v15(
    holdout_by_date: dict[str, dict[str, Any]],
):
    """Wrap post-decision teacher append with strict holdout coverage audit."""
    def append(
        observations: list[dict[str, Any]],
        races: list[dict[str, Any]],
        *,
        closing_forecasts: dict[str, dict[str, float]],
        probability_lcb: dict[str, Any],
        evaluation_date: str,
    ) -> int:
        artifact = _fit_closing_envelope(
            observations, evaluation_date=evaluation_date
        )
        start = len(observations)
        appended = append_closing_envelope_observations_v15(
            observations,
            races,
            closing_forecasts=closing_forecasts,
            probability_lcb=probability_lcb,
            evaluation_date=evaluation_date,
        )
        holdout_by_date[evaluation_date] = (
            evaluate_closing_envelope_holdout_v15(
                observations[start:],
                artifact=artifact,
                evaluation_date=evaluation_date,
            )
        )
        return appended

    return append


def build_strict_prior_prewarm_observations_v15(
    races: Iterable[Mapping[str, Any]], *, min_calibration_days: int
) -> list[dict[str, Any]]:
    """Build selection-free T300 baseline teachers for calibration-only days."""
    if min_calibration_days < 0:
        raise ValueError("min_calibration_days must not be negative")
    materialized = [dict(race) for race in races]
    prewarm_dates = set(sorted({
        str(race.get("race_date") or "") for race in materialized
    })[:min_calibration_days])
    observations: list[dict[str, Any]] = []
    for race in sorted(
        (
            row
            for row in materialized
            if str(row.get("race_date") or "") in prewarm_dates
        ),
        key=lambda row: (
            str(row.get("race_date") or ""),
            str(row.get("race_id") or ""),
        ),
    ):
        checkpoints = normalize_labeled_checkpoints(
            race, as_of_offset_seconds=DECISION_OFFSET_SECONDS
        )
        snapshot = checkpoints.get(DECISION_CHECKPOINT) or {}
        predicted = _odds_mapping(snapshot.get("odds"))
        actual_raw, _source = select_teacher_final_odds(race)
        observations.append({
            "race_date": str(race.get("race_date") or ""),
            "race_id": str(race.get("race_id") or ""),
            "predicted_closing_odds": predicted,
            "actual_closing_odds": _odds_mapping(actual_raw),
            "teacher_population": "all_120_complete_combinations",
            "teacher_source": "strict_prior_t300_current_odds_baseline",
            "teacher_appended_after_purchase_decision": False,
            "strict_prior_prewarm": True,
        })
    return observations


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


def _aggregate_holdout_coverage_v15(
    folds: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    holdouts = [
        dict(fold.get("closing_envelope_holdout_coverage") or {})
        for fold in folds
    ]
    targets = {
        float(item["target_coverage"])
        for item in holdouts
        if item.get("target_coverage") is not None
    }
    evaluated = sum(
        int(item.get("evaluated_observations") or 0) for item in holdouts
    )
    covered = sum(
        int(item.get("covered_observations") or 0) for item in holdouts
    )
    return {
        "holdout_evaluation_folds": len(holdouts),
        "holdout_complete_folds": sum(
            bool(item.get("complete")) for item in holdouts
        ),
        "holdout_input_races": sum(
            int(item.get("input_races") or 0) for item in holdouts
        ),
        "holdout_accepted_races": sum(
            int(item.get("accepted_races") or 0) for item in holdouts
        ),
        "holdout_rejected_races": sum(
            int(item.get("rejected_races") or 0) for item in holdouts
        ),
        "holdout_expected_observations": sum(
            int(item.get("expected_observations") or 0) for item in holdouts
        ),
        "holdout_evaluated_observations": evaluated,
        "holdout_missing_observations": sum(
            int(item.get("missing_observations") or 0) for item in holdouts
        ),
        "holdout_covered_observations": covered,
        "holdout_coverage": covered / evaluated if evaluated else None,
        "holdout_target_coverage": (
            next(iter(targets)) if len(targets) == 1 else None
        ),
        "holdout_target_consistent": len(targets) == 1,
        "holdout_actual_closing_odds_role": (
            "evaluation_only_after_purchase_decision"
        ),
        "holdout_result_used_for_decision": False,
        "holdout_payout_used_for_decision": False,
    }


def _aggregate_closing_envelopes(
    folds: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    folds = list(folds)
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
        **_aggregate_holdout_coverage_v15(folds),
    }


def _closing_envelope_promotion_gate(
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    evaluation_folds = int(summary.get("evaluation_folds") or 0)
    ready_folds = int(summary.get("ready_folds") or 0)
    input_races = int(summary.get("missing_audit_input_races") or 0)
    rejected_races = int(summary.get("missing_audit_rejected_races") or 0)
    holdout_folds = int(summary.get("holdout_evaluation_folds") or 0)
    holdout_complete_folds = int(summary.get("holdout_complete_folds") or 0)
    holdout_races = int(summary.get("holdout_input_races") or 0)
    holdout_rejected = int(summary.get("holdout_rejected_races") or 0)
    holdout_coverage = summary.get("holdout_coverage")
    holdout_target = summary.get("holdout_target_coverage")
    return {
        "closing_envelope_evaluation_folds": evaluation_folds,
        "closing_envelope_ready_folds": ready_folds,
        "closing_envelope_input_races": input_races,
        "closing_envelope_rejected_races": rejected_races,
        "closing_envelope_holdout_folds": holdout_folds,
        "closing_envelope_holdout_races": holdout_races,
        "closing_envelope_holdout_rejected_races": holdout_rejected,
        "closing_envelope_holdout_coverage": holdout_coverage,
        "closing_envelope_holdout_target_coverage": holdout_target,
        "closing_envelope_ready_pass": (
            evaluation_folds > 0 and ready_folds == evaluation_folds
        ),
        "closing_envelope_no_missing_races_pass": (
            input_races > 0 and rejected_races == 0
        ),
        "closing_envelope_holdout_complete_pass": (
            holdout_folds > 0
            and holdout_complete_folds == holdout_folds
            and holdout_races > 0
            and holdout_rejected == 0
        ),
        "closing_envelope_holdout_coverage_pass": (
            bool(summary.get("holdout_target_consistent"))
            and holdout_coverage is not None
            and holdout_target is not None
            and float(holdout_coverage) >= float(holdout_target)
        ),
    }



_LEGACY_V12_COVERAGE_GATE_KEYS = frozenset({
    "selection_conditional_coverage_pass",
    "selection_conditional_complete_pass",
    "quantile_coverage_pass",
})


def _replace_legacy_coverage_gate_v15(
    gate: Mapping[str, Any], envelope_summary: Mapping[str, Any]
) -> dict[str, Any]:
    normalized = {
        key: value
        for key, value in gate.items()
        if key not in _LEGACY_V12_COVERAGE_GATE_KEYS
    }
    normalized["closing_envelope_replaced_legacy_gate_keys"] = sorted(
        _LEGACY_V12_COVERAGE_GATE_KEYS.intersection(gate)
    )
    normalized.update(_closing_envelope_promotion_gate(envelope_summary))
    return normalized


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
    prewarm_observations = build_strict_prior_prewarm_observations_v15(
        races, min_calibration_days=min_calibration_days
    )
    holdout_by_date: dict[str, dict[str, Any]] = {}
    result = walk_forward_evaluate_v12(
        races,
        daily_budget_yen=daily_budget_yen,
        min_calibration_days=min_calibration_days,
        evaluation_dates=evaluation_dates,
        closing_fallback_policy=closing_fallback_policy,
        closing_forecast_field="point_final_odds",
        selection_conformal_fit=_fit_closing_envelope,
        selection_observation_append=(
            _holdout_auditing_observation_append_v15(holdout_by_date)
        ),
        initial_selection_observations=prewarm_observations,
    )

    folds = [
        _rename_envelope_keys(fold) for fold in list(result.get("folds") or [])
    ]
    for fold in folds:
        evaluation_date = str(fold.get("evaluation_date") or "")
        if evaluation_date in holdout_by_date:
            fold["closing_envelope_holdout_coverage"] = dict(
                holdout_by_date[evaluation_date]
            )
        else:
            fold.setdefault("closing_envelope_holdout_coverage", {})
        fold["selected_policy"] = _v15_policy(fold.get("selected_policy"))
        guard = dict(fold.get("leakage_guard") or {})
        guard.update({
            "closing_envelope_population": "all_120_complete_combinations",
            "closing_envelope_selection_free": True,
            "closing_teacher_appended_after_purchase_decision": True,
            "actual_closing_odds_used_for_holdout_evaluation_only": True,
            "result_used_for_closing_envelope_decision": False,
            "payout_used_for_closing_envelope_decision": False,
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
    prospective_gate = _replace_legacy_coverage_gate_v15(
        dict(prospective.get("promotion_gate") or {}), prospective_envelope
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
