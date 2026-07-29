from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Iterable, Mapping

import numpy as np

from .edge_conditional_probability_lcb_v14 import (
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_SEED,
    METHOD as PROBABILITY_LCB_METHOD,
    REGISTERED_DIVERGENCE_LABEL,
    REGISTERED_DIVERGENCE_LOWER,
    REGISTERED_DIVERGENCE_UPPER,
    TARGET_LOWER_QUANTILE,
    fit_edge_conditional_probability_lcb_v14,
    t300_snapshot_consistency,
)
from .odds_path_role_integrated_v12 import (
    CLOSING_FALLBACK_V11,
    DISCRETE_POLICY_V12,
    PROSPECTIVE_OUTPUT_KEY as V12_PROSPECTIVE_OUTPUT_KEY,
    walk_forward_evaluate_v12,
)
from .odds_path_role_integrated_v13 import _remove_closing_estimators
from .strict_prior_divergence_diagnostics import (
    aggregate_strict_prior_divergence_band_metrics,
    strict_prior_divergence_band_metrics,
)


MODEL_NAME = "odds_path_role_integrated_registered_band_lcb_v14"
STRATEGY_NAME = MODEL_NAME
REGISTERED_AFTER = "2026-07-29"
PROSPECTIVE_OUTPUT_KEY = "prospective_role_integrated_v14_walk_forward"
HISTORICAL_OUTPUT_KEY = "historical_exploratory_role_integrated_v14"
DAILY_BUDGET_YEN = 10_000
MIN_PROMOTION_EVALUATION_DAYS = 5
MIN_PROMOTION_CANDIDATES = 300
MIN_PROMOTION_HITS = 20

DISCRETE_POLICY_V14: dict[str, Any] = {
    **DISCRETE_POLICY_V12,
    "name": "v14_registered_divergence_band_v12_closing_discrete_log",
    "probability_lcb_method": PROBABILITY_LCB_METHOD,
    "registered_after": REGISTERED_AFTER,
    "registered_divergence_definition": (
        "log(model_probability / normalized_T300_market_probability)"
    ),
    "registered_divergence_lower_inclusive": REGISTERED_DIVERGENCE_LOWER,
    "registered_divergence_upper_exclusive": REGISTERED_DIVERGENCE_UPPER,
    "registered_divergence_label": REGISTERED_DIVERGENCE_LABEL,
    "daily_budget_yen": DAILY_BUDGET_YEN,
    "historical_results_are_exploratory_only": True,
    "zero_bet_allowed": True,
}


def _candidate_population_fingerprint(rows: Iterable[Mapping[str, Any]]) -> str:
    keys = sorted(
        (str(row.get("race_id")), str(row.get("combination")))
        for row in rows
    )
    return hashlib.sha256(
        json.dumps(keys, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _selected_population_metrics(
    races: list[dict[str, Any]],
    *,
    closing_forecasts: Mapping[str, Mapping[str, float]],
    probability_lcb: Mapping[str, Any],
    selected_candidates: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Score exactly the post-threshold population sent to the allocator."""
    del closing_forecasts, probability_lcb
    selected = [dict(row) for row in selected_candidates]
    race_by_id = {str(race["race_id"]): race for race in races}
    raw_predicted = 0.0
    adjusted_predicted = 0.0
    observed_hits = 0
    brier_sum = 0.0
    log_loss_sum = 0.0
    daily: dict[str, dict[str, Any]] = {}
    for candidate in selected:
        race_id = str(candidate["race_id"])
        race = race_by_id[race_id]
        combination = str(candidate["combination"])
        detail = candidate.get("probability_lcb_detail") or {}
        raw_probability = float(
            detail.get("raw_probability", candidate.get("probability", 0.0))
        )
        adjusted_probability = float(candidate.get("probability") or 0.0)
        hit = int(combination == str(race.get("actual_combination") or ""))
        raw_predicted += raw_probability
        adjusted_predicted += adjusted_probability
        observed_hits += hit
        brier_sum += (adjusted_probability - hit) ** 2
        clipped = float(np.clip(adjusted_probability, 1e-15, 1.0 - 1e-15))
        log_loss_sum += -(
            hit * math.log(clipped) + (1 - hit) * math.log(1.0 - clipped)
        )
        race_date = str(race["race_date"])
        row = daily.setdefault(race_date, {
            "race_date": race_date,
            "candidate_count": 0,
            "raw_predicted_hits": 0.0,
            "adjusted_predicted_hits": 0.0,
            "observed_hits": 0,
        })
        row["candidate_count"] += 1
        row["raw_predicted_hits"] += raw_probability
        row["adjusted_predicted_hits"] += adjusted_probability
        row["observed_hits"] += hit

    inconsistent = []
    for race in races:
        consistency = t300_snapshot_consistency(race)
        if not consistency["consistent"]:
            inconsistent.append({
                "race_id": str(race.get("race_id")),
                "reason": consistency["reason"],
            })
    count = len(selected)
    return {
        "evaluation_days": len(daily),
        "candidate_count": count,
        "raw_predicted_hits": raw_predicted,
        "adjusted_predicted_hits": adjusted_predicted,
        "observed_hits": observed_hits,
        "observed_hits_to_raw_predicted_hits_ratio": (
            observed_hits / raw_predicted if raw_predicted > 0.0 else None
        ),
        "observed_hits_to_adjusted_predicted_hits_ratio": (
            observed_hits / adjusted_predicted
            if adjusted_predicted > 0.0
            else None
        ),
        "adjusted_predicted_hits_to_observed_hits_ratio": (
            adjusted_predicted / observed_hits if observed_hits > 0 else None
        ),
        "candidate_binary_brier_score": brier_sum / count if count else None,
        "candidate_binary_log_loss": log_loss_sum / count if count else None,
        "candidate_population_stage": (
            "after_adjusted_probability_and_ev_threshold_before_allocation"
        ),
        "candidate_population_fingerprint": _candidate_population_fingerprint(
            selected
        ),
        "candidate_keys": [
            [str(row["race_id"]), str(row["combination"])] for row in selected
        ],
        "daily": [daily[key] for key in sorted(daily)],
        "inconsistent_t300_snapshot_races": len(inconsistent),
        "inconsistent_t300_snapshot_details": inconsistent,
    }


def _daily_bootstrap_lower_ratio(
    daily: list[Mapping[str, Any]],
    *,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_SEED,
) -> float | None:
    if not daily:
        return None
    predicted = np.asarray(
        [float(row.get("adjusted_predicted_hits") or 0.0) for row in daily],
        dtype=np.float64,
    )
    observed = np.asarray(
        [float(row.get("observed_hits") or 0.0) for row in daily],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(daily), size=(bootstrap_samples, len(daily)))
    sampled_predicted = predicted[indices].sum(axis=1)
    sampled_observed = observed[indices].sum(axis=1)
    ratios = np.zeros(bootstrap_samples, dtype=np.float64)
    valid = sampled_predicted > 0.0
    ratios[valid] = sampled_observed[valid] / sampled_predicted[valid]
    return float(np.quantile(ratios, TARGET_LOWER_QUANTILE))


def _aggregate_calibration(
    folds: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = []
    for fold in folds:
        metrics = fold.get("probability_lcb_metrics") or {}
        if isinstance(metrics, Mapping):
            row = metrics.get("selected_candidate_calibration")
            if isinstance(row, Mapping):
                rows.append(dict(row))
    daily = [
        dict(item)
        for row in rows
        for item in (row.get("daily") or [])
        if isinstance(item, Mapping)
    ]
    candidates = sum(int(row.get("candidate_count") or 0) for row in rows)
    raw_predicted = sum(float(row.get("raw_predicted_hits") or 0.0) for row in rows)
    adjusted_predicted = sum(
        float(row.get("adjusted_predicted_hits") or 0.0) for row in rows
    )
    observed = sum(int(row.get("observed_hits") or 0) for row in rows)
    brier_weighted = sum(
        float(row["candidate_binary_brier_score"])
        * int(row.get("candidate_count") or 0)
        for row in rows
        if row.get("candidate_binary_brier_score") is not None
    )
    log_loss_weighted = sum(
        float(row["candidate_binary_log_loss"])
        * int(row.get("candidate_count") or 0)
        for row in rows
        if row.get("candidate_binary_log_loss") is not None
    )
    return {
        "evaluation_days": len({str(row.get("race_date")) for row in daily}),
        "candidate_count": candidates,
        "raw_predicted_hits": raw_predicted,
        "adjusted_predicted_hits": adjusted_predicted,
        "observed_hits": observed,
        "observed_hits_to_raw_predicted_hits_ratio": (
            observed / raw_predicted if raw_predicted > 0.0 else None
        ),
        "observed_hits_to_adjusted_predicted_hits_ratio": (
            observed / adjusted_predicted if adjusted_predicted > 0.0 else None
        ),
        "adjusted_predicted_hits_to_observed_hits_ratio": (
            adjusted_predicted / observed if observed > 0 else None
        ),
        "day_bootstrap_observed_to_adjusted_predicted_ratio_lower_95": (
            _daily_bootstrap_lower_ratio(daily)
        ),
        "candidate_binary_brier_score": (
            brier_weighted / candidates if candidates else None
        ),
        "candidate_binary_log_loss": (
            log_loss_weighted / candidates if candidates else None
        ),
        "candidate_population_stage": (
            "after_adjusted_probability_and_ev_threshold_before_allocation"
        ),
        "candidate_population_fingerprint": hashlib.sha256(
            "".join(str(row.get("candidate_population_fingerprint") or "") for row in rows)
            .encode("utf-8")
        ).hexdigest(),
        "inconsistent_t300_snapshot_races": sum(
            int(row.get("inconsistent_t300_snapshot_races") or 0) for row in rows
        ),
        "daily": daily,
    }


def _settlement_probability_metrics(
    races: list[dict[str, Any]],
    *,
    closing_forecasts: Mapping[str, Mapping[str, float]],
    probability_lcb: Mapping[str, Any],
    selected_candidates: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "selected_candidate_calibration": _selected_population_metrics(
            races,
            closing_forecasts=closing_forecasts,
            probability_lcb=probability_lcb,
            selected_candidates=selected_candidates,
        ),
        "strict_prior_divergence_bands": strict_prior_divergence_band_metrics(
            races
        ),
        "result_and_payout_usage": "post_purchase_settlement_metrics_only",
    }


def _promotion_calibration_gate(summary: Mapping[str, Any]) -> dict[str, Any]:
    days = int(summary.get("evaluation_days") or 0)
    candidates = int(summary.get("candidate_count") or 0)
    hits = int(summary.get("observed_hits") or 0)
    lower = summary.get(
        "day_bootstrap_observed_to_adjusted_predicted_ratio_lower_95"
    )
    mismatches = int(summary.get("inconsistent_t300_snapshot_races") or 0)
    return {
        "minimum_probability_calibration_evaluation_days": (
            MIN_PROMOTION_EVALUATION_DAYS
        ),
        "probability_calibration_evaluation_days": days,
        "probability_calibration_evaluation_days_pass": (
            days >= MIN_PROMOTION_EVALUATION_DAYS
        ),
        "minimum_probability_calibration_candidates": MIN_PROMOTION_CANDIDATES,
        "probability_calibration_candidates": candidates,
        "probability_calibration_candidates_pass": (
            candidates >= MIN_PROMOTION_CANDIDATES
        ),
        "minimum_probability_calibration_observed_hits": MIN_PROMOTION_HITS,
        "probability_calibration_observed_hits": hits,
        "probability_calibration_observed_hits_pass": hits >= MIN_PROMOTION_HITS,
        "probability_calibration_day_bootstrap_ratio_lower_target": 1.0,
        "probability_calibration_day_bootstrap_ratio_lower": lower,
        "probability_calibration_day_bootstrap_ratio_lower_pass": (
            lower is not None and float(lower) > 1.0
        ),
        "t300_snapshot_mismatch_count": mismatches,
        "t300_snapshot_consistency_pass": mismatches == 0,
    }


def _aggregate_divergence(folds: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = []
    for fold in folds:
        row = fold.get("probability_lcb_metrics") or {}
        if isinstance(row, Mapping):
            metrics.append(row.get("strict_prior_divergence_bands") or {})
    return aggregate_strict_prior_divergence_band_metrics(metrics)


def walk_forward_evaluate_v14(
    races: list[dict[str, Any]],
    *,
    daily_budget_yen: int,
    min_calibration_days: int,
    evaluation_dates: Iterable[str] | None = None,
    closing_fallback_policy: str = CLOSING_FALLBACK_V11,
) -> dict[str, Any]:
    if int(daily_budget_yen) != DAILY_BUDGET_YEN:
        raise ValueError("V14 pre-registered shadow policy requires JPY10000/day")
    accepted = []
    rejected = []
    for race in races:
        consistency = t300_snapshot_consistency(race)
        if consistency["consistent"]:
            accepted.append(race)
        else:
            rejected.append({
                "race_id": str(race.get("race_id")),
                "race_date": str(race.get("race_date")),
                "reason": consistency["reason"],
            })
    result = walk_forward_evaluate_v12(
        accepted,
        daily_budget_yen=daily_budget_yen,
        min_calibration_days=min_calibration_days,
        evaluation_dates=evaluation_dates,
        closing_fallback_policy=closing_fallback_policy,
        probability_lcb_fit=fit_edge_conditional_probability_lcb_v14,
        probability_lcb_metrics=_settlement_probability_metrics,
        probability_lcb_metrics_use_preallocation_population=True,
    )
    folds = list(result.get("folds") or [])
    for fold in folds:
        _remove_closing_estimators(fold)
        fold["selected_policy"] = (
            dict(fold.get("selected_policy") or {})
            if (fold.get("selected_policy") or {}).get("no_bet")
            else {**dict(fold.get("selected_policy") or {}), **DISCRETE_POLICY_V14}
        )
        guard = dict(fold.get("leakage_guard") or {})
        guard.update({
            "probability_ratio_bootstrap_unit": "whole_race_day",
            "optimistic_probability_pseudo_counts": False,
            "probability_double_shrinkage": False,
            "global_all_ticket_factor_used": False,
            "child_factor_never_above_rank_parent": True,
            "candidate_evaluation_population": (
                "exact_preallocation_selected_population"
            ),
            "registered_divergence_band": "[0.5,1.0)",
            "result_payout_in_purchase_features": False,
        })
        fold["leakage_guard"] = guard

    prospective_folds = [
        fold for fold in folds
        if str(fold.get("evaluation_date")) > REGISTERED_AFTER
    ]
    historical_folds = [
        fold for fold in folds
        if str(fold.get("evaluation_date")) <= REGISTERED_AFTER
    ]
    prospective = dict(result.pop(V12_PROSPECTIVE_OUTPUT_KEY, {}) or {})
    prospective_calibration = _aggregate_calibration(prospective_folds)
    prospective_rejections = [
        row for row in rejected
        if str(row.get("race_date")) > REGISTERED_AFTER
    ]
    prospective_calibration["inconsistent_t300_snapshot_races"] += len(
        prospective_rejections
    )
    prospective_calibration["inconsistent_t300_snapshot_details"] = list(
        prospective_rejections
    )
    prospective.update({
        "status": (
            "prospective_shadow_evaluating"
            if prospective_folds
            else "waiting_for_first_prospective_day"
        ),
        "registered_after": REGISTERED_AFTER,
        "fixed_divergence_band": "[0.5,1.0)",
        "selected_candidate_calibration": prospective_calibration,
        "strict_prior_divergence_bands": _aggregate_divergence(
            prospective_folds
        ),
        "historical_results_included_in_promotion": False,
    })
    gate = dict(prospective.get("promotion_gate") or {})
    gate.update(_promotion_calibration_gate(prospective_calibration))
    checks = [value for key, value in gate.items() if key.endswith("_pass")]
    prospective["promotion_gate"] = gate
    prospective["promotion_eligible"] = bool(checks) and all(checks)

    historical_calibration = _aggregate_calibration(historical_folds)
    historical_rejections = [
        row for row in rejected
        if str(row.get("race_date")) <= REGISTERED_AFTER
    ]
    historical_calibration["inconsistent_t300_snapshot_races"] += len(
        historical_rejections
    )
    historical_calibration["inconsistent_t300_snapshot_details"] = list(
        historical_rejections
    )
    historical = {
        "status": "historical_exploratory_non_promotion_evidence",
        "research_only": True,
        "promotion_evidence": False,
        "registered_after": REGISTERED_AFTER,
        "evaluation_days": len(historical_folds),
        "evaluation_dates": [
            str(fold.get("evaluation_date")) for fold in historical_folds
        ],
        "selected_candidate_calibration": historical_calibration,
        "strict_prior_divergence_bands": _aggregate_divergence(
            historical_folds
        ),
        "interpretation": (
            "The [0.5,1.0) band was chosen from historical diagnostics; all "
            "results through 2026-07-29 are exploratory and cannot promote V14"
        ),
    }

    deployment = dict(result.get("deployment_configuration") or {})
    _remove_closing_estimators(deployment)
    deployment.update({
        "calibrator_strategy": STRATEGY_NAME,
        "candidate_policy": dict(DISCRETE_POLICY_V14),
        "selected_policy": {"name": "no_bet", "no_bet": True},
        "operational_status": "shadow_only_until_v14_prospective_gate",
        "real_betting_enabled": False,
    })
    aggregate_calibration = _aggregate_calibration(folds)
    aggregate_calibration["inconsistent_t300_snapshot_races"] += len(rejected)
    aggregate_calibration["inconsistent_t300_snapshot_details"] = list(rejected)
    result.update({
        "model": MODEL_NAME,
        "calibrator_strategy": STRATEGY_NAME,
        "comparison_role": "v12_closing_registered_band_probability_lcb_v14_shadow",
        "registered_after": REGISTERED_AFTER,
        "fixed_policy": dict(DISCRETE_POLICY_V14),
        "input_snapshot_consistency": {
            "accepted_races": len(accepted),
            "rejected_races": len(rejected),
            "historical_rejected_races": len(historical_rejections),
            "prospective_rejected_races": len(prospective_rejections),
            "rejections": rejected,
        },
        "folds": folds,
        "selected_candidate_calibration": aggregate_calibration,
        "strict_prior_divergence_bands": _aggregate_divergence(folds),
        HISTORICAL_OUTPUT_KEY: historical,
        PROSPECTIVE_OUTPUT_KEY: prospective,
        "promotion_gate": gate,
        "promotion_eligible": prospective["promotion_eligible"],
        "deployment_configuration": deployment,
    })
    return result
