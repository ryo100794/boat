from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import numpy as np

from boatrace_ai.discrete_log_allocation import allocate_discrete_log_day

from .odds_path_conservative_v7 import (
    CLOSING_QUANTILE,
    MAX_DAILY_EXPOSURE_FRACTION,
    MAX_TICKETS_PER_RACE,
    RACE_CAP_FRACTION,
    SAFE_EV_DIAGNOSTIC_THRESHOLDS,
    SAFE_EV_THRESHOLD,
    STAKE_GRANULARITY_YEN,
    TICKET_CAP_FRACTION,
    _new_purchase_diagnostic_accumulator,
    _policy_candidate,
    _summarize_bankroll,
    _walk_forward_evaluate_conservative_ev,
)
from .odds_path_probability_v8 import (
    attach_odds_path_probability_v8,
    fit_odds_path_probability_v8,
)
from .selection_conformal import (
    METHOD,
    MIN_TRAINING_CANDIDATES,
    MIN_TRAINING_DAYS,
    TARGET_COVERAGE,
    build_prequential_selection_conformal,
    selected_safe_ev_candidates,
    selection_coverage_metrics,
)
from .edge_conditional_probability_lcb_v13 import probability_lower_bound_details


MODEL_NAME = "odds_path_market_offset_selection_conformal_discrete_ev_v10"
STRATEGY_NAME = MODEL_NAME
REGISTERED_AFTER = "2026-07-29"
PROSPECTIVE_OUTPUT_KEY = (
    "prospective_market_offset_selection_conformal_discrete_ev_v10_walk_forward"
)

DISCRETE_POLICY: dict[str, Any] = {
    "name": "v10_selection_conformal_safe_ev105_discrete_log_100yen",
    "safe_ev_threshold": SAFE_EV_THRESHOLD,
    "closing_quantile": CLOSING_QUANTILE,
    "selection_rule": "safe_ev105_then_top2_per_race",
    "selection_conformal_method": METHOD,
    "selection_conformal_target_coverage": TARGET_COVERAGE,
    "selection_conformal_minimum_training_days": MIN_TRAINING_DAYS,
    "selection_conformal_minimum_training_candidates": MIN_TRAINING_CANDIDATES,
    "max_tickets_per_race": MAX_TICKETS_PER_RACE,
    "allocation_method": "discrete_conservative_expected_log",
    "daily_budget_yen": 10_000,
    "max_daily_exposure_fraction": MAX_DAILY_EXPOSURE_FRACTION,
    "race_cap_fraction": RACE_CAP_FRACTION,
    "ticket_cap_fraction": TICKET_CAP_FRACTION,
    "stake_granularity_yen": STAKE_GRANULARITY_YEN,
    "zero_bet_allowed": True,
}


def _zero_reason(
    *,
    conformal_ready: bool,
    total_races: int,
    raw_candidates: int,
    guarded_candidates: int,
    allocation_candidates: int,
) -> str:
    if total_races == 0:
        return "no_evaluated_races"
    if not conformal_ready:
        return "selection_conformal_not_ready"
    if raw_candidates == 0:
        return "no_safe_ev_threshold_candidate"
    if guarded_candidates == 0:
        return "no_candidate_after_selection_conformal"
    if allocation_candidates == 0:
        return "no_positive_discrete_log_growth"
    return "day_portfolio_constraints"


def _simulate_selection_conformal_policy(
    races: list[dict[str, Any]],
    *,
    closing_forecasts: dict[str, dict[str, float]],
    probability_lcb: dict[str, Any],
    daily_budget_yen: int,
    selection_conformal: dict[str, Any],
    capture_preallocation_candidates: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    dates = {str(race["race_date"]) for race in races}
    if len(dates) > 1:
        raise ValueError("v10 purchase simulation requires a single outer day")
    by_day_races: dict[str, set[str]] = defaultdict(set)
    by_day_candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    diagnostic = _new_purchase_diagnostic_accumulator()
    diagnostic.update({
        "candidates_before_allocation": 0,
        "allocation_candidate_tickets": 0,
        "zero_purchase_days": 0,
        "zero_reason_counts": {},
        "raw_selected_candidates": 0,
        "guarded_threshold_candidates": 0,
        "selection_conformal": dict(selection_conformal),
    })
    diagnostic["total_races"] = len(races)
    lcb_ready = bool(probability_lcb.get("ready"))
    conformal_ready = bool(selection_conformal.get("ready"))
    haircut = (
        float(selection_conformal["haircut"])
        if conformal_ready
        else None
    )
    raw_selected = selected_safe_ev_candidates(
        races,
        closing_forecasts=closing_forecasts,
        probability_lcb=probability_lcb,
    )
    raw_by_key = {
        (str(row["race_id"]), str(row["combination"])): row
        for row in raw_selected
    }
    diagnostic["raw_selected_candidates"] = len(raw_selected)
    diagnostic["threshold_pass_candidates"] = len(raw_selected)
    diagnostic["candidates_after_race_cap"] = len(raw_selected)

    for race in races:
        date = str(race["race_date"])
        race_id = str(race["race_id"])
        by_day_races[date].add(race_id)
        closing = closing_forecasts.get(race_id) or {}
        if len(closing) != 120:
            diagnostic["closing_forecast_missing_races"] += 1
        if not lcb_ready:
            diagnostic["lcb_not_ready_races"] += 1
        if len(closing) != 120 or not lcb_ready or not conformal_ready:
            continue
        probabilities = race["model_probabilities"]
        for combination, raw_odds in closing.items():
            lcb_detail = probability_lower_bound_details(
                race, str(combination), probability_lcb
            )
            safe_probability = float(lcb_detail["probability"])
            raw_safe_ev = safe_probability * float(raw_odds)
            diagnostic["evaluated_combinations"] += 1
            diagnostic["safe_ev_values"].append(raw_safe_ev)
            for threshold in SAFE_EV_DIAGNOSTIC_THRESHOLDS:
                if raw_safe_ev >= threshold:
                    diagnostic["safe_ev_at_least"][f"{threshold:.2f}"] += 1
            if (race_id, str(combination)) not in raw_by_key:
                continue
            guarded_odds = float(raw_odds) * float(haircut)
            guarded_safe_ev = safe_probability * guarded_odds
            if guarded_safe_ev < SAFE_EV_THRESHOLD:
                continue
            candidate = _policy_candidate(
                race,
                combination=str(combination),
                probability=safe_probability,
                estimated_odds=guarded_odds,
                safe_ev=guarded_safe_ev,
            )
            candidate.update({
                "predicted_closing": float(raw_odds),
                "raw_predicted_closing_odds": float(raw_odds),
                "selection_conformal_haircut": float(haircut),
                "raw_safe_ev": raw_safe_ev,
                "probability_lcb_detail": lcb_detail,
                "odds_source": "strict_prior_q20_times_selection_conformal_haircut",
            })
            by_day_candidates[date].append(candidate)
            diagnostic["guarded_threshold_candidates"] += 1
            diagnostic["candidates_before_allocation"] += 1

    allocator_input_candidates = [
        candidate
        for date in sorted(by_day_candidates)
        for candidate in by_day_candidates[date]
    ]
    if capture_preallocation_candidates:
        diagnostic["_preallocation_candidates"] = [
            {
                key: candidate.get(key)
                for key in (
                    "race_id", "race_date", "jcd", "rno", "combination",
                    "probability", "estimated_odds", "estimated_ev", "safe_ev",
                    "predicted_closing", "raw_predicted_closing_odds",
                    "selection_conformal_haircut", "raw_safe_ev",
                    "probability_lcb_detail", "odds_source",
                )
            }
            for candidate in allocator_input_candidates
        ]
    coverage = selection_coverage_metrics(
        races, allocator_input_candidates, haircut=haircut
    )
    diagnostic["selection_conformal"].update(coverage)

    daily = []
    cumulative_profit = peak_profit = max_drawdown = 0
    zero_reasons: dict[str, int] = defaultdict(int)
    for date in sorted(by_day_races):
        row = allocate_discrete_log_day(
            date,
            by_day_candidates.get(date, []),
            by_day_races[date],
            daily_budget_yen=daily_budget_yen,
            max_daily_exposure_fraction=MAX_DAILY_EXPOSURE_FRACTION,
            race_cap_fraction=RACE_CAP_FRACTION,
            ticket_cap_fraction=TICKET_CAP_FRACTION,
            max_daily_tickets=None,
            stake_granularity_yen=STAKE_GRANULARITY_YEN,
            min_stake_yen=STAKE_GRANULARITY_YEN,
            max_tickets_per_race=MAX_TICKETS_PER_RACE,
        )
        allocation_candidates = int(row["allocation_candidate_tickets"])
        diagnostic["allocation_candidate_tickets"] += allocation_candidates
        if int(row["tickets"]) == 0:
            diagnostic["zero_purchase_days"] += 1
            reason = _zero_reason(
                conformal_ready=conformal_ready,
                total_races=len(by_day_races[date]),
                raw_candidates=len(raw_selected),
                guarded_candidates=len(by_day_candidates.get(date, [])),
                allocation_candidates=allocation_candidates,
            )
            zero_reasons[reason] += 1
            row["zero_purchase_reason"] = reason
        cumulative_profit += int(row["profit_yen"])
        peak_profit = max(peak_profit, cumulative_profit)
        max_drawdown = max(max_drawdown, peak_profit - cumulative_profit)
        row["cumulative_profit_yen"] = cumulative_profit
        daily.append(row)
    diagnostic["purchases_after_allocation"] = sum(
        int(row.get("tickets") or 0) for row in daily
    )
    diagnostic["zero_reason_counts"] = dict(sorted(zero_reasons.items()))
    bankroll = _summarize_bankroll(
        daily,
        evaluated_races=len(races),
        max_drawdown_yen=max_drawdown,
        purchase_diagnostic_accumulators=[diagnostic],
    )
    bankroll["selection_conformal"] = dict(
        diagnostic["selection_conformal"]
    )
    return bankroll, diagnostic


def _aggregate_selection_conformal(folds: list[dict[str, Any]]) -> dict[str, Any]:
    artifacts = [
        fold.get("selection_conformal") or {}
        for fold in folds
    ]
    candidates = sum(
        int(item.get("selection_evaluation_candidates") or 0)
        for item in artifacts
    )
    observed = sum(
        int(item.get("selection_observed_closing_candidates") or 0)
        for item in artifacts
    )
    missing = sum(
        int(item.get("selection_closing_missing_candidates") or 0)
        for item in artifacts
    )
    raw_covered = sum(
        int(item.get("selection_raw_covered_candidates") or 0)
        for item in artifacts
    )
    guarded_covered = sum(
        int(item.get("selection_guarded_covered_candidates") or 0)
        for item in artifacts
    )
    ratios = [
        float(value)
        for item in artifacts
        for value in item.get("selection_closing_ratios") or []
    ]
    ready = [item for item in artifacts if item.get("ready")]
    latest = artifacts[-1] if artifacts else {}
    haircuts = [float(item["haircut"]) for item in ready]
    return {
        "method": METHOD,
        "target_coverage": TARGET_COVERAGE,
        "evaluation_folds": len(artifacts),
        "ready_folds": len(ready),
        "selection_evaluation_candidates": candidates,
        "selection_observed_closing_candidates": observed,
        "selection_closing_missing_candidates": missing,
        "selection_closing_complete": candidates > 0 and missing == 0,
        "selection_raw_covered_candidates": raw_covered,
        "selection_guarded_covered_candidates": guarded_covered,
        "selection_raw_closing_coverage": (
            raw_covered / candidates if candidates else None
        ),
        "selection_guarded_closing_coverage": (
            guarded_covered / candidates if candidates else None
        ),
        "selection_closing_ratio_mean": (
            float(np.mean(ratios)) if ratios else None
        ),
        "selection_closing_ratio_p10": (
            float(np.quantile(ratios, 0.10)) if ratios else None
        ),
        "selection_closing_ratio_median": (
            float(np.median(ratios)) if ratios else None
        ),
        "haircut_latest": latest.get("haircut"),
        "haircut_min": min(haircuts) if haircuts else None,
        "haircut_max": max(haircuts) if haircuts else None,
        "training_days_latest": latest.get("training_days"),
        "training_candidates_latest": latest.get("training_candidates"),
        "trained_through_date_latest": latest.get("trained_through_date"),
    }


def _selection_coverage_gate(summary: dict[str, Any]) -> dict[str, bool]:
    coverage = summary.get("selection_guarded_closing_coverage")
    candidates = int(summary.get("selection_evaluation_candidates") or 0)
    missing = int(summary.get("selection_closing_missing_candidates") or 0)
    return {
        "selection_conditional_coverage_pass": (
            coverage is not None and 0.75 <= float(coverage) <= 0.95
        ),
        "selection_conditional_complete_pass": (
            candidates > 0 and missing == 0
        ),
    }


def walk_forward_evaluate_v10(
    races: list[dict[str, Any]],
    *,
    daily_budget_yen: int,
    min_calibration_days: int,
    evaluation_dates: Iterable[str] | None = None,
) -> dict[str, Any]:
    prequential = build_prequential_selection_conformal(
        races,
        min_calibration_days=min_calibration_days,
        probability_fit=fit_odds_path_probability_v8,
        probability_attach=attach_odds_path_probability_v8,
    )
    artifacts = prequential["artifacts_by_date"]

    def simulate(races_for_day: list[dict[str, Any]], **kwargs: Any):
        date = str(races_for_day[0]["race_date"]) if races_for_day else ""
        return _simulate_selection_conformal_policy(
            races_for_day,
            selection_conformal=artifacts.get(date) or {
                "ready": False,
                "reason": "missing_prequential_selection_conformal_artifact",
            },
            **kwargs,
        )

    policy = {**DISCRETE_POLICY, "daily_budget_yen": daily_budget_yen}
    result = _walk_forward_evaluate_conservative_ev(
        races,
        daily_budget_yen=daily_budget_yen,
        min_calibration_days=min_calibration_days,
        evaluation_dates=evaluation_dates,
        model_name=MODEL_NAME,
        strategy_name=STRATEGY_NAME,
        registered_after=REGISTERED_AFTER,
        prospective_output_key=PROSPECTIVE_OUTPUT_KEY,
        comparison_role="real_t5_market_offset_selection_conformal_discrete_log_shadow",
        prospective_comparison_role="pre_registered_strict_outer_day_v10_shadow",
        probability_fit=fit_odds_path_probability_v8,
        probability_attach=attach_odds_path_probability_v8,
        deployment_waiting_status="shadow_only_until_v10_promotion_gate",
        purchase_simulator=simulate,
        fixed_policy=policy,
    )
    for fold in result.get("folds") or []:
        bankroll = fold.get("bankroll") or {}
        artifact = dict(bankroll.get("selection_conformal") or {})
        fold["selection_conformal"] = artifact
        trained_through = artifact.get("trained_through_date")
        guard = fold.get("leakage_guard") or {}
        guard["selection_conformal_trained_through"] = trained_through
        guard["selection_conformal_pass"] = (
            trained_through is None
            or str(trained_through) < str(fold["evaluation_date"])
        )
        guard["pass"] = bool(guard.get("pass")) and bool(
            guard["selection_conformal_pass"]
        )
        fold["leakage_guard"] = guard
    summary = _aggregate_selection_conformal(result.get("folds") or [])
    result["selection_conformal"] = summary
    result.update({
        key: value
        for key, value in summary.items()
        if key.startswith("selection_") or key.startswith("haircut_")
    })
    deployment = result.get("deployment_configuration") or {}
    deployment["selection_conformal"] = prequential["deployment_artifact"]
    if not prequential["deployment_artifact"].get("ready"):
        deployment["selected_policy"] = {"name": "no_bet", "no_bet": True}
        deployment["operational_status"] = "selection_conformal_not_ready"
    result["deployment_configuration"] = deployment
    prospective = result.get(PROSPECTIVE_OUTPUT_KEY) or {}
    prospective_folds = [
        fold
        for fold in result.get("folds") or []
        if str(fold.get("evaluation_date")) > REGISTERED_AFTER
    ]
    prospective_summary = _aggregate_selection_conformal(prospective_folds)
    prospective["selection_conformal"] = prospective_summary
    gate = prospective.get("promotion_gate") or {}
    gate.update(_selection_coverage_gate(prospective_summary))
    checks = [value for key, value in gate.items() if key.endswith("_pass")]
    prospective["promotion_gate"] = gate
    prospective["promotion_eligible"] = bool(checks) and all(checks)
    result[PROSPECTIVE_OUTPUT_KEY] = prospective
    result["promotion_gate"] = gate
    result["promotion_eligible"] = prospective["promotion_eligible"]
    return result
