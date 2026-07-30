from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

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
    _rank_groups,
    _summarize_bankroll,
    _walk_forward_evaluate_conservative_ev,
)
from .odds_path_probability_v8 import (
    attach_odds_path_probability_v8,
    fit_odds_path_probability_v8,
)


MODEL_NAME = "odds_path_market_offset_discrete_log_ev_v9"
STRATEGY_NAME = "odds_path_market_offset_discrete_log_ev_v9"
REGISTERED_AFTER = "2026-07-29"
PROSPECTIVE_OUTPUT_KEY = (
    "prospective_market_offset_discrete_log_ev_v9_walk_forward"
)

DISCRETE_POLICY: dict[str, Any] = {
    "name": "v9_safe_ev105_q20_lcb_discrete_log_100yen",
    "safe_ev_threshold": SAFE_EV_THRESHOLD,
    "closing_quantile": CLOSING_QUANTILE,
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
    total_races: int,
    ready_races: int,
    threshold_candidates: int,
    allocation_candidates: int,
) -> str:
    if total_races == 0:
        return "no_evaluated_races"
    if ready_races == 0:
        return "closing_or_lcb_not_ready"
    if threshold_candidates == 0:
        return "no_safe_ev_threshold_candidate"
    if allocation_candidates == 0:
        return "no_positive_discrete_log_growth"
    return "day_portfolio_constraints"


def _simulate_discrete_log_ev_policy(
    races: list[dict[str, Any]],
    *,
    closing_forecasts: dict[str, dict[str, float]],
    probability_lcb: dict[str, Any],
    daily_budget_yen: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    by_day_candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_day_races: dict[str, set[str]] = defaultdict(set)
    ready_races_by_day: dict[str, int] = defaultdict(int)
    threshold_by_day: dict[str, int] = defaultdict(int)
    diagnostic = _new_purchase_diagnostic_accumulator()
    diagnostic.update({
        "candidates_before_allocation": 0,
        "allocation_candidate_tickets": 0,
        "zero_purchase_days": 0,
        "zero_reason_counts": {},
    })
    diagnostic["total_races"] = len(races)
    lcb_ready = bool(probability_lcb.get("ready"))

    for race in races:
        date = str(race["race_date"])
        race_id = str(race["race_id"])
        by_day_races[date].add(race_id)
        closing = closing_forecasts.get(race_id) or {}
        if len(closing) != 120:
            diagnostic["closing_forecast_missing_races"] += 1
        if not lcb_ready:
            diagnostic["lcb_not_ready_races"] += 1
        if len(closing) != 120 or not lcb_ready:
            continue
        ready_races_by_day[date] += 1
        probabilities = race["model_probabilities"]
        rank_groups = _rank_groups(probabilities)
        factors = probability_lcb.get("factors") or {}
        candidates = []
        for combination, odds in closing.items():
            safe_probability = float(probabilities[combination]) * float(
                factors.get(rank_groups[combination], 0.0)
            )
            safe_ev = safe_probability * float(odds)
            diagnostic["evaluated_combinations"] += 1
            diagnostic["safe_ev_values"].append(safe_ev)
            for threshold in SAFE_EV_DIAGNOSTIC_THRESHOLDS:
                if safe_ev >= threshold:
                    diagnostic["safe_ev_at_least"][f"{threshold:.2f}"] += 1
            if safe_ev < SAFE_EV_THRESHOLD:
                continue
            candidates.append(
                _policy_candidate(
                    race,
                    combination=combination,
                    probability=safe_probability,
                    estimated_odds=float(odds),
                    safe_ev=safe_ev,
                )
            )
        diagnostic["threshold_pass_candidates"] += len(candidates)
        threshold_by_day[date] += len(candidates)
        diagnostic["candidates_after_race_cap"] += min(
            len(candidates), MAX_TICKETS_PER_RACE
        )
        diagnostic["candidates_before_allocation"] += len(candidates)
        by_day_candidates[date].extend(candidates)

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
                total_races=len(by_day_races[date]),
                ready_races=ready_races_by_day[date],
                threshold_candidates=threshold_by_day[date],
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
    return (
        _summarize_bankroll(
            daily,
            evaluated_races=len(races),
            max_drawdown_yen=max_drawdown,
            purchase_diagnostic_accumulators=[diagnostic],
        ),
        diagnostic,
    )


def walk_forward_evaluate_v9(
    races: list[dict[str, Any]],
    *,
    daily_budget_yen: int,
    min_calibration_days: int,
    evaluation_dates: Iterable[str] | None = None,
) -> dict[str, Any]:
    policy = {**DISCRETE_POLICY, "daily_budget_yen": daily_budget_yen}
    return _walk_forward_evaluate_conservative_ev(
        races,
        daily_budget_yen=daily_budget_yen,
        min_calibration_days=min_calibration_days,
        evaluation_dates=evaluation_dates,
        model_name=MODEL_NAME,
        strategy_name=STRATEGY_NAME,
        registered_after=REGISTERED_AFTER,
        prospective_output_key=PROSPECTIVE_OUTPUT_KEY,
        comparison_role="real_t5_market_offset_q20_lcb_discrete_log_shadow",
        prospective_comparison_role=(
            "pre_registered_strict_outer_day_v9_shadow"
        ),
        probability_fit=fit_odds_path_probability_v8,
        probability_attach=attach_odds_path_probability_v8,
        deployment_waiting_status="shadow_only_until_v9_promotion_gate",
        purchase_simulator=_simulate_discrete_log_ev_policy,
        fixed_policy=policy,
    )
