from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from ..adaptive_allocation import (
    allocate_adaptive_day,
    append_day_result,
    zero_totals,
)
from ..bankroll_backtest import _build_payout_model
from ..fast_math import TRIFECTA_COMBINATIONS


COMBINATION_LABELS = tuple(
    "-".join(str(lane) for lane in combination)
    for combination in TRIFECTA_COMBINATIONS
)


def standard_direct_policy() -> dict[str, Any]:
    return {
        "daily_budget_yen": 10_000,
        "ev_threshold": 1.20,
        "payout_prior_weight": 30.0,
        "fractional_kelly": 0.25,
        "max_daily_exposure_fraction": 0.60,
        "min_daily_exposure_fraction": 0.40,
        "race_cap_fraction": 0.10,
        "ticket_cap_fraction": 0.03,
        "max_daily_tickets": 30,
        "allocation_mode": "normalized_kelly",
        "stake_granularity_yen": 100,
        "min_stake_yen": 100,
        "bet_type": "3連単",
        "include_odds": False,
        "payout_estimator": (
            "training-period trifecta payout mean with global prior"
        ),
        "selection": "fixed standard_365d_v2 policy; no holdout tuning",
    }


def direct_candidates(
    probabilities: np.ndarray,
    *,
    race_key: tuple[str, str, str, int],
    actual: dict[str, Any],
    payout_model: dict[str, dict[str, float]],
    ev_threshold: float,
) -> list[dict[str, Any]]:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.shape != (len(COMBINATION_LABELS),):
        raise ValueError("probabilities must contain all 120 trifecta combinations")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("probabilities must be finite and non-negative")
    total = float(values.sum())
    if total <= 0.0:
        raise ValueError("probabilities must have positive mass")
    values = values / total
    race_id, race_date, jcd, rno = race_key
    candidates = []
    for combination, probability in zip(COMBINATION_LABELS, values):
        estimate = payout_model.get(combination)
        if not estimate:
            continue
        estimated_odds = float(estimate["estimated_odds"])
        estimated_ev = float(probability) * estimated_odds
        if estimated_ev < float(ev_threshold):
            continue
        candidates.append(
            {
                "race_id": race_id,
                "race_date": race_date,
                "jcd": jcd,
                "rno": int(rno),
                "combination": combination,
                "probability": float(probability),
                "estimated_odds": estimated_odds,
                "estimated_payout_yen": float(
                    estimate["estimated_payout_yen"]
                ),
                "estimated_ev": estimated_ev,
                "payout_history_count": int(estimate["history_count"]),
                "odds_source": "payout_model",
                "actual_combination": str(actual["combination"]),
                "actual_payout_yen": int(actual["payout_yen"]),
                "hit": combination == str(actual["combination"]),
            }
        )
    return candidates


def simulate_direct_bankroll(
    probabilities: np.ndarray,
    *,
    race_keys: list[tuple[str, str, str, int]],
    payouts: dict[str, dict[str, Any]],
    training_races: set[str],
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.shape != (len(race_keys), len(COMBINATION_LABELS)):
        raise ValueError("probability matrix and race keys must align")
    selected_policy = dict(policy or standard_direct_policy())
    payout_model = _build_payout_model(
        payouts,
        train_races=training_races,
        prior_weight=float(selected_policy["payout_prior_weight"]),
    )
    candidates_by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    evaluated_by_day: dict[str, set[str]] = defaultdict(set)
    for row_index, race_key in enumerate(race_keys):
        race_id, race_date, _jcd, _rno = race_key
        actual = payouts.get(race_id)
        if actual is None:
            continue
        evaluated_by_day[race_date].add(race_id)
        candidates_by_day[race_date].extend(
            direct_candidates(
                values[row_index],
                race_key=race_key,
                actual=actual,
                payout_model=payout_model,
                ev_threshold=float(selected_policy["ev_threshold"]),
            )
        )

    totals = zero_totals()
    daily = []
    state = (0, 0, 0)
    for race_date in sorted({str(row[1]) for row in race_keys}):
        result = allocate_adaptive_day(
            race_date,
            candidates_by_day.get(race_date, []),
            evaluated_by_day.get(race_date, set()),
            daily_budget_yen=int(selected_policy["daily_budget_yen"]),
            fractional_kelly=float(selected_policy["fractional_kelly"]),
            max_daily_exposure_fraction=float(
                selected_policy["max_daily_exposure_fraction"]
            ),
            min_daily_exposure_fraction=float(
                selected_policy["min_daily_exposure_fraction"]
            ),
            race_cap_fraction=float(selected_policy["race_cap_fraction"]),
            ticket_cap_fraction=float(selected_policy["ticket_cap_fraction"]),
            max_daily_tickets=int(selected_policy["max_daily_tickets"]),
            allocation_mode=str(selected_policy["allocation_mode"]),
            stake_granularity_yen=int(
                selected_policy["stake_granularity_yen"]
            ),
            min_stake_yen=int(selected_policy["min_stake_yen"]),
        )
        state = append_day_result(
            daily,
            totals,
            result,
            cumulative_profit=state[0],
            peak_profit=state[1],
            max_drawdown=state[2],
        )
    stake_yen = int(totals["stake_yen"])
    return_yen = int(totals["return_yen"])
    return {
        "policy": selected_policy,
        "evaluation_days": len(daily),
        "evaluated_races": int(totals["evaluated_races"]),
        "candidate_tickets": int(totals["candidate_tickets"]),
        "selected_tickets": int(totals["tickets"]),
        "races_bet": int(totals["races_bet"]),
        "hit_tickets": int(totals["hit_tickets"]),
        "stake_yen": stake_yen,
        "return_yen": return_yen,
        "profit_yen": return_yen - stake_yen,
        "roi": return_yen / stake_yen if stake_yen else 0.0,
        "winning_days": int(totals["winning_days"]),
        "losing_days": int(totals["losing_days"]),
        "max_drawdown_yen": int(state[2]),
        "daily": daily,
    }
