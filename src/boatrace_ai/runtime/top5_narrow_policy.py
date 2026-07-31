from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


POLICY_NAME = "registered_top5_ev1.00_to1.05_flat100_v1"
REGISTERED_AFTER = "2026-07-28"
MAX_RANK = 5
MIN_ESTIMATED_EV = 1.0
MAX_ESTIMATED_EV = 1.05
STAKE_YEN = 100


def select_top5_narrow_candidates(
    probabilities: Mapping[str, float],
    forecast_odds: Mapping[str, float],
    *,
    race_id: str,
    race_date: str,
    jcd: str,
    rno: int,
    snapshot_id: int,
    captured_at: str,
    available_capital_yen: int,
) -> tuple[dict[str, Any], ...]:
    """Select fixed-unit V23 tickets using decision-time values only."""
    if isinstance(available_capital_yen, bool) or available_capital_yen < 0:
        raise ValueError("available_capital_yen must be non-negative")
    combinations = set(probabilities)
    if len(combinations) != 120 or combinations != set(forecast_odds):
        raise ValueError("V23 requires aligned 120-outcome probabilities and odds")
    probability_values = [float(probabilities[key]) for key in combinations]
    odds_values = [float(forecast_odds[key]) for key in combinations]
    if (
        any(not math.isfinite(value) or value <= 0.0 for value in probability_values)
        or not math.isclose(sum(probability_values), 1.0, abs_tol=1e-8)
        or any(not math.isfinite(value) or value <= 0.0 for value in odds_values)
    ):
        raise ValueError("V23 probabilities or forecast odds are invalid")

    ranked = sorted(
        combinations,
        key=lambda combination: (-float(probabilities[combination]), combination),
    )[:MAX_RANK]
    capacity = available_capital_yen // STAKE_YEN
    selected: list[dict[str, Any]] = []
    for rank, combination in enumerate(ranked, start=1):
        probability = float(probabilities[combination])
        odds = float(forecast_odds[combination])
        estimated_ev = probability * odds
        if not MIN_ESTIMATED_EV <= estimated_ev <= MAX_ESTIMATED_EV:
            continue
        if len(selected) >= capacity:
            break
        selected.append(
            {
                "race_id": race_id,
                "race_date": race_date,
                "jcd": jcd,
                "rno": int(rno),
                "combination": combination,
                "probability": probability,
                "probability_rank": rank,
                "estimated_odds": odds,
                "predicted_closing": odds,
                "estimated_ev": estimated_ev,
                "stake_yen": STAKE_YEN,
                "real_odds_snapshot_id": int(snapshot_id),
                "real_odds_captured_at": captured_at,
                "real_odds_combinations": 120,
                "odds_source": "strict_prior_forecast_final_from_real_t5",
                "policy_name": POLICY_NAME,
            }
        )
    return tuple(selected)


def daily_capital_limits(
    rows: list[Mapping[str, Any]],
    *,
    bankroll_yen: int,
    starting_bankroll_yen: int = 10_000,
) -> dict[str, int]:
    """Compute live capital without using settlement fields for ticket ranking."""
    gross_stake = 0
    realized_profit = 0
    for row in rows:
        gross_stake += int(row.get("total_stake_yen") or 0)
        if row.get("profit_yen") is not None:
            realized_profit += int(row["profit_yen"])
    gross_allowance = starting_bankroll_yen + max(0, realized_profit)
    remaining_gross = max(0, gross_allowance - gross_stake)
    return {
        "gross_stake_yen": gross_stake,
        "realized_cumulative_profit_yen": realized_profit,
        "gross_stake_allowance_yen": gross_allowance,
        "remaining_gross_stake_allowance_yen": remaining_gross,
        "allocatable_bankroll_yen": max(
            0, min(int(bankroll_yen), remaining_gross)
        ),
    }
