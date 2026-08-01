from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import Any

from ..chronological_bankroll import (
    settlement_events_from_races,
    simulate_chronological_bankroll_day,
    summarize_chronological_bankroll_days,
)
from .closing_odds import decision_odds


MODEL_NAME = "dual_head_conformal_top5_v32"
POLICY_NAME = "registered_dual_head_top5_conservative_ev_v32"
REGISTERED_AFTER = "2026-08-01"
MAX_MODEL_RANK = 5
MIN_CONSERVATIVE_EV = 1.0
MAX_LOWER_ODDS = 80.0
STAKE_YEN = 100

POLICY = {
    "name": POLICY_NAME,
    "max_model_rank": MAX_MODEL_RANK,
    "ev_threshold": MIN_CONSERVATIVE_EV,
    "max_odds": MAX_LOWER_ODDS,
    "stake_per_ticket_yen": STAKE_YEN,
    "ranking_source": "ranking_head",
    "probability_source": "probability_head",
    "odds_source": "strict_prior_conformal_lower",
}


def simulate_dual_head_conformal_policy_v32(
    races: list[dict[str, Any]],
    *,
    probability_calibrator: dict[str, float],
    ranking_calibrator: dict[str, float],
    probability_blender: Callable[..., dict[str, float]],
    initial_bankroll_yen: int = 10_000,
) -> dict[str, Any]:
    """Replay the immutable V32 policy with strictly separated model roles."""
    races_by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for race in races:
        races_by_day[str(race["race_date"])].append(race)

    daily: list[dict[str, Any]] = []
    for race_date in sorted(races_by_day):
        day_races = races_by_day[race_date]
        candidates: list[dict[str, Any]] = []
        for race in day_races:
            probability_output = probability_blender(
                race["model_probabilities"],
                race["market_probabilities"],
                model_weight=float(probability_calibrator["model_weight"]),
                temperature=float(probability_calibrator["temperature"]),
            )
            ranking_output = probability_blender(
                race["model_probabilities"],
                race["market_probabilities"],
                model_weight=float(ranking_calibrator["model_weight"]),
                temperature=float(ranking_calibrator["temperature"]),
            )
            lower_odds = decision_odds(race)
            combinations = set(probability_output)
            if (
                len(combinations) != 120
                or combinations != set(ranking_output)
                or combinations != set(lower_odds)
            ):
                raise ValueError("V32 diagnostic requires aligned 120-outcome inputs")
            ranked = sorted(
                combinations,
                key=lambda combination: (
                    -float(ranking_output[combination]),
                    combination,
                ),
            )[:MAX_MODEL_RANK]
            for rank, combination in enumerate(ranked, start=1):
                probability = float(probability_output[combination])
                odds = float(lower_odds[combination])
                conservative_ev = probability * odds
                if odds > MAX_LOWER_ODDS or conservative_ev < MIN_CONSERVATIVE_EV:
                    continue
                candidates.append(
                    {
                        "race_id": str(race["race_id"]),
                        "race_date": race_date,
                        "jcd": race.get("jcd"),
                        "rno": int(race.get("rno") or 0),
                        "combination": combination,
                        "probability": probability,
                        "probability_rank": rank,
                        "ranking_score": float(ranking_output[combination]),
                        "estimated_odds": odds,
                        "estimated_ev": conservative_ev,
                        "decision_at": race.get("captured_at")
                        or race.get("odds_deadline_at"),
                        "policy_name": POLICY_NAME,
                    }
                )

        def allocate_fixed_unit(
            _race_date: str,
            race_candidates: list[dict[str, Any]],
            _eligible_races: set[str],
            **kwargs: Any,
        ) -> dict[str, Any]:
            budget = int(kwargs["daily_budget_yen"])
            settlements = dict(kwargs.get("settlements") or {})
            selected = []
            materialized = list(race_candidates)
            for candidate in materialized:
                if budget < STAKE_YEN:
                    break
                row = dict(candidate)
                returned = int(
                    settlements.get(
                        (str(row["race_id"]), str(row["combination"])), 0
                    )
                )
                row.update(
                    {
                        "stake_yen": STAKE_YEN,
                        "return_yen": returned,
                        "hit": returned > 0,
                    }
                )
                selected.append(row)
                budget -= STAKE_YEN
            return {
                "selected_sample": selected,
                "allocation_candidate_tickets": len(materialized),
            }

        result = simulate_chronological_bankroll_day(
            race_date,
            candidates,
            (str(race["race_id"]) for race in day_races),
            settlement_events=settlement_events_from_races(day_races),
            initial_bankroll_yen=initial_bankroll_yen,
            daily_stake_limit_fraction=1.0,
            max_decision_exposure_fraction=1.0,
            race_cap_fraction=1.0,
            ticket_cap_fraction=1.0,
            max_tickets_per_race=MAX_MODEL_RANK,
            schedule=day_races,
            allocate_day=allocate_fixed_unit,
            allocation_method="chronological_fixed100_v32_dual_head_conformal",
        )
        result.pop("ledger", None)
        daily.append(result)

    summary = summarize_chronological_bankroll_days(daily)
    return {
        **summary,
        "model": MODEL_NAME,
        "policy": dict(POLICY),
        "evaluated_races": len(races),
        "evaluation_days": len(daily),
        "winning_days": summary["winning_days"],
        "daily": daily,
    }
