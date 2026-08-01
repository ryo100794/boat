from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


POLICY_NAME = "registered_dual_head_top5_conservative_ev_v31"
REGISTERED_AFTER = "2026-08-01"
MAX_RANK = 5
MIN_CONSERVATIVE_EV = 1.0
MAX_POINT_ODDS = 80.0
STAKE_YEN = 100


def select_uncertainty_adjusted_top5_candidates(
    ranking_scores: Mapping[str, float],
    probabilities: Mapping[str, float],
    lower_forecast_odds: Mapping[str, float],
    *,
    race_id: str,
    race_date: str,
    jcd: str,
    rno: int,
    snapshot_id: int,
    captured_at: str,
    available_capital_yen: int,
) -> tuple[dict[str, Any], ...]:
    """Rank with the ranking head, but price tickets with calibrated probabilities."""
    if isinstance(available_capital_yen, bool) or available_capital_yen < 0:
        raise ValueError("available_capital_yen must be non-negative")
    combinations = set(ranking_scores)
    if (
        len(combinations) != 120
        or combinations != set(probabilities)
        or combinations != set(lower_forecast_odds)
    ):
        raise ValueError("V31 requires aligned 120-outcome model outputs and odds")
    for name, values in (
        ("ranking scores", ranking_scores),
        ("probabilities", probabilities),
    ):
        numeric = [float(values[key]) for key in combinations]
        if (
            any(not math.isfinite(value) or value <= 0.0 for value in numeric)
            or not math.isclose(sum(numeric), 1.0, abs_tol=1e-8)
        ):
            raise ValueError(f"V31 {name} are invalid")
    odds_values = [float(lower_forecast_odds[key]) for key in combinations]
    if any(not math.isfinite(value) or value <= 0.0 for value in odds_values):
        raise ValueError("V31 lower forecast odds are invalid")

    ranked = sorted(
        combinations,
        key=lambda combination: (-float(ranking_scores[combination]), combination),
    )[:MAX_RANK]
    capacity = available_capital_yen // STAKE_YEN
    selected: list[dict[str, Any]] = []
    for rank, combination in enumerate(ranked, start=1):
        probability = float(probabilities[combination])
        lower_odds = float(lower_forecast_odds[combination])
        conservative_ev = probability * lower_odds
        if lower_odds > MAX_POINT_ODDS or conservative_ev < MIN_CONSERVATIVE_EV:
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
                "ranking_score": float(ranking_scores[combination]),
                "estimated_odds": lower_odds,
                "predicted_closing": lower_odds,
                "estimated_ev": conservative_ev,
                "stake_yen": STAKE_YEN,
                "real_odds_snapshot_id": int(snapshot_id),
                "real_odds_captured_at": captured_at,
                "real_odds_combinations": 120,
                "odds_source": "strict_prior_t300_conformal_lower_v31",
                "policy_name": POLICY_NAME,
            }
        )
    return tuple(selected)
