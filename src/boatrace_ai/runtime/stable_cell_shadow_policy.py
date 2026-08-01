from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .top5_narrow_policy import STAKE_YEN, select_top5_narrow_candidates


POLICY_NAME = "preregistered_top5_lt20_ev1.00_1.05_flat100_20260802"
REGISTERED_AFTER = "2026-08-01"
MAX_FORECAST_ODDS = 20.0
SOURCE_EVALUATION_JOB_ID = 10_730
DIAGNOSTIC_KEY = "stable_cell_top5_lt20"


def select_stable_cell_candidates(
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
    """Apply the preregistered low-odds stable cell using decision-time values."""
    if isinstance(available_capital_yen, bool) or available_capital_yen < 0:
        raise ValueError("available_capital_yen must be non-negative")
    capacity = int(available_capital_yen) // STAKE_YEN
    broad = select_top5_narrow_candidates(
        probabilities,
        forecast_odds,
        race_id=race_id,
        race_date=race_date,
        jcd=jcd,
        rno=rno,
        snapshot_id=snapshot_id,
        captured_at=captured_at,
        available_capital_yen=5 * STAKE_YEN,
    )
    selected = []
    for row in broad:
        if float(row["estimated_odds"]) >= MAX_FORECAST_ODDS:
            continue
        selected.append({**row, "policy_name": POLICY_NAME})
        if len(selected) >= capacity:
            break
    return tuple(selected)


def registration() -> dict[str, Any]:
    return {
        "policy_name": POLICY_NAME,
        "registered_after": REGISTERED_AFTER,
        "source_evaluation_job_id": SOURCE_EVALUATION_JOB_ID,
        "probability_rank": {"minimum": 1, "maximum": 5},
        "forecast_odds": {"minimum_inclusive": 0.0, "maximum_exclusive": 20.0},
        "estimated_ev": {"minimum_inclusive": 1.0, "maximum_inclusive": 1.05},
        "stake_yen": STAKE_YEN,
        "real_betting_enabled": False,
        "development_holdout_used_to_choose_policy": True,
        "promotion_evidence_start_date": "2026-08-02",
    }
